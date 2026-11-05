#include "gpu/metal/MetalBackend.hpp"

#include "core/Log.hpp"
#include "core/ResultFail.hpp"
#include "gpu/ShaderLibrary.hpp"

#include <cstring>
#include <fstream>
#include <unordered_set>

namespace quantiloom::gpu::metal {

namespace {

id<MTLLibrary> NewLibrary(id<MTLDevice> device, const ShaderCode& code, String& error) {
    dispatch_data_t data = dispatch_data_create(code.bytes.data(), code.bytes.size(), nullptr,
                                                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError* nsError = nil;
    id<MTLLibrary> library = [device newLibraryWithData:data error:&nsError];
    if (library == nil) {
        error = "cannot load " + code.path.string() + ": " + NsErrorText(nsError);
    }
    return library;
}

}  // namespace

void MetalBuffer::Upload(const void* data, u64 bytes, u64 offset) {
    if (offset > size_ || bytes > size_ - offset) {
        QL_LOG_ERROR("Buffer upload of {} bytes at offset {} overflows a {}-byte buffer", bytes, offset, size_);
        return;
    }
    if ([buffer_ contents] == nullptr) {
        QL_LOG_ERROR("Buffer upload into device-local memory the host cannot map ({} bytes)", bytes);
        return;
    }
    std::memcpy(static_cast<std::byte*>([buffer_ contents]) + offset, data, bytes);
}

TextureRef MetalTexture::Ref() const {
    TextureRef ref;
    ref.image = ToHandle(texture_);
    ref.view = 0;
    ref.format = desc_.format;
    ref.width = desc_.width;
    ref.height = desc_.height;
    ref.layers = desc_.layers;
    ref.layout = HasAny(desc_.usage, TextureUsage::Storage) ? TextureLayoutPolicy::General
                                                            : TextureLayoutPolicy::Optimal;
    return ref;
}

Result<std::unique_ptr<BindingLayout>, String> MetalBindingLayout::Create(const BindingLayoutDesc& desc) {
    std::unordered_set<u32> seen;
    for (const BindingSlot& slot : desc.slots) {
        if (!seen.insert(slot.slot).second) {
            return Fail<std::unique_ptr<BindingLayout>>("binding layout space " + std::to_string(desc.space) +
                                                        " declares slot " + std::to_string(slot.slot) + " twice");
        }
    }
    auto layout = std::unique_ptr<MetalBindingLayout>(new MetalBindingLayout());
    layout->space_ = desc.space;
    layout->slots_.assign(desc.slots.begin(), desc.slots.end());
    return std::unique_ptr<BindingLayout>(std::move(layout));
}

MetalKernelLayout::MetalKernelLayout(const KernelLayoutDesc& desc)
    : spaces_(desc.spaces.begin(), desc.spaces.end()),
      pushBytes_(desc.pushConstantBytes),
      pushVisibility_(desc.pushConstantVisibility) {}

Result<Vector<BindGroup>, String> MetalBindGroupPool::Allocate(u32 count) {
    if (groups_.size() + count > maxGroups_) {
        return Fail<Vector<BindGroup>>("bind group pool exhausted: " + std::to_string(groups_.size()) + " of " +
                                       std::to_string(maxGroups_) + " in use, " + std::to_string(count) +
                                       " more asked for");
    }
    Vector<BindGroup> groups;
    groups.reserve(count);
    for (u32 i = 0; i < count; ++i) {
        auto group = std::make_unique<MetalBindGroup>();
        group->layout = &layout_;
        for (const BindingSlot& slot : layout_.Slots()) {
            MetalBinding& binding = group->bindings[slot.slot];
            binding.kind = slot.kind;
            binding.count = slot.count;
            if (slot.count > 1) {
                if (slot.kind == BindingKind::Sampler) {
                    binding.samplers.resize(slot.count);
                } else {
                    binding.textures.resize(slot.count);
                }
            }
        }
        groups.push_back(BindGroup{reinterpret_cast<NativeHandle>(group.get())});
        groups_.push_back(std::move(group));
    }
    return groups;
}

Result<std::unique_ptr<ComputeKernel>, String> MetalComputeKernel::Create(id<MTLDevice> device,
                                                                          const ComputeKernelDesc& desc) {
    using CreateResult = std::unique_ptr<ComputeKernel>;
    if (desc.code == nullptr || desc.layout == nullptr) {
        return Fail<CreateResult>("compute kernel needs code and a layout");
    }
    auto reflection = LoadReflection(desc.code->path);
    if (!reflection) return Fail<CreateResult>(reflection.error());

    String error;
    id<MTLLibrary> library = NewLibrary(device, *desc.code, error);
    if (library == nil) return Fail<CreateResult>(error);
    id<MTLFunction> function =
        [library newFunctionWithName:[NSString stringWithUTF8String:reflection.value().entryPoint.c_str()]];
    if (function == nil) {
        return Fail<CreateResult>("no function '" + reflection.value().entryPoint + "' in " +
                                  desc.code->path.string());
    }
    NSError* nsError = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&nsError];
    if (pipeline == nil) {
        return Fail<CreateResult>("compute pipeline " + desc.code->name + ": " + NsErrorText(nsError));
    }
    auto kernel = std::unique_ptr<MetalComputeKernel>(new MetalComputeKernel());
    kernel->pipeline_ = pipeline;
    kernel->reflection_ = std::move(reflection).value();
    kernel->layout_ = desc.layout;
    kernel->name_ = desc.code->name;
    return CreateResult(std::move(kernel));
}

Result<std::unique_ptr<PipelineCache>, String> MetalPipelineCache::Create(id<MTLDevice> device,
                                                                          std::span<const std::byte> initialData) {
    using CreateResult = std::unique_ptr<PipelineCache>;
    auto cache = std::unique_ptr<MetalPipelineCache>(new MetalPipelineCache());
    MTLBinaryArchiveDescriptor* descriptor = [MTLBinaryArchiveDescriptor new];
    NSError* nsError = nil;
    if (!initialData.empty()) {
        // A binary archive opens from a URL only, so the bytes go through a temporary file.
        const std::filesystem::path path =
            std::filesystem::temp_directory_path() /
            ("ql_pipeline_cache_" + std::to_string(reinterpret_cast<uintptr_t>(cache.get())) + ".metallib");
        {
            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(initialData.data()),
                      static_cast<std::streamsize>(initialData.size()));
        }
        descriptor.url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
        cache->archive_ = [device newBinaryArchiveWithDescriptor:descriptor error:&nsError];
        std::error_code ec;
        std::filesystem::remove(path, ec);
        if (cache->archive_ == nil) {
            QL_LOG_WARN("Pipeline cache rejected ({}); starting an empty one", NsErrorText(nsError));
        }
    }
    if (cache->archive_ == nil) {
        descriptor.url = nil;
        cache->archive_ = [device newBinaryArchiveWithDescriptor:descriptor error:&nsError];
        if (cache->archive_ == nil) {
            return Fail<CreateResult>("cannot create a pipeline cache: " + NsErrorText(nsError));
        }
    }
    return CreateResult(std::move(cache));
}

Result<Vector<std::byte>, String> MetalPipelineCache::Serialize() const {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("ql_pipeline_cache_out_" + std::to_string(reinterpret_cast<uintptr_t>(this)) + ".metallib");
    NSError* nsError = nil;
    if (![archive_ serializeToURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]]
                            error:&nsError]) {
        return Fail<Vector<std::byte>>("pipeline cache serialization failed: " + NsErrorText(nsError));
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto size = file.is_open() ? static_cast<size_t>(file.tellg()) : 0;
    Vector<std::byte> bytes(size);
    if (size != 0) {
        file.seekg(0);
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    }
    file.close();
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (bytes.empty()) return Fail<Vector<std::byte>>("pipeline cache serialized to zero bytes");
    return bytes;
}

Result<std::unique_ptr<TraceKernel>, String> MetalTraceKernel::Create(id<MTLDevice> device,
                                                                      const TraceKernelDesc& desc) {
    using CreateResult = std::unique_ptr<TraceKernel>;
    if (desc.rayGen == nullptr || desc.layout == nullptr) {
        return Fail<CreateResult>("trace kernel needs a ray generation shader and a layout");
    }
    auto reflection = LoadReflection(desc.rayGen->path);
    if (!reflection) return Fail<CreateResult>(reflection.error());

    String error;
    id<MTLLibrary> library = NewLibrary(device, *desc.rayGen, error);
    if (library == nil) return Fail<CreateResult>(error);

    // Every function constant the converted raygen declares gets a value: the one the
    // caller specialised, or 0 for one it did not name (there is no default to fall back
    // on once a constant is a Metal function constant; ql_bind.hlsli).
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    for (const FunctionConstantInfo& constant : reflection.value().functionConstants) {
        u32 value = 0;
        bool named = false;
        for (const SpecializationConstant& s : desc.specialization) {
            if (s.id == constant.id) {
                value = s.value;
                named = true;
            }
        }
        if (!named) {
            return Fail<CreateResult>("trace kernel " + desc.rayGen->name + " declares function constant " +
                                      constant.name + ", which the caller did not specialise");
        }
        // Always uint4, whatever the reflection reports: MSC declares each constant
        // as a four-lane vector and its manual requires MTLDataTypeUInt4 here.
        const std::array<u32, 4> lanes{value, 0, 0, 0};
        [values setConstantValue:lanes.data()
                            type:MTLDataTypeUInt4
                        withName:[NSString stringWithUTF8String:constant.name.c_str()]];
    }
    NSError* nsError = nil;
    id<MTLFunction> function =
        [library newFunctionWithName:[NSString stringWithUTF8String:reflection.value().entryPoint.c_str()]
                      constantValues:values
                               error:&nsError];
    if (function == nil) {
        return Fail<CreateResult>("ray generation function of " + desc.rayGen->path.string() + ": " +
                                  NsErrorText(nsError));
    }

    MTLComputePipelineDescriptor* pipelineDesc = [MTLComputePipelineDescriptor new];
    pipelineDesc.computeFunction = function;
    const auto* cache = static_cast<const MetalPipelineCache*>(desc.cache);
    if (cache != nullptr) pipelineDesc.binaryArchives = @[ cache->Archive() ];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pipelineDesc
                                                                                 options:MTLPipelineOptionNone
                                                                              reflection:nil
                                                                                   error:&nsError];
    if (pipeline == nil) {
        return Fail<CreateResult>("ray tracing pipeline " + desc.rayGen->name + ": " + NsErrorText(nsError));
    }
    if (cache != nullptr) {
        NSError* addError = nil;
        if (![cache->Archive() addComputePipelineFunctionsWithDescriptor:pipelineDesc error:&addError]) {
            QL_LOG_WARN("Pipeline cache did not take {}: {}", desc.rayGen->name, NsErrorText(addError));
        }
    }

    auto kernel = std::unique_ptr<MetalTraceKernel>(new MetalTraceKernel());
    kernel->pipeline_ = pipeline;
    kernel->reflection_ = std::move(reflection).value();
    kernel->layout_ = desc.layout;
    kernel->name_ = desc.name.empty() ? desc.rayGen->name : String(desc.name);
    return CreateResult(std::move(kernel));
}

}  // namespace quantiloom::gpu::metal
