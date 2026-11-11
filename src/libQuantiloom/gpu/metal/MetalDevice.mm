#include "gpu/metal/MetalBackend.hpp"

#include "core/Log.hpp"
#include "core/ResultFail.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace quantiloom::gpu::metal {

namespace {

constexpr u64 kArenaChunkBytes = 4ull << 20;

// The fullscreen pass BlitTexture draws: a triangle covering the viewport, sampling the
// source rectangle at one mip level. p runs 0..1 across the viewport from its top-left.
constexpr const char* kBlitSource = R"(
#include <metal_stdlib>
using namespace metal;
struct BlitArgs { float2 uvMin; float2 uvMax; float lod; };
struct VOut { float4 position [[position]]; float2 uv; };
vertex VOut ql_blit_vertex(uint vid [[vertex_id]], constant BlitArgs& args [[buffer(0)]]) {
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    VOut out;
    out.position = float4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    out.uv = mix(args.uvMin, args.uvMax, p);
    return out;
}
fragment float4 ql_blit_fragment(VOut in [[stage_in]], texture2d<float> source [[texture(0)]],
                                 sampler s [[sampler(0)]], constant BlitArgs& args [[buffer(0)]]) {
    return source.sample(s, in.uv, level(args.lod));
}
)";

MTLSamplerMinMagFilter ToMtl(Filter filter) {
    return filter == Filter::Linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
}

MTLSamplerAddressMode ToMtl(AddressMode mode) {
    switch (mode) {
        case AddressMode::Repeat: return MTLSamplerAddressModeRepeat;
        case AddressMode::MirroredRepeat: return MTLSamplerAddressModeMirrorRepeat;
        case AddressMode::ClampToEdge: return MTLSamplerAddressModeClampToEdge;
        case AddressMode::ClampToBorder: return MTLSamplerAddressModeClampToBorderColor;
    }
    return MTLSamplerAddressModeRepeat;
}

MTLSamplerBorderColor ToMtl(BorderColor color) {
    switch (color) {
        case BorderColor::FloatTransparentBlack:
        case BorderColor::IntTransparentBlack: return MTLSamplerBorderColorTransparentBlack;
        case BorderColor::FloatOpaqueBlack:
        case BorderColor::IntOpaqueBlack: return MTLSamplerBorderColorOpaqueBlack;
        case BorderColor::FloatOpaqueWhite:
        case BorderColor::IntOpaqueWhite: return MTLSamplerBorderColorOpaqueWhite;
    }
    return MTLSamplerBorderColorTransparentBlack;
}

MTLCompareFunction ToMtl(CompareOp op) {
    switch (op) {
        case CompareOp::Never: return MTLCompareFunctionNever;
        case CompareOp::Less: return MTLCompareFunctionLess;
        case CompareOp::Equal: return MTLCompareFunctionEqual;
        case CompareOp::LessOrEqual: return MTLCompareFunctionLessEqual;
        case CompareOp::Greater: return MTLCompareFunctionGreater;
        case CompareOp::NotEqual: return MTLCompareFunctionNotEqual;
        case CompareOp::GreaterOrEqual: return MTLCompareFunctionGreaterEqual;
        case CompareOp::Always: return MTLCompareFunctionAlways;
    }
    return MTLCompareFunctionNever;
}

MTLResourceOptions ResourceOptions(MemoryClass memory) {
    // DeviceLocal is private, as VMA's GPU_ONLY is device memory the host cannot map:
    // callers stage into it through a command list, which orders the write after the
    // GPU work recorded before it. A mappable DeviceLocal buffer would let a host
    // write land under work that is recorded but not yet run. The CPU-written classes
    // are write-combined.
    switch (memory) {
        case MemoryClass::DeviceLocal: return MTLResourceStorageModePrivate;
        case MemoryClass::Upload: return MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
        case MemoryClass::Readback:
        case MemoryClass::HostOnly: return MTLResourceStorageModeShared;
    }
    return MTLResourceStorageModeShared;
}

// Nanoseconds per GPU timestamp tick, from two CPU/GPU correlations taken 20 ms apart.
// The CPU side is checked against the steady clock as well, so the answer does not depend
// on which units sampleTimestamps reports the CPU time in.
f64 MeasureTimestampPeriod(id<MTLDevice> device) {
    MTLTimestamp cpu0 = 0, gpu0 = 0, cpu1 = 0, gpu1 = 0;
    const auto wall0 = std::chrono::steady_clock::now();
    [device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    [device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
    const auto wall1 = std::chrono::steady_clock::now();
    if (gpu1 <= gpu0 || cpu1 <= cpu0) return 1.0;
    const f64 wallNs = static_cast<f64>(std::chrono::duration_cast<std::chrono::nanoseconds>(wall1 - wall0).count());
    const f64 nsPerCpuTick = wallNs / static_cast<f64>(cpu1 - cpu0);
    // Round the CPU scale to a whole tick length: it is 1 (ns) or a timebase (e.g. 41.67).
    const f64 cpuNs = static_cast<f64>(cpu1 - cpu0) * (std::abs(nsPerCpuTick - 1.0) < 0.2 ? 1.0 : nsPerCpuTick);
    return cpuNs / static_cast<f64>(gpu1 - gpu0);
}

}  // namespace

MetalDevice::MetalDevice(id<MTLDevice> device, id<MTLCommandQueue> queue) : device_(device), queue_(queue) {
    arenas_->device = device_;
    orderFence_ = [device_ newFence];
    info_.backend = Backend::Metal;
    info_.name = String([[device_ name] UTF8String]);
    info_.vendorId = 0x106B;  // Apple
    info_.deviceId = static_cast<u32>([device_ registryID] & 0xFFFFFFFFu);
    info_.maxStorageBufferRange = static_cast<u64>([device_ maxBufferLength]);
    info_.maxComputeSharedMemoryBytes = static_cast<u32>([device_ maxThreadgroupMemoryLength]);
    info_.hardwareRayTracing = [device_ supportsRaytracing];
    info_.maxBindlessTextures = 1024;
    info_.maxTraceRecursionDepth = 31;
    info_.rtScratchAlignment = 256;
    info_.timestampsOnAllQueues = [device_ supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
    info_.timestampPeriodNs = MeasureTimestampPeriod(device_);
    info_.linearFilterRGBA32Float = [device_ supports32BitFloatFiltering];
    info_.apiVersion = 0;

    MTLResidencySetDescriptor* residency = [MTLResidencySetDescriptor new];
    residency.label = @"Quantiloom acceleration structures";
    NSError* error = nil;
    residency_ = [device_ newResidencySetWithDescriptor:residency error:&error];
    if (residency_ != nil) {
        [queue_ addResidencySet:residency_];
    } else {
        QL_LOG_WARN("Metal residency set unavailable ({}): ray tracing cannot reach bottom levels",
                    NsErrorText(error));
    }

    MTLSamplerDescriptor* sampler = [MTLSamplerDescriptor new];
    sampler.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sampler.tAddressMode = MTLSamplerAddressModeClampToEdge;
    blitNearest_ = [device_ newSamplerStateWithDescriptor:sampler];
    sampler.minFilter = MTLSamplerMinMagFilterLinear;
    sampler.magFilter = MTLSamplerMinMagFilterLinear;
    blitLinear_ = [device_ newSamplerStateWithDescriptor:sampler];
}

MetalDevice::~MetalDevice() {
    (void)WaitIdle();
    if (residency_ != nil) [queue_ removeResidencySet:residency_];
}

Result<std::unique_ptr<Device>, String> MetalDevice::Create(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    if (device == nil) return Fail<std::unique_ptr<Device>>("no Metal device");
    if (queue == nil) queue = [device newCommandQueue];
    if (queue == nil) return Fail<std::unique_ptr<Device>>("cannot create a Metal command queue");
    return std::unique_ptr<Device>(new MetalDevice(device, queue));
}

Result<std::unique_ptr<Buffer>, String> MetalDevice::CreateBuffer(const BufferDesc& desc) {
    const u64 size = std::max<u64>(desc.size, 1);
    id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:ResourceOptions(desc.memory)];
    if (buffer == nil) {
        return Fail<std::unique_ptr<Buffer>>("Failed to create buffer of " + std::to_string(desc.size) + " bytes");
    }
    return std::unique_ptr<Buffer>(std::make_unique<MetalBuffer>(buffer, desc.size));
}

Result<std::unique_ptr<Texture>, String> MetalDevice::CreateTexture(const TextureDesc& desc) {
    MTLTextureDescriptor* d = [MTLTextureDescriptor new];
    d.pixelFormat = ToMtl(desc.format);
    d.width = desc.width;
    d.height = desc.height;
    d.mipmapLevelCount = std::max(1u, desc.mipLevels);
    switch (desc.dimension) {
        case TextureDimension::Tex2D:
            d.textureType = desc.layers > 1 ? MTLTextureType2DArray : MTLTextureType2D;
            d.arrayLength = std::max(1u, desc.layers);
            break;
        case TextureDimension::Tex2DArray:
            d.textureType = MTLTextureType2DArray;
            d.arrayLength = std::max(1u, desc.layers);
            break;
        case TextureDimension::Cube:
            d.textureType = desc.layers > 6 ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
            d.arrayLength = std::max(1u, desc.layers / 6);
            break;
    }
    MTLTextureUsage usage = MTLTextureUsageShaderRead;
    if (HasAny(desc.usage, TextureUsage::Storage)) usage |= MTLTextureUsageShaderWrite;
    // Render target: BlitTexture draws into its destination.
    if (!IsBlockCompressed(desc.format)) usage |= MTLTextureUsageRenderTarget;
    d.usage = usage;
    const bool hostVisible = desc.memory != MemoryClass::DeviceLocal && !IsBlockCompressed(desc.format);
    d.storageMode = hostVisible ? MTLStorageModeShared : MTLStorageModePrivate;
    id<MTLTexture> texture = [device_ newTextureWithDescriptor:d];
    if (texture == nil) {
        return Fail<std::unique_ptr<Texture>>("Failed to create image " + std::to_string(desc.width) + "x" +
                                              std::to_string(desc.height));
    }
    return std::unique_ptr<Texture>(std::make_unique<MetalTexture>(texture, desc));
}

Result<std::unique_ptr<BindingLayout>, String> MetalDevice::CreateBindingLayout(const BindingLayoutDesc& desc) {
    return MetalBindingLayout::Create(desc);
}

Result<std::unique_ptr<KernelLayout>, String> MetalDevice::CreateKernelLayout(const KernelLayoutDesc& desc) {
    return std::unique_ptr<KernelLayout>(std::make_unique<MetalKernelLayout>(desc));
}

Result<std::unique_ptr<ComputeKernel>, String> MetalDevice::CreateComputeKernel(const ComputeKernelDesc& desc) {
    return MetalComputeKernel::Create(device_, desc);
}

Result<std::unique_ptr<BindGroupPool>, String> MetalDevice::CreateBindGroupPool(const BindingLayout& layout,
                                                                                u32 maxGroups) {
    return std::unique_ptr<BindGroupPool>(
        std::make_unique<MetalBindGroupPool>(static_cast<const MetalBindingLayout&>(layout), maxGroups));
}

void MetalDevice::WriteBindGroup(BindGroup group, std::span<const BindingWrite> writes) {
    auto* record = reinterpret_cast<MetalBindGroup*>(group.handle);
    for (const BindingWrite& w : writes) {
        MetalBinding& binding = record->bindings[w.slot];
        binding.kind = w.kind;
        switch (w.kind) {
            case BindingKind::ReadOnlyBuffer:
            case BindingKind::ReadWriteBuffer:
            case BindingKind::ConstantBuffer:
                binding.buffer = FromHandle<id<MTLBuffer>>(w.buffer.handle);
                binding.offset = w.offset;
                binding.range = w.range == kWholeSize ? w.buffer.size - w.offset : w.range;
                break;
            case BindingKind::StorageTexture:
                binding.texture = FromHandle<id<MTLTexture>>(w.texture.image);
                break;
            case BindingKind::SampledTexture:
                if (w.textures.empty()) {
                    if (binding.count > 1) {
                        if (binding.textures.size() <= w.arrayElement) binding.textures.resize(w.arrayElement + 1);
                        binding.textures[w.arrayElement] = FromHandle<id<MTLTexture>>(w.texture.image);
                    } else {
                        binding.texture = FromHandle<id<MTLTexture>>(w.texture.image);
                    }
                } else {
                    if (binding.textures.size() < w.arrayElement + w.textures.size()) {
                        binding.textures.resize(w.arrayElement + w.textures.size());
                    }
                    for (size_t i = 0; i < w.textures.size(); ++i) {
                        binding.textures[w.arrayElement + i] = FromHandle<id<MTLTexture>>(w.textures[i].image);
                    }
                }
                break;
            case BindingKind::Sampler:
                if (w.samplers.empty()) {
                    if (binding.count > 1) {
                        if (binding.samplers.size() <= w.arrayElement) binding.samplers.resize(w.arrayElement + 1);
                        binding.samplers[w.arrayElement] = FromHandle<id<MTLSamplerState>>(w.sampler.handle);
                    } else {
                        binding.sampler = FromHandle<id<MTLSamplerState>>(w.sampler.handle);
                    }
                } else {
                    if (binding.samplers.size() < w.arrayElement + w.samplers.size()) {
                        binding.samplers.resize(w.arrayElement + w.samplers.size());
                    }
                    for (size_t i = 0; i < w.samplers.size(); ++i) {
                        binding.samplers[w.arrayElement + i] = FromHandle<id<MTLSamplerState>>(w.samplers[i].handle);
                    }
                }
                break;
            case BindingKind::AccelerationStructure:
                binding.scene = w.scene.handle;
                break;
        }
    }
}

void MetalDevice::ExecuteImmediate(const std::function<void(CommandList&)>& record) {
    id<MTLCommandBuffer> commandBuffer = [queue_ commandBuffer];
    if (commandBuffer == nil) throw std::runtime_error("Failed to create a Metal command buffer");
    {
        MetalCommandList list(*this, commandBuffer);
        record(list);
        list.Finish();
    }
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if ([commandBuffer status] == MTLCommandBufferStatusError) {
        throw std::runtime_error("Metal command buffer failed: " + NsErrorText([commandBuffer error]));
    }
}

Result<void, String> MetalDevice::WaitIdle() {
    return WaitQueueIdle();
}

Result<std::unique_ptr<Sampler>, String> MetalDevice::CreateSampler(const SamplerDesc& desc) {
    MTLSamplerDescriptor* d = [MTLSamplerDescriptor new];
    d.magFilter = ToMtl(desc.magFilter);
    d.minFilter = ToMtl(desc.minFilter);
    d.mipFilter = desc.mipmapMode == Filter::Linear ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
    d.sAddressMode = ToMtl(desc.addressModeU);
    d.tAddressMode = ToMtl(desc.addressModeV);
    d.rAddressMode = ToMtl(desc.addressModeW);
    d.borderColor = ToMtl(desc.borderColor);
    d.lodMinClamp = desc.minLod;
    d.lodMaxClamp = desc.maxLod;
    d.maxAnisotropy = desc.anisotropyEnable
                          ? static_cast<NSUInteger>(std::clamp(desc.maxAnisotropy, 1.0f, 16.0f))
                          : 1;
    if (desc.compareEnable) d.compareFunction = ToMtl(desc.compareOp);
    d.normalizedCoordinates = !desc.unnormalizedCoordinates;
    d.supportArgumentBuffers = YES;
    id<MTLSamplerState> sampler = [device_ newSamplerStateWithDescriptor:d];
    if (sampler == nil) return Fail<std::unique_ptr<Sampler>>("cannot create a Metal sampler");
    return std::unique_ptr<Sampler>(std::make_unique<MetalSampler>(sampler));
}

Result<std::unique_ptr<Event>, String> MetalDevice::CreateGpuEvent() {
    id<MTLSharedEvent> event = [device_ newSharedEvent];
    if (event == nil) return Fail<std::unique_ptr<Event>>("cannot create a Metal shared event");
    return std::unique_ptr<Event>(std::make_unique<MetalEvent>(event));
}

Result<std::unique_ptr<TimestampPool>, String> MetalDevice::CreateTimestampPool(u32 count) {
    if (!info_.timestampsOnAllQueues) {
        return Fail<std::unique_ptr<TimestampPool>>("this Metal device cannot sample timestamps at encoder boundaries");
    }
    id<MTLCounterSet> timestamps = nil;
    for (id<MTLCounterSet> set in [device_ counterSets]) {
        if ([[set name] isEqualToString:MTLCommonCounterSetTimestamp]) timestamps = set;
    }
    if (timestamps == nil) return Fail<std::unique_ptr<TimestampPool>>("this Metal device has no timestamp counters");
    MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
    d.counterSet = timestamps;
    d.sampleCount = count;
    d.storageMode = MTLStorageModeShared;
    NSError* error = nil;
    id<MTLCounterSampleBuffer> samples = [device_ newCounterSampleBufferWithDescriptor:d error:&error];
    if (samples == nil) {
        return Fail<std::unique_ptr<TimestampPool>>("cannot create a timestamp pool: " + NsErrorText(error));
    }
    return std::unique_ptr<TimestampPool>(std::make_unique<MetalTimestampPool>(samples, count));
}

Result<std::unique_ptr<CommandPool>, String> MetalDevice::CreateCommandPool(const CommandPoolDesc&) {
    return std::unique_ptr<CommandPool>(std::make_unique<MetalCommandPool>(*this));
}

Result<std::unique_ptr<Fence>, String> MetalDevice::CreateFence() {
    id<MTLSharedEvent> event = [device_ newSharedEvent];
    if (event == nil) return Fail<std::unique_ptr<Fence>>("cannot create a Metal shared event");
    return std::unique_ptr<Fence>(std::make_unique<MetalFence>(event));
}

Result<void, String> MetalDevice::Submit(CommandBuffer& commandBuffer, Fence* signal) {
    id<MTLCommandBuffer> native = static_cast<MetalCommandBuffer&>(commandBuffer).Native();
    if (native == nil) return Fail<void>("submitted a command buffer that was never begun");
    if (signal != nullptr) {
        auto* fence = static_cast<MetalFence*>(signal);
        [native encodeSignalEvent:fence->Native() value:fence->Target()];
    }
    [native commit];
    return Result<void, String>::Ok();
}

Result<void, String> MetalDevice::WaitQueueIdle() {
    // A queue runs its command buffers in commit order, so an empty one committed now
    // completes after everything committed before it.
    id<MTLCommandBuffer> marker = [queue_ commandBuffer];
    if (marker == nil) return Fail<void>("cannot create a Metal command buffer");
    [marker commit];
    [marker waitUntilCompleted];
    if ([marker status] == MTLCommandBufferStatusError) return Fail<void>(NsErrorText([marker error]));
    return Result<void, String>::Ok();
}

std::unique_ptr<CommandList> MetalDevice::WrapRecording(NativeHandle commandBuffer) {
    return std::make_unique<MetalCommandList>(*this, FromHandle<id<MTLCommandBuffer>>(commandBuffer));
}

TextureRef MetalDevice::ImportHostTexture(NativeHandle image, Format format, u32 width, u32 height) {
    TextureRef ref;
    ref.image = image;
    ref.view = 0;
    ref.format = format;
    ref.width = width;
    ref.height = height;
    ref.layers = 1;
    ref.layout = TextureLayoutPolicy::Optimal;
    return ref;
}

Result<std::unique_ptr<PipelineCache>, String> MetalDevice::CreatePipelineCache(
    std::span<const std::byte> initialData) {
    return MetalPipelineCache::Create(device_, initialData);
}

Result<std::unique_ptr<TraceKernel>, String> MetalDevice::CreateTraceKernel(const TraceKernelDesc& desc) {
    return MetalTraceKernel::Create(device_, desc);
}

u64 MetalDevice::BufferDeviceAddress(BufferRef buffer) const {
    if (!buffer.IsValid()) return 0;
    return [FromHandle<id<MTLBuffer>>(buffer.handle) gpuAddress];
}

void MetalDevice::RegisterAccelerationStructure(MetalRtAccelerationStructure* structure) {
    std::lock_guard lock(mutex_);
    structures_[ToHandle(structure->Native())] = structure;
    if (residency_ != nil) {
        [residency_ addAllocation:structure->Native()];
        [residency_ commit];
    }
}

void MetalDevice::UnregisterAccelerationStructure(MetalRtAccelerationStructure* structure) {
    std::lock_guard lock(mutex_);
    structures_.erase(ToHandle(structure->Native()));
    if (residency_ != nil) {
        [residency_ removeAllocation:structure->Native()];
        [residency_ commit];
    }
}

MetalRtAccelerationStructure* MetalDevice::FindAccelerationStructure(NativeHandle handle) const {
    std::lock_guard lock(mutex_);
    const auto it = structures_.find(handle);
    return it == structures_.end() ? nullptr : it->second;
}

id<MTLBuffer> ArenaPool::Take(u64 minBytes) {
    {
        std::lock_guard lock(mutex);
        for (auto it = free.begin(); it != free.end(); ++it) {
            if ([*it length] >= minBytes) {
                id<MTLBuffer> chunk = *it;
                free.erase(it);
                return chunk;
            }
        }
    }
    return [device newBufferWithLength:std::max(minBytes, kArenaChunkBytes)
                               options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
}

void ArenaPool::Give(id<MTLBuffer> chunk) {
    std::lock_guard lock(mutex);
    if (free.size() < 64) free.push_back(chunk);
}

id<MTLRenderPipelineState> MetalDevice::BlitPipeline(MTLPixelFormat format) {
    std::lock_guard lock(mutex_);
    const auto key = static_cast<u64>(format);
    if (const auto it = blitPipelines_.find(key); it != blitPipelines_.end()) return it->second;
    NSError* error = nil;
    if (blitLibrary_ == nil) {
        blitLibrary_ = [device_ newLibraryWithSource:[NSString stringWithUTF8String:kBlitSource]
                                             options:nil
                                               error:&error];
        if (blitLibrary_ == nil) {
            QL_LOG_ERROR("Metal blit shaders failed to compile: {}", NsErrorText(error));
            return nil;
        }
    }
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [blitLibrary_ newFunctionWithName:@"ql_blit_vertex"];
    d.fragmentFunction = [blitLibrary_ newFunctionWithName:@"ql_blit_fragment"];
    d.colorAttachments[0].pixelFormat = format;
    id<MTLRenderPipelineState> pipeline = [device_ newRenderPipelineStateWithDescriptor:d error:&error];
    if (pipeline == nil) {
        QL_LOG_ERROR("Metal blit pipeline for format {} failed: {}", static_cast<u64>(format), NsErrorText(error));
        return nil;
    }
    blitPipelines_[key] = pipeline;
    return pipeline;
}

}  // namespace quantiloom::gpu::metal
