#include "gpu/metal/MetalBackend.hpp"

#include "core/Log.hpp"

#include <metal_irconverter_runtime/metal_irconverter_runtime.h>

#include <algorithm>
#include <cstring>

namespace quantiloom::gpu::metal {

namespace {

u64 AlignUp(u64 value, u64 alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

size_t KindIndex(PipelineKind kind) {
    return kind == PipelineKind::Trace ? 1 : 0;
}

bool IsSrvKind(BindingKind kind) {
    return kind == BindingKind::ReadOnlyBuffer || kind == BindingKind::SampledTexture ||
           kind == BindingKind::AccelerationStructure;
}

bool IsUavKind(BindingKind kind) {
    return kind == BindingKind::ReadWriteBuffer || kind == BindingKind::StorageTexture;
}

bool Matches(ArgumentSlot::Type type, BindingKind kind) {
    switch (type) {
        case ArgumentSlot::Type::Srv: return IsSrvKind(kind);
        case ArgumentSlot::Type::Uav: return IsUavKind(kind);
        case ArgumentSlot::Type::Cbv: return kind == BindingKind::ConstantBuffer;
        case ArgumentSlot::Type::Sampler: return kind == BindingKind::Sampler;
        case ArgumentSlot::Type::Table: return false;
    }
    return false;
}

/// Everything one dispatch's argument buffer names, made resident and hazard-tracked with
/// useResources before the dispatch.
struct Residency {
    Vector<id<MTLResource>> read;
    Vector<id<MTLResource>> readWrite;

    void Add(id<MTLResource> resource, bool writes) {
        if (resource == nil) return;
        (writes ? readWrite : read).push_back(resource);
    }

    void Apply(id<MTLComputeCommandEncoder> encoder) const {
        if (!read.empty()) {
            [encoder useResources:read.data() count:read.size() usage:MTLResourceUsageRead];
        }
        if (!readWrite.empty()) {
            [encoder useResources:readWrite.data()
                            count:readWrite.size()
                            usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        }
    }
};

}  // namespace

// ---- UploadArena ---------------------------------------------------------------------------

UploadArena::~UploadArena() {
    for (id<MTLBuffer> chunk : chunks_) pool_->Give(chunk);
}

UploadArena::Slice UploadArena::Allocate(u64 bytes, u64 alignment) {
    bytes = std::max<u64>(bytes, 1);
    u64 offset = AlignUp(used_, alignment);
    if (chunks_.empty() || offset + bytes > [chunks_.back() length]) {
        chunks_.push_back(pool_->Take(bytes));
        offset = 0;
    }
    used_ = offset + bytes;
    id<MTLBuffer> chunk = chunks_.back();
    Slice slice;
    slice.buffer = chunk;
    slice.offset = offset;
    slice.cpu = static_cast<std::byte*>([chunk contents]) + offset;
    slice.gpuAddress = [chunk gpuAddress] + offset;
    return slice;
}

void UploadArena::RetireWith(id<MTLCommandBuffer> commandBuffer) {
    if (chunks_.empty()) return;
    Vector<id<MTLBuffer>> chunks = std::move(chunks_);
    chunks_.clear();
    used_ = 0;
    std::shared_ptr<ArenaPool> pool = pool_;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
        for (id<MTLBuffer> chunk : chunks) pool->Give(chunk);
    }];
}

// ---- MetalCommandList ----------------------------------------------------------------------

MetalCommandList::MetalCommandList(MetalDevice& device, id<MTLCommandBuffer> commandBuffer)
    : device_(device), commandBuffer_(commandBuffer), arena_(device.Arenas()) {}

MetalCommandList::~MetalCommandList() {
    if (!finished_) Finish();
}

void MetalCommandList::Finish() {
    EndEncoder();
    arena_.RetireWith(commandBuffer_);
    finished_ = true;
}

void MetalCommandList::EndEncoder() {
    id<MTLFence> fence = device_.OrderFence();
    if (compute_ != nil) {
        [compute_ updateFence:fence];
        [compute_ endEncoding];
        compute_ = nil;
    }
    if (blit_ != nil) {
        [blit_ updateFence:fence];
        [blit_ endEncoding];
        blit_ = nil;
    }
    if (accel_ != nil) {
        [accel_ updateFence:fence];
        [accel_ endEncoding];
        accel_ = nil;
    }
}

void MetalCommandList::Ordered(id<MTLCommandEncoder> encoder) {
    // Hazard tracking orders what an encoder declares; a TLAS reaches its bottom
    // levels by resource ID, which nothing declares, so a build or a trace could
    // overlap the BLAS build before it. One fence threaded through every encoder
    // makes recording order execution order, as a Vulkan pipeline barrier does.
    if ([encoder conformsToProtocol:@protocol(MTLComputeCommandEncoder)]) {
        [(id<MTLComputeCommandEncoder>)encoder waitForFence:device_.OrderFence()];
    } else if ([encoder conformsToProtocol:@protocol(MTLBlitCommandEncoder)]) {
        [(id<MTLBlitCommandEncoder>)encoder waitForFence:device_.OrderFence()];
    } else if ([encoder conformsToProtocol:@protocol(MTLAccelerationStructureCommandEncoder)]) {
        [(id<MTLAccelerationStructureCommandEncoder>)encoder waitForFence:device_.OrderFence()];
    }
}

id<MTLComputeCommandEncoder> MetalCommandList::Compute() {
    if (compute_ == nil) {
        EndEncoder();
        compute_ = [commandBuffer_ computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        Ordered(compute_);
    }
    return compute_;
}

id<MTLBlitCommandEncoder> MetalCommandList::Blit() {
    if (blit_ == nil) {
        EndEncoder();
        blit_ = [commandBuffer_ blitCommandEncoder];
        Ordered(blit_);
    }
    return blit_;
}

id<MTLAccelerationStructureCommandEncoder> MetalCommandList::Accel() {
    if (accel_ == nil) {
        EndEncoder();
        accel_ = [commandBuffer_ accelerationStructureCommandEncoder];
        Ordered(accel_);
    }
    return accel_;
}

void MetalCommandList::SetKernel(const ComputeKernel& kernel) {
    kernel_ = static_cast<const MetalComputeKernel*>(&kernel);
}

void MetalCommandList::SetBindGroup(const KernelLayout&, u32 space, BindGroup group, PipelineKind kind) {
    auto& groups = groups_[KindIndex(kind)];
    if (groups.size() <= space) groups.resize(space + 1, nullptr);
    groups[space] = reinterpret_cast<const MetalBindGroup*>(group.handle);
}

void MetalCommandList::PushConstants(const KernelLayout&, std::span<const std::byte> bytes) {
    push_.assign(bytes.begin(), bytes.end());
}

u64 MetalCommandList::EncodeArguments(const ShaderReflection& reflection, PipelineKind kind) {
    const auto& groups = groups_[KindIndex(kind)];
    const UploadArena::Slice top = arena_.Allocate(std::max<u64>(reflection.topLevelBytes, 16));
    std::memset(top.cpu, 0, std::max<u64>(reflection.topLevelBytes, 16));
    Residency residency;

    auto pushAddress = [&]() -> u64 {
        if (push_.empty()) return 0;
        const UploadArena::Slice slice = arena_.Allocate(AlignUp(push_.size(), 16));
        std::memcpy(slice.cpu, push_.data(), push_.size());
        return slice.gpuAddress;
    };

    auto encode = [&](IRDescriptorTableEntry* entry, const MetalBinding& binding) {
        switch (binding.kind) {
            case BindingKind::ReadOnlyBuffer:
            case BindingKind::ReadWriteBuffer:
            case BindingKind::ConstantBuffer:
                if (binding.buffer == nil) return;
                IRDescriptorTableSetBuffer(entry, [binding.buffer gpuAddress] + binding.offset,
                                           std::min<u64>(binding.range, 0xFFFFFFFFull));
                residency.Add(binding.buffer, binding.kind == BindingKind::ReadWriteBuffer);
                return;
            case BindingKind::StorageTexture:
            case BindingKind::SampledTexture:
                if (binding.texture == nil) return;
                IRDescriptorTableSetTexture(entry, binding.texture, 0.0f, 0);
                residency.Add(binding.texture, binding.kind == BindingKind::StorageTexture);
                return;
            case BindingKind::Sampler:
                if (binding.sampler == nil) return;
                IRDescriptorTableSetSampler(entry, binding.sampler, 0.0f);
                return;
            case BindingKind::AccelerationStructure: {
                MetalRtAccelerationStructure* structure = device_.FindAccelerationStructure(binding.scene);
                if (structure == nullptr || structure->Header() == nil) {
                    QL_LOG_ERROR("Metal: a bind group names an acceleration structure that is not a built top level");
                    return;
                }
                IRDescriptorTableSetAccelerationStructure(entry, [structure->Header() gpuAddress]);
                residency.Add(structure->Native(), false);
                residency.Add(structure->Header(), false);
                return;
            }
        }
    };

    if (!reflection.rootSignature) {
        // MSC's linear layout: one descriptor per resource, named by register and space.
        for (const ArgumentSlot& slot : reflection.topLevel) {
            auto* entry = reinterpret_cast<IRDescriptorTableEntry*>(top.cpu + slot.offset);
            if (slot.type == ArgumentSlot::Type::Cbv && slot.space == 1 && slot.slot == 0) {
                IRDescriptorTableSetBuffer(entry, pushAddress(), push_.size());
                continue;
            }
            if (slot.space >= groups.size() || groups[slot.space] == nullptr) continue;
            const auto& bindings = groups[slot.space]->bindings;
            const auto it = bindings.find(slot.slot);
            if (it == bindings.end() || !Matches(slot.type, it->second.kind)) continue;
            encode(entry, it->second);
        }
    } else {
        // QL_ROOT_SIGNATURE: four descriptor tables, then the push constants as a root CBV.
        u32 srvUav = 0, samplers = 0, bindlessTextures = 0, bindlessSamplers = 0;
        const MetalBinding* textureArray = nullptr;
        const MetalBinding* samplerArray = nullptr;
        const MetalBindGroup* space0 = groups.empty() ? nullptr : groups[0];
        if (space0 != nullptr) {
            for (const auto& [slot, binding] : space0->bindings) {
                if (binding.count > 1) {
                    if (binding.kind == BindingKind::Sampler) {
                        samplerArray = &binding;
                        bindlessSamplers = static_cast<u32>(binding.samplers.size());
                    } else {
                        textureArray = &binding;
                        bindlessTextures = static_cast<u32>(binding.textures.size());
                    }
                }
            }
            srvUav = kRootSrvCount + kRootUavCount;
            samplers = kRootSamplerCount;
        }
        auto table = [&](u32 entries) -> UploadArena::Slice {
            const UploadArena::Slice slice = arena_.Allocate(std::max<u64>(entries, 1) * sizeof(IRDescriptorTableEntry));
            std::memset(slice.cpu, 0, std::max<u64>(entries, 1) * sizeof(IRDescriptorTableEntry));
            return slice;
        };
        const UploadArena::Slice t0 = table(srvUav);
        const UploadArena::Slice t1 = table(samplers);
        const UploadArena::Slice t2 = table(bindlessTextures);
        const UploadArena::Slice t3 = table(bindlessSamplers);
        auto* e0 = reinterpret_cast<IRDescriptorTableEntry*>(t0.cpu);
        auto* e1 = reinterpret_cast<IRDescriptorTableEntry*>(t1.cpu);
        auto* e2 = reinterpret_cast<IRDescriptorTableEntry*>(t2.cpu);
        auto* e3 = reinterpret_cast<IRDescriptorTableEntry*>(t3.cpu);
        if (space0 != nullptr) {
            for (const auto& [slot, binding] : space0->bindings) {
                if (binding.count > 1) continue;
                if (IsSrvKind(binding.kind) && slot < kRootSrvCount) {
                    encode(e0 + slot, binding);
                } else if (IsUavKind(binding.kind) && slot < kRootUavCount) {
                    encode(e0 + kRootSrvCount + slot, binding);
                } else if (binding.kind == BindingKind::Sampler && slot < kRootSamplerCount) {
                    encode(e1 + slot, binding);
                } else {
                    QL_LOG_ERROR("Metal: slot {} of space 0 is outside QL_ROOT_SIGNATURE's ranges", slot);
                }
            }
        }
        if (textureArray != nullptr) {
            for (u32 i = 0; i < bindlessTextures; ++i) {
                id<MTLTexture> texture = textureArray->textures[i];
                if (texture == nil) continue;
                IRDescriptorTableSetTexture(e2 + i, texture, 0.0f, 0);
                residency.Add(texture, false);
            }
        }
        if (samplerArray != nullptr) {
            for (u32 i = 0; i < bindlessSamplers; ++i) {
                id<MTLSamplerState> sampler = samplerArray->samplers[i];
                if (sampler != nil) IRDescriptorTableSetSampler(e3 + i, sampler, 0.0f);
            }
        }
        const u64 tables[4] = {t0.gpuAddress, t1.gpuAddress, t2.gpuAddress, t3.gpuAddress};
        u32 tableIndex = 0;
        for (const ArgumentSlot& slot : reflection.topLevel) {
            auto* root = reinterpret_cast<u64*>(top.cpu + slot.offset);
            if (slot.type == ArgumentSlot::Type::Table) {
                // A fifth table is the function-constant space, which holds nothing.
                if (tableIndex < 4) *root = tables[tableIndex];
                ++tableIndex;
            } else if (slot.type == ArgumentSlot::Type::Cbv && slot.space == 1 && slot.slot == 0) {
                *root = pushAddress();
            }
        }
    }

    residency.Apply(compute_);
    [compute_ setBuffer:top.buffer offset:top.offset atIndex:kIRArgumentBufferBindPoint];
    return top.gpuAddress;
}

void MetalCommandList::Dispatch(u32 groupsX, u32 groupsY, u32 groupsZ) {
    if (kernel_ == nullptr) {
        QL_LOG_ERROR("Metal: Dispatch with no kernel set");
        return;
    }
    id<MTLComputeCommandEncoder> encoder = Compute();
    [encoder setComputePipelineState:kernel_->Pipeline()];
    (void)EncodeArguments(kernel_->Reflection(), PipelineKind::Compute);
    const auto& tg = kernel_->Reflection().threadgroup;
    [encoder dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, groupsZ)
            threadsPerThreadgroup:MTLSizeMake(tg[0], tg[1], tg[2])];
}

void MetalCommandList::SetTraceKernel(const TraceKernel& kernel) {
    trace_ = static_cast<const MetalTraceKernel*>(&kernel);
}

void MetalCommandList::Trace(const TraceKernel& kernel, u32 width, u32 height, u32 depth) {
    trace_ = static_cast<const MetalTraceKernel*>(&kernel);
    id<MTLComputeCommandEncoder> encoder = Compute();
    [encoder setComputePipelineState:trace_->Pipeline()];
    const u64 topLevel = EncodeArguments(trace_->Reflection(), PipelineKind::Trace);

    // Ray generation converted as a kernel reads its launch from the dispatch-rays
    // argument: the grid it reports as DispatchRaysDimensions, and the global root
    // signature. There is no shader binding table: every ray is inline (Trace.hpp).
    IRDispatchRaysArgument arguments{};
    arguments.DispatchRaysDesc.Width = width;
    arguments.DispatchRaysDesc.Height = height;
    arguments.DispatchRaysDesc.Depth = depth;
    arguments.GRS = topLevel;
    [encoder setBytes:&arguments length:sizeof(arguments) atIndex:kIRRayDispatchArgumentsBindPoint];

    id<MTLComputePipelineState> pipeline = trace_->Pipeline();
    const NSUInteger maxThreads = [pipeline maxTotalThreadsPerThreadgroup];
    const MTLSize group = maxThreads >= 64 ? MTLSizeMake(8, 8, 1) : MTLSizeMake([pipeline threadExecutionWidth], 1, 1);
    [encoder dispatchThreads:MTLSizeMake(width, height, depth) threadsPerThreadgroup:group];
}

void MetalCommandList::BuildRtAccelerationStructure(const RtBuild& build) {
    MTLAccelerationStructureDescriptor* descriptor = device_.DescribeGeometry(build.geometry);
    id<MTLAccelerationStructure> dst = FromHandle<id<MTLAccelerationStructure>>(build.dst.handle);
    id<MTLBuffer> scratch = FromHandle<id<MTLBuffer>>(build.scratch.handle);
    if (build.geometry.level == RtLevel::Top) {
        if (MetalRtAccelerationStructure* top = device_.FindAccelerationStructure(build.dst.handle)) {
            top->PrepareHeader(build.geometry.instances.count);
        }
    }
    id<MTLAccelerationStructureCommandEncoder> encoder = Accel();
    if (build.mode == RtBuildMode::Build) {
        [encoder buildAccelerationStructure:dst descriptor:descriptor scratchBuffer:scratch scratchBufferOffset:0];
    } else {
        id<MTLAccelerationStructure> src = FromHandle<id<MTLAccelerationStructure>>(build.src.handle);
        [encoder refitAccelerationStructure:src
                                 descriptor:descriptor
                                destination:(src == dst ? nil : dst)
                              scratchBuffer:scratch
                        scratchBufferOffset:0];
    }
}

void MetalCommandList::Barrier(const BarrierBatch&) {
    // Every resource is hazard-tracked and every resource an argument buffer names is
    // declared with useResources, so ending the encoder orders what follows after what
    // came before. Layouts, stages and host layouts have no Metal meaning.
    EndEncoder();
}

void MetalCommandList::FillBuffer(BufferRef dst, u64 offset, u64 size, u32 value) {
    id<MTLBuffer> buffer = FromHandle<id<MTLBuffer>>(dst.handle);
    if (size == kWholeSize) size = dst.size - offset;
    const u8 byte = static_cast<u8>(value & 0xFF);
    if (value == byte * 0x01010101u) {
        [Blit() fillBuffer:buffer range:NSMakeRange(offset, size) value:byte];
        return;
    }
    // Metal fills bytes, Vulkan words: a word pattern is staged and copied.
    const UploadArena::Slice slice = arena_.Allocate(size, 16);
    for (u64 i = 0; i + 4 <= size; i += 4) std::memcpy(slice.cpu + i, &value, 4);
    [Blit() copyFromBuffer:slice.buffer sourceOffset:slice.offset toBuffer:buffer destinationOffset:offset size:size];
}

void MetalCommandList::UpdateBuffer(BufferRef dst, u64 offset, std::span<const std::byte> bytes) {
    if (bytes.empty()) return;
    const UploadArena::Slice slice = arena_.Allocate(bytes.size(), 16);
    std::memcpy(slice.cpu, bytes.data(), bytes.size());
    [Blit() copyFromBuffer:slice.buffer
              sourceOffset:slice.offset
                  toBuffer:FromHandle<id<MTLBuffer>>(dst.handle)
         destinationOffset:offset
                      size:bytes.size()];
}

void MetalCommandList::CopyBuffer(BufferRef src, BufferRef dst, std::span<const BufferCopyRegion> regions) {
    id<MTLBlitCommandEncoder> encoder = Blit();
    for (const BufferCopyRegion& region : regions) {
        [encoder copyFromBuffer:FromHandle<id<MTLBuffer>>(src.handle)
                   sourceOffset:region.srcOffset
                       toBuffer:FromHandle<id<MTLBuffer>>(dst.handle)
              destinationOffset:region.dstOffset
                           size:region.size];
    }
}

void MetalCommandList::CopyTexture(TextureRef src, TextureRef dst, const TextureCopyRegion& region) {
    [Blit() copyFromTexture:FromHandle<id<MTLTexture>>(src.image)
                sourceSlice:0
                sourceLevel:0
               sourceOrigin:MTLOriginMake(0, 0, 0)
                 sourceSize:MTLSizeMake(region.width, region.height, 1)
                  toTexture:FromHandle<id<MTLTexture>>(dst.image)
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
}

namespace {

struct BufferPitch {
    u64 row = 0;
    u64 image = 0;
};

BufferPitch Pitch(Format format, const BufferTextureCopy& region) {
    const u32 rowLength = region.bufferRowLength != 0 ? region.bufferRowLength : region.width;
    const u32 imageHeight = region.bufferImageHeight != 0 ? region.bufferImageHeight : region.height;
    BufferPitch pitch;
    if (IsBlockCompressed(format)) {
        pitch.row = static_cast<u64>((rowLength + 3) / 4) * FormatBytes(format);
        pitch.image = pitch.row * ((imageHeight + 3) / 4);
    } else {
        pitch.row = static_cast<u64>(rowLength) * FormatBytes(format);
        pitch.image = pitch.row * imageHeight;
    }
    return pitch;
}

}  // namespace

void MetalCommandList::CopyBufferToTexture(BufferRef src, TextureRef dst, const BufferTextureCopy& region) {
    const BufferPitch pitch = Pitch(dst.format, region);
    id<MTLBlitCommandEncoder> encoder = Blit();
    for (u32 layer = 0; layer < region.layerCount; ++layer) {
        [encoder copyFromBuffer:FromHandle<id<MTLBuffer>>(src.handle)
                   sourceOffset:region.bufferOffset + layer * pitch.image
              sourceBytesPerRow:pitch.row
            sourceBytesPerImage:pitch.image
                     sourceSize:MTLSizeMake(region.width, region.height, 1)
                      toTexture:FromHandle<id<MTLTexture>>(dst.image)
               destinationSlice:region.baseLayer + layer
               destinationLevel:region.mipLevel
              destinationOrigin:MTLOriginMake(static_cast<NSUInteger>(region.x), static_cast<NSUInteger>(region.y), 0)];
    }
}

void MetalCommandList::CopyTextureToBuffer(TextureRef src, BufferRef dst, const BufferTextureCopy& region) {
    const BufferPitch pitch = Pitch(src.format, region);
    id<MTLBlitCommandEncoder> encoder = Blit();
    for (u32 layer = 0; layer < region.layerCount; ++layer) {
        [encoder copyFromTexture:FromHandle<id<MTLTexture>>(src.image)
                         sourceSlice:region.baseLayer + layer
                         sourceLevel:region.mipLevel
                        sourceOrigin:MTLOriginMake(static_cast<NSUInteger>(region.x),
                                                   static_cast<NSUInteger>(region.y), 0)
                          sourceSize:MTLSizeMake(region.width, region.height, 1)
                            toBuffer:FromHandle<id<MTLBuffer>>(dst.handle)
                   destinationOffset:region.bufferOffset + layer * pitch.image
              destinationBytesPerRow:pitch.row
            destinationBytesPerImage:pitch.image];
    }
}

void MetalCommandList::BlitTexture(TextureRef src, TextureRef dst, const BlitRegion& region, Filter filter) {
    id<MTLTexture> source = FromHandle<id<MTLTexture>>(src.image);
    id<MTLTexture> target = FromHandle<id<MTLTexture>>(dst.image);
    const i32 srcW = region.srcX1 - region.srcX0;
    const i32 srcH = region.srcY1 - region.srcY0;
    const i32 dstW = region.dstX1 - region.dstX0;
    const i32 dstH = region.dstY1 - region.dstY0;

    // An unscaled, unmirrored blit between textures of one format is a copy.
    if (srcW == dstW && srcH == dstH && srcW > 0 && srcH > 0 && source != target &&
        [source pixelFormat] == [target pixelFormat]) {
        [Blit() copyFromTexture:source
                    sourceSlice:0
                    sourceLevel:region.srcMip
                   sourceOrigin:MTLOriginMake(static_cast<NSUInteger>(region.srcX0), static_cast<NSUInteger>(region.srcY0), 0)
                     sourceSize:MTLSizeMake(static_cast<NSUInteger>(srcW), static_cast<NSUInteger>(srcH), 1)
                      toTexture:target
               destinationSlice:0
               destinationLevel:region.dstMip
              destinationOrigin:MTLOriginMake(static_cast<NSUInteger>(region.dstX0), static_cast<NSUInteger>(region.dstY0), 0)];
        return;
    }

    // Otherwise a scaling or converting blit is a draw (CommandList.hpp): a fullscreen
    // triangle over the destination rectangle, sampling the source rectangle.
    id<MTLRenderPipelineState> pipeline = device_.BlitPipeline([target pixelFormat]);
    if (pipeline == nil) return;
    EndEncoder();

    const f32 levelW = static_cast<f32>(std::max<NSUInteger>(1, [source width] >> region.srcMip));
    const f32 levelH = static_cast<f32>(std::max<NSUInteger>(1, [source height] >> region.srcMip));
    struct BlitArgs {
        f32 uvMin[2];
        f32 uvMax[2];
        f32 lod;
    } args{};
    args.uvMin[0] = static_cast<f32>(dstW >= 0 ? region.srcX0 : region.srcX1) / levelW;
    args.uvMax[0] = static_cast<f32>(dstW >= 0 ? region.srcX1 : region.srcX0) / levelW;
    args.uvMin[1] = static_cast<f32>(dstH >= 0 ? region.srcY0 : region.srcY1) / levelH;
    args.uvMax[1] = static_cast<f32>(dstH >= 0 ? region.srcY1 : region.srcY0) / levelH;
    args.lod = static_cast<f32>(region.srcMip);

    // Reading one mip of the texture being drawn into goes through a view of that mip.
    id<MTLTexture> sampled = source;
    if (source == target) {
        sampled = [source newTextureViewWithPixelFormat:[source pixelFormat]
                                            textureType:MTLTextureType2D
                                                 levels:NSMakeRange(region.srcMip, 1)
                                                 slices:NSMakeRange(0, 1)];
        args.lod = 0.0f;
    }

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].level = region.dstMip;
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> encoder = [commandBuffer_ renderCommandEncoderWithDescriptor:pass];
    [encoder waitForFence:device_.OrderFence() beforeStages:MTLRenderStageVertex | MTLRenderStageFragment];
    [encoder setRenderPipelineState:pipeline];
    [encoder setViewport:MTLViewport{static_cast<f64>(std::min(region.dstX0, region.dstX1)),
                                     static_cast<f64>(std::min(region.dstY0, region.dstY1)),
                                     static_cast<f64>(std::abs(dstW)), static_cast<f64>(std::abs(dstH)), 0.0, 1.0}];
    [encoder setVertexBytes:&args length:sizeof(args) atIndex:0];
    [encoder setFragmentBytes:&args length:sizeof(args) atIndex:0];
    [encoder setFragmentTexture:sampled atIndex:0];
    [encoder setFragmentSamplerState:device_.BlitSampler(filter) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder updateFence:device_.OrderFence() afterStages:MTLRenderStageFragment];
    [encoder endEncoding];
}

void MetalCommandList::ResetTimestamps(TimestampPoolRef, u32, u32) {
    // A counter sample buffer has no reset: every sample is overwritten when taken again.
}

void MetalCommandList::WriteTimestamp(TimestampPoolRef pool, u32 index, Stage) {
    // Apple GPUs sample counters at encoder boundaries only, so a timestamp is a blit
    // pass that samples as it starts. It fills four bytes of the arena: a pass with no
    // command in it is dropped, and its sample with it.
    EndEncoder();
    MTLBlitPassDescriptor* pass = [MTLBlitPassDescriptor blitPassDescriptor];
    pass.sampleBufferAttachments[0].sampleBuffer = FromHandle<id<MTLCounterSampleBuffer>>(pool.handle);
    pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = index;
    pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = MTLCounterDontSample;
    const UploadArena::Slice anchor = arena_.Allocate(4, 4);
    id<MTLBlitCommandEncoder> encoder = [commandBuffer_ blitCommandEncoderWithDescriptor:pass];
    [encoder waitForFence:device_.OrderFence()];
    [encoder fillBuffer:anchor.buffer range:NSMakeRange(anchor.offset, 4) value:0];
    [encoder updateFence:device_.OrderFence()];
    [encoder endEncoding];
}

void MetalCommandList::SetEvent(EventRef event, Stage) {
    auto* metalEvent = reinterpret_cast<MetalEvent*>(event.handle);
    EndEncoder();
    [commandBuffer_ encodeSignalEvent:metalEvent->Native() value:metalEvent->ArmSet()];
}

void MetalCommandList::ResetEvent(EventRef event, Stage) {
    reinterpret_cast<MetalEvent*>(event.handle)->ArmReset();
}

}  // namespace quantiloom::gpu::metal
