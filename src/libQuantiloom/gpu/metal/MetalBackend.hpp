/**
 * @file MetalBackend.hpp
 * @brief The Metal implementation of the compute RHI (internal, Objective-C++ only)
 *
 * Included by the gpu/metal/*.mm files and by the Metal side of the test support,
 * nothing else: it names Metal types. See gpu/AGENTS.md for the contract and
 * gpu/metal/AGENTS.md for how each Vulkan concept lands on Metal.
 */

#pragma once

#if !defined(__OBJC__)
#error "gpu/metal/MetalBackend.hpp is Objective-C++: include it from a .mm file"
#endif

#import <Metal/Metal.h>

#include "gpu/Device.hpp"

#include <array>
#include <filesystem>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace quantiloom::gpu::metal {

// ---- Handles -------------------------------------------------------------------------------

/// An Objective-C object as the u64 the neutral headers carry. Not retained: the owner
/// keeps the object alive.
inline NativeHandle ToHandle(id object) {
    return static_cast<NativeHandle>(reinterpret_cast<uintptr_t>((__bridge void*)object));
}

template <class T>
inline T FromHandle(NativeHandle handle) {
    return (__bridge T)reinterpret_cast<void*>(static_cast<uintptr_t>(handle));
}

[[nodiscard]] MTLPixelFormat ToMtl(Format format);
[[nodiscard]] bool IsBlockCompressed(Format format);
/// Bytes per texel, or per 4x4 block for a block-compressed format.
[[nodiscard]] u32 FormatBytes(Format format);
[[nodiscard]] String NsErrorText(NSError* error);

// ---- Shader reflection ---------------------------------------------------------------------

/// What Metal Shader Converter's reflection JSON (<name>.reflection.json, beside the metallib)
/// says about one converted shader.
struct ArgumentSlot {
    enum class Type : u8 { Srv, Uav, Cbv, Sampler, Table };
    Type type = Type::Srv;
    u32 offset = 0;  // bytes into the top-level argument buffer
    u32 size = 0;    // 24 for a descriptor entry, 8 for a root pointer
    u32 slot = 0;    // register number (~0 for a table)
    u32 space = 0;   // register space (~0 for a table)
};

struct FunctionConstantInfo {
    String name;  // MSC_FC_<id>
    u32 id = 0;
};

struct ShaderReflection {
    String entryPoint = "main";
    Vector<ArgumentSlot> topLevel;
    u32 topLevelBytes = 0;
    bool rootSignature = false;  // tables + root descriptors, rather than MSC's linear layout
    std::array<u32, 3> threadgroup{1, 1, 1};
    Vector<FunctionConstantInfo> functionConstants;
};

[[nodiscard]] Result<ShaderReflection, String> LoadReflection(const std::filesystem::path& metallibPath);

// ---- The root-signature layout (src/shaders/ql_bind.hlsli QL_ROOT_SIGNATURE) --------------

/// Table 0 holds t0..t37 then u0..u36; table 1 s0..s21; tables 2 and 3 the unbounded
/// space-1 texture and sampler arrays; then the push constants as a root CBV.
inline constexpr u32 kRootSrvCount = 38;
inline constexpr u32 kRootUavCount = 37;
inline constexpr u32 kRootSamplerCount = 22;

// ---- Resources -----------------------------------------------------------------------------

class MetalDevice;

class MetalBuffer final : public Buffer {
public:
    MetalBuffer(id<MTLBuffer> buffer, u64 size) : buffer_(buffer), size_(size) {}
    [[nodiscard]] BufferRef Ref() const override { return BufferRef{ToHandle(buffer_), size_}; }
    void* Map() override { return [buffer_ contents]; }
    const void* MapRead() override { return [buffer_ contents]; }
    void Unmap() override {}
    void Upload(const void* data, u64 bytes, u64 offset = 0) override;
    [[nodiscard]] id<MTLBuffer> Native() const { return buffer_; }

private:
    id<MTLBuffer> buffer_;
    u64 size_;
};

class MetalTexture final : public Texture {
public:
    MetalTexture(id<MTLTexture> texture, const TextureDesc& desc) : texture_(texture), desc_(desc) {}
    [[nodiscard]] TextureRef Ref() const override;
    [[nodiscard]] id<MTLTexture> Native() const { return texture_; }

private:
    id<MTLTexture> texture_;
    TextureDesc desc_;
};

class MetalSampler final : public Sampler {
public:
    explicit MetalSampler(id<MTLSamplerState> sampler) : sampler_(sampler) {}
    [[nodiscard]] SamplerRef Ref() const override { return SamplerRef{ToHandle(sampler_)}; }

private:
    id<MTLSamplerState> sampler_;
};

class MetalBindingLayout final : public BindingLayout {
public:
    [[nodiscard]] static Result<std::unique_ptr<BindingLayout>, String> Create(const BindingLayoutDesc& desc);
    [[nodiscard]] u32 Space() const override { return space_; }
    [[nodiscard]] std::span<const BindingSlot> Slots() const override { return slots_; }

private:
    u32 space_ = 0;
    Vector<BindingSlot> slots_;
};

class MetalKernelLayout final : public KernelLayout {
public:
    explicit MetalKernelLayout(const KernelLayoutDesc& desc);
    [[nodiscard]] u32 PushConstantBytes() const override { return pushBytes_; }
    [[nodiscard]] ShaderStages PushConstantVisibility() const override { return pushVisibility_; }
    [[nodiscard]] u32 SpaceCount() const { return static_cast<u32>(spaces_.size()); }

private:
    Vector<const BindingLayout*> spaces_;
    u32 pushBytes_ = 0;
    ShaderStages pushVisibility_ = ShaderStages::Compute;
};

/// One bound resource of a bind group. Strong references: a group keeps what it names alive.
struct MetalBinding {
    BindingKind kind{};
    u32 count = 1;  // > 1: an array (the bindless tables)
    id<MTLBuffer> buffer = nil;
    u64 offset = 0;
    u64 range = kWholeSize;
    id<MTLTexture> texture = nil;
    id<MTLSamplerState> sampler = nil;
    NativeHandle scene = 0;  // a Top-level acceleration structure (MetalDevice::FindAccelerationStructure)
    Vector<id<MTLTexture>> textures;
    Vector<id<MTLSamplerState>> samplers;
};

/// What a BindGroup handle points at: the CPU-side record of one group's writes, encoded
/// into a top-level argument buffer at dispatch.
struct MetalBindGroup {
    const MetalBindingLayout* layout = nullptr;
    std::unordered_map<u32, MetalBinding> bindings;  // by slot
};

class MetalBindGroupPool final : public BindGroupPool {
public:
    MetalBindGroupPool(const MetalBindingLayout& layout, u32 maxGroups) : layout_(layout), maxGroups_(maxGroups) {}
    [[nodiscard]] Result<Vector<BindGroup>, String> Allocate(u32 count) override;
    void Reset() override { groups_.clear(); }

private:
    const MetalBindingLayout& layout_;
    u32 maxGroups_;
    Vector<std::unique_ptr<MetalBindGroup>> groups_;
};

class MetalComputeKernel final : public ComputeKernel {
public:
    [[nodiscard]] static Result<std::unique_ptr<ComputeKernel>, String> Create(id<MTLDevice> device,
                                                                               const ComputeKernelDesc& desc);
    [[nodiscard]] const KernelLayout& Layout() const override { return *layout_; }
    [[nodiscard]] StringView Name() const override { return name_; }
    [[nodiscard]] id<MTLComputePipelineState> Pipeline() const { return pipeline_; }
    [[nodiscard]] const ShaderReflection& Reflection() const { return reflection_; }

private:
    id<MTLComputePipelineState> pipeline_ = nil;
    ShaderReflection reflection_;
    const KernelLayout* layout_ = nullptr;
    String name_;
};

class MetalPipelineCache final : public PipelineCache {
public:
    [[nodiscard]] static Result<std::unique_ptr<PipelineCache>, String> Create(id<MTLDevice> device,
                                                                               std::span<const std::byte> initialData);
    [[nodiscard]] Result<Vector<std::byte>, String> Serialize() const override;
    [[nodiscard]] id<MTLBinaryArchive> Archive() const { return archive_; }

private:
    id<MTLBinaryArchive> archive_ = nil;
};

class MetalTraceKernel final : public TraceKernel {
public:
    [[nodiscard]] static Result<std::unique_ptr<TraceKernel>, String> Create(id<MTLDevice> device,
                                                                             const TraceKernelDesc& desc);
    [[nodiscard]] const KernelLayout& Layout() const override { return *layout_; }
    [[nodiscard]] StringView Name() const override { return name_; }
    [[nodiscard]] id<MTLComputePipelineState> Pipeline() const { return pipeline_; }
    [[nodiscard]] const ShaderReflection& Reflection() const { return reflection_; }

private:
    id<MTLComputePipelineState> pipeline_ = nil;
    ShaderReflection reflection_;
    const KernelLayout* layout_ = nullptr;
    String name_;
};

class MetalRtAccelerationStructure final : public RtAccelerationStructure {
public:
    MetalRtAccelerationStructure(MetalDevice& device, id<MTLAccelerationStructure> structure, RtLevel level);
    ~MetalRtAccelerationStructure() override;
    [[nodiscard]] RtSceneRef Ref() const override;
    [[nodiscard]] RtLevel Level() const override { return level_; }
    [[nodiscard]] id<MTLAccelerationStructure> Native() const { return structure_; }
    /// The IRRaytracingAccelerationStructureGPUHeader a converted shader reads a TLAS
    /// through, sized for at least `instances` instance contributions. Top level only.
    void PrepareHeader(u32 instances);
    [[nodiscard]] id<MTLBuffer> Header() const { return header_; }

private:
    MetalDevice& device_;
    id<MTLAccelerationStructure> structure_;
    RtLevel level_;
    id<MTLBuffer> header_ = nil;
    u32 headerCapacity_ = 0;
};

// ---- Synchronisation -----------------------------------------------------------------------

class MetalFence final : public Fence {
public:
    explicit MetalFence(id<MTLSharedEvent> event) : event_(event) {}
    [[nodiscard]] Result<FenceStatus, String> Status() override;
    [[nodiscard]] Result<void, String> Wait(u64 timeoutNs) override;
    [[nodiscard]] Result<void, String> Reset() override;
    /// What a submission signalling this fence encodes.
    [[nodiscard]] u64 Target() const { return target_; }
    [[nodiscard]] id<MTLSharedEvent> Native() const { return event_; }

private:
    id<MTLSharedEvent> event_;
    u64 target_ = 1;  // created unsignaled: the event is 0
};

class MetalEvent final : public Event {
public:
    explicit MetalEvent(id<MTLSharedEvent> event) : event_(event) {}
    [[nodiscard]] EventRef Ref() const override { return EventRef{reinterpret_cast<NativeHandle>(this)}; }
    [[nodiscard]] bool IsSet() const override { return [event_ signaledValue] >= target_; }
    /// SetEvent: the value the command buffer signals. ResetEvent: a target no signal has
    /// reached yet. Metal events only count up, so "reset" moves the target, not the event.
    u64 ArmSet() { target_ = ++next_; return target_; }
    void ArmReset() { target_ = next_ + 1; }
    [[nodiscard]] id<MTLSharedEvent> Native() const { return event_; }

private:
    id<MTLSharedEvent> event_;
    u64 target_ = 1;
    u64 next_ = 0;
};

class MetalTimestampPool final : public TimestampPool {
public:
    MetalTimestampPool(id<MTLCounterSampleBuffer> samples, u32 count) : samples_(samples), count_(count) {}
    [[nodiscard]] TimestampPoolRef Ref() const override {
        return TimestampPoolRef{ToHandle(samples_), count_};
    }
    [[nodiscard]] Result<ResolveStatus, String> Resolve(u32 first, u32 count, TimestampResolve mode,
                                                        std::span<u64> out) override;

private:
    id<MTLCounterSampleBuffer> samples_;
    u32 count_;
};

class MetalCommandList;

class MetalCommandBuffer final : public CommandBuffer {
public:
    explicit MetalCommandBuffer(MetalDevice& device) : device_(device) {}
    ~MetalCommandBuffer() override;
    [[nodiscard]] Result<void, String> Reset() override;
    [[nodiscard]] Result<CommandList*, String> Begin() override;
    [[nodiscard]] Result<void, String> End() override;
    [[nodiscard]] id<MTLCommandBuffer> Native() const;

private:
    MetalDevice& device_;
    std::unique_ptr<MetalCommandList> list_;
};

class MetalCommandPool final : public CommandPool {
public:
    explicit MetalCommandPool(MetalDevice& device) : device_(device) {}
    [[nodiscard]] Result<std::unique_ptr<CommandBuffer>, String> Allocate() override {
        return std::unique_ptr<CommandBuffer>(std::make_unique<MetalCommandBuffer>(device_));
    }

private:
    MetalDevice& device_;
};

// ---- Command recording ---------------------------------------------------------------------

/// The device's free list of upload chunks. Shared with the completion handlers that give
/// chunks back, so a command buffer finishing after the device is gone frees into a pool
/// that still exists.
struct ArenaPool {
    id<MTLDevice> device = nil;
    std::mutex mutex;
    Vector<id<MTLBuffer>> free;
    [[nodiscard]] id<MTLBuffer> Take(u64 minBytes);
    void Give(id<MTLBuffer> chunk);
};

/// Shared, CPU-written memory a command buffer reads: argument buffers, push constants,
/// UpdateBuffer data. Chunks go back to the pool when the command buffer completes.
class UploadArena {
public:
    struct Slice {
        id<MTLBuffer> buffer = nil;
        u64 offset = 0;
        std::byte* cpu = nullptr;
        u64 gpuAddress = 0;
    };
    explicit UploadArena(std::shared_ptr<ArenaPool> pool) : pool_(std::move(pool)) {}
    ~UploadArena();
    [[nodiscard]] Slice Allocate(u64 bytes, u64 alignment = 256);
    /// Hands the chunks to `commandBuffer`'s completion; the arena is empty afterwards.
    void RetireWith(id<MTLCommandBuffer> commandBuffer);

private:
    std::shared_ptr<ArenaPool> pool_;
    Vector<id<MTLBuffer>> chunks_;
    u64 used_ = 0;
};

class MetalCommandList final : public CommandList {
public:
    MetalCommandList(MetalDevice& device, id<MTLCommandBuffer> commandBuffer);
    ~MetalCommandList() override;

    [[nodiscard]] Backend GetBackend() const override { return Backend::Metal; }
    [[nodiscard]] id<MTLCommandBuffer> Native() const { return commandBuffer_; }
    /// Ends the open encoder and hands the arena to the command buffer's completion. After
    /// this the command buffer has no open encoder and may be committed.
    void Finish();

    void SetKernel(const ComputeKernel& kernel) override;
    void SetBindGroup(const KernelLayout& layout, u32 space, BindGroup group, PipelineKind kind) override;
    using CommandList::SetBindGroup;
    void PushConstants(const KernelLayout& layout, std::span<const std::byte> bytes) override;
    using CommandList::PushConstants;
    void Dispatch(u32 groupsX, u32 groupsY, u32 groupsZ) override;

    void SetTraceKernel(const TraceKernel& kernel) override;
    void Trace(const TraceKernel& kernel, u32 width, u32 height, u32 depth) override;
    void BuildRtAccelerationStructure(const RtBuild& build) override;

    void Barrier(const BarrierBatch& batch) override;
    using CommandList::Barrier;

    void FillBuffer(BufferRef dst, u64 offset, u64 size, u32 value) override;
    void UpdateBuffer(BufferRef dst, u64 offset, std::span<const std::byte> bytes) override;
    void CopyBuffer(BufferRef src, BufferRef dst, std::span<const BufferCopyRegion> regions) override;
    void CopyTexture(TextureRef src, TextureRef dst, const TextureCopyRegion& region) override;
    void CopyBufferToTexture(BufferRef src, TextureRef dst, const BufferTextureCopy& region) override;
    void CopyTextureToBuffer(TextureRef src, BufferRef dst, const BufferTextureCopy& region) override;
    void BlitTexture(TextureRef src, TextureRef dst, const BlitRegion& region, Filter filter) override;

    void ResetTimestamps(TimestampPoolRef pool, u32 first, u32 count) override;
    void WriteTimestamp(TimestampPoolRef pool, u32 index, Stage after) override;

    void SetEvent(EventRef event, Stage after) override;
    void ResetEvent(EventRef event, Stage after) override;

private:
    void EndEncoder();
    /// Orders a just-opened encoder after every encoder before it (OrderFence).
    void Ordered(id<MTLCommandEncoder> encoder);
    id<MTLComputeCommandEncoder> Compute();
    id<MTLBlitCommandEncoder> Blit();
    id<MTLAccelerationStructureCommandEncoder> Accel();
    /// Writes the top-level argument buffer `reflection` describes from the bind groups set
    /// for `kind` and the push constants, makes every resource it names resident, and
    /// returns the GPU address the kernel reads it from.
    u64 EncodeArguments(const ShaderReflection& reflection, PipelineKind kind);

    MetalDevice& device_;
    id<MTLCommandBuffer> commandBuffer_;
    id<MTLComputeCommandEncoder> compute_ = nil;
    id<MTLBlitCommandEncoder> blit_ = nil;
    id<MTLAccelerationStructureCommandEncoder> accel_ = nil;
    const MetalComputeKernel* kernel_ = nullptr;
    const MetalTraceKernel* trace_ = nullptr;
    std::array<Vector<const MetalBindGroup*>, 2> groups_;  // [PipelineKind][space]
    Vector<std::byte> push_;
    UploadArena arena_;
    bool finished_ = false;
};

// ---- Device --------------------------------------------------------------------------------

class MetalDevice final : public Device {
public:
    /// Creates its own queue when `queue` is nil. Err when the device has no ray tracing.
    [[nodiscard]] static Result<std::unique_ptr<Device>, String> Create(id<MTLDevice> device,
                                                                        id<MTLCommandQueue> queue);
    ~MetalDevice() override;

    [[nodiscard]] const DeviceInfo& Info() const override { return info_; }
    [[nodiscard]] const ShaderLibrary& Shaders() const override { return shaders_; }

    [[nodiscard]] Result<std::unique_ptr<Buffer>, String> CreateBuffer(const BufferDesc& desc) override;
    [[nodiscard]] Result<std::unique_ptr<Texture>, String> CreateTexture(const TextureDesc& desc) override;
    [[nodiscard]] Result<std::unique_ptr<BindingLayout>, String> CreateBindingLayout(
        const BindingLayoutDesc& desc) override;
    [[nodiscard]] Result<std::unique_ptr<KernelLayout>, String> CreateKernelLayout(
        const KernelLayoutDesc& desc) override;
    [[nodiscard]] Result<std::unique_ptr<ComputeKernel>, String> CreateComputeKernel(
        const ComputeKernelDesc& desc) override;
    [[nodiscard]] Result<std::unique_ptr<BindGroupPool>, String> CreateBindGroupPool(const BindingLayout& layout,
                                                                                     u32 maxGroups) override;

    void WriteBindGroup(BindGroup group, std::span<const BindingWrite> writes) override;
    void ExecuteImmediate(const std::function<void(CommandList&)>& record) override;
    [[nodiscard]] Result<void, String> WaitIdle() override;

    [[nodiscard]] Result<std::unique_ptr<Sampler>, String> CreateSampler(const SamplerDesc& desc) override;
    [[nodiscard]] Result<std::unique_ptr<Event>, String> CreateGpuEvent() override;
    [[nodiscard]] Result<std::unique_ptr<TimestampPool>, String> CreateTimestampPool(u32 count) override;
    [[nodiscard]] Result<std::unique_ptr<CommandPool>, String> CreateCommandPool(
        const CommandPoolDesc& desc) override;
    [[nodiscard]] Result<std::unique_ptr<Fence>, String> CreateFence() override;
    [[nodiscard]] Result<void, String> Submit(CommandBuffer& commandBuffer, Fence* signal) override;
    [[nodiscard]] Result<void, String> WaitQueueIdle() override;
    [[nodiscard]] std::unique_ptr<CommandList> WrapRecording(NativeHandle commandBuffer) override;
    [[nodiscard]] TextureRef ImportHostTexture(NativeHandle image, Format format, u32 width, u32 height) override;

    [[nodiscard]] Result<std::unique_ptr<PipelineCache>, String> CreatePipelineCache(
        std::span<const std::byte> initialData) override;
    [[nodiscard]] Result<std::unique_ptr<TraceKernel>, String> CreateTraceKernel(
        const TraceKernelDesc& desc) override;
    [[nodiscard]] RtBuildSizes GetRtBuildSizes(const RtGeometry& geometry) const override;
    [[nodiscard]] Result<std::unique_ptr<RtAccelerationStructure>, String> CreateRtAccelerationStructure(
        RtLevel level, u64 bytes) override;
    [[nodiscard]] u32 RtInstanceStride() const override;
    void PackRtInstances(std::span<const RtInstance> instances, std::span<std::byte> out) const override;
    [[nodiscard]] u64 BufferDeviceAddress(BufferRef buffer) const override;

    [[nodiscard]] id<MTLDevice> NativeDevice() const { return device_; }
    [[nodiscard]] id<MTLCommandQueue> NativeQueue() const { return queue_; }
    /// Every encoder this device's command lists open waits on this fence first and
    /// updates it last, so encoders run in recording order even for what hazard
    /// tracking cannot see: the bottom levels a TLAS names by resource ID.
    [[nodiscard]] id<MTLFence> OrderFence() const { return orderFence_; }

    // For the command lists and acceleration structures of this device.
    void RegisterAccelerationStructure(MetalRtAccelerationStructure* structure);
    void UnregisterAccelerationStructure(MetalRtAccelerationStructure* structure);
    [[nodiscard]] MetalRtAccelerationStructure* FindAccelerationStructure(NativeHandle handle) const;
    [[nodiscard]] const std::shared_ptr<ArenaPool>& Arenas() const { return arenas_; }
    /// The fullscreen render pipeline BlitTexture draws with, per destination format.
    [[nodiscard]] id<MTLRenderPipelineState> BlitPipeline(MTLPixelFormat format);
    [[nodiscard]] id<MTLSamplerState> BlitSampler(Filter filter) const {
        return filter == Filter::Linear ? blitLinear_ : blitNearest_;
    }
    [[nodiscard]] MTLAccelerationStructureDescriptor* DescribeGeometry(const RtGeometry& geometry) const;

private:
    MetalDevice(id<MTLDevice> device, id<MTLCommandQueue> queue);

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLResidencySet> residency_ = nil;  // every acceleration structure: TLAS instances name BLAS by ID
    id<MTLFence> orderFence_ = nil;
    DeviceInfo info_;
    ShaderLibrary shaders_{ShaderFormat::MetalLib};

    mutable std::mutex mutex_;
    std::unordered_map<NativeHandle, MetalRtAccelerationStructure*> structures_;
    std::shared_ptr<ArenaPool> arenas_ = std::make_shared<ArenaPool>();
    id<MTLLibrary> blitLibrary_ = nil;
    std::unordered_map<u64, id<MTLRenderPipelineState>> blitPipelines_;
    id<MTLSamplerState> blitNearest_ = nil;
    id<MTLSamplerState> blitLinear_ = nil;
};

}  // namespace quantiloom::gpu::metal
