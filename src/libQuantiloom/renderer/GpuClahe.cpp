/**
 * @file GpuClahe.cpp
 * @brief The CLAHE display-enhancement passes on the gpu/ compute RHI
 */

#include "renderer/GpuClahe.hpp"

#include "core/Log.hpp"
#include "gpu/Device.hpp"
#include "gpu/Resources.hpp"

#include <algorithm>
#include <vector>


namespace quantiloom::rendercore {

namespace {

/// Push constants (must match CLAHEPushConstants in clahe.comp.hlsl). The
/// pipeline-layout range and the record-time payload share this one 48-byte
/// block; the words past passIndex are padding in the layout range and the
/// tone/palette selectors at record time.
struct CLAHEPushConstants {
    u32 imageWidth;
    u32 imageHeight;
    u32 tileCountX;
    u32 tileCountY;
    f32 clipLimit;
    u32 luminanceOnly;
    f32 inputMin;
    f32 inputMax;
    u32 passIndex;
    u32 toneMode;
    u32 palette;
    u32 padding;
};
static_assert(sizeof(CLAHEPushConstants) == 48);

}  // namespace

struct GpuClahe::Impl {
    gpu::Device& device;

    // Owned gpu:: resources, committed together by Initialize(). Destruction
    // order is pool, kernels, layouts, so a later destruction never names an
    // object an earlier one still needs.
    std::unique_ptr<gpu::BindingLayout> bindingLayout;
    std::unique_ptr<gpu::KernelLayout> kernelLayout;
    std::unique_ptr<gpu::ComputeKernel> histogramKernel;
    std::unique_ptr<gpu::ComputeKernel> cdfKernel;
    std::unique_ptr<gpu::ComputeKernel> applyKernel;
    std::unique_ptr<gpu::BindGroupPool> pool;
    gpu::BindGroup group;

    std::unique_ptr<gpu::Buffer> histogramBuffer;  // Per-tile histograms
    std::unique_ptr<gpu::Buffer> cdfBuffer;        // Per-tile CDFs
    std::unique_ptr<gpu::Buffer> minMaxBuffer;     // Per-tile min/max for normalization

    // Last input view written to the group's binding 0. The view only
    // changes when the render target is recreated or the camera chain starts
    // feeding CLAHE -- moments that drain the device first -- so the write is
    // skipped while it is current: a write to a set a submitted frame may
    // still be executing is a VUID-03047 violation, and every frame used to
    // take it for a view that had not moved.
    gpu::NativeHandle boundInputIdentity = 0;

    explicit Impl(gpu::Device& dev) : device(dev) {}

    Result<void, String> Initialize() {
        QL_LOG_INFO("Creating CLAHE compute pipeline...");

        const StringView kShaderNames[] = {
            "clahe_histogram",
            "clahe_cdf",
            "clahe_apply",
        };
        auto shaders = device.Shaders().Load(kShaderNames);
        if (!shaders.has_value()) {
            return Result<void, String>::Err(
                "CLAHE: Could not load shader files, CLAHE disabled");
        }
        QL_LOG_DEBUG("CLAHE: Loaded shaders from {}",
                     shaders.value().front().path.parent_path().string());

        // One slot per binding clahe.comp.hlsl declares (:84-97): u0
        // inputImage, u1 outputImage, u2 histogramBuffer, u3 cdfBuffer,
        // u4 tileMinMaxBuffer.
        constexpr gpu::BindingSlot kSlots[] = {
            {0, gpu::BindingKind::StorageTexture, 1, false},
            {1, gpu::BindingKind::StorageTexture, 1, false},
            {2, gpu::BindingKind::ReadWriteBuffer, 1, false},
            {3, gpu::BindingKind::ReadWriteBuffer, 1, false},
            {4, gpu::BindingKind::ReadWriteBuffer, 1, false},
        };

        auto layoutResult = device.CreateBindingLayout(
            gpu::BindingLayoutDesc{0, std::span<const gpu::BindingSlot>(kSlots)});
        if (!layoutResult.has_value()) {
            return Result<void, String>::Err("CLAHE: Failed to create descriptor set layout");
        }

        const gpu::BindingLayout* spaces[] = {layoutResult.value().get()};
        auto kernelLayoutResult = device.CreateKernelLayout(gpu::KernelLayoutDesc{
            std::span<const gpu::BindingLayout* const>(spaces), sizeof(CLAHEPushConstants)});
        if (!kernelLayoutResult.has_value()) {
            return Result<void, String>::Err("CLAHE: Failed to create pipeline layout");
        }

        auto histogramResult = device.CreateComputeKernel(gpu::ComputeKernelDesc{
            &shaders.value()[0], kernelLayoutResult.value().get()});
        if (!histogramResult.has_value()) {
            return Result<void, String>::Err("CLAHE: Failed to create compute pipelines");
        }
        auto cdfResult = device.CreateComputeKernel(gpu::ComputeKernelDesc{
            &shaders.value()[1], kernelLayoutResult.value().get()});
        if (!cdfResult.has_value()) {
            return Result<void, String>::Err("CLAHE: Failed to create compute pipelines");
        }
        auto applyResult = device.CreateComputeKernel(gpu::ComputeKernelDesc{
            &shaders.value()[2], kernelLayoutResult.value().get()});
        if (!applyResult.has_value()) {
            return Result<void, String>::Err("CLAHE: Failed to create compute pipelines");
        }

        auto poolResult = device.CreateBindGroupPool(*layoutResult.value(), 1);
        if (!poolResult.has_value()) {
            return Result<void, String>::Err("CLAHE: Failed to create descriptor pool");
        }

        // Max 64x64 tiles, 256 bins per tile.
        constexpr u32 maxTiles = 64 * 64;
        constexpr u32 histogramBins = 256;

        auto histogramBufferResult = device.CreateBuffer(gpu::BufferDesc{
            maxTiles * histogramBins * sizeof(u32),
            gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst,
            gpu::MemoryClass::DeviceLocal, 0});
        auto cdfBufferResult = device.CreateBuffer(gpu::BufferDesc{
            maxTiles * histogramBins * sizeof(f32),
            gpu::BufferUsage::Storage,
            gpu::MemoryClass::DeviceLocal, 0});
        auto minMaxBufferResult = device.CreateBuffer(gpu::BufferDesc{
            maxTiles * 2 * sizeof(f32),
            gpu::BufferUsage::Storage,
            gpu::MemoryClass::DeviceLocal, 0});
        if (!histogramBufferResult.has_value() || !cdfBufferResult.has_value() ||
            !minMaxBufferResult.has_value()) {
            return Result<void, String>::Err("CLAHE: Failed to create compute buffers");
        }

        auto groups = poolResult.value()->Allocate(1);
        if (!groups.has_value()) {
            return Result<void, String>::Err("CLAHE: Failed to allocate descriptor set");
        }

        bindingLayout = std::move(layoutResult.value());
        kernelLayout = std::move(kernelLayoutResult.value());
        histogramKernel = std::move(histogramResult.value());
        cdfKernel = std::move(cdfResult.value());
        applyKernel = std::move(applyResult.value());
        pool = std::move(poolResult.value());
        group = groups.value()[0];
        histogramBuffer = std::move(histogramBufferResult.value());
        cdfBuffer = std::move(cdfBufferResult.value());
        minMaxBuffer = std::move(minMaxBufferResult.value());
        return Result<void, String>::Ok();
    }

    void BindAll(gpu::TextureRef input, gpu::TextureRef output) {
        const gpu::BindingWrite writes[] = {
            gpu::BindingWrite::StorageTexture(0, input),
            gpu::BindingWrite::StorageTexture(1, output),
            gpu::BindingWrite::Buffer(2, gpu::BindingKind::ReadWriteBuffer,
                                      histogramBuffer->Ref()),
            gpu::BindingWrite::Buffer(3, gpu::BindingKind::ReadWriteBuffer,
                                      cdfBuffer->Ref()),
            gpu::BindingWrite::Buffer(4, gpu::BindingKind::ReadWriteBuffer,
                                      minMaxBuffer->Ref()),
        };
        device.WriteBindGroup(group, std::span<const gpu::BindingWrite>(writes));
    }

    void SetOutput(gpu::TextureRef output) {
        const gpu::BindingWrite write = gpu::BindingWrite::StorageTexture(1, output);
        device.WriteBindGroup(group, std::span<const gpu::BindingWrite>{&write, 1});
    }

    void Record(gpu::CommandList& cmd, gpu::TextureRef input, const ClaheRun& run) {
        // The input view is rewritten only when it actually moved; see
        // boundInputIdentity above.
        const gpu::NativeHandle inputIdentity = input.Identity();
        if (inputIdentity != boundInputIdentity) {
            const gpu::BindingWrite write = gpu::BindingWrite::StorageTexture(0, input);
            device.WriteBindGroup(group, std::span<const gpu::BindingWrite>{&write, 1});
            boundInputIdentity = inputIdentity;
        }

        // tileSize from UI represents tile grid dimension (e.g., 8 = 8x8 grid)
        // NOT pixels per tile
        u32 tileCountX = static_cast<u32>(run.params.tileSize);
        u32 tileCountY = static_cast<u32>(run.params.tileSize);

        // Clamp to max tiles (matching buffer allocation)
        tileCountX = std::min(tileCountX, 64u);
        tileCountY = std::min(tileCountY, 64u);

        // Ensure at least 1 tile
        tileCountX = std::max(tileCountX, 1u);
        tileCountY = std::max(tileCountY, 1u);

        CLAHEPushConstants pushConstants{};
        pushConstants.imageWidth = run.width;
        pushConstants.imageHeight = run.height;
        pushConstants.tileCountX = tileCountX;
        pushConstants.tileCountY = tileCountY;
        pushConstants.clipLimit = run.params.clipLimit;
        pushConstants.luminanceOnly = run.params.luminanceOnly ? 1 : 0;
        pushConstants.toneMode = static_cast<u32>(run.params.toneMode);
        pushConstants.palette = static_cast<u32>(run.params.palette);
        pushConstants.inputMin = run.inputMin;
        pushConstants.inputMax = run.inputMax;

        // A linear stretch is the window and nothing else, so the two passes that
        // exist to build a CDF have nothing to contribute -- and skipping them
        // makes the cheapest tone operator also the cheapest to run.
        const bool needsHistogram = run.params.toneMode != DisplayToneMode::Linear;

        // Memory barrier: wait for the input's last writer and make the write
        // visible to the compute passes that read the image -- pass 1's histogram
        // is a compute read in both tone modes, so the destination is the compute
        // stage with shader read|write even when a buffer fill (transfer) is the
        // next command: the fill is ordered by its own buffer barrier below.
        cmd.Barrier(gpu::GlobalBarrier{run.inputProducer,
                                       gpu::Access::ComputeRead | gpu::Access::ComputeWrite});

        if (needsHistogram) {
            // Clear histogram buffer to zero before Pass 1. Without this,
            // garbage data from uninitialized GPU memory causes corrupted
            // histograms and eventual TDR timeout (GPU crash).
            cmd.FillBuffer(histogramBuffer->Ref(), 0, gpu::kWholeSize, 0u);
            cmd.Barrier(gpu::BufferBarrier{histogramBuffer->Ref(),
                                           gpu::Access::TransferWrite,
                                           gpu::Access::ComputeRead |
                                               gpu::Access::ComputeWrite});

            // Pass 1: Build histograms. Dispatch one workgroup per tile.
            pushConstants.passIndex = 0;
            cmd.SetKernel(*histogramKernel);
            cmd.SetBindGroup(*kernelLayout, 0, group);
            cmd.PushConstants(*kernelLayout, pushConstants);
            cmd.Dispatch(tileCountX, tileCountY, 1);
            cmd.Barrier(gpu::GlobalBarrier{gpu::Access::ComputeWrite,
                                           gpu::Access::ComputeRead |
                                               gpu::Access::ComputeWrite});

            // Pass 2: Clip, redistribute, compute CDF. No set rebind.
            pushConstants.passIndex = 1;
            cmd.SetKernel(*cdfKernel);
            cmd.PushConstants(*kernelLayout, pushConstants);
            cmd.Dispatch(tileCountX, tileCountY, 1);
            cmd.Barrier(gpu::GlobalBarrier{gpu::Access::ComputeWrite,
                                           gpu::Access::ComputeRead |
                                               gpu::Access::ComputeWrite});
        }

        // Pass 3: Apply the mapping, then the palette.
        pushConstants.passIndex = 2;
        cmd.SetKernel(*applyKernel);
        // Bound here as well as in pass 1, because pass 1 does not always run.
        cmd.SetBindGroup(*kernelLayout, 0, group);
        cmd.PushConstants(*kernelLayout, pushConstants);
        // Dispatch one thread per pixel
        const u32 groupCountX = (run.width + 15) / 16;
        const u32 groupCountY = (run.height + 15) / 16;
        cmd.Dispatch(groupCountX, groupCountY, 1);

        // Final barrier: CLAHE output ready for blit
        cmd.Barrier(gpu::GlobalBarrier{gpu::Access::ComputeWrite, gpu::Access::TransferRead});
    }
};

GpuClahe::GpuClahe(std::unique_ptr<Impl> impl) : m_impl(std::move(impl)) {}

GpuClahe::~GpuClahe() = default;

Result<std::unique_ptr<GpuClahe>, String> GpuClahe::Create(gpu::Device& device) {
    auto impl = std::make_unique<Impl>(device);
    auto initialized = impl->Initialize();
    if (!initialized) {
        return Result<std::unique_ptr<GpuClahe>, String>::Err(initialized.error());
    }
    QL_LOG_INFO("CLAHE: Compute pipeline created successfully");
    return std::unique_ptr<GpuClahe>(new GpuClahe(std::move(impl)));
}

void GpuClahe::BindAll(gpu::TextureRef input, gpu::TextureRef output) {
    m_impl->BindAll(input, output);
}

void GpuClahe::SetOutput(gpu::TextureRef output) {
    m_impl->SetOutput(output);
}

void GpuClahe::Record(gpu::CommandList& cmd, gpu::TextureRef input, const ClaheRun& run) {
    m_impl->Record(cmd, input, run);
}

}  // namespace quantiloom::rendercore
