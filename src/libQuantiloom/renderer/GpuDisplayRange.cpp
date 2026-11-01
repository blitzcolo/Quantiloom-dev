#include "renderer/GpuDisplayRange.hpp"

#include "gpu/Binding.hpp"
#include "gpu/CommandList.hpp"
#include "gpu/Device.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

// RangePush, defined once for display_range.comp.hlsl and this file. By a path
// relative to this file, as RtBindings.hpp includes its manifest.
#include "../../shaders/ql_shared_compute.hlsli"


namespace quantiloom::rendercore {
namespace {

u32 FloatToOrdered(f32 value) {
    const u32 bits = std::bit_cast<u32>(value);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

f32 OrderedToFloat(u32 ordered) {
    const u32 bits = (ordered & 0x80000000u)
        ? (ordered ^ 0x80000000u) : ~ordered;
    return std::bit_cast<f32>(bits);
}

/// The push block of display_range.comp.hlsl, which calls it RangePush.
using DisplayRangePush = shader::RangePush;

}  // namespace

struct GpuDisplayRange::Impl {
    gpu::Device& device;

    // Owned gpu:: resources, committed together by Initialize(). Destruction
    // order is pool, kernels, layouts, so a later destruction never names an
    // object an earlier one still needs.
    std::unique_ptr<gpu::BindingLayout> bindingLayout;
    std::unique_ptr<gpu::KernelLayout> kernelLayout;
    std::unique_ptr<gpu::ComputeKernel> extentsKernel;
    std::unique_ptr<gpu::ComputeKernel> histogramKernel;
    std::unique_ptr<gpu::BindGroupPool> pool;
    gpu::BindGroup group;

    std::unique_ptr<gpu::Buffer> scratch;
    std::unique_ptr<gpu::Buffer> histogram;
    std::unique_ptr<gpu::Buffer> scratchReadback;

    explicit Impl(gpu::Device& dev) : device(dev) {}

    Result<void, String> Initialize() {
        const StringView kShaderNames[] = {
            "display_range_extents.comp",
            "display_range_histogram.comp",
        };
        auto shaders = device.Shaders().Load(kShaderNames);
        if (!shaders.has_value())
            return Result<void, String>::Err(
                "display-range compute shaders are missing or invalid");

        // One slot per binding display_range.comp.hlsl declares (:29-34):
        // u0 inputImage, u1 rangeScratch, u2 histogram.
        constexpr gpu::BindingSlot kSlots[] = {
            {0, gpu::BindingKind::StorageTexture, 1, false},
            {1, gpu::BindingKind::ReadWriteBuffer, 1, false},
            {2, gpu::BindingKind::ReadWriteBuffer, 1, false},
        };
        auto layoutResult = device.CreateBindingLayout(
            gpu::BindingLayoutDesc{0, std::span<const gpu::BindingSlot>(kSlots)});
        if (!layoutResult.has_value())
            return Result<void, String>::Err(
                "cannot create display-range descriptor layout");

        const gpu::BindingLayout* spaces[] = {layoutResult.value().get()};
        auto kernelLayoutResult = device.CreateKernelLayout(gpu::KernelLayoutDesc{
            std::span<const gpu::BindingLayout* const>(spaces), sizeof(DisplayRangePush)});
        if (!kernelLayoutResult.has_value())
            return Result<void, String>::Err(
                "cannot create display-range pipeline layout");

        auto extentsResult = device.CreateComputeKernel(gpu::ComputeKernelDesc{
            &shaders.value()[0], kernelLayoutResult.value().get()});
        if (!extentsResult.has_value())
            return Result<void, String>::Err(
                "cannot create display-range compute pipelines");
        auto histogramResult = device.CreateComputeKernel(gpu::ComputeKernelDesc{
            &shaders.value()[1], kernelLayoutResult.value().get()});
        if (!histogramResult.has_value())
            return Result<void, String>::Err(
                "cannot create display-range compute pipelines");

        auto poolResult = device.CreateBindGroupPool(*layoutResult.value(), 1);
        if (!poolResult.has_value())
            return Result<void, String>::Err(
                "cannot create display-range descriptor pool");
        auto groups = poolResult.value()->Allocate(1);
        if (!groups.has_value())
            return Result<void, String>::Err(
                "cannot allocate display-range descriptor set");

        const u64 scratchBytes = kScratchWords * sizeof(u32);
        const u64 histogramBytes =
            static_cast<u64>(GpuDisplayRange::kHistogramBins) * sizeof(u32);
        auto scratchResult = device.CreateBuffer(gpu::BufferDesc{
            scratchBytes,
            gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst |
                gpu::BufferUsage::TransferSrc,
            gpu::MemoryClass::DeviceLocal, 0});
        auto histogramBufferResult = device.CreateBuffer(gpu::BufferDesc{
            histogramBytes,
            gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst |
                gpu::BufferUsage::TransferSrc,
            gpu::MemoryClass::DeviceLocal, 0});
        auto readbackResult = device.CreateBuffer(gpu::BufferDesc{
            scratchBytes, gpu::BufferUsage::TransferDst, gpu::MemoryClass::Readback, 0});
        if (!scratchResult.has_value() || !histogramBufferResult.has_value() ||
            !readbackResult.has_value())
            return Result<void, String>::Err("cannot allocate display-range buffers");

        const gpu::BindingWrite staticWrites[] = {
            gpu::BindingWrite::Buffer(1, gpu::BindingKind::ReadWriteBuffer,
                                      scratchResult.value()->Ref()),
            gpu::BindingWrite::Buffer(2, gpu::BindingKind::ReadWriteBuffer,
                                      histogramBufferResult.value()->Ref()),
        };

        this->bindingLayout = std::move(layoutResult.value());
        this->kernelLayout = std::move(kernelLayoutResult.value());
        this->extentsKernel = std::move(extentsResult.value());
        this->histogramKernel = std::move(histogramResult.value());
        this->pool = std::move(poolResult.value());
        group = groups.value()[0];
        this->scratch = std::move(scratchResult.value());
        this->histogram = std::move(histogramBufferResult.value());
        this->scratchReadback = std::move(readbackResult.value());

        // Bindings 1 and 2 are written once here; binding 0 (the input view)
        // is rewritten on every Compute.
        device.WriteBindGroup(group, std::span<const gpu::BindingWrite>(staticWrites));
        return Result<void, String>::Ok();
    }
};

GpuDisplayRange::GpuDisplayRange(std::unique_ptr<Impl> impl)
    : m_impl(std::move(impl)) {}

GpuDisplayRange::~GpuDisplayRange() = default;

Result<std::unique_ptr<GpuDisplayRange>, String>
GpuDisplayRange::Create(gpu::Device& device) {
    auto impl = std::make_unique<Impl>(device);
    auto initialized = impl->Initialize();
    if (!initialized)
        return Result<std::unique_ptr<GpuDisplayRange>, String>::Err(
            initialized.error());
    return std::unique_ptr<GpuDisplayRange>(
        new GpuDisplayRange(std::move(impl)));
}

Result<DisplayRange, String> GpuDisplayRange::Compute(
    gpu::TextureRef image, u32 width, u32 height,
    f32 percentileLow, f32 percentileHigh) {
    if (!image.IsValid() || image.format != gpu::Format::RGBA32Float)
        return Result<DisplayRange, String>::Err(
            "display range requires an RGBA32F image");
    if (width == 0 || height == 0 || width > image.width || height > image.height)
        return Result<DisplayRange, String>::Err(
            "display-range extent is empty or exceeds its image");

    gpu::Device& device = m_impl->device;
    const gpu::BindingWrite inputWrite =
        gpu::BindingWrite::StorageTexture(0, image);
    device.WriteBindGroup(m_impl->group,
                          std::span<const gpu::BindingWrite>{&inputWrite, 1});

    const std::array<u32, kScratchWords> initialScratch{
        FloatToOrdered(std::numeric_limits<f32>::max()),
        FloatToOrdered(std::numeric_limits<f32>::lowest()),
        0u, 0u, 0u, 0u, 0u, 0u};
    const DisplayRangePush extentsPush{width, height, 0.0f, 0.0f, 0u, 0u};
    const u32 groupsX = (width + 15u) / 16u;
    const u32 groupsY = (height + 15u) / 16u;

    device.ExecuteImmediate([&](gpu::CommandList& cmd) {
        cmd.Barrier(gpu::TextureBarrier{image, gpu::Access::AnyWrite,
                                        gpu::Access::ComputeRead});

        cmd.UpdateBuffer(m_impl->scratch->Ref(), 0,
                         std::span<const std::byte>(
                             reinterpret_cast<const std::byte*>(initialScratch.data()),
                             sizeof(initialScratch)));
        cmd.Barrier(gpu::BufferBarrier{m_impl->scratch->Ref(),
                                       gpu::Access::TransferWrite,
                                       gpu::Access::ComputeRead |
                                           gpu::Access::ComputeWrite});

        // SetBindGroup and PushConstants precede SetKernel here; the
        // interface takes the layout explicitly so that order is legal.
        cmd.SetBindGroup(*m_impl->kernelLayout, 0, m_impl->group);
        cmd.PushConstants(*m_impl->kernelLayout, extentsPush);
        cmd.SetKernel(*m_impl->extentsKernel);
        cmd.Dispatch(groupsX, groupsY, 1);

        cmd.Barrier(gpu::BufferBarrier{m_impl->scratch->Ref(),
                                       gpu::Access::ComputeWrite,
                                       gpu::Access::TransferRead});
        const gpu::BufferCopyRegion scratchCopy{0, 0, m_impl->scratch->Ref().size};
        cmd.CopyBuffer(m_impl->scratch->Ref(), m_impl->scratchReadback->Ref(),
                       std::span<const gpu::BufferCopyRegion>{&scratchCopy, 1});
    });

    std::array<u32, kScratchWords> scratch{};
    const void* scratchData = m_impl->scratchReadback->MapRead();
    if (!scratchData)
        return Result<DisplayRange, String>::Err(
            "cannot map display-range scalar readback");
    std::memcpy(scratch.data(), scratchData, sizeof(scratch));
    m_impl->scratchReadback->Unmap();

    if (scratch[3] != 0u)
        return Result<DisplayRange, String>::Err(
            "display range input needs CPU small-value arithmetic");
    const u32 validCount = scratch[2];
    const f32 absMin = OrderedToFloat(scratch[0]);
    const f32 absMax = OrderedToFloat(scratch[1]);
    // No valid pixels at all: the {0,1} default window is the only answer.
    if (validCount == 0) return DisplayRange{};
    // A flat image is still an image. Stretching it would turn rounding noise
    // into a picture, so it keeps its own range, widened by an epsilon so the
    // window is nonzero -- not the arbitrary [0,1] an empty input reports.
    if (absMin >= absMax) {
        const f32 epsilon = std::max(std::abs(absMin), 1.0f) * 1e-6f;
        return DisplayRange{absMin - epsilon, absMax + epsilon};
    }

    // This is deliberately host arithmetic: Vulkan permits a less accurate
    // floating-point division than MSVC's f32 division, enough to move an
    // endpoint across a 65,536-bin boundary. Reading these three words first
    // preserves the existing bin definition without reading back the image.
    const f32 span = absMax - absMin;
    const f32 scale = static_cast<f32>(kHistogramBins - 1) / span;
    if (!std::isfinite(span) || !(span > 0.0f) || !std::isfinite(scale))
        return Result<DisplayRange, String>::Err(
            "display range input needs CPU scale arithmetic");
    // Percentile targets, as counts: the histogram pass's last workgroup
    // finds the first bin whose inclusive cumulative count reaches each
    // target and writes the two bins into the scratch words, so the bins
    // never cross to the host.
    const f64 lowFraction =
        std::clamp(static_cast<f64>(percentileLow), 0.0, 100.0) / 100.0;
    const f64 highFraction =
        std::clamp(static_cast<f64>(percentileHigh), 0.0, 100.0) / 100.0;
    const usize targetLow =
        static_cast<usize>(static_cast<f64>(validCount) * lowFraction);
    const usize targetHigh =
        static_cast<usize>(static_cast<f64>(validCount) * highFraction);
    const DisplayRangePush histogramPush{width, height, absMin, scale,
                                         static_cast<u32>(targetLow),
                                         static_cast<u32>(targetHigh)};
    device.ExecuteImmediate([&](gpu::CommandList& cmd) {
        cmd.FillBuffer(m_impl->histogram->Ref(), 0, gpu::kWholeSize, 0u);
        cmd.Barrier(gpu::BufferBarrier{m_impl->histogram->Ref(),
                                       gpu::Access::TransferWrite,
                                       gpu::Access::ComputeRead |
                                           gpu::Access::ComputeWrite});

        cmd.SetBindGroup(*m_impl->kernelLayout, 0, m_impl->group);
        cmd.PushConstants(*m_impl->kernelLayout, histogramPush);
        cmd.SetKernel(*m_impl->histogramKernel);
        cmd.Dispatch(groupsX, groupsY, 1);

        // The last workgroup wrote the crossing bins into the scratch words;
        // those words are all the host needs.
        cmd.Barrier(gpu::BufferBarrier{m_impl->scratch->Ref(),
                                       gpu::Access::ComputeWrite,
                                       gpu::Access::TransferRead});
        const gpu::BufferCopyRegion scratchCopyOut{0, 0, m_impl->scratch->Ref().size};
        cmd.CopyBuffer(m_impl->scratch->Ref(), m_impl->scratchReadback->Ref(),
                       std::span<const gpu::BufferCopyRegion>{&scratchCopyOut, 1});
    });

    std::array<u32, kScratchWords> resultWords{};
    const void* resultData = m_impl->scratchReadback->MapRead();
    if (!resultData)
        return Result<DisplayRange, String>::Err(
            "cannot map display-range percentile readback");
    std::memcpy(resultWords.data(), resultData, sizeof(resultWords));
    m_impl->scratchReadback->Unmap();

    const f32 invScale =
        (absMax - absMin) / static_cast<f32>(kHistogramBins - 1);
    DisplayRange result{
        absMin + static_cast<f32>(resultWords[5]) * invScale,
        absMin + static_cast<f32>(resultWords[6]) * invScale};
    if (!(result.max - result.min > (absMax - absMin) * 1e-3f))
        result = {absMin, absMax};
    return result;
}

}  // namespace quantiloom::rendercore
