/**
 * @file GpuSpectralReconstructor.cpp
 * @brief Implementation of GPU-accelerated spectral cube reconstruction
 *
 * @author blitzcolo
 */

#include "hs_core/GpuSpectralReconstructor.hpp"

#include "core/Log.hpp"
#include "gpu/CommandList.hpp"
#include "gpu/Device.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <span>
#include <utility>
#include <vector>
#include <cmath>

namespace quantiloom {

// ============================================================================
// Status String Conversion
// ============================================================================

const char* GpuReconstructorStatusToString(GpuReconstructorStatus status) {
    switch (status) {
        case GpuReconstructorStatus::Success:            return "Success";
        case GpuReconstructorStatus::NotInitialized:     return "Not initialized";
        case GpuReconstructorStatus::InvalidInput:       return "Invalid input";
        case GpuReconstructorStatus::ShaderCompileFailed: return "Shader compile failed";
        case GpuReconstructorStatus::PipelineCreateFailed: return "Pipeline create failed";
        case GpuReconstructorStatus::BufferAllocFailed:  return "Buffer alloc failed";
        case GpuReconstructorStatus::DispatchFailed:     return "Dispatch failed";
        case GpuReconstructorStatus::ReadbackFailed:     return "Readback failed";
        case GpuReconstructorStatus::OutOfMemory:        return "Out of memory";
        case GpuReconstructorStatus::Timeout:            return "Timeout";
        default:                                         return "Unknown status";
    }
}

// ============================================================================
// Push Constants Structure (must match shader)
// ============================================================================

struct ReconstructPushConstants {
    u32 width;
    u32 height;
    u32 srcBands;
    u32 dstBands;
    u32 interpolationMode;
    u32 _pad0;
    u32 _pad1;
    u32 _pad2;
};
static_assert(sizeof(ReconstructPushConstants) == 8u * sizeof(u32));

// ============================================================================
// Reconstruction Mapping (must match spectral_reconstruct.comp, binding 4)
// ============================================================================
namespace {

// One entry per target band, packed as 6 x u32 (uint4 + float2) exactly as
// the shader's MappingBuffer declares it.
struct GpuMappingEntry {
    u32 targetBand;
    u32 sourceBand;
    u32 sourceNextBand;
    u32 wasRendered;
    f32 targetWavelength;
    f32 interpWeight;
};
static_assert(sizeof(GpuMappingEntry) == 6u * sizeof(u32));

// Builds one mapping entry with the same index search the shader used to do
// per pixel: an exact match within 0.01 nm wins, otherwise the interval
// whose upper edge is the first source wavelength at or above the target.
// All interpolation modes share those indices; only the interpolant differs.
GpuMappingEntry BuildMappingEntry(u32 targetBand, f32 targetWl,
                                  const Vector<f32>& srcWavelengths) {
    GpuMappingEntry entry{};
    entry.targetBand = targetBand;
    entry.targetWavelength = targetWl;
    const u32 numSrc = static_cast<u32>(srcWavelengths.size());

    for (u32 i = 0; i < numSrc; ++i) {
        if (std::fabs(srcWavelengths[i] - targetWl) < 0.01f) {
            entry.wasRendered = 1u;
            entry.sourceBand = i;
            entry.sourceNextBand = i;
            return entry;
        }
    }

    u32 idx = 0;
    for (u32 i = 0; i + 1 < numSrc; ++i) {
        if (srcWavelengths[i + 1] < targetWl) {
            idx = i + 1;
        } else {
            break;
        }
    }
    const u32 next = idx + 1 < numSrc ? idx + 1 : idx;
    entry.sourceBand = idx;
    entry.sourceNextBand = next;
    if (next > idx) {
        entry.interpWeight = std::clamp(
            (targetWl - srcWavelengths[idx]) /
                (srcWavelengths[next] - srcWavelengths[idx]),
            0.0f, 1.0f);
    }
    return entry;
}

/**
 * @brief Convert InterpolationMethod to shader mode value
 */
u32 InterpolationModeValue(InterpolationMethod method) {
    switch (method) {
        case InterpolationMethod::Linear:     return 0;
        case InterpolationMethod::CatmullRom: return 1;
        case InterpolationMethod::Akima:      return 2;
        default:                              return 1;  // Default to Catmull-Rom
    }
}

/**
 * @brief Create a buffer, reporting an allocation failure as OutOfMemory
 *
 * That is the status ReconstructCustom returns for a failed buffer allocation.
 */
Result<std::unique_ptr<gpu::Buffer>, GpuReconstructorStatus> CreateBuffer(
    gpu::Device& device, u64 size, gpu::BufferUsage usage, gpu::MemoryClass memory) {
    auto buffer = device.CreateBuffer(gpu::BufferDesc{size, usage, memory, 0});
    if (!buffer.has_value()) {
        LOG_ERROR("GpuSpectralReconstructor: Buffer allocation failed: {}", buffer.error());
        return Result<std::unique_ptr<gpu::Buffer>, GpuReconstructorStatus>::Err(
            GpuReconstructorStatus::OutOfMemory);
    }
    return std::move(buffer.value());
}

} // namespace

// ============================================================================
// GpuSpectralReconstructor::Impl
// ============================================================================

struct GpuSpectralReconstructor::Impl {
    gpu::Device& device;

    // Owned gpu:: resources; destruction order is pool, kernel, layouts, so a
    // later destruction never names an object an earlier one still needs.
    std::unique_ptr<gpu::BindingLayout> bindingLayout;
    std::unique_ptr<gpu::KernelLayout> kernelLayout;
    std::unique_ptr<gpu::ComputeKernel> kernel;
    std::unique_ptr<gpu::BindGroupPool> pool;

    bool initialized = false;

    // Statistics
    f64 lastReconstructionTime = 0.0;
    u64 lastMemoryUsage = 0;
    u64 lastPixelCount = 0;

    explicit Impl(gpu::Device& dev) : device(dev) {}

    ~Impl() {
        Cleanup();
    }

    void Cleanup() {
        pool.reset();
        kernel.reset();
        kernelLayout.reset();
        bindingLayout.reset();
        initialized = false;
    }
};

// ============================================================================
// GpuSpectralReconstructor - Public Interface
// ============================================================================

GpuSpectralReconstructor::GpuSpectralReconstructor(gpu::Device& device)
    : m_impl(std::make_unique<Impl>(device)) {
    LOG_DEBUG("GpuSpectralReconstructor created");
}

GpuSpectralReconstructor::~GpuSpectralReconstructor() = default;

GpuSpectralReconstructor::GpuSpectralReconstructor(GpuSpectralReconstructor&&) noexcept = default;
GpuSpectralReconstructor& GpuSpectralReconstructor::operator=(GpuSpectralReconstructor&&) noexcept = default;

GpuReconstructorStatus GpuSpectralReconstructor::Initialize() {
    if (m_impl->initialized) {
        return GpuReconstructorStatus::Success;
    }

    LOG_INFO("GpuSpectralReconstructor: Initializing...");

    // The one resource locator serves every shader the SDK loads, and the
    // logical name is the CMake OUT_NAME, so the same lookup later finds a
    // .metallib when a Metal device asks for the set.
    const StringView kShaderNames[] = {"spectral_reconstruct"};
    auto shaders = m_impl->device.Shaders().Load(kShaderNames);
    if (!shaders.has_value()) {
        LOG_ERROR("GpuSpectralReconstructor: Could not load spectral_reconstruct: {}", shaders.error());
        return GpuReconstructorStatus::ShaderCompileFailed;
    }

    // One slot per HLSL declaration in spectral_reconstruct.comp.hlsl, all
    // buffers: t0 sparseCube :49, u1 fullCube :52, t2 srcWavelengths :55,
    // t3 dstWavelengths :59, t4 mappingData :72.
    constexpr gpu::BindingSlot kSlots[] = {
        {0, gpu::BindingKind::ReadOnlyBuffer, 1, false},
        {1, gpu::BindingKind::ReadWriteBuffer, 1, false},
        {2, gpu::BindingKind::ReadOnlyBuffer, 1, false},
        {3, gpu::BindingKind::ReadOnlyBuffer, 1, false},
        {4, gpu::BindingKind::ReadOnlyBuffer, 1, false},
    };

    auto bindingLayout = m_impl->device.CreateBindingLayout(
        gpu::BindingLayoutDesc{0, std::span<const gpu::BindingSlot>(kSlots)});
    if (!bindingLayout.has_value()) {
        LOG_ERROR("GpuSpectralReconstructor: Failed to create descriptor set layout: {}",
                  bindingLayout.error());
        return GpuReconstructorStatus::PipelineCreateFailed;
    }

    const gpu::BindingLayout* spaces[] = {bindingLayout.value().get()};
    auto kernelLayout = m_impl->device.CreateKernelLayout(gpu::KernelLayoutDesc{
        std::span<const gpu::BindingLayout* const>(spaces), sizeof(ReconstructPushConstants)});
    if (!kernelLayout.has_value()) {
        LOG_ERROR("GpuSpectralReconstructor: Failed to create pipeline layout: {}", kernelLayout.error());
        return GpuReconstructorStatus::PipelineCreateFailed;
    }

    auto kernel = m_impl->device.CreateComputeKernel(
        gpu::ComputeKernelDesc{&shaders.value()[0], kernelLayout.value().get()});
    if (!kernel.has_value()) {
        LOG_ERROR("GpuSpectralReconstructor: Failed to create compute pipeline: {}", kernel.error());
        return GpuReconstructorStatus::PipelineCreateFailed;
    }

    auto pool = m_impl->device.CreateBindGroupPool(*bindingLayout.value(), 1);
    if (!pool.has_value()) {
        LOG_ERROR("GpuSpectralReconstructor: Failed to create descriptor pool: {}", pool.error());
        return GpuReconstructorStatus::PipelineCreateFailed;
    }

    m_impl->bindingLayout = std::move(bindingLayout.value());
    m_impl->kernelLayout = std::move(kernelLayout.value());
    m_impl->kernel = std::move(kernel.value());
    m_impl->pool = std::move(pool.value());

    m_impl->initialized = true;
    LOG_INFO("GpuSpectralReconstructor: Initialization complete");

    return GpuReconstructorStatus::Success;
}

bool GpuSpectralReconstructor::IsReady() const {
    return m_impl->initialized;
}

void GpuSpectralReconstructor::Shutdown() {
    m_impl->Cleanup();
}

std::pair<GpuReconstructorStatus, SpectralCube> GpuSpectralReconstructor::Reconstruct(
    const SpectralCube& sparseCube,
    const AdaptiveGridInfo& gridInfo,
    const HyperspectralConfig& targetConfig,
    const GpuReconstructorConfig& config
) {
    // Generate target wavelength list from config
    Vector<f32> targetWavelengths;
    u32 numBands = targetConfig.GetNumBands();
    targetWavelengths.reserve(numBands);

    for (u32 i = 0; i < numBands; ++i) {
        targetWavelengths.push_back(targetConfig.GetWavelength(i));
    }

    return ReconstructCustom(sparseCube, targetWavelengths, config);
}

std::pair<GpuReconstructorStatus, SpectralCube> GpuSpectralReconstructor::ReconstructCustom(
    const SpectralCube& sparseCube,
    const Vector<f32>& targetWavelengths,
    const GpuReconstructorConfig& config
) {
    using Clock = std::chrono::high_resolution_clock;
    auto startTime = Clock::now();

    // Validate input
    if (!sparseCube.IsValid()) {
        LOG_ERROR("GpuSpectralReconstructor: Invalid sparse cube");
        return {GpuReconstructorStatus::InvalidInput, SpectralCube{}};
    }

    if (targetWavelengths.empty()) {
        LOG_ERROR("GpuSpectralReconstructor: Empty target wavelengths");
        return {GpuReconstructorStatus::InvalidInput, SpectralCube{}};
    }

    if (!m_impl->initialized) {
        auto status = Initialize();
        if (status != GpuReconstructorStatus::Success) {
            return {status, SpectralCube{}};
        }
    }

    u32 width = sparseCube.width;
    u32 height = sparseCube.height;
    u32 srcBands = sparseCube.nbands;
    u32 dstBands = static_cast<u32>(targetWavelengths.size());

    if (config.verbose) {
        LOG_INFO("GpuSpectralReconstructor: Reconstructing {}x{}x{} from {} source bands",
                 width, height, dstBands, srcBands);
    }

    // Calculate buffer sizes. Dimensions convert to u64 before multiplying:
    // as u32 the cube-size product only stays exact below 2^32 elements, and
    // the check belongs here, not at the overflow.
    const u64 srcCubeSize = static_cast<u64>(width) * height * srcBands * sizeof(f32);
    const u64 dstCubeSize = static_cast<u64>(width) * height * dstBands * sizeof(f32);
    const u64 srcWlSize = srcBands * sizeof(f32);
    const u64 dstWlSize = dstBands * sizeof(f32);
    // One mapping entry per target band (uint4 + float2), built once on the
    // CPU: the shader reads indices and weights from it instead of scanning
    // the source grid for every pixel and band.
    const u64 mappingSize = dstBands * 6 * sizeof(u32);

    m_impl->lastMemoryUsage = srcCubeSize + dstCubeSize + srcWlSize + dstWlSize + mappingSize;

    try {
        gpu::Device& device = m_impl->device;

        // 5 device-local buffers.
        auto srcCubeBuffer = CreateBuffer(device, srcCubeSize,
            gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst, gpu::MemoryClass::DeviceLocal);
        if (!srcCubeBuffer.has_value()) return {srcCubeBuffer.error(), SpectralCube{}};
        auto dstCubeBuffer = CreateBuffer(device, dstCubeSize,
            gpu::BufferUsage::Storage | gpu::BufferUsage::TransferSrc, gpu::MemoryClass::DeviceLocal);
        if (!dstCubeBuffer.has_value()) return {dstCubeBuffer.error(), SpectralCube{}};
        auto srcWlBuffer = CreateBuffer(device, srcWlSize,
            gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst, gpu::MemoryClass::DeviceLocal);
        if (!srcWlBuffer.has_value()) return {srcWlBuffer.error(), SpectralCube{}};
        auto dstWlBuffer = CreateBuffer(device, dstWlSize,
            gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst, gpu::MemoryClass::DeviceLocal);
        if (!dstWlBuffer.has_value()) return {dstWlBuffer.error(), SpectralCube{}};

        // Pre-computed band mapping (binding 4)
        auto mappingBuffer = CreateBuffer(device, mappingSize,
            gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst, gpu::MemoryClass::DeviceLocal);
        if (!mappingBuffer.has_value()) return {mappingBuffer.error(), SpectralCube{}};

        auto mappingStaging = CreateBuffer(device, mappingSize,
            gpu::BufferUsage::TransferSrc, gpu::MemoryClass::HostOnly);
        if (!mappingStaging.has_value()) return {mappingStaging.error(), SpectralCube{}};

        // Staging buffers for upload
        auto srcCubeStaging = CreateBuffer(device, srcCubeSize,
            gpu::BufferUsage::TransferSrc, gpu::MemoryClass::HostOnly);
        if (!srcCubeStaging.has_value()) return {srcCubeStaging.error(), SpectralCube{}};

        auto srcWlStaging = CreateBuffer(device, srcWlSize,
            gpu::BufferUsage::TransferSrc, gpu::MemoryClass::HostOnly);
        if (!srcWlStaging.has_value()) return {srcWlStaging.error(), SpectralCube{}};

        auto dstWlStaging = CreateBuffer(device, dstWlSize,
            gpu::BufferUsage::TransferSrc, gpu::MemoryClass::HostOnly);
        if (!dstWlStaging.has_value()) return {dstWlStaging.error(), SpectralCube{}};

        std::unique_ptr<gpu::Buffer> srcCube = std::move(srcCubeBuffer.value());
        std::unique_ptr<gpu::Buffer> dstCube = std::move(dstCubeBuffer.value());
        std::unique_ptr<gpu::Buffer> srcWl = std::move(srcWlBuffer.value());
        std::unique_ptr<gpu::Buffer> dstWl = std::move(dstWlBuffer.value());
        std::unique_ptr<gpu::Buffer> mapping = std::move(mappingBuffer.value());

        // Upload source cube data
        {
            f32* mapped = static_cast<f32*>(srcCubeStaging.value()->Map());
            std::memcpy(mapped, sparseCube.data.data(), srcCubeSize);
            srcCubeStaging.value()->Unmap();
        }

        // Upload source wavelengths
        {
            f32* mapped = static_cast<f32*>(srcWlStaging.value()->Map());
            std::memcpy(mapped, sparseCube.wavelengths.data(), srcWlSize);
            srcWlStaging.value()->Unmap();
        }

        // Upload target wavelengths
        {
            f32* mapped = static_cast<f32*>(dstWlStaging.value()->Map());
            std::memcpy(mapped, targetWavelengths.data(), dstWlSize);
            dstWlStaging.value()->Unmap();
        }

        // Build and upload the per-band mapping: the search depends only on
        // the two wavelength grids, so it happens once here, not per pixel.
        Vector<GpuMappingEntry> mappingEntries(dstBands);
        for (u32 b = 0; b < dstBands; ++b)
            mappingEntries[b] =
                BuildMappingEntry(b, targetWavelengths[b], sparseCube.wavelengths);
        {
            void* mapped = mappingStaging.value()->Map();
            std::memcpy(mapped, mappingEntries.data(), mappingSize);
            mappingStaging.value()->Unmap();
        }

        // Copy staging to GPU buffers, then a barrier before compute. Each
        // copy is its own CopyBuffer call: one recording call is one native
        // command (gpu/AGENTS.md), which the command-stream diff depends on.
        device.ExecuteImmediate([&](gpu::CommandList& cmd) {
            const gpu::BufferCopyRegion srcCubeCopy{0, 0, srcCubeSize};
            cmd.CopyBuffer(srcCubeStaging.value()->Ref(), srcCube->Ref(),
                           std::span<const gpu::BufferCopyRegion>{&srcCubeCopy, 1});

            const gpu::BufferCopyRegion srcWlCopy{0, 0, srcWlSize};
            cmd.CopyBuffer(srcWlStaging.value()->Ref(), srcWl->Ref(),
                           std::span<const gpu::BufferCopyRegion>{&srcWlCopy, 1});

            const gpu::BufferCopyRegion dstWlCopy{0, 0, dstWlSize};
            cmd.CopyBuffer(dstWlStaging.value()->Ref(), dstWl->Ref(),
                           std::span<const gpu::BufferCopyRegion>{&dstWlCopy, 1});

            const gpu::BufferCopyRegion mappingCopy{0, 0, mappingSize};
            cmd.CopyBuffer(mappingStaging.value()->Ref(), mapping->Ref(),
                           std::span<const gpu::BufferCopyRegion>{&mappingCopy, 1});

            cmd.Barrier(gpu::GlobalBarrier{gpu::Access::TransferWrite, gpu::Access::ComputeRead});
        });

        // Allocate the descriptor set for this call. The pool holds one set
        // and is reset at the end of the call.
        auto groups = m_impl->pool->Allocate(1);
        if (!groups.has_value()) {
            LOG_ERROR("GpuSpectralReconstructor: Failed to allocate descriptor set");
            return {GpuReconstructorStatus::BufferAllocFailed, SpectralCube{}};
        }
        const gpu::BindGroup group = groups.value()[0];

        // Write the descriptor set; each range is its buffer's size.
        const gpu::BindingWrite writes[] = {
            gpu::BindingWrite::Buffer(0, gpu::BindingKind::ReadOnlyBuffer, srcCube->Ref(), 0, srcCubeSize),
            gpu::BindingWrite::Buffer(1, gpu::BindingKind::ReadWriteBuffer, dstCube->Ref(), 0, dstCubeSize),
            gpu::BindingWrite::Buffer(2, gpu::BindingKind::ReadOnlyBuffer, srcWl->Ref(), 0, srcWlSize),
            gpu::BindingWrite::Buffer(3, gpu::BindingKind::ReadOnlyBuffer, dstWl->Ref(), 0, dstWlSize),
            gpu::BindingWrite::Buffer(4, gpu::BindingKind::ReadOnlyBuffer, mapping->Ref(), 0, mappingSize),
        };
        device.WriteBindGroup(group, std::span<const gpu::BindingWrite>(writes));

        // Setup push constants
        ReconstructPushConstants pushConstants{};
        pushConstants.width = width;
        pushConstants.height = height;
        pushConstants.srcBands = srcBands;
        pushConstants.dstBands = dstBands;
        pushConstants.interpolationMode = InterpolationModeValue(config.method);

        // Dispatch compute shader
        device.ExecuteImmediate([&](gpu::CommandList& cmd) {
            cmd.SetKernel(*m_impl->kernel);
            cmd.SetBindGroup(*m_impl->kernelLayout, 0, group);
            cmd.PushConstants(*m_impl->kernelLayout, pushConstants);

            // Calculate workgroup count
            u32 groupCountX = (width + config.workgroupSizeX - 1) / config.workgroupSizeX;
            u32 groupCountY = (height + config.workgroupSizeY - 1) / config.workgroupSizeY;

            cmd.Dispatch(groupCountX, groupCountY, 1);

            cmd.Barrier(gpu::GlobalBarrier{gpu::Access::ComputeWrite, gpu::Access::TransferRead});
        });

        // Read back results
        auto dstCubeStaging = CreateBuffer(device, dstCubeSize,
            gpu::BufferUsage::TransferDst, gpu::MemoryClass::HostOnly);
        if (!dstCubeStaging.has_value()) return {dstCubeStaging.error(), SpectralCube{}};

        device.ExecuteImmediate([&](gpu::CommandList& cmd) {
            const gpu::BufferCopyRegion readbackCopy{0, 0, dstCubeSize};
            cmd.CopyBuffer(dstCube->Ref(), dstCubeStaging.value()->Ref(),
                           std::span<const gpu::BufferCopyRegion>{&readbackCopy, 1});
        });

        // Create result cube
        SpectralCube result(width, height, dstBands,
            targetWavelengths.front(), targetWavelengths.back());
        result.wavelengths = targetWavelengths;

        // Copy data from staging buffer. Deliberately Map() and not MapRead():
        // the copy completed before ExecuteImmediate returned, and this read
        // must not pay for a cache invalidation.
        {
            f32* mapped = static_cast<f32*>(dstCubeStaging.value()->Map());
            std::memcpy(result.data.data(), mapped, dstCubeSize);
            dstCubeStaging.value()->Unmap();
        }

        // Add metadata
        result.metadata = sparseCube.metadata;
        result.metadata["reconstruction_method"] =
            (config.method == InterpolationMethod::Linear) ? "gpu_linear" :
            (config.method == InterpolationMethod::CatmullRom) ? "gpu_catmull_rom" : "gpu_akima";

        // Reset descriptor pool for next use
        m_impl->pool->Reset();

        // Calculate statistics
        auto endTime = Clock::now();
        m_impl->lastReconstructionTime = std::chrono::duration<f64>(endTime - startTime).count();
        m_impl->lastPixelCount = width * height;

        if (config.verbose) {
            LOG_INFO("GpuSpectralReconstructor: Complete in {:.3f}s ({:.2f} Mpixels/s)",
                     m_impl->lastReconstructionTime,
                     (m_impl->lastPixelCount / 1e6) / m_impl->lastReconstructionTime);
        }

        return {GpuReconstructorStatus::Success, std::move(result)};
    }
    catch (const std::exception& e) {
        LOG_ERROR("GpuSpectralReconstructor: Exception: {}", e.what());
        return {GpuReconstructorStatus::OutOfMemory, SpectralCube{}};
    }
}

f64 GpuSpectralReconstructor::GetLastReconstructionTime() const {
    return m_impl->lastReconstructionTime;
}

u64 GpuSpectralReconstructor::GetLastMemoryUsage() const {
    return m_impl->lastMemoryUsage;
}

f64 GpuSpectralReconstructor::GetPixelsPerSecond() const {
    if (m_impl->lastReconstructionTime > 0) {
        return static_cast<f64>(m_impl->lastPixelCount) / m_impl->lastReconstructionTime;
    }
    return 0.0;
}

// ============================================================================
// Utility Functions
// ============================================================================

bool IsGpuReconstructionSupported(gpu::Device& device) {
    // Check for compute shader support (always available in Vulkan) and for
    // sufficient buffer size limits.
    // Need at least 256MB for typical hyperspectral cubes
    constexpr u64 kMinBufferSize = 256ull * 1024 * 1024;

    return device.Info().maxStorageBufferRange >= kMinBufferSize;
}

u64 EstimateReconstructionMemory(const SpectralCube& sparseCube, u32 targetBands) {
    u64 srcSize = sparseCube.width * sparseCube.height * sparseCube.nbands * sizeof(f32);
    u64 dstSize = sparseCube.width * sparseCube.height * targetBands * sizeof(f32);
    u64 wlSize = (sparseCube.nbands + targetBands) * sizeof(f32);
    u64 stagingSize = srcSize + dstSize;  // For upload/download

    return srcSize + dstSize + wlSize + stagingSize;
}

} // namespace quantiloom
