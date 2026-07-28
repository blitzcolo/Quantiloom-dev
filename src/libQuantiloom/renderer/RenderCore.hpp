/**
 * @file RenderCore.hpp
 * @brief Render orchestration shared by the offline CLI and the GUI's ExternalRenderContext
 *
 * Internal on purpose: this header is not in include/quantiloom/ and nothing here
 * carries QL_API. Consumers reach it through a facade -- ExternalRenderContext for
 * the GUI today, an offline context for the CLI once the migration below finishes.
 * The unit tests link quantiloom_core and see it directly.
 *
 * ## Why this exists
 *
 * `src/app/main.cpp` drives the Vulkan layer itself -- GpuBuffer, CommandHelper,
 * BLAS/TLAS, RayTracingPipeline -- as a ~25-stage pipeline that runs in parallel with
 * the equivalent private steps inside ExternalRenderContext::Impl. Two orchestrators
 * for one renderer means a fix lands in one and not the other, and they have already
 * drifted: the GUI runs the sensor chain on the GPU while the CLI runs GenericSensor
 * on the CPU, and the two equirectangular-to-cubemap conversions disagreed on which
 * way is up (see EquirectToCubemap below).
 *
 * Stages move here one at a time, each verified by a render before the next one
 * starts. When the set is complete, ExternalRenderContext becomes a thin adapter over
 * it and the CLI stops needing library internals -- which is what still keeps ~274
 * symbols exported for its sake alone.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Config.hpp"
#include "core/Image.hpp"
#include "core/Types.hpp"
#include "renderer/AccelerationStructure.hpp"
#include "renderer/BRDFLutGenerator.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/MaterialGpuData.hpp"
#include "scene/Scene.hpp"

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

namespace quantiloom {
class GpuImage;
class VulkanContext;
}

namespace quantiloom::rendercore {

/**
 * @brief Build a Scene from a parsed TOML scene configuration
 *
 * Resolves `scene.usd`, then `scene.gltf`. A config naming neither is an error:
 * quietly rendering something else turns a misspelt key into a wrong picture rather
 * than a message. Loading files is all this does -- no GPU resources are touched, so
 * both the offline path and the interactive context can call it before they have a
 * device.
 *
 * @note Merged from two implementations that had drifted. The CLI also accepted a
 *       `scene.preset` naming one of three meshes built in code, and fell back to a
 *       procedural Cornell box when a config named no scene at all;
 *       ExternalRenderContext::LoadScene did neither. Both went away rather than
 *       being adopted: assets/models/cornell_box/ holds a committed Cornell box
 *       built to the original specification with ECOSTRESS spectral reflectances,
 *       which the cornell_box_{vis,swir,mwir,lwir} configs use, and a scene whose
 *       materials carry no spectral data is of little use to a spectral renderer.
 * @note The two disagreed on precedence when a config names both a USD and a glTF:
 *       the CLI took the USD, the context the glTF. No config in assets/configs does
 *       both, so the case was unreachable either way; the CLI's order is kept
 *       because it is the one covered by a render test.
 */
Result<Scene, String> LoadSceneFromConfig(const Config& config);

/**
 * @brief Convert an equirectangular (latitude-longitude) environment map to cubemap faces
 *
 * @param equirect Source map. Row 0 is the zenith, which is what OpenEXR's scanline
 *                 order gives and what ImageIO::ReadEXR passes through unchanged.
 * @param faceSize Edge length of each square face
 * @return Six RGB faces in Vulkan order: +X, -X, +Y, -Y, +Z, -Z
 *
 * @note This replaced two implementations that were vertically mirrored relative to
 *       each other. The CLI mapped v = acos(y)/pi, so +Y sampled row 0; the GUI mapped
 *       v = (asin(y) + pi/2)/pi, so +Y sampled the last row -- the ground. Since
 *       acos(y)/pi == 1 - (asin(y) + pi/2)/pi exactly, the GUI rendered every
 *       environment map upside down. The CLI's mapping is the one kept.
 */
Vector<Image> EquirectToCubemap(const Image& equirect, u32 faceSize);

/**
 * @brief The split-sum BRDF integration LUT for image-based lighting, on the GPU
 *
 * Owns the image and its sampler together, because they are bound together and
 * because both call sites previously tracked the sampler by hand -- one of them
 * checking the create result, the other dropping it.
 *
 * Generation is Monte Carlo on the CPU and costs seconds at the default 512x1024,
 * so Create() reads a cached binary when one is present and writes one when it is
 * not. The cache is keyed on the config, so a changed resolution or sample count
 * regenerates rather than loading a mismatched LUT.
 */
class BrdfLut {
public:
    BrdfLut() = default;
    ~BrdfLut();

    BrdfLut(BrdfLut&&) noexcept;
    BrdfLut& operator=(BrdfLut&&) noexcept;
    BrdfLut(const BrdfLut&) = delete;
    BrdfLut& operator=(const BrdfLut&) = delete;

    /**
     * @brief Load or generate the LUT, upload it, and create its sampler
     *
     * @param ctx       Device to allocate on
     * @param cachePath Binary cache, read when present and written when not
     * @param config    Resolution and sample count; the image follows
     *                  config.resolution rather than a hardcoded 512, which is what
     *                  both call sites did -- a changed resolution would have
     *                  uploaded into a mismatched image
     * @return An invalid BrdfLut if the sampler could not be created
     */
    static BrdfLut Create(VulkanContext& ctx,
                          const String& cachePath = "assets/luts/brdf_lut_512_ggx.bin",
                          const BRDFLutGenerator::Config& config = {});

    [[nodiscard]] bool IsValid() const { return m_sampler != VK_NULL_HANDLE; }
    [[nodiscard]] VkImageView View() const;
    [[nodiscard]] VkSampler Sampler() const { return m_sampler; }
    [[nodiscard]] u32 Resolution() const { return m_resolution; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    std::unique_ptr<GpuImage> m_image;
    VkSampler m_sampler = VK_NULL_HANDLE;
    u32 m_resolution = 0;
};

/**
 * @brief The IBL environment cubemap: six faces with a mip chain, ready to bind
 *
 * The mip chain stands in for GGX prefiltering, which neither implementation this
 * replaced actually did -- both carried a TODO. The shader reads the level count at
 * runtime (`GetDimensions`) and maps `lod = roughness * (numMips - 1)`, so the level
 * count is a quality knob rather than a contract: more levels means a rougher
 * surface reaches a blurrier level.
 *
 * Carries its own sampler. It used to share the one bound for the BRDF LUT, whose
 * maxLod is 0 -- right for a single-level LUT, and enough to pin every environment
 * lookup to mip 0, so the chain above was built and never read.
 */
class EnvironmentCubemap {
public:
    struct Params {
        u32 faceSize = 512;
        /// Clamped to what faceSize supports -- see MipLevels() for what was used.
        u32 mipLevels = 8;
    };

    /// What a scene with no environment map configured gets: uniform sky blue.
    static constexpr Params kFallbackParams{256, 5};

    EnvironmentCubemap() = default;
    // Out of line: GpuImage is only forward declared here, so the destructor cannot
    // be implicit -- it would need the complete type at every use site.
    ~EnvironmentCubemap();
    EnvironmentCubemap(EnvironmentCubemap&&) noexcept;
    EnvironmentCubemap& operator=(EnvironmentCubemap&&) noexcept;
    EnvironmentCubemap(const EnvironmentCubemap&) = delete;
    EnvironmentCubemap& operator=(const EnvironmentCubemap&) = delete;

    /**
     * @brief Load an equirectangular HDR image and convert it to a cubemap
     *
     * @note Reads through ImageIO::ReadImage, so .exr, .hdr and the LDR formats all
     *       work. The CLI used to call ReadEXR directly and silently fall back to
     *       sky blue for anything else.
     */
    static Result<EnvironmentCubemap, String> Load(VulkanContext& ctx,
                                                   const String& path,
                                                   const Params& requested = {});

    /// Uniform sky blue, for a scene that configures no environment map.
    static EnvironmentCubemap Fallback(VulkanContext& ctx,
                                       const Params& requested = kFallbackParams);

    [[nodiscard]] bool IsValid() const { return m_image != nullptr && m_sampler != VK_NULL_HANDLE; }
    [[nodiscard]] VkImageView View() const;
    /// Trilinear and unclamped, so `lod` from the shader actually selects a level.
    [[nodiscard]] VkSampler Sampler() const { return m_sampler; }
    [[nodiscard]] u32 FaceSize() const { return m_faceSize; }
    [[nodiscard]] u32 MipLevels() const { return m_mipLevels; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    std::unique_ptr<GpuImage> m_image;
    VkSampler m_sampler = VK_NULL_HANDLE;
    u32 m_faceSize = 0;
    u32 m_mipLevels = 0;
};

/**
 * @brief CPU mirror of the shader's InstanceGeometryInfo (descriptor binding 18)
 *
 * One entry per TLAS instance, indexed by InstanceIndex(). It carries where that
 * instance's slice of each merged global buffer starts, which is how the closest-hit
 * shader finds its vertices after every primitive was concatenated into one buffer.
 *
 * @note Defined once. It used to be declared separately in ExternalRenderContext.cpp
 *       and in the CLI's main.cpp, two copies of a layout the shader depends on.
 */
struct InstanceGeometryInfo {
    u32 vertexOffset;   // Offset into the merged vertex buffer, in vertices
    u32 indexOffset;    // Offset into the merged index buffer, in indices
    u32 normalOffset;   // Offset into the merged normal buffer, in normals
    u32 uvOffset;       // Offset into the merged UV buffer, in UVs
    u32 tangentOffset;  // Offset into the merged tangent buffer, in tangents
    u32 materialId;     // Index into Scene::materials
    u32 pad[2];         // Pad to 32 bytes
};

static_assert(sizeof(InstanceGeometryInfo) == 32, "InstanceGeometryInfo size mismatch");

/**
 * @brief Everything the ray tracing pipeline needs to trace a scene's geometry
 *
 * Concatenates every primitive's attributes into one buffer per attribute, builds a
 * BLAS per primitive and a TLAS over the node instances, and records where each
 * instance's slice begins so the shader can index back.
 *
 * Missing attributes are filled rather than omitted, so every buffer is indexable at
 * a primitive's vertex offset regardless of what the asset supplied: normals become
 * +Y, UVs (0,0), tangents (1,0,0,1).
 */
class SceneGeometry {
public:
    SceneGeometry() = default;
    ~SceneGeometry();
    SceneGeometry(SceneGeometry&&) noexcept;
    SceneGeometry& operator=(SceneGeometry&&) noexcept;
    SceneGeometry(const SceneGeometry&) = delete;
    SceneGeometry& operator=(const SceneGeometry&) = delete;

    /// Merge, build and upload. A scene with no primitives yields an invalid result
    /// rather than empty buffers, since there is nothing to trace.
    static SceneGeometry Build(VulkanContext& ctx, const Scene& scene);

    /// Rebuild only the TLAS, from the BLAS already built, after node transforms
    /// changed. A transform moves an instance; it does not touch per-primitive
    /// structures or the merged buffers, so neither is redone.
    void RebuildTlas(VulkanContext& ctx, const Scene& scene);

    [[nodiscard]] bool IsValid() const { return m_tlas != nullptr; }

    [[nodiscard]] const TLAS& Tlas() const { return *m_tlas; }
    [[nodiscard]] const GpuBuffer& Vertices() const { return *m_vertices; }
    [[nodiscard]] const GpuBuffer& Indices() const { return *m_indices; }
    [[nodiscard]] const GpuBuffer& Normals() const { return *m_normals; }
    [[nodiscard]] const GpuBuffer& UVs() const { return *m_uvs; }
    [[nodiscard]] const GpuBuffer& Tangents() const { return *m_tangents; }
    [[nodiscard]] const GpuBuffer& InstanceInfo() const { return *m_instanceInfo; }

    /// One per TLAS instance -- node count times primitives per mesh, not BLAS count.
    [[nodiscard]] u32 InstanceCount() const { return m_instanceCount; }
    /// One per primitive in the scene, shared by every node that instances the mesh.
    [[nodiscard]] u32 BlasCount() const { return static_cast<u32>(m_blas.size()); }
    [[nodiscard]] u32 VertexCount() const { return m_vertexCount; }
    [[nodiscard]] u32 IndexCount() const { return m_indexCount; }

    /// The offsets written to InstanceInfo(), kept for tests and diagnostics.
    [[nodiscard]] const Vector<InstanceGeometryInfo>& Instances() const { return m_instances; }

private:
    Vector<std::unique_ptr<BLAS>> m_blas;
    std::unique_ptr<TLAS> m_tlas;
    std::unique_ptr<GpuBuffer> m_vertices;
    std::unique_ptr<GpuBuffer> m_indices;
    std::unique_ptr<GpuBuffer> m_normals;
    std::unique_ptr<GpuBuffer> m_uvs;
    std::unique_ptr<GpuBuffer> m_tangents;
    std::unique_ptr<GpuBuffer> m_instanceInfo;
    Vector<InstanceGeometryInfo> m_instances;
    u32 m_instanceCount = 0;
    u32 m_vertexCount = 0;
    u32 m_indexCount = 0;
};

/**
 * @brief Where a material's spectral curve and refractive index live in their buffers
 *
 * Resolved by the caller because the two front ends know different things: the
 * interactive context reads the indices a Material already carries, while the CLI
 * resolves material names against the curves it loaded from the scene config.
 */
struct MaterialGpuIndices {
    i32 spectralReflectanceCurve = -1;  // -1: no curve, shader falls back to RGB
    i32 complexRefractiveIndex = -1;    // -1: no CRI, shader uses the F0 approximation
};

/**
 * @brief Convert a Material into the layout the closest-hit shader reads
 *
 * @param wavelengthNm Wavelength the IR curves are evaluated at
 *
 * @note The IR emissivity and transmittance are interpolated at `wavelengthNm`.
 *       ExternalRenderContext used to average each curve over its whole range
 *       instead, because the conversion was a file-static function with no way to
 *       reach the current wavelength -- which made the GUI's thermal response
 *       wavelength-independent, in a renderer whose reason to exist is that it is
 *       not. The two agree only for a flat curve.
 * @note Both are clamped to [0, 1]. Emissivity above 1 is unphysical, and the clamp
 *       was already on the context's side.
 */
MaterialDataCPU ConvertMaterial(const Material& material, f32 wavelengthNm,
                                const MaterialGpuIndices& indices = {});

/**
 * @brief Convert and upload every material in the scene
 *
 * @param indices One entry per scene material. Empty means read each Material's own
 *                spectralReflectanceCurveIndex and complexRefractiveIndexIndex.
 * @return null when the scene has no materials -- there is nothing to bind
 */
std::unique_ptr<GpuBuffer> BuildMaterialBuffer(VulkanContext& ctx, const Scene& scene,
                                               f32 wavelengthNm,
                                               const Vector<MaterialGpuIndices>& indices = {});

/**
 * @brief Create the ray tracing target: an RGBA32F storage image in GENERAL layout
 *
 * Recreated rather than resized, since a Vulkan image's extent is fixed at creation.
 * The caller rebinds it — the descriptor still points at the old view.
 *
 * @note One shape, three call sites: the context creates it at start-up and again on
 *       every Resize(), and the CLI creates one per run. All three specified the same
 *       format, usage and initial transition; only the surrounding code differed.
 */
std::unique_ptr<GpuImage> CreateRenderTarget(VulkanContext& ctx, u32 width, u32 height);

} // namespace quantiloom::rendercore
