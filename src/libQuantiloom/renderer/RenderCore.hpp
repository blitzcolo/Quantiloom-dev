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
 * ## Config reading came next, and it is done
 *
 * The other half of the same problem was that the two hosts also *read the config*
 * separately -- the CLI inside OfflineRenderer, Quantiloom Studio in its own
 * ConfigManager -- and those had drifted further than the orchestrators had: opposite
 * shadow-ray defaults, a band-centre wavelength rule only one of them had, a
 * normalisation key only one of them honoured, the NMF basis and [refractive_index]
 * read by one alone. That reading is now ConfigResolve.{hpp,cpp}, and both hosts go
 * through it: the CLI directly, Studio via ExternalRenderContext::ApplyConfig.
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
class RayTracingPipeline;
class TextureManager;
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
 *
 * @param baseDir Directory a relative path is tried against before being used
 *        as written. Empty -- what the CLI passes -- means as written, i.e.
 *        relative to the working directory. Both conventions are in use; see
 *        ResolveConfigPath.
 */
Result<Scene, String> LoadSceneFromConfig(const Config& config,
                                          const String& baseDir = {});

/**
 * @brief Resolve a path a config named, against the config's own directory
 *
 * Tries @p baseDir first and falls back to the path as written, so a
 * self-contained scene folder resolves against itself while a repo-root-relative
 * path keeps resolving against the working directory. An empty @p baseDir skips
 * the attempt. Defined in ConfigResolve.cpp, next to the rest of the reading.
 */
String ResolveConfigPath(const String& path, const String& baseDir);

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
 * @brief Whether traversal may skip the any-hit shader for a material
 *
 * Opaque geometry keeps the hardware fast path; alphaMode MASK and BLEND give
 * it up so a per-texel coverage test can run. A transmissive material counts as
 * opaque here on purpose -- KHR_materials_transmission is refractive
 * transparency, alpha is coverage, and running both removes the surface twice.
 *
 * Public because the answer is baked into the acceleration structure at build
 * time, so the interactive host has to ask it again after a material edit to
 * know whether the geometry went stale. Out of range means opaque.
 */
[[nodiscard]] bool IsOpaqueForRayTracing(const Scene& scene, u32 materialId);

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

    /// A placeholder, not an environment. The ray tracing pipeline declares
    /// binding 10 with descriptorCount 1 and no partially-bound flag, so a
    /// descriptor must be written there before the pipeline can be used, even
    /// for a scene that has no environment map at all. One black texel is the
    /// smallest thing that satisfies that. It used to be 256x256x5 mips of sky
    /// blue, which meant a scene that named no map was lit by an invented sky --
    /// visible in a preview, and a 46-84% contribution to the signal in a
    /// quantitative render. Nothing should ever sample this: the lighting flag
    /// enableEnvironmentMap is 0 whenever this is what is bound.
    static constexpr Params kFallbackParams{1, 1};

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

    /// A black placeholder to keep binding 10 valid for a scene that configures
    /// no environment map. See kFallbackParams for why it is not a sky.
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
    /// First thermal element of this instance, or 0xFFFFFFFF where no solve
    /// ran. The closest-hit shader adds PrimitiveIndex() to it, which is the
    /// whole of how a triangle finds the temperature the balance gave it.
    u32 thermalElementBase;
    u32 pad;            // Pad to 32 bytes
};

static_assert(sizeof(InstanceGeometryInfo) == 32, "InstanceGeometryInfo size mismatch");

/**
 * @brief One emissive triangle, in world space, for next-event estimation
 *
 * Light sampling needs to put a point on an emitter without tracing towards it
 * first, which means the emitters have to exist as geometry the shader can
 * address directly -- the acceleration structure only answers questions about
 * rays that already have a direction.
 *
 * World space rather than object space because that is what the shader needs
 * and nothing else would save work: the transform is applied once here per
 * build instead of once per sample. The cost is that a moved node invalidates
 * the buffer, which is why RefitTlas rebuilds it.
 *
 * `cumulativePower` is the running sum of `luminance(emissive) * area` up to
 * and including this triangle, so the last entry holds the total and a single
 * uniform variate selects a triangle by scanning for the first entry above
 * `u * total`. Choosing in proportion to power and then uniformly over the
 * triangle makes the area-measure density `luminance(emissive) / total`, in
 * which the area has cancelled -- see LightingParams::emissiveTotalPower.
 */
/*
 * `emissiveCurveIndex` took the first padding word, so the struct is still 64
 * bytes. It is the emitter's spectral radiance curve, or -1 when the emitter is
 * an RGB triple like every emitter used to be. Sampling is unaffected either
 * way: the CDF and the density are built from `emissive`, which the material
 * resolution guarantees is the linear-sRGB that same curve integrates to. Only
 * the radiance the shader returns changes, from an upsampled triple to the
 * curve -- which is the whole point, since the density may be approximate but
 * the radiance may not.
 */
struct EmissiveTriangleGPU {
    glm::vec3 v0;         float cumulativePower;  //  0..16
    glm::vec3 edge1;      float area;             // 16..32
    glm::vec3 edge2;      i32   emissiveCurveIndex; // 32..48
    glm::vec3 emissive;   float _pad1;            // 48..64
};

static_assert(sizeof(EmissiveTriangleGPU) == 64, "EmissiveTriangleGPU size mismatch");

/**
 * @brief The emissive triangles of a scene, in world space, power-ordered CDF built
 *
 * Walks the same node/primitive order as the geometry build, so a triangle's
 * material is the one the shader would find through InstanceGeometryInfo.
 *
 * Returns an empty vector when the scene has no emissive material, which is the
 * common case and the one that must stay free: with no triangles the shader
 * skips light sampling entirely and every sun-and-sky scene renders exactly as
 * it did before this existed.
 */
Vector<EmissiveTriangleGPU> CollectEmissiveTriangles(const Scene& scene);

/**
 * @brief Upload emissive triangles for binding 23
 *
 * Never returns null: an empty list still gets a one-element zero-filled buffer,
 * because the descriptor has to be written with something and
 * LightingParams::emissiveTriangleCount is what says whether to read it.
 */
std::unique_ptr<GpuBuffer> CreateEmissiveTriangleBuffer(
    VulkanContext& ctx, const Vector<EmissiveTriangleGPU>& triangles);

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

    /// Rebuild only the TLAS, from the BLAS already built, after node
    /// transforms or the node set changed. Handles topology edits
    /// (DuplicateNode/RemoveNode): the instance tables and the GPU instance
    /// info buffer are regenerated from the current walk, so the caller must
    /// rebind both the TLAS and InstanceInfo() afterwards. Per-primitive
    /// structures and the merged buffers are geometry, not placement, and
    /// are never redone.
    void RebuildTlas(VulkanContext& ctx, const Scene& scene);

    /**
     * @brief Rebuild any BLAS whose material changed its ray-tracing opacity
     *
     * VK_GEOMETRY_OPAQUE_BIT_KHR is baked into the acceleration structure at
     * build time, so flipping a material between OPAQUE and MASK/BLEND -- or
     * switching transmission on -- leaves the geometry behaving as it was
     * built until the structure is rebuilt. This finds the primitives that
     * disagree with their material, rebuilds those, and regenerates the TLAS.
     *
     * Returns true if anything was rebuilt, so a caller can skip resetting
     * accumulation when nothing moved. Cheap when nothing changed: one
     * comparison per primitive and no GPU work.
     */
    bool RefreshMaterialOpacity(VulkanContext& ctx, const Scene& scene);

    /// Refit the TLAS in place for transform-only edits: no allocation, no
    /// teardown, same handle -- cheap enough to run per mouse-move during a
    /// drag. Returns false without touching anything when a refit is not
    /// possible (nothing built yet, or the instance count changed, i.e. a
    /// topology edit); the caller falls back to a full rebuild.
    [[nodiscard]] bool RefitTlas(VulkanContext& ctx, const Scene& scene);

    [[nodiscard]] bool IsValid() const { return m_tlas != nullptr; }

    [[nodiscard]] const TLAS& Tlas() const { return *m_tlas; }
    [[nodiscard]] const GpuBuffer& Vertices() const { return *m_vertices; }
    [[nodiscard]] const GpuBuffer& Indices() const { return *m_indices; }
    [[nodiscard]] const GpuBuffer& Normals() const { return *m_normals; }
    [[nodiscard]] const GpuBuffer& UVs() const { return *m_uvs; }
    [[nodiscard]] const GpuBuffer& Tangents() const { return *m_tangents; }
    [[nodiscard]] const GpuBuffer& InstanceInfo() const { return *m_instanceInfo; }

    /// Point each instance at its first thermal element and re-upload.
    ///
    /// Separate from Build() because the solver runs after the geometry does:
    /// it needs the acceleration structure this built in order to find out who
    /// sees whom. Passing fewer bases than there are instances leaves the rest
    /// at the sentinel, which the shader reads as "no solver here".
    void SetThermalElementBases(const Vector<u32>& bases);

    /// One per TLAS instance -- node count times primitives per mesh, not BLAS count.
    [[nodiscard]] u32 InstanceCount() const { return m_instanceCount; }
    /// One per primitive in the scene, shared by every node that instances the mesh.
    [[nodiscard]] u32 BlasCount() const { return static_cast<u32>(m_blas.size()); }
    [[nodiscard]] u32 VertexCount() const { return m_vertexCount; }
    [[nodiscard]] u32 IndexCount() const { return m_indexCount; }

    /// The offsets written to InstanceInfo(), kept for tests and diagnostics.
    [[nodiscard]] const Vector<InstanceGeometryInfo>& Instances() const { return m_instances; }

    /// Scene::nodes index for each TLAS instance, in InstanceIndex() order.
    /// Filled by Build and regenerated by RebuildTlas, which is what keeps it
    /// current across topology edits (DuplicateNode/RemoveNode).
    [[nodiscard]] const Vector<u32>& InstanceToNode() const { return m_instanceToNode; }

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
    Vector<u32> m_instanceToNode;
    /// Per-primitive (globalPrim-indexed) geometry offsets from Build's first
    /// pass, kept so RebuildTlas can regenerate m_instances after a node is
    /// added or tombstoned. Geometry never changes at runtime, so this does
    /// not go stale.
    Vector<InstanceGeometryInfo> m_primitiveOffsets;
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

    // Endmember mixing. spectralReflectanceCurve above is endmember 0.
    // -1 for the weight texture means w = (1, 0, 0, 0), i.e. that curve alone.
    i32 endmemberCurve1 = -1;
    i32 endmemberCurve2 = -1;
    i32 endmemberCurve3 = -1;
    i32 weightTexture = -1;

    // Measured sheen reflectance, resolved from the same curve buffer as the
    // rest. -1: no curve, and the shader falls back to the RGB sheen factor
    // (visible bands only). Append new slots at the end -- IndicesFromMaterial
    // initialises this aggregate positionally.
    i32 sheenReflectanceCurve = -1;

    // Measured clearcoat reflectance. The only path by which a coat reaches
    // MWIR or LWIR, where a dielectric 0.04 would be fiction.
    i32 clearcoatReflectanceCurve = -1;

    // Measured diffuse transmission colour. The only path by which it reaches
    // NIR or SWIR, for the same reason sheen's curve is: a visible-basis RGB
    // factor says nothing past ~1400nm.
    i32 diffuseTransmissionColorCurve = -1;

    // Measured spectral radiance for self-emission. The emission-side twin of
    // lighting.solar_lut: the only path by which a light in the scene is
    // described by data rather than by an RGB triple expanded through D65.
    i32 emissiveRadianceCurve = -1;

    // Fluorescence, rank one: what the surface absorbs into the fluorescent
    // channel, and what it gives back. Both or neither -- one alone is a shape
    // with no partner and the material does not fluoresce. The yield rides on
    // the Material itself rather than here, being a scalar and not an index.
    i32 fluorescenceExcitationCurve = -1;
    i32 fluorescenceEmissionCurve = -1;
};

/**
 * @brief The indices a Material already carries, as a MaterialGpuIndices
 *
 * The interactive path resolves spectra into the Material itself and then has
 * to hand them back here, in two different places. One function so the two
 * cannot disagree about which fields count -- adding a slot and updating only
 * one caller is otherwise a silent partial upload.
 */
[[nodiscard]] inline MaterialGpuIndices IndicesFromMaterial(const Material& material) {
    return MaterialGpuIndices{material.spectralReflectanceCurveIndex,
                              material.complexRefractiveIndexIndex,
                              material.endmemberCurveIndex1,
                              material.endmemberCurveIndex2,
                              material.endmemberCurveIndex3,
                              material.weightTextureIndex,
                              material.sheenReflectanceCurveIndex,
                              material.clearcoatReflectanceCurveIndex,
                              material.diffuseTransmissionColorCurveIndex,
                              material.emissiveRadianceCurveIndex,
                              material.fluorescenceExcitationCurveIndex,
                              material.fluorescenceEmissionCurveIndex};
}

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
std::unique_ptr<GpuImage> CreateRenderTarget(VulkanContext& ctx, u32 width, u32 height,
                                             VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT);

/**
 * @brief Everything the ray tracing pipeline binds, in one place
 *
 * The two front ends bound the same sixteen resources in two hand-written
 * sequences. Naming the set once means a new binding is added where it is declared
 * rather than in whichever sequence the author happened to be reading.
 *
 * Null members are skipped. `RayTracingPipeline`'s buffer binders already accept a
 * null pointer for the optional spectral resources; the rest are simply not written.
 */
struct PipelineBindings {
    const GpuImage* outputImage = nullptr;
    const GpuImage* depthImage = nullptr;  // primary-hit depth AOV (binding 22)
    const SceneGeometry* geometry = nullptr;
    const GpuBuffer* lightingParams = nullptr;
    const GpuBuffer* materials = nullptr;
    const TextureManager* textures = nullptr;
    const EnvironmentCubemap* environment = nullptr;
    const BrdfLut* brdfLut = nullptr;

    // Optional quantitative-spectral resources. The CLI loads these from the scene
    // config; the interactive context binds zero-filled placeholders (see
    // CreateDummyBuffers) except for the refractive index, which Qt fills through
    // ExternalRenderContext::AddComplexRefractiveIndex.
    const GpuBuffer* spectralCurves = nullptr;
    const GpuBuffer* complexRefractiveIndex = nullptr;
    const GpuBuffer* solarLut = nullptr;
    const GpuBuffer* atmosphereHeader = nullptr;
    const GpuBuffer* atmosphereData = nullptr;
    const GpuBuffer* cieColourMatching = nullptr;
    /// Jakob-Hanika RGB -> spectrum coefficients. Always bound: binding 25 has
    /// no partially-bound flag, and every spectral band that meets an RGB
    /// reflectance without a measured curve reads it.
    const GpuBuffer* rgbToSpectrum = nullptr;
    const GpuBuffer* emissiveTriangles = nullptr;
    /// Per-element surface temperatures from the thermal solver. Always bound;
    /// a scene with no solve gets a single zero entry.
    const GpuBuffer* thermalTemperatures = nullptr;
    /// How those temperatures respond to the sun: a header record with the
    /// solve's sun direction, then (dT/dv, v_element) per element. Always
    /// bound; one zeroed record when there is no solve, which the header's
    /// w = 0 turns off.
    const GpuBuffer* thermalSunResponse = nullptr;
};

/**
 * @brief Build the ray tracing pipeline and bind its resources
 *
 * @param cache Pipeline cache to build against; may be VK_NULL_HANDLE. The caller
 *              owns it, because the two front ends keep it for different spans --
 *              the context across scene reloads, the CLI for one run.
 */
std::unique_ptr<RayTracingPipeline> CreateRayTracingPipeline(
    VulkanContext& ctx, VkPipelineCache cache, const PipelineBindings& bindings);

/**
 * @brief How a fused-band render has to be conditioned before the sensor chain
 *
 * `core/Types.hpp` states the contract next to the band table: a fused-band render
 * writes per-nm **average** spectral radiance, while the sensor chain wants
 * band-**integrated** radiance and the band's own photon energy. Getting either
 * wrong is not subtle -- for LWIR the two together are a factor of ~7e4.
 *
 * The CLI applied this inline and ExternalRenderContext did not apply it at all, so
 * the same scene through the same SensorParams produced a black frame in the GUI and
 * a correct one from the CLI.
 */
struct SensorBandAdjustment {
    /// Multiply radiance by this before the chain. 1.0 outside the fused IR modes.
    f32 radianceScale = 1.0f;
    /// Wavelength for photon energy, or 0 to leave the caller's value alone.
    f32 wavelengthNm = 0.0f;
};

/**
 * @brief Derive the sensor conditioning for a spectral mode
 *
 * @param hostSetWavelength True when the caller supplied a wavelength on purpose, in
 *                          which case it is left alone. The CLI reads this as "the
 *                          config named spectral.wavelength_nm"; the interactive
 *                          context as "the host passed something other than the
 *                          SensorParams default".
 */
SensorBandAdjustment SensorAdjustmentForMode(SpectralMode mode, bool hostSetWavelength);

/**
 * @brief Upload the CIE 1931 2-degree colour matching functions
 *
 * `x̄, ȳ, z̄` at 1 nm over 380-780 nm, padded to vec4 for the shader's structured
 * buffer stride. Needed by the VIS_Fused mode to integrate a spectrum to XYZ.
 *
 * @note The table is compiled in (`core/CIE_CMF_Data.hpp`) rather than read from
 *       assets/luts/CIE_xyz_1931_2deg.csv, which is what the CLI used to parse with
 *       a hand-rolled reader. The two agree to 2e-5 across all 401 shared samples --
 *       the CSV merely also covers 360-380 and 780-830, which the renderer clips
 *       away. A published constant does not need a load path or a file that can go
 *       missing.
 */
std::unique_ptr<GpuBuffer> CreateCieColourMatchingBuffer(VulkanContext& ctx);

/**
 * @brief Fit (or load) the RGB -> spectrum coefficient table and upload it
 *
 * Cached to @p cachePath, defaulting beside the BRDF LUT. A miss costs a few
 * seconds of fitting, once per machine; see core/RgbToSpectrum.hpp for why the
 * table is built rather than shipped.
 */
std::unique_ptr<GpuBuffer> CreateRgbToSpectrumBuffer(
    VulkanContext& ctx, const String& cachePath = "assets/luts/rgb2spec_srgb_64.bin");

} // namespace quantiloom::rendercore
