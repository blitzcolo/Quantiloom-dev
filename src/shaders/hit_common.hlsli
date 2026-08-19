// ============================================================================
// Quantiloom - Hit-side bindings and surface sampling
// ============================================================================
// What both hit-group stages need to look at a surface: the geometry indirection
// (indices, UVs, per-instance offsets), the material buffer, the bindless
// texture arrays, and the two helpers that turn a slot plus a UV into a texel.
//
// Included ONLY by closesthit.rchit and anyhit.rahit. Deliberately not folded
// into common.hlsli, which raygen, the miss shaders and every compute shader
// also include -- those stages have no business declaring a bindless texture
// array they never sample, and the descriptor set layout names the stages that
// may reach each binding.
//
// Bindings 1/2/3/9/16 and up stay in closesthit.rchit: the acceleration
// structure, lighting, positions, tangents and normals are closest-hit's alone.
// An any-hit answers one question -- is this intersection there at all -- and
// that question needs indices, UVs and a material, nothing more.
// ============================================================================

#ifndef QUANTILOOM_HIT_COMMON_HLSLI
#define QUANTILOOM_HIT_COMMON_HLSLI

#include "common.hlsli"

// ============================================================================
// Bindings
// ============================================================================

[[vk::binding(4, 0)]] StructuredBuffer<uint> indexBuffer;       // Triangle indices
[[vk::binding(5, 0)]] StructuredBuffer<MaterialData> materials; // Material properties
[[vk::binding(6, 0)]] Texture2D textures[];                     // Bindless texture array
[[vk::binding(7, 0)]] SamplerState samplers[];                  // Bindless sampler array
[[vk::binding(8, 0)]] StructuredBuffer<float2> uvBuffer;        // UV coordinates (optional)

// ============================================================================
// Instance Geometry Info Buffer (Binding 18)
// ============================================================================
// Per-TLAS-instance geometry offset information for multi-BLAS support.
// When scene has multiple BLAS, each instance's geometry data is merged into
// global buffers. This buffer tells us where each instance's data starts.
//
// USAGE:
//   uint instIdx = InstanceIndex();
//   InstanceGeometryInfo geo = instanceGeometryInfo[instIdx];
//   uint idx = indexBuffer[geo.indexOffset + PrimitiveIndex() * 3 + i];
//   float3 pos = vertexBuffer[geo.vertexOffset + idx];
//   float3 nrm = normalBuffer[geo.normalOffset + idx];
// ============================================================================

[[vk::binding(18, 0)]] StructuredBuffer<InstanceGeometryInfo> instanceGeometryInfo;

// ============================================================================
// Hit Attributes
// ============================================================================
// Barycentric coordinates of hit point within triangle.
//
// Shared because both stages of one hit group must agree on the attribute
// type: the closest hit and the any-hit are handed the same record.
// ============================================================================

struct HitAttributes {
    [[vk::location(0)]] float2 bary : SV_Barycentrics;  // Barycentric coordinates (b1, b2), where b0 = 1 - b1 - b2
};

// ============================================================================
// Texture sampling
// ============================================================================

// Maximum valid texture index (must match MAX_TEXTURES in RayTracingPipeline.cpp)
// CRITICAL: This bounds check prevents GPU hangs from invalid descriptor access
static const int MAX_TEXTURE_INDEX = 1024;

// Ray tracing has no screen-space derivatives, so the mip level has to be
// stated rather than inferred -- and every fetch here states 0.
//
// There used to be a ray-differential LOD path: the payload carried dD/dx and
// dD/dy, and ComputeUVDifferentialX/Y turned them into a UV footprint. It never
// worked and never ran. Both of those functions returned a hardcoded heuristic
// (`float2(t * length(dDdx), 0) * 0.001`) rather than an actual UV gradient,
// neither had a single caller, and the only path into the LOD computation passed
// literal zeros -- whose log2 is -inf, so the clamp handed back LOD 0 anyway.
// 24 of the payload's 56 bytes existed to feed it.
//
// Filtering the indirect bounces would want this back, and would want it
// computed rather than guessed. Until then LOD 0 is what the renderer does, said
// once, in one place.
float4 SampleTexture(int textureIndex, int samplerIndex, float2 uv, float4 fallback) {
    // Check both lower AND upper bounds to prevent invalid descriptor access
    // Invalid indices (negative or out-of-range) can cause GPU hangs with PARTIALLY_BOUND descriptors
    if (textureIndex < 0 || textureIndex >= MAX_TEXTURE_INDEX) {
        return fallback;
    }
    // Ensure sampler index is also valid (use same index as texture for 1:1 mapping)
    if (samplerIndex < 0 || samplerIndex >= MAX_TEXTURE_INDEX) {
        return fallback;
    }
    return textures[NonUniformResourceIndex(textureIndex)].SampleLevel(
        samplers[NonUniformResourceIndex(samplerIndex)], uv, 0.0
    );
}

// ============================================================================
// KHR_texture_transform
// ============================================================================
// One slot's UV transform, pre-multiplied by ConvertMaterial into the 2x3
// affine this applies. The identity is (1,0,0,1) and (0,0), so a slot with no
// transform returns its argument exactly -- no arithmetic that could round.
//
// Applied per slot rather than once to the interpolated UV because glTF puts
// the transform on the textureInfo, not on the material: SheenChair's fabric
// scales its base colour by 7 and its normal map by 2 in one material.
// ============================================================================
float2 TransformUV(MaterialData mat, int slot, float2 uv) {
    const float4 m = mat.uvTransformMat[slot];
    const float2 t = mat.uvTransformOffset[slot];
    return float2(m.x * uv.x + m.y * uv.y, m.z * uv.x + m.w * uv.y) + t;
}

// ============================================================================
// glTF alphaMode
// ============================================================================
// The coverage of a surface at one texel: baseColorFactor's alpha times the
// base colour texture's, with that slot's KHR_texture_transform applied. The
// fallback's alpha of 1 is what lets an untextured material fall through
// unchanged.
//
// Note this reads the same texel the closest-hit shader's baseColor does, and
// the sRGB transfer applies to RGB only -- alpha stays linear, so the two agree
// on what the author wrote. BC7 is off in this build (build_wsl.sh passes
// -DQUANTILOOM_USE_BC7ENC=OFF); if it is ever turned back on, its per-block
// alpha loss would land exactly on a MASK cutoff and want a look.
// ============================================================================
float SurfaceAlpha(MaterialData mat, float2 uv) {
    return mat.baseColorFactor.a *
           SampleTexture(mat.baseColorTextureIndex, mat.baseColorTextureIndex,
                         TransformUV(mat, UV_SLOT_BASE_COLOR, uv),
                         float4(1.0, 1.0, 1.0, 1.0)).a;
}

#endif // QUANTILOOM_HIT_COMMON_HLSLI
