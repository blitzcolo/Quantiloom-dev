// ============================================================================
// Quantiloom - binding manifest of the ray tracing set
// ============================================================================
// The one list of the 38 bindings of the ray tracing descriptor set: Vulkan
// set 0, HLSL space 0, shared by raygen, closest-hit, any-hit, miss and the
// fusion geometry kernel. Plain preprocessor text that DXC and MSVC both read:
//
//   - HLSL. Every declaration of one of these resources takes its register
//     from its entry, through the macros in ql_bind.hlsli:
//         RWTexture2D<float4> outputImage : QL_RT_REGISTER(outputImage);
//     preprocesses to `register(u0, space0)`, the spelling it replaced, so the
//     .spv and the DXIL are the ones the hand-written register gave.
//   - C++. renderer/RtBindings.hpp expands every entry, in QL_RT_BINDINGS
//     order, into the gpu::BindingSlot table RayTracingPipeline creates the
//     descriptor set layout from, and static_asserts the rows.
//
// An entry is a function-like macro that hands its row to whichever row macro
// it is given, so each language expands the same entry its own way:
//
//   #define QL_RT_<role>(ROW)  ROW(class, slot, kind, count, visibility, partial)
//
//   class       the HLSL register letter: t (read-only buffer, sampled
//               texture, acceleration structure), u (read-write buffer,
//               storage texture), s (sampler)
//   slot        register number, Vulkan binding and table row at once
//   kind        gpu::BindingKind (gpu/Binding.hpp)                   C++ only
//   count       1, or QL_BINDLESS for the two unbounded arrays, which the
//               host replaces by the device's texture limit         C++ only
//   visibility  QL_VIS_* bits (RGen, CHit, AHit, Miss, Compute): the stage
//               set the Vulkan layout carries                        C++ only
//   partial     1 if partially bound: the unbounded arrays, and the
//               bindings only one of the three descriptor sets writes  C++ only
//
// Every HLSL row macro drops the C++-only columns, so their tokens (QL_VIS_R,
// QL_BINDLESS, StorageTexture, ...) never reach a shader; neither language may
// define them as macros. The entry is invoked with ROW(...) written out here
// rather than as a bare tuple behind an APPLY macro, so expanding it needs no
// rescan trick that MSVC's traditional preprocessor might read differently.
//
// The role names the binding, not one variable: slot 3 is declared as
// vertexBuffer, fusionVertices and geometryVertices, each with
// QL_RT_REGISTER(vertexPositions). A declaration keeps its own variable name,
// because the .spv take their debug names from it. Roles are lowerCamelCase;
// QL_RT_ followed by an upper-case word is an expansion macro.
//
// Adding a binding: an entry here at the next slot, its role at the end of
// QL_RT_BINDINGS, the declaration with QL_RT_REGISTER(role) and, past
// t37 / u36 / s21, the ranges of QL_ROOT_SIGNATURE in ql_bind.hlsli (a shader
// that declares beyond them fails the build's portability step).
// RtBindings.hpp rejects a role missing from QL_RT_BINDINGS or out of slot
// order, and a class that does not fit its kind, at compile time.
//
// Metal. A gpu/metal/ backend derives its argument-buffer layout from these
// same rows -- by expanding QL_RT_BINDINGS with a row macro of its own, or by
// reading kRtBindingTemplate -- and keeps no list of its own. Metal Shader
// Converter lays the top-level argument buffer out from QL_ROOT_SIGNATURE,
// whose first table is the SRV range t0..t37 followed by the UAV range
// u0..u36, so a row lands at:
//
//   class t, slot n      entry n of that table
//   class u, slot n      entry 38 + n of that table (38: the SRV range's length)
//   class s, slot n      entry n of the sampler table
//   count QL_BINDLESS    the unbounded space-1 table of its class (textures,
//                        samplers)
//
// kind gives the entry's resource type; visibility is dropped, an argument
// buffer not being per stage; partial means the entry may be null, which the
// residency bookkeeping has to allow for.
// ============================================================================

#ifndef QUANTILOOM_QL_RT_BINDINGS_HLSLI
#define QUANTILOOM_QL_RT_BINDINGS_HLSLI

//                                                 clsslotkind                   count        visibility                                 partial    declared as
#define QL_RT_outputImage(ROW)                 ROW(u, 0,  StorageTexture,        1,           QL_VIS_R | QL_VIS_X,                       0)  // outputImage (raygen.rgen)
#define QL_RT_scene(ROW)                       ROW(t, 1,  AccelerationStructure, 1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // scene (ql_trace.hlsli), geometryScene (fusion_geometry.rayq.hlsl)
#define QL_RT_lightingParams(ROW)              ROW(t, 2,  ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X, 0)  // lightingParams (shade_resources.hlsli)
#define QL_RT_vertexPositions(ROW)             ROW(t, 3,  ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_A | QL_VIS_X, 0)  // vertexBuffer (ql_trace.hlsli), fusionVertices (anyhit.rahit), geometryVertices (fusion_geometry.rayq.hlsl)
#define QL_RT_indexBuffer(ROW)                 ROW(t, 4,  ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_A | QL_VIS_X, 0)  // indexBuffer (hit_common.hlsli)
#define QL_RT_materials(ROW)                   ROW(t, 5,  ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_A | QL_VIS_X, 0)  // materials (hit_common.hlsli)
#define QL_RT_textures(ROW)                    ROW(t, 6,  SampledTexture,        QL_BINDLESS, QL_VIS_R | QL_VIS_C | QL_VIS_A | QL_VIS_X, 1)  // textures[] (hit_common.hlsli)
#define QL_RT_samplers(ROW)                    ROW(s, 7,  Sampler,               QL_BINDLESS, QL_VIS_R | QL_VIS_C | QL_VIS_A | QL_VIS_X, 1)  // samplers[] (hit_common.hlsli)
#define QL_RT_uvBuffer(ROW)                    ROW(t, 8,  ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_A | QL_VIS_X, 0)  // uvBuffer (hit_common.hlsli)
#define QL_RT_tangentBuffer(ROW)               ROW(t, 9,  ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // tangentBuffer (shade_vertex.hlsli)
#define QL_RT_prefilteredEnvMap(ROW)           ROW(t, 10, SampledTexture,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // prefilteredEnvMap (shade_vertex.hlsli)
#define QL_RT_brdfLUT(ROW)                     ROW(t, 11, SampledTexture,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // brdfLUT (shade_vertex.hlsli)
#define QL_RT_iblSampler(ROW)                  ROW(s, 12, Sampler,               1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // iblSampler (shade_vertex.hlsli)
#define QL_RT_spectralCurves(ROW)              ROW(t, 13, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // spectralCurves (shade_vertex.hlsli)
#define QL_RT_complexRefractiveIndices(ROW)    ROW(t, 14, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // complexRefractiveIndices (pbr.hlsli)
#define QL_RT_solarSpectralLUT(ROW)            ROW(t, 15, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X, 0)  // solarSpectralLUT (shade_resources.hlsli)
#define QL_RT_normalBuffer(ROW)                ROW(t, 16, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // normalBuffer (shade_vertex.hlsli)
#define QL_RT_atmosNNHeader(ROW)               ROW(t, 17, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X, 0)  // atmosNNHeader (shade_resources.hlsli)
#define QL_RT_instanceGeometryInfo(ROW)        ROW(t, 18, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_A | QL_VIS_X, 0)  // instanceGeometryInfo (hit_common.hlsli)
#define QL_RT_cieCMF_LUT(ROW)                  ROW(t, 19, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X, 0)  // cieCMF_LUT (shade_resources.hlsli)
#define QL_RT_atmosNNData(ROW)                 ROW(t, 20, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X, 0)  // atmosNNData (shade_resources.hlsli)
#define QL_RT_envSampler(ROW)                  ROW(s, 21, Sampler,               1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // envSampler (shade_vertex.hlsli)
#define QL_RT_depthImage(ROW)                  ROW(u, 22, StorageTexture,        1,           QL_VIS_R | QL_VIS_X,                       0)  // depthImage (raygen.rgen)
#define QL_RT_emissiveTriangles(ROW)           ROW(t, 23, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // emissiveTriangles (shade_vertex.hlsli)
#define QL_RT_thermalTemperatures(ROW)         ROW(t, 24, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // thermalTemperatures (shade_vertex.hlsli)
#define QL_RT_rgbToSpectrumTable(ROW)          ROW(t, 25, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X, 0)  // rgbToSpectrumTable (shade_resources.hlsli)
#define QL_RT_thermalSunResponse(ROW)          ROW(t, 26, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // thermalSunResponse (shade_vertex.hlsli)
#define QL_RT_thermalParameterTangent(ROW)     ROW(t, 27, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // thermalParameterTangent (shade_vertex.hlsli)
#define QL_RT_cameraMeasurementImage(ROW)      ROW(u, 28, StorageTexture,        1,           QL_VIS_R | QL_VIS_X,                       0)  // cameraMeasurementImage (raygen.rgen)
#define QL_RT_cameraResponseTable(ROW)         ROW(t, 29, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X, 0)  // cameraResponseTable (shade_resources.hlsli)
#define QL_RT_cameraDynamicCounters(ROW)       ROW(u, 30, ReadWriteBuffer,       1,           QL_VIS_R | QL_VIS_X,                       1)  // cameraDynamicCounters (raygen.rgen)
#define QL_RT_cameraMeasurementDepthImage(ROW) ROW(u, 31, StorageTexture,        1,           QL_VIS_R | QL_VIS_X,                       1)  // cameraMeasurementDepthImage (raygen.rgen)
#define QL_RT_projectionConfig(ROW)            ROW(t, 32, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_X,                       0)  // projectionConfig (camera_projection.hlsli)
#define QL_RT_geometryHits(ROW)                ROW(u, 33, ReadWriteBuffer,       1,           QL_VIS_X,                                  1)  // geometryHits (fusion_geometry.rayq.hlsl)
#define QL_RT_geometryQueryPixels(ROW)         ROW(t, 34, ReadOnlyBuffer,        1,           QL_VIS_X,                                  1)  // geometryQueryPixels (fusion_geometry.rayq.hlsl)
#define QL_RT_fusionTransport(ROW)             ROW(t, 35, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // fusionTransport (fusion_transport.hlsli)
#define QL_RT_fusionRecords(ROW)               ROW(u, 36, ReadWriteBuffer,       1,           QL_VIS_R | QL_VIS_C | QL_VIS_M | QL_VIS_X, 0)  // fusionRecords (fusion_records.hlsli)
#define QL_RT_fusionProbeRays(ROW)             ROW(t, 37, ReadOnlyBuffer,        1,           QL_VIS_R | QL_VIS_C | QL_VIS_X,            0)  // fusionProbeRays (raygen.rgen)

// Every role, in slot order. RtBindings.hpp expands the rows in this order and
// asserts that row i is slot i, so a role left out or out of place fails the
// C++ build.
#define QL_RT_BINDINGS(X)                                                                    \
    X(outputImage) X(scene) X(lightingParams) X(vertexPositions) X(indexBuffer) X(materials) \
    X(textures) X(samplers) X(uvBuffer) X(tangentBuffer) X(prefilteredEnvMap) X(brdfLUT)     \
    X(iblSampler) X(spectralCurves) X(complexRefractiveIndices) X(solarSpectralLUT)          \
    X(normalBuffer) X(atmosNNHeader) X(instanceGeometryInfo) X(cieCMF_LUT) X(atmosNNData)    \
    X(envSampler) X(depthImage) X(emissiveTriangles) X(thermalTemperatures)                  \
    X(rgbToSpectrumTable) X(thermalSunResponse) X(thermalParameterTangent)                   \
    X(cameraMeasurementImage) X(cameraResponseTable) X(cameraDynamicCounters)                \
    X(cameraMeasurementDepthImage) X(projectionConfig) X(geometryHits)                       \
    X(geometryQueryPixels) X(fusionTransport) X(fusionRecords) X(fusionProbeRays)

#endif // QUANTILOOM_QL_RT_BINDINGS_HLSLI
