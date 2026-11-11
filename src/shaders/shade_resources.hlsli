#ifndef QUANTILOOM_SHADE_RESOURCES_HLSLI
#define QUANTILOOM_SHADE_RESOURCES_HLSLI

#include "ql_bind.hlsli"

// Resources the surface and the sky evaluators both read. Declared once so
// that one stage can include both libraries without redeclaring a binding.
StructuredBuffer<LightingParams> lightingParams : QL_RT_REGISTER(lightingParams);
StructuredBuffer<SolarSpectralLUT> solarSpectralLUT : QL_RT_REGISTER(solarSpectralLUT);
ByteAddressBuffer cameraResponseTable : QL_RT_REGISTER(cameraResponseTable);
StructuredBuffer<AtmosNNHeader> atmosNNHeader : QL_RT_REGISTER(atmosNNHeader);
StructuredBuffer<float> atmosNNData : QL_RT_REGISTER(atmosNNData);
StructuredBuffer<float4> cieCMF_LUT : QL_RT_REGISTER(cieCMF_LUT);
StructuredBuffer<float4> rgbToSpectrumTable : QL_RT_REGISTER(rgbToSpectrumTable);

#endif // QUANTILOOM_SHADE_RESOURCES_HLSLI
