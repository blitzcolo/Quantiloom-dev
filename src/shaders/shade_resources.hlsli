#ifndef QUANTILOOM_SHADE_RESOURCES_HLSLI
#define QUANTILOOM_SHADE_RESOURCES_HLSLI

// Resources the surface and the sky evaluators both read. Declared once so
// that one stage can include both libraries without redeclaring a binding.
StructuredBuffer<LightingParams> lightingParams : register(t2, space0);
StructuredBuffer<SolarSpectralLUT> solarSpectralLUT : register(t15, space0);
ByteAddressBuffer cameraResponseTable : register(t29, space0);
StructuredBuffer<AtmosNNHeader> atmosNNHeader : register(t17, space0);
StructuredBuffer<float> atmosNNData : register(t20, space0);
StructuredBuffer<float4> cieCMF_LUT : register(t19, space0);
StructuredBuffer<float4> rgbToSpectrumTable : register(t25, space0);

#endif // QUANTILOOM_SHADE_RESOURCES_HLSLI
