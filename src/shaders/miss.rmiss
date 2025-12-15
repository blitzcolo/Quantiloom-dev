// ============================================================================
// Quantiloom M1 - Miss Shader (Spectral Rendering)
// ============================================================================
// Returns sky background radiance when ray misses all geometry
// Supports both RGB and spectral modes with SolarSpectralLUT integration
// ============================================================================

#include "common.hlsli"

// PI constant (from pbr.hlsli, duplicated here to avoid heavy include)
static const float PI = 3.14159265358979323846;

// ============================================================================
// Bindings
// ============================================================================

[[vk::binding(2, 0)]] StructuredBuffer<LightingParams> lightingParams;

// ============================================================================
// Solar Spectral LUT Buffer (Binding 15)
// ============================================================================
// ASTM G-173 solar irradiance curves for spectral sky background
// When numSamples > 0, use true spectral sky radiance
// Otherwise, fall back to LightingParams scalar/RGB values
// ============================================================================

[[vk::binding(15, 0)]] StructuredBuffer<SolarSpectralLUT> solarSpectralLUT;

// ============================================================================
// Push Constants
// ============================================================================
// Camera data for accessing spectral mode and wavelength
// ============================================================================

[[vk::push_constant]] CameraData camera;

// ============================================================================
// Miss Entry Point
// ============================================================================

[shader("miss")]
void main(inout Payload payload) {
    // Fetch sky radiance from LightingParams (fallback values)
    LightingParams lut = lightingParams[0];

    // Check if spectral solar LUT is available
    bool hasSpectralSolarLUT = (solarSpectralLUT[0].skyIrradiance.numSamples > 0);

    // Choose sky radiance based on spectral mode
    if (camera.spectral_mode == SPECTRAL_MODE_RGB_FUSED) {
        // RGB mode: Use full RGB sky radiance or integrate spectral
        if (hasSpectralSolarLUT) {
            // For miss shader, use average sky radiance across visible spectrum
            // This is a simplified approach - full integration would be expensive
            // Use center of visible spectrum (550nm) as representative
            float sky_irr = SampleSkyIrradiance(solarSpectralLUT[0], 550.0);
            float sky_radiance = sky_irr / PI;  // Convert irradiance to radiance (diffuse hemisphere)
            payload.radiance = float3(sky_radiance, sky_radiance, sky_radiance);
        } else {
            // Fallback: Use LightingParams RGB values
            payload.radiance = lut.skyRadiance_rgb;
        }
    } else if (camera.spectral_mode == SPECTRAL_MODE_SINGLE) {
        // Single wavelength mode: Query spectral sky at current wavelength
        float radiance_spectral;

        if (hasSpectralSolarLUT) {
            float sky_irr = SampleSkyIrradiance(solarSpectralLUT[0], camera.wavelength_nm);
            radiance_spectral = sky_irr / PI;
        } else {
            radiance_spectral = lut.skyRadiance_spectral;
        }

        payload.radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);
    } else {
        // Other spectral modes (MWIR, LWIR, etc.): Use scalar fallback
        float radiance_spectral = lut.skyRadiance_spectral;
        payload.radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);
    }
}
