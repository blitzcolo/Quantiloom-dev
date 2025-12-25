// ============================================================================
// Quantiloom M1 - Miss Shader (Spectral Rendering)
// ============================================================================
// Returns sky background radiance when ray misses all geometry
// Supports both RGB and spectral modes with SolarSpectralLUT integration
// Now with Delta-Tracking volumetric atmospheric scattering
// ============================================================================

#include "common.hlsli"
#include "pbr.hlsli"
#include "atmospheric.hlsli"

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
// Atmospheric Parameters Buffer (Binding 17)
// ============================================================================
// Physical parameters for Delta-Tracking volumetric atmospheric rendering
// When beta_rayleigh_550nm.x > 0, Delta-Tracking is enabled
// Otherwise, fall back to simple sky radiance
// ============================================================================

[[vk::binding(17, 0)]] StructuredBuffer<AtmosphericParams> atmosphericParams;

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
    // Fetch atmospheric and lighting parameters
    LightingParams lut = lightingParams[0];
    AtmosphericParams atmo = atmosphericParams[0];

    // Check if Delta-Tracking is enabled (beta_rayleigh > 0 indicates enabled)
    bool atmosphereEnabled = (atmo.beta_rayleigh_550nm.x > 1e-9);

    // Check if spectral solar LUT is available
    bool hasSpectralSolarLUT = (solarSpectralLUT[0].skyIrradiance.numSamples > 0);

    // Get ray information
    float3 ray_origin = WorldRayOrigin();
    float3 ray_dir = WorldRayDirection();

    // ========================================================================
    // Atmospheric Scattering (Delta-Tracking)
    // ========================================================================
    if (atmosphereEnabled) {
        // Planet center: assume camera at surface, planet center below
        // TODO: Make this configurable via uniform buffer
        float3 planet_center = float3(0.0, -atmo.planet_radius, 0.0);

        // Compute ray-atmosphere intersection
        float t_atmo_near, t_atmo_far;
        bool hits_atmosphere = RaySphereIntersection(
            ray_origin, ray_dir, planet_center,
            atmo.planet_radius + atmo.atmosphere_height,
            t_atmo_near, t_atmo_far);

        if (hits_atmosphere && t_atmo_far > 0.0) {
            // Ray enters atmosphere
            float t_min = max(t_atmo_near, 0.0);
            float t_max = t_atmo_far;

            // Initialize random state using ray coordinates
            // Use a simple hash of ray origin + direction for seed
            uint seed = uint(dot(ray_origin, float3(127.1, 311.7, 74.7))) +
                        uint(dot(ray_dir, float3(269.5, 183.3, 246.1)) * 1000.0);
            uint random_state = seed ^ 0xDEADBEEFu;

            // Perform Delta-Tracking
            float t_scatter;
            float transmittance;
            bool scattered = DeltaTracking(
                ray_origin, ray_dir,
                t_min, t_max,
                camera.wavelength_nm,
                atmo,
                planet_center,
                random_state,
                t_scatter,
                transmittance);

            if (scattered) {
                // Scattering event occurred
                float3 scatter_pos = ray_origin + ray_dir * t_scatter;
                float3 sun_dir = normalize(lut.sunDirection);

                // Get sun radiance at current wavelength
                float sun_radiance;
                if (hasSpectralSolarLUT) {
                    float sun_irr = SampleSunIrradiance(solarSpectralLUT[0], camera.wavelength_nm);
                    sun_radiance = SunIrradianceToRadiance(sun_irr);
                } else {
                    sun_radiance = lut.sunRadiance_spectral;
                }

                // Compute single scattering
                float3 scattered_radiance = SingleScattering(
                    scatter_pos, ray_dir, sun_dir,
                    camera.wavelength_nm,
                    atmo, planet_center,
                    sun_radiance);

                payload.radiance = scattered_radiance;
                return;
            }

            // No scattering: ray escaped atmosphere without collision
            // Fall through to simple sky radiance (attenuated by transmittance if needed)
        }
    }

    // ========================================================================
    // Fallback: Simple Sky Radiance (no atmosphere or ray missed atmosphere)
    // ========================================================================

    // Choose sky radiance based on spectral mode
    if (camera.spectral_mode == SPECTRAL_MODE_RGB_FUSED) {
        // RGB mode: Use full RGB sky radiance or integrate spectral
        if (hasSpectralSolarLUT) {
            // For miss shader, use average sky radiance across visible spectrum
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
