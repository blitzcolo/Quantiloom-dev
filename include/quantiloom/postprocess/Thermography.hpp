#pragma once

#include "../core/Platform.hpp"
#include "../core/Types.hpp"
#include "SensorModel.hpp"

// ============================================================================
// Thermography - band radiance back to temperature
// ============================================================================
// A thermal camera does not report radiance. It measures one, assumes the
// scene is a blackbody unless told otherwise, and displays the temperature
// that would have produced what it measured. Two surfaces at the same
// temperature and different emissivities therefore read differently, and a
// measured thermogram cannot be compared with a rendered radiance field until
// the render has been put through the same arithmetic.
//
// This is that arithmetic, in the renderer's own quadrature, so that a scene
// which renders its own blackbody -- an isothermal cavity does -- inverts to
// the temperature it was given.
//
// Aguerre et al., "Physically Based Simulation and Rendering of Urban
// Thermography", Computer Graphics Forum 39(6), 2020, eq. 8-11, restated per
// band rather than over the whole spectrum: their sigma T^4 is the total flux
// of a blackbody, which is not what a 7-14 um camera collects, and using it
// would put the error of the out-of-band tail into every temperature.
// ============================================================================

namespace quantiloom {

/// What the camera was told about the surface it is looking at. The defaults
/// are the apparent-temperature setting: emissivity 1, no reflected source, no
/// atmosphere -- what a camera reports when nobody has characterised the
/// scene, and what a measurement campaign records when it wants a number that
/// does not depend on an assumed emissivity.
struct ThermographyParams {
    /// Emissivity the camera assumes. 1 gives apparent temperature.
    f32 emissivity = 1.0f;
    /// Temperature of whatever the surface reflects. Ignored when emissivity
    /// is 1, since a blackbody reflects nothing.
    f32 reflectedTemperature_K = 0.0f;
    /// Path transmittance between surface and lens. 1 removes the atmosphere
    /// from the model, which is right for a short measurement distance.
    f32 atmosphereTransmittance = 1.0f;
    /// Temperature of that path, used only when the transmittance is below 1.
    f32 atmosphereTemperature_K = 0.0f;
};

/// Temperature of the blackbody whose band-average radiance is @p radiance.
///
/// @param radiance   band-average spectral radiance, W sr^-1 m^-2 nm^-1 --
///                   the unit the renderer writes into an EXR for a fused
///                   thermal band
/// @param lambdaMinNm,lambdaMaxNm  band edges, from GetFusedBandInfo
/// @return kelvin, clamped to [100, 3000]; the floor is also what a
///         non-positive radiance returns
[[nodiscard]] QL_API f64 InvertApparentTemperatureK(f64 radiance, f64 lambdaMinNm,
                                                    f64 lambdaMaxNm);

/// Surface temperature, having removed what the camera model attributes to
/// reflection and to the path between surface and lens.
///
/// Solves the band form of Aguerre eq. 8 for the surface's own blackbody:
///
///   L = tau [ eps B(T_s) + (1 - eps) B(T_refl) ] + (1 - tau) B(T_atm)
///   B(T_s) = [ L/tau - (1 - eps) B(T_refl) - ((1 - tau)/tau) B(T_atm) ] / eps
///
/// and inverts it. With the default params this is exactly
/// InvertApparentTemperatureK.
///
/// Subtracting more than the pixel received leaves a negative blackbody, which
/// means the parameters are inconsistent with the image rather than that the
/// surface is cold; the floor is returned rather than a complex temperature.
[[nodiscard]] QL_API f64 InvertSurfaceTemperatureK(f64 radiance, f64 lambdaMinNm,
                                                   f64 lambdaMaxNm,
                                                   const ThermographyParams& params);

/// d/dT of the band-average radiance at @p temperatureK: how much radiance one
/// kelvin is worth in this band. A noise-equivalent temperature difference is
/// a radiance noise divided by this.
[[nodiscard]] QL_API f64 BandRadianceDerivativePerK(f64 lambdaMinNm, f64 lambdaMaxNm,
                                                    f64 temperatureK);

/// Band-average radiance of a blackbody, on the renderer's quadrature. The
/// forward direction of the two above, exposed because a host that wants to
/// show what a temperature would read needs the same rule the inversion uses.
[[nodiscard]] QL_API f64 BlackbodyBandRadiance(f64 lambdaMinNm, f64 lambdaMaxNm,
                                               f64 temperatureK);

/// Smallest temperature difference this sensor can distinguish at
/// @p sceneTemperatureK, in kelvin: the temporal noise referred back to the
/// scene through the band's own dL/dT.
///
///   NETD = sigma_L / (dL/dT),  sigma_L = sigma_e / responsivity
///
/// The noise is shot, read and dark, each counted only if the params enable
/// it -- fixed pattern noise is excluded because it does not vary between
/// frames, which is what a NETD measurement averages away.
///
/// Returns 0 when nothing in the params is noisy, and infinity where the band
/// has no sensitivity left (dL/dT underflows below roughly 50 K in the LWIR).
///
/// Quoted on axis and WITHOUT the well capacity: a configuration whose signal
/// exceeds wellCapacity_e gets the sensitivity that integration time would
/// have bought, not the one the detector can deliver. A thermal band puts an
/// enormous DC term on the detector -- everything in view is near 300 K -- so
/// that gap is easy to reach and worth checking against the raw DN before
/// quoting a figure.
///
/// The responsivity here must stay the one GenericSensor::RadianceToElectrons
/// applies; a test asserts the two agree rather than trusting the comment.
[[nodiscard]] QL_API f64 NoiseEquivalentTemperatureDifferenceK(const SensorParams& sensor,
                                                               f64 lambdaMinNm,
                                                               f64 lambdaMaxNm,
                                                               f64 sceneTemperatureK);

// ============================================================================
// Clear sky
// ============================================================================
// A thermal camera pointed up does not see the air temperature: it sees a
// partly transparent atmosphere against a background near 3 K, and how cold
// that reads depends on the water vapour in the way. These three are what the
// renderer derives its clear-sky model from, exposed so a host can show the
// derived numbers rather than reimplement the correlation to display them.
//
// Berdahl & Fromberg, "The thermal radiance of clear skies", Solar Energy
// 29(4), 1982.

/// Dew point in Celsius, Magnus with the WMO coefficients. Exact at RH = 100,
/// where it returns the air temperature.
[[nodiscard]] QL_API f64 DewPointC(f64 airTemperatureC, f64 relativeHumidityPercent);

/// Clear-sky emissivity from the dew point:
/// eps = 0.711 + 0.56 (Tdp/100) + 0.73 (Tdp/100)^2, clamped to [0, 1].
[[nodiscard]] QL_API f64 ClearSkyEmissivity(f64 dewPointC);

/// The blackbody that radiates what this sky does: T_air * eps^(1/4).
[[nodiscard]] QL_API f64 EffectiveSkyTemperatureK(f64 airTemperatureK, f64 emissivity);

}  // namespace quantiloom
