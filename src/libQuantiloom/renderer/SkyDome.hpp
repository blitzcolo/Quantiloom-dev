/**
 * @file SkyDome.hpp
 * @brief Where the visible sky's light comes from: the clear-sky distribution
 *        behind lighting.sky_distribution = "clear"
 *
 * The illuminant says how much light the sky sends -- the solar LUT's diffuse
 * curve E_sky(lambda) -- and says nothing about direction. Every spectral band
 * used to read that as a uniform dome, E_sky/pi from everywhere, so a primary
 * ray that missed the scene came back one flat colour: the hemispherical
 * average of a clear sky, which is a dim grey-blue nobody has ever seen
 * overhead. A real one is blue at the zenith and pale at the horizon, and
 * brightest around the sun.
 *
 * The model is the textbook clear sky of a plane-parallel atmosphere seen from
 * the ground:
 *
 *   - single scattering of the direct beam by molecules (Rayleigh, optical
 *     depth after Hansen & Travis 1974) and by aerosol (Angstrom law, optical
 *     depth 0.084 at 500 nm -- the rural aerosol ASTM G-173 was computed for --
 *     Henyey-Greenstein phase, g = 0.7, single-scattering albedo 0.95):
 *
 *       L_ss = F0 mu0 [tau_R P_R + w_A tau_A P_A] / tau
 *                     * (exp(-tau/mu) - exp(-tau/mu0)) / (mu - mu0)
 *
 *   - and the first-order multiple-scattering term, an isotropic source of
 *     strength J filling the same layer, J (1 - exp(-tau/mu)), with J set so
 *     that the layer's own diffuse field feeds it (see ClearSkyNode). Without
 *     it the horizon carries only singly scattered sunlight and renders the
 *     colour of the sun; with it the horizon goes the pale grey-white it is.
 *
 * Below the horizon there is no sky, and no scene either where a ray reaches
 * that far: it has gone past the edge of the geometry and stands for ground
 * too distant to see, which a real atmosphere hides behind the horizon's own
 * haze. So it takes the horizon at its azimuth -- the same choice the thermal
 * sky makes (AtmosSkyRadianceIR) -- and the dome is continuous there. Mirroring
 * the sky instead painted a second, upside-down sky beyond every tile edge.
 *
 * NORMALISATION is the whole point. Per wavelength the dome integrates, cosine
 * weighted over the upper hemisphere, to exactly the illuminant's E_sky -- so
 * a horizontal surface under an open sky receives what it always received and
 * the radiometry of every scene stands. Only the direction of the light moves.
 * The model cannot simply be scaled onto E_sky wavelength by wavelength,
 * though: G-173's "diffuse" is global tilt minus direct, measured on a 37 deg
 * tilted plane that sees the ground, and the ground-reflected part makes it
 * markedly redder than any sky -- scaling the model onto it paints that red
 * into the horizon. So the model is anchored to the direct beam instead: it is
 * computed from the LUT's own sun spectrum, scaled once by the largest factor
 * that keeps it under E_sky everywhere in 420-680 nm, and whatever E_sky holds
 * beyond it at a given wavelength stays the uniform dome it always was:
 *
 *   L(lambda, w) = E_sky/pi * [ f (pi/N) S(lambda, w) + (1 - f) ]
 *   f(lambda)    = min(1, c DNI(lambda) N(lambda) exp(tau/mu0) / E_sky(lambda))
 *
 * where S is the model above divided by F0, N its cosine integral, and c the
 * anchoring scale. f and c are evaluated on the GPU, which holds both curves;
 * N and J depend only on the sun and the aerosol and are tabulated here, on 16
 * wavelengths, every time the lighting is uploaded.
 *
 * What it changes downstream: the analytic ambient terms keep the uniform dome
 * as their base, and the traced environment bounce carries the difference
 * between the sky it actually reaches and that base. That correction is
 * unbiased, but no longer zero in an open scene -- see src/shaders/AGENTS.md.
 *
 * Every function here has a twin in src/shaders/common.hlsli. The tables are
 * only right if the two compute the same shape; the illumination suite's
 * open:vis:clear arm is what notices when they do not.
 */

#pragma once

#include "core/Types.hpp"
#include "renderer/LightingParams.hpp"

#include <glm/glm.hpp>

namespace quantiloom::rendercore {

namespace skydome {
/// Wavelength grid of the N and J tables: 380-780 nm in 16 nodes.
inline constexpr u32 kNodes = 16;
inline constexpr f32 kLambdaStartNm = 380.0f;
inline constexpr f32 kLambdaStepNm = 400.0f / 15.0f;
/// Aerosol: Angstrom exponent, single-scattering albedo, asymmetry. Rural
/// continental values; the optical depth itself is a config key.
inline constexpr f32 kAngstromExponent = 1.3f;
inline constexpr f32 kAerosolAlbedo = 0.95f;
inline constexpr f32 kAerosolAsymmetry = 0.7f;
/// The sun is held at least 3 deg above the horizon for the shape: the
/// plane-parallel formula has no twilight, and E_sky still says how much
/// light there is.
inline constexpr f32 kMinSunSin = 0.0523f;
/// A grazing view is held this far above the horizon.
inline constexpr f32 kMinViewCos = 1e-4f;
/// Where the model is anchored to the direct beam.
inline constexpr f32 kAnchorMinNm = 420.0f;
inline constexpr f32 kAnchorMaxNm = 680.0f;
}  // namespace skydome

/// Rayleigh optical depth of a sea-level standard atmosphere, Hansen & Travis
/// (1974): 0.008569 l^-4 (1 + 0.0113 l^-2 + 0.00013 l^-4), l in micrometres.
[[nodiscard]] f32 SkyRayleighOpticalDepth(f32 lambdaNm);

/// Aerosol optical depth at a wavelength, from its value at 500 nm.
[[nodiscard]] f32 SkyAerosolOpticalDepth(f32 lambdaNm, f32 aod500);

/// The sun direction the shape uses: normalised, and held at least
/// kMinSunSin above the horizon with its azimuth kept.
[[nodiscard]] glm::vec3 SkySunDirection(const glm::vec3& sunDirection);

/// Exponential integral E3(x), x >= 0. E3(0) = 1/2.
[[nodiscard]] f64 ExponentialIntegralE3(f64 x);

/// The unnormalised clear sky toward @p direction (unit, toward the sky), in
/// units of the extraterrestrial irradiance F0. @p sun must come from
/// SkySunDirection; @p multipleScattering is the node's J.
[[nodiscard]] f32 ClearSkyShape(f32 lambdaNm, const glm::vec3& direction,
                                const glm::vec3& sun, f32 aod500,
                                f32 multipleScattering);

/// N and J at one wavelength: the cosine integral of the shape over the upper
/// hemisphere, and the multiple-scattering source that goes into it.
struct ClearSkyNodeValues {
    f32 cosineIntegral = 0.0f;      ///< N
    f32 multipleScattering = 0.0f;  ///< J
};
[[nodiscard]] ClearSkyNodeValues ClearSkyNode(f32 lambdaNm, const glm::vec3& sun, f32 aod500);

/// Fill the sky-dome grid and tables of @p params from its sunDirection,
/// skyDistribution and skyAerosolOpticalDepth. A uniform sky gets zeroed tables. The
/// one function every upload of LightingParams passes through.
void FillSkyDome(LightingParams& params);

}  // namespace quantiloom::rendercore
