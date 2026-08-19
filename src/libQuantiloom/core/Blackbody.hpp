/**
 * @file Blackbody.hpp
 * @brief Planck's law in double precision, and its inverse
 *
 * The renderer's blackbody lives in blackbody.hlsli and runs in fp32 on the
 * GPU. This is the CPU's copy, and it exists for the direction the shader does
 * not go: from a band radiance back to the temperature that produced it, which
 * is what a thermal camera reports and what makes a rendered thermogram
 * comparable to a measured one.
 *
 * THE QUADRATURE IS THE SHADER'S. The fused thermal bands integrate 16
 * wavelengths by the trapezoid rule with half-weight endpoints and divide by
 * the band width; PlanckBandAverage does exactly that, with the same node
 * count, so inverting a rendered pixel returns the temperature that was
 * rendered rather than the temperature a finer quadrature would have needed.
 * An isothermal cavity is the test: it renders its own blackbody, so it must
 * invert to its own temperature. Changing NUM_IR_SAMPLES in the shader means
 * changing the default here.
 *
 * Double throughout. The inversion divides by dL/dT, which for a 10 um band
 * near 300 K is of order 1e-11 of the radiance in SI units, and fp32 has no
 * room for that.
 */

#pragma once

#include "core/Types.hpp"

namespace quantiloom::blackbody {

/// Node count the fused thermal bands use. NUM_IR_SAMPLES / NUM_SWIR_SAMPLES
/// in closesthit.rchit, which are the same number for every thermal band.
inline constexpr i32 kBandNodes = 16;

/// Coldest and hottest the inversion will report. Wider than any scene the
/// renderer takes (materials are clamped to [150, 1000] K) so the bracket is
/// never the thing that limits an answer, and finite so a pixel of noise
/// cannot walk the iteration off to infinity.
inline constexpr f64 kMinInvertibleK = 100.0;
inline constexpr f64 kMaxInvertibleK = 3000.0;

/**
 * @brief Spectral radiance of a blackbody, per nanometre
 *
 * L(lambda, T) = 2hc^2 / (lambda^5 (exp(hc / (lambda k T)) - 1))
 *
 * @param lambdaNm wavelength in nanometres
 * @param temperatureK temperature in kelvin
 * @return W sr^-1 m^-2 nm^-1, the renderer's unit; zero for T <= 0
 */
[[nodiscard]] f64 SpectralRadiancePerNm(f64 lambdaNm, f64 temperatureK);

/**
 * @brief d/dT of SpectralRadiancePerNm, analytically
 *
 * dL/dT = L * (x / T) * e^x / (e^x - 1),  x = hc / (lambda k T)
 *
 * Analytic rather than a difference quotient because it is the Newton step's
 * denominator: a difference quotient at fp64 over a 0.01 K step loses half the
 * digits of a quantity the iteration divides by.
 */
[[nodiscard]] f64 SpectralRadianceDerivativePerNmPerK(f64 lambdaNm, f64 temperatureK);

/**
 * @brief Band-averaged spectral radiance, on the shader's quadrature
 *
 * The same trapezoid the fused bands use: @p nodes wavelengths spanning
 * [lambdaMinNm, lambdaMaxNm], endpoints at half weight, divided by the band
 * width. The result is what the renderer writes into an EXR for a thermal
 * band, so it can be compared with one pixel for pixel.
 */
[[nodiscard]] f64 BandAverageRadiance(f64 lambdaMinNm, f64 lambdaMaxNm, f64 temperatureK,
                                      i32 nodes = kBandNodes);

/**
 * @brief d/dT of BandAverageRadiance
 *
 * Also the sensitivity a noise-equivalent temperature difference divides by:
 * NETD = sigma_L / (dL/dT).
 */
[[nodiscard]] f64 BandAverageRadianceDerivative(f64 lambdaMinNm, f64 lambdaMaxNm,
                                                f64 temperatureK, i32 nodes = kBandNodes);

/**
 * @brief Temperature whose band average is @p bandAverageRadiance
 *
 * Newton from a Wien-based first guess, with a bisection fallback for the
 * cases Newton is bad at (radiance near zero, where dL/dT collapses). Returns
 * kMinInvertibleK for a non-positive radiance, since a pixel that received
 * nothing is not evidence of a temperature.
 *
 * Inverse of BandAverageRadiance to better than 1e-6 K over [150, 1000] K,
 * which is what the round-trip test asserts.
 */
[[nodiscard]] f64 InvertBandAverageRadiance(f64 bandAverageRadiance, f64 lambdaMinNm,
                                            f64 lambdaMaxNm, i32 nodes = kBandNodes);

/**
 * @brief Planck-weighted band-averaged reflectance of a spectral curve
 *
 * <rho>_B = integral(rho(lambda) * B(lambda, T)) / integral(B(lambda, T))
 *
 * Uses the same trapezoid quadrature as BandAverageRadiance (matching the
 * shader's integration) and evaluates the curve via linear interpolation.
 * T0 = 300 K is representative of surface temperatures in this experiment
 * (260-320 K); the normalised Planck weight changes shape slowly over that
 * range, making the result insensitive to T within ~0.005 for realistic
 * urban-surface curves.
 *
 * @tparam Curve  Anything with an `f32 Evaluate(f32 lambda_nm) const` method
 *                (e.g. SpectralCurve)
 * @param curve          Reflectance curve
 * @param lambdaMinNm    Lower edge of the integration band (nm)
 * @param lambdaMaxNm    Upper edge of the integration band (nm)
 * @param temperatureK   Planck weighting temperature (default 300 K)
 * @param nodes          Quadrature node count (default kBandNodes = 16)
 * @return Band-averaged reflectance in [0, 1]
 */
template <typename Curve>
[[nodiscard]] f64 PlanckWeightedBandAverage(
    const Curve& curve,
    f64 lambdaMinNm, f64 lambdaMaxNm,
    f64 temperatureK = 300.0, i32 nodes = kBandNodes)
{
    if (nodes < 2) return static_cast<f64>(curve.Evaluate(static_cast<f32>(0.5 * (lambdaMinNm + lambdaMaxNm))));

    const f64 step = (lambdaMaxNm - lambdaMinNm) / (nodes - 1);
    f64 sumRhoB = 0.0;
    f64 sumB = 0.0;

    for (i32 i = 0; i < nodes; ++i) {
        const f64 lambda = lambdaMinNm + i * step;
        const f64 B = SpectralRadiancePerNm(lambda, temperatureK);
        const f64 rho = static_cast<f64>(curve.Evaluate(static_cast<f32>(lambda)));
        const f64 w = (i == 0 || i == nodes - 1) ? 0.5 : 1.0;
        sumRhoB += w * rho * B;
        sumB += w * B;
    }

    if (sumB <= 0.0) return 0.0;
    return sumRhoB / sumB;
}

}  // namespace quantiloom::blackbody
