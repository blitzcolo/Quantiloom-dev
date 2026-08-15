/**
 * @file Blackbody.cpp
 * @brief Planck's law in double precision, and its inverse
 */

#include "core/Blackbody.hpp"

#include <algorithm>
#include <cmath>

namespace quantiloom::blackbody {

namespace {

// The same two constants blackbody.hlsli carries, in double.
//   C1_NM = 2hc^2 * 1e36, so that lambda may be given in nm and the result is
//           per nm rather than per m.
//   C2    = hc/k, in m*K; the exponent is C2 / (lambda_m * T).
constexpr f64 kC1Nm = 1.191042972e20;  // W nm^4 sr^-1 m^-2
constexpr f64 kC2 = 1.438776877e-2;    // m K

/// exp() overflows to inf for a cold enough surface at a short enough
/// wavelength -- x reaches 700 around 20 K in the LWIR band -- and inf-1 is
/// inf, so the quotient would be 0 anyway. Returning zero early keeps the
/// derivative finite too, where inf/inf would not be.
constexpr f64 kMaxExponent = 700.0;

struct Exponent {
    f64 x;      ///< hc / (lambda k T)
    bool huge;  ///< x is past where exp() is representable
};

Exponent ExponentAt(const f64 lambdaNm, const f64 temperatureK) {
    const f64 lambdaM = lambdaNm * 1e-9;
    const f64 x = kC2 / (lambdaM * temperatureK);
    return {x, x > kMaxExponent};
}

}  // namespace

f64 SpectralRadiancePerNm(const f64 lambdaNm, const f64 temperatureK) {
    if (temperatureK <= 0.0 || lambdaNm <= 0.0) {
        return 0.0;
    }
    const auto e = ExponentAt(lambdaNm, temperatureK);
    if (e.huge) {
        return 0.0;
    }
    const f64 lambda5 = lambdaNm * lambdaNm * lambdaNm * lambdaNm * lambdaNm;
    return (kC1Nm / lambda5) / (std::exp(e.x) - 1.0);
}

f64 SpectralRadianceDerivativePerNmPerK(const f64 lambdaNm, const f64 temperatureK) {
    if (temperatureK <= 0.0 || lambdaNm <= 0.0) {
        return 0.0;
    }
    const auto e = ExponentAt(lambdaNm, temperatureK);
    if (e.huge) {
        return 0.0;
    }
    const f64 lambda5 = lambdaNm * lambdaNm * lambdaNm * lambdaNm * lambdaNm;
    const f64 expX = std::exp(e.x);
    const f64 denom = expX - 1.0;
    // L * (x/T) * e^x / (e^x - 1), written as one expression so the two
    // occurrences of (e^x - 1) cancel in the same order every time.
    return (kC1Nm / lambda5) * (e.x / temperatureK) * expX / (denom * denom);
}

namespace {

/// The shader's trapezoid, parameterised by what is being integrated: nodes
/// spanning the band, endpoints at half weight, divided by the band width.
/// Both the radiance and its derivative go through it, which is the only way
/// the derivative is guaranteed to be the derivative of the quadrature rather
/// than of the exact integral.
template <typename SampleFn>
f64 BandAverage(const f64 lambdaMinNm, const f64 lambdaMaxNm, const i32 nodes,
                SampleFn&& sample) {
    if (nodes < 2 || lambdaMaxNm <= lambdaMinNm) {
        return 0.0;
    }
    const f64 step = (lambdaMaxNm - lambdaMinNm) / static_cast<f64>(nodes - 1);
    f64 accum = 0.0;
    for (i32 i = 0; i < nodes; ++i) {
        const f64 lambda = lambdaMinNm + static_cast<f64>(i) * step;
        const f64 weight = (i == 0 || i == nodes - 1) ? 0.5 : 1.0;
        accum += sample(lambda) * weight;
    }
    // accum * step / (lambdaMax - lambdaMin), and the band width is
    // (nodes - 1) * step, so the steps cancel.
    return accum / static_cast<f64>(nodes - 1);
}

}  // namespace

f64 BandAverageRadiance(const f64 lambdaMinNm, const f64 lambdaMaxNm, const f64 temperatureK,
                        const i32 nodes) {
    if (temperatureK <= 0.0) {
        return 0.0;
    }
    return BandAverage(lambdaMinNm, lambdaMaxNm, nodes,
                       [temperatureK](const f64 lambda) {
                           return SpectralRadiancePerNm(lambda, temperatureK);
                       });
}

f64 BandAverageRadianceDerivative(const f64 lambdaMinNm, const f64 lambdaMaxNm,
                                  const f64 temperatureK, const i32 nodes) {
    if (temperatureK <= 0.0) {
        return 0.0;
    }
    return BandAverage(lambdaMinNm, lambdaMaxNm, nodes,
                       [temperatureK](const f64 lambda) {
                           return SpectralRadianceDerivativePerNmPerK(lambda, temperatureK);
                       });
}

f64 InvertBandAverageRadiance(const f64 bandAverageRadiance, const f64 lambdaMinNm,
                              const f64 lambdaMaxNm, const i32 nodes) {
    if (!(bandAverageRadiance > 0.0) || lambdaMaxNm <= lambdaMinNm || nodes < 2) {
        // A pixel that received nothing, or received a negative number because
        // a sensor's noise took it there, is not evidence of a temperature.
        return kMinInvertibleK;
    }

    // Bracket first. Band radiance is strictly increasing in T, so if the
    // target is outside what the bracket spans the answer is the bracket's
    // own end and no iteration can improve on it.
    if (bandAverageRadiance <=
        BandAverageRadiance(lambdaMinNm, lambdaMaxNm, kMinInvertibleK, nodes)) {
        return kMinInvertibleK;
    }
    if (bandAverageRadiance >=
        BandAverageRadiance(lambdaMinNm, lambdaMaxNm, kMaxInvertibleK, nodes)) {
        return kMaxInvertibleK;
    }

    // Newton, kept inside a shrinking bracket. Newton alone converges in three
    // or four steps over the whole range, but its step is unbounded where
    // dL/dT underflows, and a single bad step there would leave the bracket
    // for good; taking the bisection point instead whenever Newton proposes
    // one costs nothing when Newton is behaving.
    f64 lo = kMinInvertibleK;
    f64 hi = kMaxInvertibleK;
    f64 T = 0.5 * (lo + hi);

    for (i32 iteration = 0; iteration < 100; ++iteration) {
        const f64 f = BandAverageRadiance(lambdaMinNm, lambdaMaxNm, T, nodes) -
                      bandAverageRadiance;
        if (f > 0.0) {
            hi = T;
        } else {
            lo = T;
        }

        const f64 df = BandAverageRadianceDerivative(lambdaMinNm, lambdaMaxNm, T, nodes);
        f64 next = (df > 0.0) ? T - f / df : 0.5 * (lo + hi);
        if (!(next > lo && next < hi)) {
            next = 0.5 * (lo + hi);
        }

        const f64 delta = std::abs(next - T);
        T = next;
        if (delta < 1e-9 * std::max(1.0, T)) {
            break;
        }
    }
    return T;
}

}  // namespace quantiloom::blackbody
