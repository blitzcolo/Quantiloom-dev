// ============================================================================
// Quantiloom - Unit Tests for core/Blackbody.hpp
// ============================================================================
// Planck in double, and the inverse the shader does not have. Two things have
// to hold for a rendered thermogram to mean anything: the forward law has to
// be the physics (checked against Wien and Stefan-Boltzmann, which are
// independent consequences of it), and the inverse has to invert THIS
// quadrature rather than the exact integral -- an isothermal cavity renders
// its own band average on 16 trapezoid nodes, and must come back at its own
// temperature, not at the temperature a finer rule would have needed.
// ============================================================================

#include <gtest/gtest.h>

#include "core/Blackbody.hpp"

#include <cmath>
#include <numbers>

using namespace quantiloom;
using namespace quantiloom::blackbody;

namespace {

// The three thermal bands GetFusedBandInfo defines, in nm.
constexpr f64 kSwirMin = 1400.0, kSwirMax = 2400.0;
constexpr f64 kMwirMin = 3000.0, kMwirMax = 5000.0;
constexpr f64 kLwirMin = 8000.0, kLwirMax = 12000.0;

}  // namespace

// ============================================================================
// The forward law, against consequences of it that were derived separately
// ============================================================================

TEST(BlackbodyTest, PeakFollowsWiensDisplacementLaw) {
    // lambda_peak * T = 2.897771955e-3 m K. Found by scanning rather than
    // asserted from the formula, so the test does not restate the code.
    for (const f64 T : {300.0, 500.0, 1000.0, 2000.0}) {
        const f64 expectedPeakNm = 2.897771955e-3 / T * 1e9;

        f64 bestLambda = 0.0;
        f64 bestValue = 0.0;
        for (f64 lambda = 0.2 * expectedPeakNm; lambda < 4.0 * expectedPeakNm;
             lambda += expectedPeakNm * 1e-4) {
            const f64 L = SpectralRadiancePerNm(lambda, T);
            if (L > bestValue) {
                bestValue = L;
                bestLambda = lambda;
            }
        }
        EXPECT_NEAR(bestLambda, expectedPeakNm, expectedPeakNm * 1e-3)
            << "peak misplaced at T = " << T << " K";
    }
}

TEST(BlackbodyTest, WholeSpectrumIntegratesToStefanBoltzmann) {
    // Integrating L over all wavelengths and all solid angle gives sigma T^4;
    // for radiance the hemispherical integral contributes pi, so the check is
    // integral L dlambda == sigma T^4 / pi.
    constexpr f64 sigma = 5.670374419e-8;

    for (const f64 T : {300.0, 800.0}) {
        // Simpson over a range wide enough to hold essentially all the energy:
        // 0.1x to 30x the Wien peak carries better than 1e-4 of the total.
        const f64 peakNm = 2.897771955e-3 / T * 1e9;
        const f64 lo = 0.1 * peakNm;
        const f64 hi = 30.0 * peakNm;
        const i32 n = 20000;  // even
        const f64 h = (hi - lo) / n;

        f64 sum = SpectralRadiancePerNm(lo, T) + SpectralRadiancePerNm(hi, T);
        for (i32 i = 1; i < n; ++i) {
            const f64 lambda = lo + i * h;
            sum += SpectralRadiancePerNm(lambda, T) * ((i % 2 == 0) ? 2.0 : 4.0);
        }
        const f64 integral = sum * h / 3.0;

        const f64 expected = sigma * T * T * T * T / std::numbers::pi;
        EXPECT_NEAR(integral, expected, expected * 1e-3)
            << "Stefan-Boltzmann mismatch at T = " << T << " K";
    }
}

TEST(BlackbodyTest, DerivativeMatchesAFiniteDifference) {
    for (const f64 T : {200.0, 300.0, 600.0}) {
        for (const f64 lambda : {2000.0, 4000.0, 10000.0}) {
            const f64 dT = 1e-4 * T;
            const f64 numeric = (SpectralRadiancePerNm(lambda, T + dT) -
                                 SpectralRadiancePerNm(lambda, T - dT)) /
                                (2.0 * dT);
            const f64 analytic = SpectralRadianceDerivativePerNmPerK(lambda, T);
            EXPECT_NEAR(analytic, numeric, std::abs(numeric) * 1e-5)
                << "at " << lambda << " nm, " << T << " K";
        }
    }
}

TEST(BlackbodyTest, ColdSurfacesAtShortWavelengthsReturnZeroRatherThanInfinity) {
    // exp() overflows past x ~ 700, which in the SWIR is reached around 15 K.
    // The quotient is zero there; what must not happen is inf or NaN reaching
    // an image.
    const f64 L = SpectralRadiancePerNm(1400.0, 5.0);
    EXPECT_EQ(L, 0.0);
    EXPECT_TRUE(std::isfinite(SpectralRadianceDerivativePerNmPerK(1400.0, 5.0)));
}

// ============================================================================
// The quadrature, and its inverse
// ============================================================================

TEST(BlackbodyTest, BandAverageIsTheShadersTrapezoid) {
    // Sixteen nodes, endpoints at half weight, divided by the band width --
    // spelled out here rather than called, so a change to the rule in
    // Blackbody.cpp fails this test instead of silently agreeing with itself.
    constexpr i32 nodes = 16;
    const f64 T = 320.0;
    const f64 step = (kLwirMax - kLwirMin) / (nodes - 1);

    f64 accum = 0.0;
    for (i32 i = 0; i < nodes; ++i) {
        const f64 lambda = kLwirMin + i * step;
        const f64 w = (i == 0 || i == nodes - 1) ? 0.5 : 1.0;
        accum += SpectralRadiancePerNm(lambda, T) * w * step;
    }
    const f64 expected = accum / (kLwirMax - kLwirMin);

    EXPECT_NEAR(BandAverageRadiance(kLwirMin, kLwirMax, T), expected, expected * 1e-12);
}

TEST(BlackbodyTest, InversionRoundTripsAcrossEveryThermalBand) {
    // The property the whole file exists for: whatever the renderer wrote for
    // a blackbody at T, inverting it returns T. Tight tolerance because both
    // directions use the same nodes -- any looser and a quadrature mismatch
    // could hide inside it.
    for (const auto& band : {std::pair{kSwirMin, kSwirMax}, std::pair{kMwirMin, kMwirMax},
                             std::pair{kLwirMin, kLwirMax}}) {
        for (f64 T = 200.0; T <= 1000.0; T += 25.0) {
            const f64 L = BandAverageRadiance(band.first, band.second, T);
            const f64 recovered = InvertBandAverageRadiance(L, band.first, band.second);
            EXPECT_NEAR(recovered, T, 1e-6)
                << "band " << band.first << "-" << band.second << " nm at " << T << " K";
        }
    }
}

TEST(BlackbodyTest, BandAverageDerivativeMatchesAFiniteDifferenceOfTheQuadrature) {
    // Of the quadrature, not of the exact integral: this derivative is the
    // Newton step's denominator, so it has to be the slope of the function
    // being inverted.
    for (const f64 T : {250.0, 300.0, 500.0}) {
        const f64 dT = 1e-4 * T;
        const f64 numeric = (BandAverageRadiance(kLwirMin, kLwirMax, T + dT) -
                             BandAverageRadiance(kLwirMin, kLwirMax, T - dT)) /
                            (2.0 * dT);
        const f64 analytic = BandAverageRadianceDerivative(kLwirMin, kLwirMax, T);
        EXPECT_NEAR(analytic, numeric, std::abs(numeric) * 1e-5) << "at " << T << " K";
    }
}

TEST(BlackbodyTest, BandRadianceIsStrictlyIncreasingInTemperature) {
    // What makes the inversion single-valued. Checked rather than assumed,
    // because the bracket-and-bisect fallback relies on it.
    f64 previous = 0.0;
    for (f64 T = 150.0; T <= 1000.0; T += 10.0) {
        const f64 L = BandAverageRadiance(kMwirMin, kMwirMax, T);
        EXPECT_GT(L, previous) << "not monotone at " << T << " K";
        previous = L;
    }
}

TEST(BlackbodyTest, NonPositiveRadianceReturnsTheFloorRatherThanNaN) {
    // A sensor's read noise can take a dark pixel below zero. That is not
    // evidence of a temperature, and it must not be evidence of a NaN either.
    EXPECT_EQ(InvertBandAverageRadiance(0.0, kLwirMin, kLwirMax), kMinInvertibleK);
    EXPECT_EQ(InvertBandAverageRadiance(-1e-4, kLwirMin, kLwirMax), kMinInvertibleK);
}

TEST(BlackbodyTest, RadianceBeyondTheBracketSaturatesAtItsEnds) {
    const f64 hot = BandAverageRadiance(kLwirMin, kLwirMax, kMaxInvertibleK) * 10.0;
    EXPECT_EQ(InvertBandAverageRadiance(hot, kLwirMin, kLwirMax), kMaxInvertibleK);

    const f64 cold = BandAverageRadiance(kLwirMin, kLwirMax, kMinInvertibleK) * 0.5;
    EXPECT_EQ(InvertBandAverageRadiance(cold, kLwirMin, kLwirMax), kMinInvertibleK);
}
