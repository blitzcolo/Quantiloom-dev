/**
 * @file BesselJ.hpp
 * @brief J_n(x) for integer order, on every standard library (internal)
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <numbers>
#include <version>

namespace quantiloom::core {

/// The cylindrical Bessel function of the first kind, integer order n.
///
/// std::cyl_bessel_j where the standard library has the C++17 special functions
/// (MSVC, libstdc++). Apple's libc++ has none of them, so there it is the
/// trapezoid rule over one period of Bessel's integral
///
///     J_n(x) = 1/(2 pi) * integral_0^{2 pi} cos(n t - x sin t) dt.
///
/// The integrand is periodic and entire, so the rule converges geometrically: its
/// error is the sum of J_{kN +- n}(x) over k >= 1, which is below 1e-17 once the
/// node count N exceeds |x| + |n| by a few dozen.
inline double CylBesselJ(int n, double x) {
#if defined(__cpp_lib_math_special_functions)
    return std::cyl_bessel_j(static_cast<double>(n), x);
#else
    const int nodes = 2 * (static_cast<int>(std::ceil(std::abs(x))) + std::abs(n) + 32);
    const double step = 2.0 * std::numbers::pi / nodes;
    double sum = 0.0;
    for (int k = 0; k < nodes; ++k) {
        const double t = step * k;
        sum += std::cos(n * t - x * std::sin(t));
    }
    return sum / nodes;
#endif
}

}  // namespace quantiloom::core
