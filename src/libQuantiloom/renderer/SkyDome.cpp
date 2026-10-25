#include "renderer/SkyDome.hpp"

#include <algorithm>
#include <cmath>

namespace quantiloom::rendercore {

namespace {

constexpr f64 kPi = 3.14159265358979323846;
constexpr f64 kEulerGamma = 0.57721566490153286061;

// E1(x), x > 0: the power series below 1, the continued fraction above
// (Numerical Recipes' expint, n = 1).
f64 ExponentialIntegralE1(f64 x) {
    if (x <= 1.0) {
        f64 sum = 0.0;
        f64 term = 1.0;
        for (int k = 1; k < 60; ++k) {
            term *= -x / k;
            const f64 add = -term / k;
            sum += add;
            if (std::abs(add) < 1e-17 * std::abs(sum)) break;
        }
        return -kEulerGamma - std::log(x) + sum;
    }
    f64 b = x + 1.0;
    f64 c = 1.0 / 1e-300;
    f64 d = 1.0 / b;
    f64 h = d;
    for (int i = 1; i < 200; ++i) {
        const f64 an = -static_cast<f64>(i) * i;
        b += 2.0;
        d = 1.0 / (an * d + b);
        c = b + an / c;
        const f64 del = c * d;
        h *= del;
        if (std::abs(del - 1.0) < 1e-16) break;
    }
    return h * std::exp(-x);
}

}  // namespace

f32 SkyRayleighOpticalDepth(f32 lambdaNm) {
    const f32 l = lambdaNm * 1e-3f;
    const f32 l2 = l * l;
    const f32 l4 = l2 * l2;
    return 0.008569f / l4 * (1.0f + 0.0113f / l2 + 0.00013f / l4);
}

f32 SkyAerosolOpticalDepth(f32 lambdaNm, f32 aod500) {
    return aod500 * std::pow(lambdaNm / 500.0f, -skydome::kAngstromExponent);
}

glm::vec3 SkySunDirection(const glm::vec3& sunDirection) {
    const f32 len = glm::length(sunDirection);
    glm::vec3 s = len > 1e-6f ? sunDirection / len : glm::vec3(0.0f, 1.0f, 0.0f);
    if (s.y < skydome::kMinSunSin) {
        glm::vec2 h(s.x, s.z);
        const f32 hl = glm::length(h);
        h = hl > 1e-6f ? h / hl : glm::vec2(1.0f, 0.0f);
        const f32 cosElevation =
            std::sqrt(std::max(1.0f - skydome::kMinSunSin * skydome::kMinSunSin, 0.0f));
        s = glm::vec3(h.x * cosElevation, skydome::kMinSunSin, h.y * cosElevation);
    }
    return s;
}

f64 ExponentialIntegralE3(f64 x) {
    if (x <= 0.0) return 0.5;
    // Two steps of the recurrence n E_{n+1}(x) = exp(-x) - x E_n(x).
    const f64 e1 = ExponentialIntegralE1(x);
    const f64 e2 = std::exp(-x) - x * e1;
    return 0.5 * (std::exp(-x) - x * e2);
}

f32 ClearSkyShape(f32 lambdaNm, const glm::vec3& direction, const glm::vec3& sun,
                  f32 aod500, f32 multipleScattering) {
    glm::vec3 d(direction.x, std::max(direction.y, 0.0f), direction.z);
    const f32 dl = glm::length(d);
    d = dl > 1e-6f ? d / dl : glm::vec3(1.0f, 0.0f, 0.0f);
    const f32 mu = std::max(d.y, skydome::kMinViewCos);
    const f32 mu0 = sun.y;

    const f32 tauR = SkyRayleighOpticalDepth(lambdaNm);
    const f32 tauA = SkyAerosolOpticalDepth(lambdaNm, aod500);
    const f32 tau = tauR + tauA;

    const f32 cosGamma = std::clamp(glm::dot(d, sun), -1.0f, 1.0f);
    const f32 pi = static_cast<f32>(kPi);
    const f32 phaseR = 3.0f / (16.0f * pi) * (1.0f + cosGamma * cosGamma);
    const f32 g = skydome::kAerosolAsymmetry;
    const f32 hg = 1.0f + g * g - 2.0f * g * cosGamma;
    const f32 phaseA = (1.0f - g * g) / (4.0f * pi * hg * std::sqrt(hg));
    const f32 phase = (tauR * phaseR + skydome::kAerosolAlbedo * tauA * phaseA) / tau;

    // mu0 (exp(-tau/mu) - exp(-tau/mu0)) / (mu - mu0), whose limit at mu = mu0
    // is tau exp(-tau/mu0) / mu0; the difference quotient cancels badly in
    // single precision close to it.
    const f32 dmu = mu - mu0;
    const f32 path = std::abs(dmu) < 1e-3f
        ? tau * std::exp(-tau / mu0) / mu0
        : mu0 * (std::exp(-tau / mu) - std::exp(-tau / mu0)) / dmu;

    return phase * path + multipleScattering * (1.0f - std::exp(-tau / mu));
}

ClearSkyNodeValues ClearSkyNode(f32 lambdaNm, const glm::vec3& sun, f32 aod500) {
    // The single-scattering part integrated over the upper hemisphere by the
    // midpoint rule in mu and in azimuth relative to the sun. The shape depends
    // on direction only through mu and the scattering angle, so the sun can be
    // put in the x-y plane and the half-circle of azimuth doubled. 64 x 64
    // agrees with a 512 x 1024 grid to 1e-4 over the visible.
    constexpr int kMu = 64;
    constexpr int kPhi = 64;
    const f32 mu0 = sun.y;
    const glm::vec3 sunLocal(std::sqrt(std::max(1.0f - mu0 * mu0, 0.0f)), mu0, 0.0f);
    f64 singleScattering = 0.0;
    for (int i = 0; i < kMu; ++i) {
        const f32 mu = (static_cast<f32>(i) + 0.5f) / kMu;
        const f32 sinTheta = std::sqrt(std::max(1.0f - mu * mu, 0.0f));
        for (int j = 0; j < kPhi; ++j) {
            const f32 phi = (static_cast<f32>(j) + 0.5f) / kPhi * static_cast<f32>(kPi);
            const glm::vec3 dir(sinTheta * std::cos(phi), mu, sinTheta * std::sin(phi));
            singleScattering += ClearSkyShape(lambdaNm, dir, sunLocal, aod500, 0.0f) * mu;
        }
    }
    singleScattering *= 2.0 * (1.0 / kMu) * (kPi / kPhi);

    // The layer's own diffuse light scattered again, isotropically. Viewed
    // from below, a uniform slab of source J and depth tau sends
    // J (1 - exp(-tau/mu)), whose cosine integral over the hemisphere is
    // 2 pi J (1/2 - E3(tau)). The source is what the scattering albedo makes
    // of the diffuse field, which includes itself:
    //   J = w E / (2 pi),  E = E_ss + 2 pi J (1/2 - E3(tau))
    // so E = E_ss / (1 - w (1/2 - E3)) and J follows.
    const f64 tauR = SkyRayleighOpticalDepth(lambdaNm);
    const f64 tauA = SkyAerosolOpticalDepth(lambdaNm, aod500);
    const f64 tau = tauR + tauA;
    const f64 albedo = (tauR + skydome::kAerosolAlbedo * tauA) / tau;
    const f64 slab = 0.5 - ExponentialIntegralE3(tau);
    const f64 total = singleScattering / (1.0 - albedo * slab);

    ClearSkyNodeValues out;
    out.cosineIntegral = static_cast<f32>(total);
    out.multipleScattering = static_cast<f32>(albedo * total / (2.0 * kPi));
    return out;
}

void FillSkyDome(LightingParams& params) {
    params.skyDomeLambdaStart_nm = skydome::kLambdaStartNm;
    params.skyDomeLambdaStep_nm = skydome::kLambdaStepNm;
    std::fill(std::begin(params.skyDomeN), std::end(params.skyDomeN), 0.0f);
    std::fill(std::begin(params.skyDomeJ), std::end(params.skyDomeJ), 0.0f);
    if (params.skyDistribution != SkyDistributionId::Clear) {
        return;
    }
    const glm::vec3 sun = SkySunDirection(params.sunDirection);
    for (u32 k = 0; k < skydome::kNodes; ++k) {
        const f32 lambda = skydome::kLambdaStartNm + static_cast<f32>(k) * skydome::kLambdaStepNm;
        const ClearSkyNodeValues node = ClearSkyNode(lambda, sun, params.skyAerosolOpticalDepth);
        params.skyDomeN[k] = node.cosineIntegral;
        params.skyDomeJ[k] = node.multipleScattering;
    }
}

}  // namespace quantiloom::rendercore
