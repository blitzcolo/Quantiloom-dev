#pragma once

#include "postprocess/CameraPhysics.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace quantiloom::camera {

// Binding 29 is one ByteAddressBuffer. All offsets below are bytes. The shader
// reads the same constants from camera_response.hlsli; keep both in step.
inline constexpr u32 kCameraResponseMagic = 0x514C4341u; // "QLCA"
inline constexpr u32 kCameraResponseGpuVersion = 1u;
inline constexpr u32 kCameraResponseHeaderWords = 16u;   // 64 bytes
inline constexpr u32 kCameraResponseChannelWords = 12u;  // 48 bytes
inline constexpr u32 kCameraResponsePdfBins = 128u;
inline constexpr f32 kCameraResponseUniformProposal = 0.05f;
inline constexpr u32 kCameraResponseMaxChannels = 3u;

namespace detail {
inline u32 FloatWord(f32 value) { return std::bit_cast<u32>(value); }
inline void PutFloat(Vector<u32>& words, size_t index, f64 value) {
    words[index] = FloatWord(static_cast<f32>(value));
}

inline const ResponseCurve& Base(const ResponseStack& stack, DetectorKind detector) {
    return stack.systemResponse ? *stack.systemResponse :
        detector == DetectorKind::Photon ? *stack.quantumEfficiency : *stack.thermalAbsorptance;
}

inline f64 CurveAt(const ResponseCurve& curve, f64 nm) {
    if (nm < curve.MinNm() || nm > curve.MaxNm()) return 0.0;
    const auto hi = std::lower_bound(curve.wavelengthNm.begin(), curve.wavelengthNm.end(), nm);
    if (hi == curve.wavelengthNm.begin()) return curve.amplitude * curve.value.front();
    if (hi == curve.wavelengthNm.end()) return curve.amplitude * curve.value.back();
    const size_t index = static_cast<size_t>(hi - curve.wavelengthNm.begin());
    const f64 x0 = curve.wavelengthNm[index - 1], x1 = curve.wavelengthNm[index];
    const f64 t = (nm - x0) / (x1 - x0);
    return curve.amplitude * (curve.value[index - 1] * (1.0 - t) + curve.value[index] * t);
}

inline f64 ResponseAt(const ResponseStack& stack, DetectorKind detector, f64 nm) {
    f64 value = CurveAt(Base(stack, detector), nm);
    if (stack.lensTransmission) value *= CurveAt(*stack.lensTransmission, nm);
    if (stack.filterTransmission) value *= CurveAt(*stack.filterTransmission, nm);
    return value;
}

inline f64 ProposalDensity(const ResponseStack& stack, DetectorKind detector, f64 nm) {
    const f64 value = ResponseAt(stack, detector, nm);
    return detector == DetectorKind::Photon ? value * nm : value;
}

inline f64 Gauss4(f64 a, f64 b, const ResponseStack& stack, DetectorKind detector) {
    constexpr f64 nodes[4] = {-0.8611363115940526, -0.3399810435848563,
                               0.3399810435848563, 0.8611363115940526};
    constexpr f64 weights[4] = {0.3478548451374539, 0.6521451548625461,
                                 0.6521451548625461, 0.3478548451374539};
    const f64 mid = 0.5 * (a + b), half = 0.5 * (b - a);
    f64 sum = 0.0;
    for (u32 i = 0; i < 4; ++i)
        sum += weights[i] * ProposalDensity(stack, detector, mid + half * nodes[i]);
    return half * sum;
}

inline f64 BinMass(const ResponseStack& stack, DetectorKind detector, f64 lo, f64 hi) {
    std::vector<f64> knots{lo, hi};
    const auto append = [&](const ResponseCurve& curve) {
        for (const f64 nm : curve.wavelengthNm)
            if (nm > lo && nm < hi) knots.push_back(nm);
    };
    append(Base(stack, detector));
    if (stack.lensTransmission) append(*stack.lensTransmission);
    if (stack.filterTransmission) append(*stack.filterTransmission);
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
    f64 sum = 0.0;
    for (size_t i = 1; i < knots.size(); ++i)
        sum += Gauss4(knots[i - 1], knots[i], stack, detector);
    return sum;
}

inline std::pair<u32, u32> AppendPoints(Vector<u32>& words,
                                         const ResponseCurve* curve) {
    if (curve == nullptr) return {0u, 0u};
    const u32 offsetBytes = static_cast<u32>(words.size() * sizeof(u32));
    for (size_t i = 0; i < curve->value.size(); ++i) {
        words.push_back(FloatWord(static_cast<f32>(curve->wavelengthNm[i])));
        words.push_back(FloatWord(static_cast<f32>(curve->amplitude * curve->value[i])));
    }
    return {offsetBytes, static_cast<u32>(curve->value.size())};
}
} // namespace detail

// All curve knots remain separate: the shader linearly interpolates base,
// lens and filter before multiplying them, exactly as CameraPhysics does.
// The CDF is only a proposal. A 5% uniform mixture ensures positive PDF at
// every supported wavelength even if a narrow feature falls in a sparse bin.
[[nodiscard]] inline Result<Vector<u32>, String> EncodeCameraResponseGpu(
    const CameraConfig& config, f32 atmosMinNm = 0.0f,
    f32 atmosStepNm = 0.0f, u32 atmosCount = 0u) {
    using Encoded = Result<Vector<u32>, String>;
    if (auto valid = ValidateCameraConfig(config); !valid)
        return Encoded::Err(valid.error());
    const size_t channelCount = config.device.channels.size();
    if (channelCount == 0 || channelCount > kCameraResponseMaxChannels)
        return Encoded::Err("GPU camera supports one to three response channels");
    if (atmosCount > 0u && (!std::isfinite(atmosMinNm) ||
                           !std::isfinite(atmosStepNm) || atmosStepNm <= 0.0f))
        return Encoded::Err("camera atmosphere grid must have finite positive spacing");
    const auto omega = ApertureSolidAngleSr(config.optics.fNumber);
    const auto area = PixelCollectionAreaM2(config.optics);
    if (!omega) return Encoded::Err(omega.error());
    if (!area) return Encoded::Err(area.error());
    if (!(static_cast<f32>(omega.value()) > 0.0f) ||
        !(static_cast<f32>(area.value()) > 0.0f))
        return Encoded::Err("camera collection geometry exceeds GPU float range");

    Vector<u32> words(kCameraResponseHeaderWords +
                      channelCount * kCameraResponseChannelWords, 0u);
    words[0] = kCameraResponseMagic;
    words[1] = kCameraResponseGpuVersion;
    words[2] = static_cast<u32>(config.device.detector);
    words[3] = static_cast<u32>(config.device.cfa);
    words[4] = static_cast<u32>(channelCount);
    words[5] = kCameraResponsePdfBins;
    words[6] = config.optics.cosFourthVignetting ? 1u : 0u;
    words[7] = config.optics.sensorWidthPx;
    detail::PutFloat(words, 8, omega.value());
    detail::PutFloat(words, 9, area.value());
    detail::PutFloat(words, 10, kCameraResponseUniformProposal);
    words[11] = config.optics.sensorHeightPx;
    detail::PutFloat(words, 12, atmosMinNm);
    detail::PutFloat(words, 13, atmosStepNm);
    words[14] = atmosCount;

    for (size_t c = 0; c < channelCount; ++c) {
        const auto& stack = config.device.channels[c].response;
        const ResponseCurve& base = detail::Base(stack, config.device.detector);
        if (!(static_cast<f32>(base.MaxNm()) > static_cast<f32>(base.MinNm())))
            return Encoded::Err("camera response span is not representable on GPU");
        const auto representable = [](const ResponseCurve& curve) {
            for (size_t i = 0; i < curve.value.size(); ++i) {
                if (!std::isfinite(static_cast<f32>(curve.wavelengthNm[i])) ||
                    !std::isfinite(static_cast<f32>(curve.amplitude * curve.value[i])))
                    return false;
            }
            return true;
        };
        if (!representable(base) ||
            (stack.lensTransmission && !representable(*stack.lensTransmission)) ||
            (stack.filterTransmission && !representable(*stack.filterTransmission)))
            return Encoded::Err("camera response knots exceed GPU float range");
        const size_t info = kCameraResponseHeaderWords + c * kCameraResponseChannelWords;
        const auto baseRef = detail::AppendPoints(words, &base);
        const auto lensRef = detail::AppendPoints(words,
            stack.lensTransmission ? &*stack.lensTransmission : nullptr);
        const auto filterRef = detail::AppendPoints(words,
            stack.filterTransmission ? &*stack.filterTransmission : nullptr);
        words[info + 0] = baseRef.first; words[info + 1] = baseRef.second;
        words[info + 2] = lensRef.first; words[info + 3] = lensRef.second;
        words[info + 4] = filterRef.first; words[info + 5] = filterRef.second;
        detail::PutFloat(words, info + 8, base.MinNm());
        detail::PutFloat(words, info + 9, base.MaxNm());

        const f64 width = (base.MaxNm() - base.MinNm()) / kCameraResponsePdfBins;
        std::vector<f64> masses(kCameraResponsePdfBins);
        f64 total = 0.0;
        for (u32 bin = 0; bin < kCameraResponsePdfBins; ++bin) {
            const f64 lo = base.MinNm() + bin * width;
            const f64 hi = bin + 1u == kCameraResponsePdfBins ? base.MaxNm() : lo + width;
            masses[bin] = std::max(0.0, detail::BinMass(
                stack, config.device.detector, lo, hi));
            total += masses[bin];
        }
        if (!(total > 0.0) || !std::isfinite(total))
            return Encoded::Err("GPU camera response has no finite positive proposal mass");
        words[info + 6] = static_cast<u32>(words.size() * sizeof(u32));
        words[info + 7] = kCameraResponsePdfBins;
        f32 cumulative = 0.0f;
        for (u32 bin = 0; bin < kCameraResponsePdfBins; ++bin) {
            const f32 mass = bin + 1u == kCameraResponsePdfBins
                ? std::max(0.0f, 1.0f - cumulative)
                : static_cast<f32>(masses[bin] / total);
            cumulative += mass;
            words.push_back(detail::FloatWord(mass));
            words.push_back(detail::FloatWord(
                bin + 1u == kCameraResponsePdfBins ? 1.0f : cumulative));
        }
    }
    if (words.size() > std::numeric_limits<u32>::max() / sizeof(u32))
        return Encoded::Err("GPU camera response table is too large");
    return Encoded(std::move(words));
}
} // namespace quantiloom::camera
