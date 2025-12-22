#pragma once

#include "SensorModel.hpp"
#include <random>

namespace quantiloom {

/// Generic sensor implementation - covers VIS/SWIR/MWIR/LWIR
/// Full chain: Optics (PSF) → Detector (QE, noise) → ADC (quantization)
class GenericSensor final : public SensorModel {
public:
    GenericSensor();
    ~GenericSensor() override = default;

    /// Apply full sensor chain to HDR input
    auto Apply(const Image& hdr, const SensorParams& params)
        -> Result<SensorOutput, String> override;

private:
    // Step 1: Apply optical PSF (Gaussian approximation)
    auto ApplyPSF(const Image& img, f32 sigma_pixels) -> Image;

    // Step 2: Convert radiance to photo-electrons
    auto RadianceToElectrons(const Image& radiance, const SensorParams& p) -> Image;

    // Step 3: Add noise sources
    auto AddNoise(Image& electrons, const SensorParams& p) -> void;

    // Step 4: Quantize to digital numbers (DN)
    auto QuantizeToDN(const Image& electrons, const SensorParams& p) -> Image;

    // Step 5: Convert noisy electrons back to radiance (for preview)
    auto ElectronsToRadiance(const Image& electrons, const SensorParams& p) -> Image;

    // Helpers
    auto MakeGaussianKernel(f32 sigma) -> Vector<f32>;
    auto ConvolveX(const Image& img, const Vector<f32>& kernel) -> Image;
    auto ConvolveY(const Image& img, const Vector<f32>& kernel) -> Image;

    // RNG for noise
    std::mt19937 m_Rng;
};

} // namespace quantiloom
