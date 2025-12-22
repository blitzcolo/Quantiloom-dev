#include "GenericSensor.hpp"
#include "../core/Log.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace quantiloom {

// ============================================================================
// Constants
// ============================================================================

// Planck constant (J·s)
constexpr f64 kPlanckConstant = 6.62607015e-34;

// Speed of light (m/s)
constexpr f64 kSpeedOfLight = 299792458.0;

// ============================================================================
// Constructor
// ============================================================================

GenericSensor::GenericSensor()
    : m_Rng(std::random_device{}()) {}

// ============================================================================
// Main Interface
// ============================================================================

auto GenericSensor::Apply(const Image& hdr, const SensorParams& params)
    -> Result<SensorOutput, String> {

    if (!hdr.IsValid()) {
        return typename Result<SensorOutput, String>::Err("Invalid input image");
    }

    Log::Debug("Sensor chain: Input {}x{} ({} channels)",
               hdr.width, hdr.height, hdr.channels);

    // Generate FPN maps if needed (lazy initialization)
    if (params.enableFPN && !m_FPNMapsGenerated) {
        GenerateFPNMaps(hdr.width, hdr.height, params);
        m_FPNMapsGenerated = true;
    }

    // Step 1: Apply PSF blur (diffraction-limited optics)
    // Calculate Airy disk radius (simplified: sigma ~ λ·f# / pixel_pitch)
    const f32 wavelength_m = params.wavelength_nm * 1e-9f;
    const f32 airyRadius_um = 1.22f * wavelength_m * 1e6f * params.fNumber;
    const f32 sigma_pixels = airyRadius_um / params.pixelPitch_um;

    Log::Debug("PSF: σ = {:.2f} pixels (Airy radius = {:.2f} μm)",
               sigma_pixels, airyRadius_um);

    Image blurred = ApplyPSF(hdr, sigma_pixels);

    // Step 2: Radiance → Photo-electrons
    Image electrons = RadianceToElectrons(blurred, params);

    // Step 3: Add noise
    AddNoise(electrons, params);

    // Step 4a: Quantize to DN (raw sensor output)
    Image rawDN = QuantizeToDN(electrons, params);

    // Step 4b: Convert noisy electrons back to radiance (enhanced preview)
    Image enhancedPreview = ElectronsToRadiance(electrons, params);

    // Add metadata
    rawDN.metadata["sensor_model"] = "GenericSensor";
    rawDN.metadata["integration_time_s"] = std::to_string(params.integrationTime_s);
    rawDN.metadata["gain"] = std::to_string(params.gain);
    rawDN.metadata["bit_depth"] = std::to_string(params.bitDepth);

    enhancedPreview.metadata["sensor_model"] = "GenericSensor";
    enhancedPreview.metadata["enhanced_preview"] = "true";

    Log::Info("Sensor chain complete: DN range [0, {}]", (1u << params.bitDepth) - 1);

    SensorOutput output;
    output.rawDN = std::move(rawDN);
    output.enhancedPreview = std::move(enhancedPreview);

    return std::move(output);  // Implicit conversion to Result
}

// ============================================================================
// Step 1: Optical PSF (Gaussian Approximation)
// ============================================================================

auto GenericSensor::ApplyPSF(const Image& img, const f32 sigma_pixels) -> Image {
    if (sigma_pixels < 0.1f) {
        // No blur needed
        return img;
    }

    // Separable Gaussian convolution: O(N·K) instead of O(N·K²)
    const auto kernel = MakeGaussianKernel(sigma_pixels);
    Image temp = ConvolveX(img, kernel);
    return ConvolveY(temp, kernel);
}

auto GenericSensor::MakeGaussianKernel(const f32 sigma) -> Vector<f32> {
    // Kernel radius: 3σ (covers 99.7% of Gaussian)
    const i32 radius = static_cast<i32>(std::ceil(3.0f * sigma));
    const i32 size = 2 * radius + 1;

    Vector<f32> kernel(size);
    f32 sum = 0.0f;

    for (i32 i = 0; i < size; ++i) {
        const f32 x = static_cast<f32>(i - radius);
        kernel[i] = std::exp(-0.5f * (x * x) / (sigma * sigma));
        sum += kernel[i];
    }

    // Normalize
    for (auto& k : kernel) {
        k /= sum;
    }

    return kernel;
}

auto GenericSensor::ConvolveX(const Image& img, const Vector<f32>& kernel) -> Image {
    Image result(img.width, img.height, img.channels);
    const i32 radius = static_cast<i32>(kernel.size()) / 2;

    for (u32 y = 0; y < img.height; ++y) {
        for (u32 x = 0; x < img.width; ++x) {
            for (u32 c = 0; c < img.channels; ++c) {
                f32 sum = 0.0f;
                for (i32 k = -radius; k <= radius; ++k) {
                    const i32 xk = std::clamp(static_cast<i32>(x) + k, 0,
                                              static_cast<i32>(img.width) - 1);
                    sum += img(xk, y, c) * kernel[k + radius];
                }
                result(x, y, c) = sum;
            }
        }
    }

    return result;
}

auto GenericSensor::ConvolveY(const Image& img, const Vector<f32>& kernel) -> Image {
    Image result(img.width, img.height, img.channels);
    const i32 radius = static_cast<i32>(kernel.size()) / 2;

    for (u32 y = 0; y < img.height; ++y) {
        for (u32 x = 0; x < img.width; ++x) {
            for (u32 c = 0; c < img.channels; ++c) {
                f32 sum = 0.0f;
                for (i32 k = -radius; k <= radius; ++k) {
                    const i32 yk = std::clamp(static_cast<i32>(y) + k, 0,
                                              static_cast<i32>(img.height) - 1);
                    sum += img(x, yk, c) * kernel[k + radius];
                }
                result(x, y, c) = sum;
            }
        }
    }

    return result;
}

// ============================================================================
// Step 2: Radiance → Photo-electrons
// ============================================================================

auto GenericSensor::RadianceToElectrons(const Image& radiance,
                                         const SensorParams& p) -> Image {
    Image electrons(radiance.width, radiance.height, radiance.channels);

    // Pixel area (m²)
    const f64 pixelArea_m2 = (p.pixelPitch_um * 1e-6) * (p.pixelPitch_um * 1e-6);

    // Photon energy: E = h·c / λ
    const f64 wavelength_m = p.wavelength_nm * 1e-9;
    const f64 photonEnergy_J = (kPlanckConstant * kSpeedOfLight) / wavelength_m;

    // Solid angle subtended by lens aperture: Ω = π / (4 × f#²)
    const f64 solidAngle_sr = std::numbers::pi / (4.0 * p.fNumber * p.fNumber);

    Log::Debug("Optics: f# = {:.1f}, Ω = {:.6e} sr, pixel area = {:.3e} m²",
               p.fNumber, solidAngle_sr, pixelArea_m2);

    // Statistics for debugging
    f64 minElectrons = 1e10, maxElectrons = 0.0, sumElectrons = 0.0;

    for (u32 i = 0; i < radiance.TotalElements(); ++i) {
        // Input: radiance L (W/m²/sr)
        // Irradiance: E = L · Ω (W/m²)
        const f64 irradiance_W_m2 = radiance.data[i] * solidAngle_sr;

        // Energy collected: E_total = E · A · t (Joules)
        const f64 energy_J = irradiance_W_m2 * pixelArea_m2 * p.integrationTime_s;

        // Number of photons: N_photons = E_total / E_photon
        const f64 numPhotons = energy_J / photonEnergy_J;

        // Number of photo-electrons: N_e = N_photons · QE
        f64 numElectrons = numPhotons * p.quantumEfficiency;

        // Add dark current
        if (p.enableDarkCurrent) {
            numElectrons += p.darkCurrent_e_s * p.integrationTime_s;
        }

        // Clamp to well capacity
        numElectrons = std::min(numElectrons, static_cast<f64>(p.wellCapacity_e));

        electrons.data[i] = static_cast<f32>(numElectrons);

        // Update statistics
        minElectrons = std::min(minElectrons, numElectrons);
        maxElectrons = std::max(maxElectrons, numElectrons);
        sumElectrons += numElectrons;
    }

    const f64 avgElectrons = sumElectrons / radiance.TotalElements();
    Log::Debug("Electrons: min={:.1f}, max={:.1f}, avg={:.1f} e-", minElectrons, maxElectrons, avgElectrons);

    return electrons;
}

// ============================================================================
// Step 4: ADC Quantization
// ============================================================================

auto GenericSensor::QuantizeToDN(const Image& electrons,
                                  const SensorParams& p) -> Image {
    Image dn(electrons.width, electrons.height, electrons.channels);

    const f32 maxDN = static_cast<f32>((1u << p.bitDepth) - 1);

    for (u32 i = 0; i < electrons.TotalElements(); ++i) {
        // Convert electrons to DN: DN = electrons / gain
        f32 dnValue = electrons.data[i] / p.gain;

        // Quantize to ADC range
        dnValue = std::clamp(dnValue, 0.0f, maxDN);

        // Floor to integer DN (ADC quantization)
        dnValue = std::floor(dnValue);

        dn.data[i] = dnValue;
    }

    return dn;
}

// ============================================================================
// Step 5: Photo-electrons → Radiance (Reverse Conversion for Preview)
// ============================================================================

auto GenericSensor::ElectronsToRadiance(const Image& electrons,
                                         const SensorParams& p) -> Image {
    Image radiance(electrons.width, electrons.height, electrons.channels);

    // Pixel area (m²)
    const f64 pixelArea_m2 = (p.pixelPitch_um * 1e-6) * (p.pixelPitch_um * 1e-6);

    // Photon energy: E = h·c / λ
    const f64 wavelength_m = p.wavelength_nm * 1e-9;
    const f64 photonEnergy_J = (kPlanckConstant * kSpeedOfLight) / wavelength_m;

    // Solid angle subtended by lens aperture: Ω = π / (4 × f#²)
    const f64 solidAngle_sr = std::numbers::pi / (4.0 * p.fNumber * p.fNumber);

    // Reverse conversion: electrons → radiance
    for (u32 i = 0; i < electrons.TotalElements(); ++i) {
        f64 numElectrons = electrons.data[i];

        // Remove dark current contribution
        if (p.enableDarkCurrent) {
            numElectrons -= p.darkCurrent_e_s * p.integrationTime_s;
            numElectrons = std::max(numElectrons, 0.0);
        }

        // Electrons → Photons: N_photons = N_e / QE
        const f64 numPhotons = numElectrons / p.quantumEfficiency;

        // Photons → Energy: E_total = N_photons × E_photon
        const f64 energy_J = numPhotons * photonEnergy_J;

        // Energy → Irradiance: E = E_total / (A × t)
        const f64 irradiance_W_m2 = energy_J / (pixelArea_m2 * p.integrationTime_s);

        // Irradiance → Radiance: L = E / Ω
        const f64 radiance_W_m2_sr = irradiance_W_m2 / solidAngle_sr;

        radiance.data[i] = static_cast<f32>(radiance_W_m2_sr);
    }

    return radiance;
}

// ============================================================================
// FPN: Generate Fixed Pattern Noise Maps (PRNU + DSNU)
// ============================================================================

auto GenericSensor::GenerateFPNMaps(const u32 width, const u32 height,
                                     const SensorParams& p) -> void {
    Log::Info("Generating FPN maps: {}x{} (PRNU sigma={:.2f}%, DSNU sigma={:.1f} e-)",
              width, height, p.prnuSigma * 100.0f, p.dsnuSigma_e);

    // Initialize maps
    m_PRNUMap.Resize(width, height, 1);  // Single channel (grayscale)
    m_DSNUMap.Resize(width, height, 1);

    // Generate PRNU map (if sigma > 0)
    // PRNU simulates column-wise gain non-uniformity from readout electronics
    // Using anisotropic filtering: smooth along Y (columns), preserve X independence
    // This creates VERTICAL STRIPES (column FPN)
    if (p.prnuSigma > 1e-6f) {
        std::normal_distribution<f32> prnuDist(0.0f, p.prnuSigma);

        // Step 1: Generate per-column random values (1D noise expanded to 2D)
        // Each column shares the same base value with slight per-pixel variation
        Vector<f32> columnNoise(width);
        for (u32 x = 0; x < width; ++x) {
            columnNoise[x] = prnuDist(m_Rng);
        }

        // Step 2: Apply 1D smoothing to column noise for wider stripes
        f32 smoothingSigma = 8.0f;  // Stripe width in pixels
        auto kernel = MakeGaussianKernel(smoothingSigma);
        const i32 radius = static_cast<i32>(kernel.size()) / 2;

        Vector<f32> smoothedColumnNoise(width);
        for (u32 x = 0; x < width; ++x) {
            f32 sum = 0.0f;
            for (i32 k = -radius; k <= radius; ++k) {
                const i32 xk = std::clamp(static_cast<i32>(x) + k, 0, static_cast<i32>(width) - 1);
                sum += columnNoise[xk] * kernel[k + radius];
            }
            smoothedColumnNoise[x] = sum;
        }

        // Step 3: Expand to 2D map (same value for entire column + small per-pixel variation)
        std::normal_distribution<f32> pixelNoise(0.0f, p.prnuSigma * 0.1f);  // 10% pixel-level noise
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                m_PRNUMap(x, y, 0) = smoothedColumnNoise[x] + pixelNoise(m_Rng);
            }
        }

        // Step 4: Renormalize to target sigma
        f32 currentMean = 0.0f;
        for (const auto& val : m_PRNUMap.data) {
            currentMean += val;
        }
        currentMean /= m_PRNUMap.TotalElements();

        f32 currentVariance = 0.0f;
        for (const auto& val : m_PRNUMap.data) {
            f32 diff = val - currentMean;
            currentVariance += diff * diff;
        }
        currentVariance /= m_PRNUMap.TotalElements();
        f32 currentSigma = std::sqrt(currentVariance);

        if (currentSigma > 1e-6f) {
            f32 scale = p.prnuSigma / currentSigma;
            for (auto& val : m_PRNUMap.data) {
                val = (val - currentMean) * scale;
            }
        }
    } else {
        std::fill(m_PRNUMap.data.begin(), m_PRNUMap.data.end(), 0.0f);
    }

    // Generate DSNU map (if sigma > 0)
    // DSNU simulates row-wise dark current non-uniformity from row addressing
    // Using anisotropic filtering: smooth along X (rows), preserve Y independence
    // This creates HORIZONTAL STRIPES (row FPN)
    if (p.dsnuSigma_e > 1e-6f) {
        std::normal_distribution<f32> dsnuDist(0.0f, p.dsnuSigma_e);

        // Step 1: Generate per-row random values
        Vector<f32> rowNoise(height);
        for (u32 y = 0; y < height; ++y) {
            rowNoise[y] = dsnuDist(m_Rng);
        }

        // Step 2: Apply 1D smoothing to row noise for wider stripes
        f32 smoothingSigma = 5.0f;  // Stripe width in pixels (narrower than PRNU)
        auto kernel = MakeGaussianKernel(smoothingSigma);
        const i32 radius = static_cast<i32>(kernel.size()) / 2;

        Vector<f32> smoothedRowNoise(height);
        for (u32 y = 0; y < height; ++y) {
            f32 sum = 0.0f;
            for (i32 k = -radius; k <= radius; ++k) {
                const i32 yk = std::clamp(static_cast<i32>(y) + k, 0, static_cast<i32>(height) - 1);
                sum += rowNoise[yk] * kernel[k + radius];
            }
            smoothedRowNoise[y] = sum;
        }

        // Step 3: Expand to 2D map (same value for entire row + small per-pixel variation)
        std::normal_distribution<f32> pixelNoise(0.0f, p.dsnuSigma_e * 0.1f);  // 10% pixel-level noise
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                m_DSNUMap(x, y, 0) = smoothedRowNoise[y] + pixelNoise(m_Rng);
            }
        }

        // Step 4: Renormalize to target sigma
        f32 currentMean = 0.0f;
        for (const auto& val : m_DSNUMap.data) {
            currentMean += val;
        }
        currentMean /= m_DSNUMap.TotalElements();

        f32 currentVariance = 0.0f;
        for (const auto& val : m_DSNUMap.data) {
            f32 diff = val - currentMean;
            currentVariance += diff * diff;
        }
        currentVariance /= m_DSNUMap.TotalElements();
        f32 currentSigma = std::sqrt(currentVariance);

        if (currentSigma > 1e-6f) {
            f32 scale = p.dsnuSigma_e / currentSigma;
            for (auto& val : m_DSNUMap.data) {
                val = (val - currentMean) * scale;
            }
        }
    } else {
        std::fill(m_DSNUMap.data.begin(), m_DSNUMap.data.end(), 0.0f);
    }

    Log::Debug("FPN maps generated: PRNU range [{:.4f}, {:.4f}], DSNU range [{:.2f}, {:.2f}] e-",
               *std::min_element(m_PRNUMap.data.begin(), m_PRNUMap.data.end()),
               *std::max_element(m_PRNUMap.data.begin(), m_PRNUMap.data.end()),
               *std::min_element(m_DSNUMap.data.begin(), m_DSNUMap.data.end()),
               *std::max_element(m_DSNUMap.data.begin(), m_DSNUMap.data.end()));
}

// ============================================================================
// FPN: Apply Fixed Pattern Noise (PRNU + DSNU) with Optional NUC
// ============================================================================

auto GenericSensor::ApplyFPN(Image& electrons, const SensorParams& p) -> void {
    if (!p.enableFPN || !m_FPNMapsGenerated) {
        return;  // FPN disabled or maps not generated
    }

    // Verify map dimensions match image dimensions
    if (m_PRNUMap.width != electrons.width || m_PRNUMap.height != electrons.height) {
        Log::Warn("FPN map size mismatch: {}x{} vs {}x{}, skipping FPN",
                  m_PRNUMap.width, m_PRNUMap.height, electrons.width, electrons.height);
        return;
    }

    // Apply FPN with optional NUC correction
    for (u32 y = 0; y < electrons.height; ++y) {
        for (u32 x = 0; x < electrons.width; ++x) {
            const u32 idx = y * electrons.width + x;

            // Get FPN values for this pixel
            f32 prnu = m_PRNUMap.data[idx];  // Multiplicative gain error
            f32 dsnu = m_DSNUMap.data[idx];  // Additive dark signal error

            // Apply NUC correction (if enabled)
            // NUC reduces FPN but leaves a residual due to imperfect calibration
            if (p.enableNUC) {
                prnu *= (1.0f - p.nucEfficiency);  // Residual PRNU after NUC
                dsnu *= (1.0f - p.nucEfficiency);  // Residual DSNU after NUC
            }

            // Apply FPN to all channels
            for (u32 c = 0; c < electrons.channels; ++c) {
                f32& signal = electrons(x, y, c);

                // PRNU: multiplicative gain non-uniformity
                // signal' = signal × (1 + prnu)
                signal *= (1.0f + prnu);

                // DSNU: additive dark signal non-uniformity
                // signal'' = signal' + dsnu
                signal += dsnu;
            }
        }
    }

    // Log statistics for verification
    f32 totalVariance = 0.0f;
    f32 mean = 0.0f;
    for (u32 i = 0; i < electrons.TotalElements(); ++i) {
        mean += electrons.data[i];
    }
    mean /= electrons.TotalElements();

    for (u32 i = 0; i < electrons.TotalElements(); ++i) {
        f32 diff = electrons.data[i] - mean;
        totalVariance += diff * diff;
    }
    totalVariance /= electrons.TotalElements();
    f32 stdDev = std::sqrt(totalVariance);

    const char* nucStatus = p.enableNUC ? " (with NUC residual)" : " (without NUC)";
    Log::Info("FPN applied{}: mean={:.1f} e-, stddev={:.1f} e- ({:.2f}%)",
              nucStatus, mean, stdDev, (stdDev / mean) * 100.0f);
}

// ============================================================================
// Step 3: Add Noise (Updated to include FPN)
// ============================================================================

auto GenericSensor::AddNoise(Image& electrons, const SensorParams& p) -> void {
    std::normal_distribution<f32> gaussianDist(0.0f, 1.0f);

    // Apply temporal noise sources (Poisson, Read Noise)
    for (u32 i = 0; i < electrons.TotalElements(); ++i) {
        f32 signal = electrons.data[i];

        // Poisson noise (shot noise): σ = sqrt(N)
        if (p.enablePoissonNoise && signal > 0.0f) {
            const f32 shotNoise_rms = std::sqrt(signal);
            signal += gaussianDist(m_Rng) * shotNoise_rms;
        }

        // Read noise (Gaussian)
        if (p.enableReadNoise) {
            signal += gaussianDist(m_Rng) * p.readNoise_e_rms;
        }

        electrons.data[i] = signal;
    }

    // Apply Fixed Pattern Noise (PRNU + DSNU) after temporal noise
    // FPN is applied last because it affects the signal AFTER photon/electron conversion
    ApplyFPN(electrons, p);

    // Final clamp to [0, well_capacity]
    for (u32 i = 0; i < electrons.TotalElements(); ++i) {
        electrons.data[i] = std::clamp(electrons.data[i], 0.0f, p.wellCapacity_e);
    }
}

} // namespace quantiloom
