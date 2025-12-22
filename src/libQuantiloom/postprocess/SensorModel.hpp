#pragma once

#include "../core/Image.hpp"
#include "../core/Types.hpp"

// ============================================================================
// Sensor Simulation - Abstract Interface
// ============================================================================
// Models the full sensor chain: Optics → Detector → ADC
// Converts radiance (W/m²/sr) to digital numbers (DN)
// ============================================================================

namespace quantiloom {

/// Sensor parameters (from TOML config)
struct SensorParams {
    // Optics
    f32 focalLength_mm = 50.0f;
    f32 fNumber = 2.8f;
    f32 pixelPitch_um = 5.0f;

    // Detector
    f32 quantumEfficiency = 0.8f;       // QE at peak wavelength [0.0-1.0]
    f32 wellCapacity_e = 50000.0f;      // Full well capacity (electrons)
    f32 readNoise_e_rms = 10.0f;        // Read noise (electrons RMS)
    f32 darkCurrent_e_s = 50.0f;        // Dark current (electrons/second)
    f32 integrationTime_s = 0.01f;      // Integration time (seconds)

    // ADC
    u32 bitDepth = 14;                  // ADC bit depth (12/14/16)
    f32 gain = 3.0f;                    // Gain (electrons/DN) - typical: 2-5

    // Noise flags
    bool enablePoissonNoise = true;     // Shot noise (photon counting)
    bool enableReadNoise = true;        // Readout noise
    bool enableDarkCurrent = true;      // Dark current noise
    bool enableFPN = false;             // Fixed pattern noise (PRNU/DSNU)

    // Temperature (for IR sensors)
    f32 detectorTemperature_K = 77.0f;  // Detector temperature (K)

    // Wavelength (for QE calculation, optional)
    f32 wavelength_nm = 550.0f;         // Peak wavelength (nm)
};

/// Sensor output containing both raw DN and enhanced preview
struct SensorOutput {
    Image rawDN;            // Raw sensor DN values [0, 2^bitDepth-1]
    Image enhancedPreview;  // Noisy radiance with PSF blur (same scale as input)
};

/// Abstract sensor model interface
class SensorModel {
public:
    virtual ~SensorModel() = default;

    /// Apply sensor effects to HDR radiance image
    /// @param hdr Input HDR image (radiance in W/m²/sr)
    /// @param params Sensor parameters
    /// @return SensorOutput containing raw DN and enhanced preview, or error
    virtual auto Apply(const Image& hdr, const SensorParams& params)
        -> Result<SensorOutput, String> = 0;
};

} // namespace quantiloom
