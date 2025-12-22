#pragma once

#include "../core/Config.hpp"
#include "SensorModel.hpp"
#include "MultibandFusion.hpp"

namespace quantiloom {

/// Helper functions to parse postprocess parameters from TOML config
class PostprocessConfig {
public:
    /// Parse sensor parameters from config
    /// @param config TOML configuration
    /// @return Parsed sensor parameters
    static auto ParseSensorParams(const Config& config) -> SensorParams {
        SensorParams p;

        // Optics
        p.focalLength_mm = config.Get<f32>("sensor.focal_length_mm", 50.0f);
        p.fNumber = config.Get<f32>("sensor.f_number", 2.8f);
        p.pixelPitch_um = config.Get<f32>("sensor.pixel_pitch_um", 5.0f);

        // Detector
        p.quantumEfficiency = config.Get<f32>("sensor.quantum_efficiency", 0.8f);
        p.wellCapacity_e = config.Get<f32>("sensor.well_capacity_e", 50000.0f);
        p.readNoise_e_rms = config.Get<f32>("sensor.read_noise_e_rms", 10.0f);
        p.darkCurrent_e_s = config.Get<f32>("sensor.dark_current_e_s", 50.0f);
        p.integrationTime_s = config.Get<f32>("sensor.integration_time_s", 0.01f);

        // ADC
        p.bitDepth = config.Get<u32>("sensor.bit_depth", 14);
        p.gain = config.Get<f32>("sensor.gain", 0.5f);

        // Noise flags
        p.enablePoissonNoise = config.Get<bool>("sensor.enable_poisson_noise", true);
        p.enableReadNoise = config.Get<bool>("sensor.enable_read_noise", true);
        p.enableDarkCurrent = config.Get<bool>("sensor.enable_dark_current", true);
        p.enableFPN = config.Get<bool>("sensor.enable_fpn", false);

        // Temperature
        p.detectorTemperature_K = config.Get<f32>("sensor.detector_temperature_k", 77.0f);

        // Wavelength (use spectral.wavelength_nm if available)
        p.wavelength_nm = config.Get<f32>("spectral.wavelength_nm", 550.0f);

        return p;
    }

    /// Parse fusion parameters from config
    /// @param config TOML configuration
    /// @return Parsed fusion parameters
    static auto ParseFusionParams(const Config& config) -> FusionParams {
        FusionParams p;

        // Method selection
        const String methodStr = config.Get<String>("fusion.method", "laplacian_pyramid");
        if (methodStr == "weighted_average") {
            p.method = FusionMethod::WeightedAverage;
        } else if (methodStr == "laplacian_pyramid") {
            p.method = FusionMethod::LaplacianPyramid;
        } else if (methodStr == "max_response") {
            p.method = FusionMethod::MaxResponse;
        } else if (methodStr == "pseudo_color") {
            p.method = FusionMethod::PseudoColor;
        }

        // Method-specific parameters
        if (p.method == FusionMethod::LaplacianPyramid) {
            p.pyramidLevels = config.Get<u32>("fusion.laplacian_pyramid.levels", 5);
        } else if (p.method == FusionMethod::WeightedAverage) {
            p.weightVis = config.Get<f32>("fusion.weighted_average.weight_vis", 0.4f);
            p.weightSwir = config.Get<f32>("fusion.weighted_average.weight_swir", 0.3f);
            p.weightMwir = config.Get<f32>("fusion.weighted_average.weight_mwir", 0.3f);
        }

        // Normalization (common to all methods)
        p.autoNormalize = config.Get<bool>("fusion.auto_normalize", true);
        p.visMin = config.Get<f32>("fusion.vis_min", 0.0f);
        p.visMax = config.Get<f32>("fusion.vis_max", 1.0f);
        p.swirMin = config.Get<f32>("fusion.swir_min", 0.0f);
        p.swirMax = config.Get<f32>("fusion.swir_max", 1e-3f);
        p.mwirMin = config.Get<f32>("fusion.mwir_min", 0.0f);
        p.mwirMax = config.Get<f32>("fusion.mwir_max", 1e-2f);

        return p;
    }

    /// Check if sensor simulation is enabled
    static auto IsSensorEnabled(const Config& config) -> bool {
        return config.Get<bool>("sensor.enabled", false);
    }

    /// Check if multiband fusion is enabled
    static auto IsFusionEnabled(const Config& config) -> bool {
        return config.Get<bool>("fusion.enabled", false);
    }

    /// Get fusion output paths
    static auto GetFusionOutputPaths(const Config& config) -> std::pair<String, String> {
        const String exrPath = config.Get<String>("fusion.output.exr",
                                                   "output/fused_vis_swir_mwir.exr");
        const String pngPath = config.Get<String>("fusion.output.png",
                                                   "output/fused_vis_swir_mwir.png");
        return {exrPath, pngPath};
    }
};

} // namespace quantiloom
