#include "RenderJob.hpp"

#include "core/Log.hpp"
#include "io/ImageIO.hpp"
#include "postprocess/GenericSensor.hpp"
#include "postprocess/PostprocessConfig.hpp"
#include "postprocess/Thermography.hpp"
#include "renderer/OfflineRenderer.hpp"

#include <algorithm>  // For std::nth_element
#include <chrono>
#include <filesystem>
#include <limits>
#include <vector>

namespace quantiloom::app {
namespace {

/**
 * @brief Run the sensor imaging chain over the traced frame
 *
 * Modifies `img` in place to the enhanced preview, and writes the raw DN image
 * beside the EXR.
 */
void ApplySensorChain(const Config& config, const OfflineRenderOutput& rendered, Image& img,
                      const String& outputPath, const u32 width, const u32 height) {
    QL_LOG_INFO("Applying sensor simulation...");

    SensorParams sensorParams = PostprocessConfig::ParseSensorParams(config);

    // ========================================================================
    // IR fused modes: unit fixup for the sensor photon budget
    // ========================================================================
    // The renderer stores per-nm AVERAGE spectral radiance (band integral /
    // band width, see closesthit.rchit) while the sensor chain expects
    // band-INTEGRATED radiance (W/sr/m^2). Multiply by the band width here, and
    // use the band center for photon energy instead of the 550 nm visible-light
    // default.
    //
    // Both numbers come back with the frame rather than being derived here: the
    // renderer is the only party that knows which band it just integrated.
    const f32 bandScale = rendered.sensorRadianceScale;
    if (rendered.sensorWavelengthNm > 0.0f) {
        sensorParams.wavelength_nm = rendered.sensorWavelengthNm;
    }
    if (bandScale != 1.0f) {
        QL_LOG_INFO("  IR sensor units: radiance x{:.0f} nm bandwidth, photon wavelength {:.0f} nm",
                    bandScale, sensorParams.wavelength_nm);
    }

    GenericSensor sensor;

    // Extract RGB/grayscale channels (drop alpha for sensor simulation)
    Image hdrInput(width, height, 3);
    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            hdrInput(x, y, 0) = img(x, y, 0) * bandScale;  // R
            hdrInput(x, y, 1) = img(x, y, 1) * bandScale;  // G
            hdrInput(x, y, 2) = img(x, y, 2) * bandScale;  // B
        }
    }

    auto sensorResult = sensor.Apply(hdrInput, sensorParams);
    if (!sensorResult.has_value()) {
        QL_LOG_ERROR("  [FAIL] Sensor simulation failed: {}", sensorResult.error());
        return;
    }
    QL_LOG_INFO("  [OK] Sensor simulation complete");

    const SensorOutput& sensorOutput = sensorResult.value();

    // Replace image with enhanced preview (noisy radiance, for PNG/visualization).
    // Divide the band scale back out so the EXR keeps the same per-nm average
    // radiance units as the sensor-off path.
    const Image& enhancedPreview = sensorOutput.enhancedPreview;
    const f32 invBandScale = 1.0f / bandScale;
    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            img(x, y, 0) = enhancedPreview(x, y, 0) * invBandScale;  // R
            img(x, y, 1) = enhancedPreview(x, y, 1) * invBandScale;  // G
            img(x, y, 2) = enhancedPreview(x, y, 2) * invBandScale;  // B
            // Alpha unchanged
        }
    }

    img.metadata["postprocess"] = "sensor_simulation_preview";

    // Save raw DN image to separate file
    const std::filesystem::path exrPath(outputPath);
    const std::string rawDnPath =
        (exrPath.parent_path() / (exrPath.stem().string() + "_rawdn.exr")).string();

    QL_LOG_INFO("Saving raw DN image to {}...", rawDnPath);
    if (ImageIO::WriteEXR(rawDnPath, sensorOutput.rawDN)) {
        QL_LOG_INFO("  [OK] Saved raw DN image");
    } else {
        QL_LOG_WARN("  [WARN] Failed to save raw DN image");
    }
}

/**
 * @brief Write the temperature a thermal camera would report for each pixel
 *
 * A thermogram is not a radiance field: the camera inverts what it measured
 * against Planck and displays a temperature. Doing the same to a render is
 * what makes it comparable with a measured thermogram pixel for pixel, which
 * is the whole point of simulating one (Aguerre et al. 2020, eq. 8-11, taken
 * per band rather than over the whole spectrum).
 *
 * Runs on the traced radiance, before the sensor chain, so the map is a
 * property of the scene rather than of a detector's calibration. Inverting the
 * sensor's own output was tried and is not usable: a thermal band carries an
 * enormous DC term -- everything in view is near 300 K -- and a converter with
 * no offset subtraction spends its whole range on it. A 14-bit chain
 * calibrated to hold a 300 K scene left 12 DN for the 60 K the scene actually
 * varied over, and the quantisation error inverted to nonsense. Real cameras
 * subtract a reference blackbody before the ADC; this sensor model does not,
 * and until it does the temperature map belongs upstream of it.
 *
 * NETD answers the other half of the question -- how small a difference this
 * detector could resolve -- and needs the sensor parameters but not its noisy
 * output, so it is reported here regardless.
 */
void WriteApparentTemperature(const Config& config, const OfflineRenderOutput& rendered,
                              const Image& img, const SpectralMode mode,
                              const String& outputPath, RenderOutcome& outcome) {
    const auto band = GetFusedBandInfo(mode);
    if (!band.has_value() || !IsIRFusedMode(mode)) {
        QL_LOG_WARN("Thermography is enabled but the render mode is {}, which carries no "
                    "band radiance to invert -- skipping the temperature map",
                    outcome.modeName);
        return;
    }

    const ThermographyParams params = PostprocessConfig::ParseThermographyParams(config);
    const f64 lambdaMin = static_cast<f64>(band->lambdaMinNm);
    const f64 lambdaMax = static_cast<f64>(band->lambdaMaxNm);

    // The EXR is per-nm average spectral radiance, which is what the band
    // routines take: no scaling on the way in, and the isothermal cavity
    // therefore comes back at its own temperature.
    Image temperature(img.width, img.height, 1);
    temperature.channelNames = {"T"};

    f64 minK = std::numeric_limits<f64>::max();
    f64 maxK = std::numeric_limits<f64>::lowest();
    std::vector<f32> values;
    values.reserve(static_cast<usize>(img.width) * img.height);
    for (u32 y = 0; y < img.height; ++y) {
        for (u32 x = 0; x < img.width; ++x) {
            const f64 T = InvertSurfaceTemperatureK(static_cast<f64>(img(x, y, 0)), lambdaMin,
                                                    lambdaMax, params);
            temperature(x, y, 0) = static_cast<f32>(T);
            values.push_back(static_cast<f32>(T));
            minK = std::min(minK, T);
            maxK = std::max(maxK, T);
        }
    }

    // The median, not the mean: a thermogram of anything outdoors has sky in
    // it, tens of kelvin below the surfaces the image is of, and a mean would
    // report a sensitivity at a temperature nothing in the scene has.
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    const f64 medianK = static_cast<f64>(*middle);

    temperature.metadata["units"] = "K";
    temperature.metadata["band_nm"] = std::to_string(band->lambdaMinNm) + "-" +
                                      std::to_string(band->lambdaMaxNm);
    temperature.metadata["emissivity"] = std::to_string(params.emissivity);
    temperature.metadata["reflected_temperature_k"] =
        std::to_string(params.reflectedTemperature_K);
    temperature.metadata["atmosphere_transmittance"] =
        std::to_string(params.atmosphereTransmittance);
    temperature.metadata["inverted_from"] = "scene_radiance";

    // NETD, at the scene's own median temperature: a sensitivity quoted at a
    // temperature the scene does not contain says nothing about this image.
    // Needs a sensor to be a sensitivity of something, so it is reported only
    // when one is configured.
    if (PostprocessConfig::IsSensorEnabled(config) &&
        PostprocessConfig::IsNetdReportEnabled(config)) {
        SensorParams sensorParams = PostprocessConfig::ParseSensorParams(config);
        if (rendered.sensorWavelengthNm > 0.0f) {
            sensorParams.wavelength_nm = rendered.sensorWavelengthNm;
        }
        const f64 netd =
            NoiseEquivalentTemperatureDifferenceK(sensorParams, lambdaMin, lambdaMax, medianK);
        temperature.metadata["netd_mk"] = std::to_string(netd * 1000.0);
        temperature.metadata["netd_reference_k"] = std::to_string(medianK);
        QL_LOG_INFO("  NETD at {:.1f} K: {:.1f} mK", medianK, netd * 1000.0);
    }

    const std::filesystem::path exrPath(outputPath);
    const String tappPath =
        (exrPath.parent_path() / (exrPath.stem().string() + "_tapp.exr")).string();

    if (ImageIO::WriteEXR(tappPath, temperature)) {
        outcome.tappPath = tappPath;
        QL_LOG_INFO("  [OK] Saved temperature map to {} ({:.1f}-{:.1f} K, median {:.1f} K)",
                    tappPath, minK, maxK, medianK);
    } else {
        QL_LOG_WARN("  [WARN] Failed to save temperature map to {}", tappPath);
    }
}

/**
 * @brief Build the displayable RGB frame for the PNG preview
 *
 * Physical radiance does not live in [0, 1], so a raw clamp makes a preview
 * that is all black or all white. IR fused modes are far below it (LWIR
 * ~5e-3 W/sr/m^2/nm); a scene lit by a measured solar spectrum is far above it,
 * since ASTM G-173 integrates to a few hundred rather than to the 5.0 someone
 * used to type into sun_radiance. Same remedy for both: stretch the 1st..99th
 * percentile to [0, 1] for the preview, and leave the EXR alone.
 *
 * Applied only when the values actually leave the range, so a scene that was
 * already displayable previews exactly as before.
 */
Image BuildPreview(const Image& img, const SpectralMode mode, const u32 width, const u32 height) {
    Image pngImg(width, height, 3);
    pngImg.channelNames = {"R", "G", "B"};

    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            pngImg(x, y, 0) = img(x, y, 0);  // R
            pngImg(x, y, 1) = img(x, y, 1);  // G
            pngImg(x, y, 2) = img(x, y, 2);  // B
        }
    }

    std::vector<f32> values(static_cast<size_t>(width) * height);
    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            values[static_cast<size_t>(y) * width + x] = pngImg(x, y, 0);
        }
    }
    const size_t loIdx = values.size() / 100;
    const size_t hiIdx = values.size() - 1 - loIdx;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(loIdx),
                     values.end());
    const f32 lo = values[loIdx];
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(hiIdx),
                     values.end());
    const f32 hi = values[hiIdx];
    const f32 range = std::max(hi - lo, 1e-12f);

    if (IsIRFusedMode(mode) || hi > 1.0f) {
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                for (u32 c = 0; c < 3; ++c) {
                    pngImg(x, y, c) = (pngImg(x, y, c) - lo) / range;
                }
            }
        }
        QL_LOG_INFO("  PNG preview stretched from [{:.4g}, {:.4g}]; "
                    "the EXR keeps the physical values", lo, hi);
    }
    QL_LOG_INFO("  IR PNG preview normalized: [{:.4e}, {:.4e}] -> [0, 1]", lo, hi);

    return pngImg;
}

}  // namespace

RenderOutcome RenderConfigToFiles(const Config& config, const String& atmosphereModelPackFallback) {
    OfflineRenderer::InitParams initParams;
    initParams.atmosphereModelPackFallback = atmosphereModelPackFallback;
    return RenderConfigToFiles(config, initParams);
}

RenderOutcome RenderConfigToFiles(const Config& config,
                                  const OfflineRenderer::InitParams& initParams) {
    RenderOutcome outcome;
    const auto started = std::chrono::steady_clock::now();

    auto rendererResult = OfflineRenderer::Create(config, initParams);
    if (!rendererResult.has_value()) {
        outcome.error = rendererResult.error();
        return outcome;
    }
    OfflineRenderer& renderer = *rendererResult.value();

    // Read rather than re-parsed. Two readers of one TOML key is the bug class
    // this project keeps closing.
    outcome.width = renderer.Params().width;
    outcome.height = renderer.Params().height;
    outcome.spp = renderer.Params().spp;
    const SpectralMode spectralMode = renderer.Params().mode;
    outcome.modeName = renderer.Params().modeName;
    outcome.wavelengthNm = renderer.Params().wavelengthNm;
    outcome.exrPath = renderer.Params().outputPath;

    OfflineRenderOutput rendered = renderer.Render();
    if (!rendered.error.empty()) {
        QL_LOG_ERROR("{}", rendered.error);
        outcome.error = rendered.error;
    }

    outcome.wroteItsOwnOutput = rendered.wroteItsOwnOutput;

    // The hyperspectral cube streams itself to disk band by band; there is no
    // frame to save. Everything below is the single-frame output stage.
    if (!rendered.wroteItsOwnOutput) {
        Image& img = rendered.radiance;

        // Before the sensor chain, which overwrites img with its own noisy
        // preview: the temperature map is of the scene, not of the detector.
        if (PostprocessConfig::IsThermographyEnabled(config)) {
            WriteApparentTemperature(config, rendered, img, spectralMode, outcome.exrPath,
                                     outcome);
        }

        if (PostprocessConfig::IsSensorEnabled(config)) {
            ApplySensorChain(config, rendered, img, outcome.exrPath, outcome.width, outcome.height);
        } else {
            QL_LOG_INFO("Sensor simulation disabled (sensor.enabled = false)");
        }

        if (ImageIO::WriteEXR(outcome.exrPath, img)) {
            QL_LOG_INFO("  [OK] Saved spectral image to {}", outcome.exrPath);
        } else {
            QL_LOG_ERROR("  [FAIL] Failed to save image to {}", outcome.exrPath);
            outcome.error = "failed to write " + outcome.exrPath;
        }

        // For fused modes (RGB, VIS_FUSED, MWIR, LWIR, SWIR), also save a PNG
        // preview: these modes output both EXR (HDR/physical) and PNG (LDR).
        const bool isFusedMode =
            (spectralMode == SpectralMode::RGB || spectralMode == SpectralMode::VIS_Fused ||
             spectralMode == SpectralMode::MWIR_Fused || spectralMode == SpectralMode::LWIR_Fused ||
             spectralMode == SpectralMode::SWIR_Fused);

        if (isFusedMode) {
            const std::filesystem::path exrPath(outcome.exrPath);
            const std::filesystem::path pngPath =
                exrPath.parent_path() / (exrPath.stem().string() + ".png");

            outcome.preview = BuildPreview(img, spectralMode, outcome.width, outcome.height);

            if (ImageIO::WritePNG(pngPath.string(), outcome.preview)) {
                QL_LOG_INFO("  [OK] Saved PNG preview to {}", pngPath.string());
                outcome.pngPath = pngPath.string();
            } else {
                QL_LOG_WARN("  [WARN] Failed to save PNG preview to {}", pngPath.string());
            }
        }
    }

    outcome.seconds = std::chrono::duration<f64>(std::chrono::steady_clock::now() - started).count();
    outcome.ok = outcome.error.empty();

    // The renderer is torn down when `rendererResult` goes out of scope here --
    // and with it the device and the pipeline cache, unless a shared RenderDevice
    // was passed in, in which case both outlive this call.
    return outcome;
}

}  // namespace quantiloom::app
