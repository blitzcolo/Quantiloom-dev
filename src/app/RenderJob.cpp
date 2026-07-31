#include "RenderJob.hpp"

#include "core/Log.hpp"
#include "io/ImageIO.hpp"
#include "postprocess/GenericSensor.hpp"
#include "postprocess/PostprocessConfig.hpp"
#include "renderer/OfflineRenderer.hpp"

#include <algorithm>  // For std::nth_element
#include <chrono>
#include <filesystem>
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
    RenderOutcome outcome;
    const auto started = std::chrono::steady_clock::now();

    OfflineRenderer::InitParams initParams;
    initParams.atmosphereModelPackFallback = atmosphereModelPackFallback;

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

    // The pipeline cache is written back and the device torn down when
    // `rendererResult` goes out of scope here.
    return outcome;
}

}  // namespace quantiloom::app
