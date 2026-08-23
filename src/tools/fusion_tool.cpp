// ============================================================================
// Quantiloom - Multiband Fusion Tool
// ============================================================================
// Fuses VIS/SWIR/MWIR bands into a single enhanced image
// Usage:
//   fusion_tool <config.toml> <vis.exr> <swir.exr> <mwir.exr> <output.exr>
// ============================================================================

#include "core/Log.hpp"
#include "core/Config.hpp"
#include "io/ImageIO.hpp"
#include "postprocess/MultibandFusion.hpp"
#include "postprocess/PostprocessConfig.hpp"

#include <iostream>
#include <filesystem>

using namespace quantiloom;

int main(int argc, char** argv) {
    // ========================================================================
    // Parse Command Line Arguments
    // ========================================================================
    if (argc < 6) {
        std::cerr << "Usage: fusion_tool <config.toml> <vis.exr> <swir.exr> <mwir.exr> <output.exr>\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  fusion_tool config.toml vis_output.exr swir_output.exr mwir_output.exr fused.exr\n";
        std::cerr << "\nThe config.toml should contain [fusion] section with fusion parameters.\n";
        return 1;
    }

    const std::string configPath = argv[1];
    const std::string visPath = argv[2];
    const std::string swirPath = argv[3];
    const std::string mwirPath = argv[4];
    const std::string outputPath = argv[5];

    // ========================================================================
    // Initialize Logging
    // ========================================================================
    Log::Init();
    QL_LOG_INFO("========================================");
    QL_LOG_INFO("  Quantiloom - Multiband Fusion Tool");
    QL_LOG_INFO("========================================");

    try {
        // ====================================================================
        // Load Configuration
        // ====================================================================
        QL_LOG_INFO("Loading configuration from {}...", configPath);
        auto configResult = Config::Load(configPath);
        if (!configResult.has_value()) {
            QL_LOG_ERROR("Failed to load config: {}", configResult.error());
            return 1;
        }
        const Config& config = configResult.value();

        // Parse fusion parameters
        FusionParams fusionParams = PostprocessConfig::ParseFusionParams(config);

        QL_LOG_INFO("Fusion method: {}",
                    fusionParams.method == FusionMethod::LaplacianPyramid ? "Laplacian Pyramid" :
                    fusionParams.method == FusionMethod::WeightedAverage ? "Weighted Average" :
                    fusionParams.method == FusionMethod::MaxResponse ? "Max Response" : "Pseudo Color");

        // ====================================================================
        // Load Input Images
        // ====================================================================
        QL_LOG_INFO("Loading input images...");

        QL_LOG_INFO("  Loading VIS: {}", visPath);
        auto visResult = ImageIO::ReadEXR(visPath);
        if (!visResult.has_value()) {
            QL_LOG_ERROR("  Failed to load VIS image");
            return 1;
        }
        const Image& visImg = visResult.value();

        QL_LOG_INFO("  Loading SWIR: {}", swirPath);
        auto swirResult = ImageIO::ReadEXR(swirPath);
        if (!swirResult.has_value()) {
            QL_LOG_ERROR("  Failed to load SWIR image");
            return 1;
        }
        const Image& swirImg = swirResult.value();

        QL_LOG_INFO("  Loading MWIR: {}", mwirPath);
        auto mwirResult = ImageIO::ReadEXR(mwirPath);
        if (!mwirResult.has_value()) {
            QL_LOG_ERROR("  Failed to load MWIR image");
            return 1;
        }
        const Image& mwirImg = mwirResult.value();

        QL_LOG_INFO("  All images loaded successfully");
        QL_LOG_INFO("  VIS: {}x{} ({} channels)", visImg.width, visImg.height, visImg.channels);
        QL_LOG_INFO("  SWIR: {}x{} ({} channels)", swirImg.width, swirImg.height, swirImg.channels);
        QL_LOG_INFO("  MWIR: {}x{} ({} channels)", mwirImg.width, mwirImg.height, mwirImg.channels);

        // Extract the radiance channel BY NAME. A render output is RGBA, and an
        // EXR comes back off disk in the name-sorted order OpenEXR keeps its
        // channel list in -- so index 0 is the alpha, a constant 1.0. Taken
        // positionally, all three bands here were the same flat image and the
        // fusion had nothing to fuse. See Image::ChannelIndex.
        const u32 visC = visImg.LuminanceChannelIndex();
        const u32 swirC = swirImg.LuminanceChannelIndex();
        const u32 mwirC = mwirImg.LuminanceChannelIndex();
        QL_LOG_INFO("  Radiance channels: VIS '{}', SWIR '{}', MWIR '{}'",
                    visImg.channelNames[visC], swirImg.channelNames[swirC],
                    mwirImg.channelNames[mwirC]);

        Image visGray(visImg.width, visImg.height, 1);
        Image swirGray(swirImg.width, swirImg.height, 1);
        Image mwirGray(mwirImg.width, mwirImg.height, 1);

        for (u32 y = 0; y < visImg.height; ++y) {
            for (u32 x = 0; x < visImg.width; ++x) {
                visGray(x, y, 0) = visImg(x, y, visC);
                swirGray(x, y, 0) = swirImg(x, y, swirC);
                mwirGray(x, y, 0) = mwirImg(x, y, mwirC);
            }
        }

        // ====================================================================
        // Perform Fusion
        // ====================================================================
        QL_LOG_INFO("Fusing VIS/SWIR/MWIR bands...");

        auto fusionResult = MultibandFusion::Fuse(visGray, swirGray, mwirGray, fusionParams);
        if (!fusionResult.has_value()) {
            QL_LOG_ERROR("Fusion failed: {}", fusionResult.error());
            return 1;
        }

        const Image& fusedImg = fusionResult.value();
        QL_LOG_INFO("  Fusion complete: {}x{} ({} channels)",
                    fusedImg.width, fusedImg.height, fusedImg.channels);

        // ====================================================================
        // Save Output
        // ====================================================================
        QL_LOG_INFO("Saving fused image to {}...", outputPath);

        if (!ImageIO::WriteEXR(outputPath, fusedImg)) {
            QL_LOG_ERROR("Failed to save fused image to {}", outputPath);
            return 1;
        }

        QL_LOG_INFO("  [OK] Saved fused image");

        // Also save PNG preview if requested
        std::filesystem::path exrPath(outputPath);
        std::filesystem::path pngPath = exrPath.parent_path() / (exrPath.stem().string() + ".png");

        QL_LOG_INFO("Saving PNG preview to {}...", pngPath.string());
        if (!ImageIO::WritePNG(pngPath.string(), fusedImg)) {
            QL_LOG_WARN("  [WARN] Failed to save PNG preview");
        } else {
            QL_LOG_INFO("  [OK] Saved PNG preview");
        }

        // ====================================================================
        // Success
        // ====================================================================
        QL_LOG_INFO("========================================");
        QL_LOG_INFO("  Fusion COMPLETED");
        QL_LOG_INFO("========================================");
        QL_LOG_INFO("  Input: VIS={}, SWIR={}, MWIR={}", visPath, swirPath, mwirPath);
        QL_LOG_INFO("  Output: {}", outputPath);
        QL_LOG_INFO("========================================");

    } catch (const std::exception& e) {
        QL_LOG_ERROR("FATAL ERROR: {}", e.what());
        Log::Shutdown();
        return 1;
    }

    Log::Shutdown();
    return 0;
}
