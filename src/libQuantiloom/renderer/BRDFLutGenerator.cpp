#include "BRDFLutGenerator.hpp"
#include "core/Log.hpp"
#include "io/ImageIO.hpp"

#include <glm/gtc/constants.hpp>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <filesystem>

namespace quantiloom {

// ============================================================================
// Binary Cache Header Structure
// ============================================================================

struct BRDFLutCacheHeader {
    u32 magic;          // CACHE_MAGIC for validation
    u32 version;        // CACHE_VERSION for compatibility
    u32 resolution;     // LUT texture size (NxN)
    u32 sampleCount;    // Monte Carlo samples used to generate
};

// ============================================================================
// Public API
// ============================================================================

Image BRDFLutGenerator::Generate(const Config& config) {
    QL_LOG_INFO("Generating BRDF LUT ({}x{}, {} samples/pixel)...",
                config.resolution, config.resolution, config.sampleCount);

    // Create image (2 channels: R=scale, G=bias)
    Image lut(config.resolution, config.resolution, 2);
    lut.channelNames = {"scale", "bias"};
    lut.metadata["generator"] = "Quantiloom_BRDFLutGenerator";
    lut.metadata["resolution"] = std::to_string(config.resolution);
    lut.metadata["samples"] = std::to_string(config.sampleCount);
    lut.metadata["method"] = "ImportanceSampling_GGX";

    // Generate LUT using Monte Carlo integration
    for (u32 y = 0; y < config.resolution; ++y) {
        // Progress logging
        if (y % (config.resolution / 10) == 0) {
            QL_LOG_INFO("  Progress: {}%", (y * 100) / config.resolution);
        }

        for (u32 x = 0; x < config.resolution; ++x) {
            // Map pixel to [0, 1]
            // x-axis: cos(theta_v) = NdotV (view angle)
            // y-axis: roughness
            const f32 NdotV = (static_cast<float>(x) + 0.5f) / static_cast<float>(config.resolution);
            const f32 roughness = (static_cast<float>(y) + 0.5f) / static_cast<float>(config.resolution);

            // Integrate BRDF
            const glm::vec2 brdf = IntegrateBRDF(NdotV, roughness, config.sampleCount);

            // Store result (R=scale, G=bias)
            lut(x, y, 0) = brdf.x;  // Scale
            lut(x, y, 1) = brdf.y;  // Bias
        }
    }

    QL_LOG_INFO("  BRDF LUT generation complete");
    return lut;
}

Image BRDFLutGenerator::Generate() {
    return Generate(Config{});
}

bool BRDFLutGenerator::GenerateAndSave(const String& filepath, const Config& config) {
    Image lut = Generate(config);

    if (ImageIO::WriteEXR(filepath, lut)) {
        QL_LOG_INFO("  Saved BRDF LUT to {}", filepath);
        return true;
    } else {
        QL_LOG_ERROR("  Failed to save BRDF LUT to {}", filepath);
        return false;
    }
}

bool BRDFLutGenerator::GenerateAndSave(const String& filepath) {
    return GenerateAndSave(filepath, Config{});
}

// ============================================================================
// Binary Cache I/O Implementation
// ============================================================================

bool BRDFLutGenerator::SaveToBinary(const String& filepath, const Image& lut, const Config& config) {
    // Validate image
    if (lut.width != config.resolution || lut.height != config.resolution || lut.channels != 2) {
        QL_LOG_ERROR("SaveToBinary: Image dimensions don't match config ({}x{}x{} vs {}x{}x2)",
                     lut.width, lut.height, lut.channels, config.resolution, config.resolution);
        return false;
    }

    // Create parent directories if they don't exist
    std::filesystem::path filePath(filepath);
    if (filePath.has_parent_path()) {
        std::filesystem::create_directories(filePath.parent_path());
    }

    // Open file for binary writing
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        QL_LOG_ERROR("SaveToBinary: Failed to open file for writing: {}", filepath);
        return false;
    }

    // Write header
    BRDFLutCacheHeader header{};
    header.magic = CACHE_MAGIC;
    header.version = CACHE_VERSION;
    header.resolution = config.resolution;
    header.sampleCount = config.sampleCount;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write raw float data
    const size_t dataSize = lut.data.size() * sizeof(f32);
    file.write(reinterpret_cast<const char*>(lut.data.data()), static_cast<std::streamsize>(dataSize));

    if (!file.good()) {
        QL_LOG_ERROR("SaveToBinary: Write error occurred");
        return false;
    }

    file.close();
    QL_LOG_INFO("SaveToBinary: Saved BRDF LUT cache to {} ({} bytes)", filepath, sizeof(header) + dataSize);
    return true;
}

std::optional<Image> BRDFLutGenerator::LoadFromBinary(const String& filepath, const Config* expectedConfig) {
    // Check if file exists
    if (!std::filesystem::exists(filepath)) {
        QL_LOG_INFO("LoadFromBinary: Cache file not found: {}", filepath);
        return std::nullopt;
    }

    // Open file for binary reading
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        QL_LOG_WARN("LoadFromBinary: Failed to open cache file: {}", filepath);
        return std::nullopt;
    }

    // Read header
    BRDFLutCacheHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (!file.good()) {
        QL_LOG_WARN("LoadFromBinary: Failed to read header from {}", filepath);
        return std::nullopt;
    }

    // Validate magic number
    if (header.magic != CACHE_MAGIC) {
        QL_LOG_WARN("LoadFromBinary: Invalid magic number in {} (expected 0x{:08X}, got 0x{:08X})",
                    filepath, CACHE_MAGIC, header.magic);
        return std::nullopt;
    }

    // Validate version
    if (header.version != CACHE_VERSION) {
        QL_LOG_WARN("LoadFromBinary: Version mismatch in {} (expected {}, got {})",
                    filepath, CACHE_VERSION, header.version);
        return std::nullopt;
    }

    // Validate against expected config if provided
    if (expectedConfig != nullptr) {
        if (header.resolution != expectedConfig->resolution) {
            QL_LOG_WARN("LoadFromBinary: Resolution mismatch (cached: {}, expected: {})",
                        header.resolution, expectedConfig->resolution);
            return std::nullopt;
        }
        if (header.sampleCount != expectedConfig->sampleCount) {
            QL_LOG_WARN("LoadFromBinary: Sample count mismatch (cached: {}, expected: {})",
                        header.sampleCount, expectedConfig->sampleCount);
            return std::nullopt;
        }
    }

    // Create image
    Image lut(header.resolution, header.resolution, 2);
    lut.channelNames = {"scale", "bias"};
    lut.metadata["generator"] = "Quantiloom_BRDFLutGenerator";
    lut.metadata["resolution"] = std::to_string(header.resolution);
    lut.metadata["samples"] = std::to_string(header.sampleCount);
    lut.metadata["source"] = "cache";

    // Read raw float data
    const size_t dataSize = lut.data.size() * sizeof(f32);
    file.read(reinterpret_cast<char*>(lut.data.data()), static_cast<std::streamsize>(dataSize));

    if (!file.good()) {
        QL_LOG_WARN("LoadFromBinary: Failed to read data from {}", filepath);
        return std::nullopt;
    }

    file.close();
    QL_LOG_INFO("LoadFromBinary: Loaded BRDF LUT from cache {} ({}x{}, {} samples)",
                filepath, header.resolution, header.resolution, header.sampleCount);
    return lut;
}

bool BRDFLutGenerator::IsCacheValid(const String& filepath, const Config& expectedConfig) {
    // Check if file exists
    if (!std::filesystem::exists(filepath)) {
        return false;
    }

    // Open file for binary reading
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read header only
    BRDFLutCacheHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    file.close();

    // Validate
    if (header.magic != CACHE_MAGIC) return false;
    if (header.version != CACHE_VERSION) return false;
    if (header.resolution != expectedConfig.resolution) return false;
    if (header.sampleCount != expectedConfig.sampleCount) return false;

    return true;
}

// ============================================================================
// Internal Implementation
// ============================================================================

glm::vec2 BRDFLutGenerator::IntegrateBRDF(const f32 NdotV, const f32 roughness, const u32 sampleCount) {
    // View direction (pointing from surface to camera)
    // Fix normal to +Z, vary view direction based on NdotV
    constexpr auto N = glm::vec3(0.0f, 0.0f, 1.0f);

    // Compute V from NdotV
    // cos(theta) = NdotV, sin(theta) = sqrt(1 - NdotV^2)
    const f32 sinThetaV = std::sqrt(std::max(0.0f, 1.0f - NdotV * NdotV));
    const auto V = glm::vec3(sinThetaV, 0.0f, NdotV);

    // Accumulate integrated terms
    f32 A = 0.0f;  // Scale term
    f32 B = 0.0f;  // Bias term

    for (u32 i = 0; i < sampleCount; ++i) {
        // Generate low-discrepancy 2D sample
        const glm::vec2 Xi = Hammersley(i, sampleCount);

        // Importance sample GGX (generates H in tangent space)
        glm::vec3 H = ImportanceSampleGGX(Xi, N, roughness);

        // Compute L (light direction) from H (halfway vector)
        // L = reflect(-V, H) = 2 * (V·H) * H - V
        glm::vec3 L = glm::normalize(2.0f * glm::dot(V, H) * H - V);

        // Only consider samples where light is in hemisphere
        if (const f32 NdotL = std::max(L.z, 0.0f); NdotL > 0.0f) {
            const f32 NdotH = std::max(H.z, 0.0f);
            const f32 VdotH = std::max(glm::dot(V, H), 0.0f);

            // Geometry term (Smith GGX with IBL correlation)
            const f32 G = GeometrySmith_GGX_IBL(NdotV, NdotL, roughness);

            // Fresnel term (Schlick approximation split into (1-VdotH)^5)
            // F = F0 + (1 - F0) * (1 - VdotH)^5
            // We separate this into: F = F0 * A + B
            const f32 G_Vis = (G * VdotH) / (NdotH * NdotV);
            const f32 Fc = std::pow(1.0f - VdotH, 5.0f);

            A += (1.0f - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }

    // Average over samples
    A /= static_cast<f32>(sampleCount);
    B /= static_cast<f32>(sampleCount);

    return {A, B};
}

glm::vec3 BRDFLutGenerator::ImportanceSampleGGX(const glm::vec2 Xi, glm::vec3 N, const f32 roughness) {
    (void)N;  // Unused: N is fixed to (0, 0, 1), tangent frame is identity
    const f32 a = roughness * roughness;

    // Spherical coordinates (GGX distribution)
    const f32 phi = 2.0f * glm::pi<f32>() * Xi.x;
    const f32 cosTheta = std::sqrt((1.0f - Xi.y) / (1.0f + (a * a - 1.0f) * Xi.y));
    const f32 sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);

    // Tangent space vector
    glm::vec3 H;
    H.x = std::cos(phi) * sinTheta;
    H.y = std::sin(phi) * sinTheta;
    H.z = cosTheta;

    // Transform from tangent space to world space
    // Since N = (0, 0, 1), tangent frame is identity, so return H directly
    return glm::normalize(H);
}

f32 BRDFLutGenerator::GeometrySmith_GGX_IBL(const f32 NdotV, const f32 NdotL, const f32 roughness) {
    // GGX geometry function (Smith) for IBL
    // Uses remapped roughness for better correlation with microfacet model
    const f32 a = roughness;
    f32 k = (a * a) / 2.0f;  // IBL variant (different from direct lighting)

    auto G1 = [k](const f32 NdotX) -> f32 {
        return NdotX / (NdotX * (1.0f - k) + k);
    };

    return G1(NdotV) * G1(NdotL);
}

glm::vec2 BRDFLutGenerator::Hammersley(const u32 i, const u32 N) {
    return {
        static_cast<f32>(i) / static_cast<f32>(N),
        RadicalInverse_VdC(i)
    };
}

f32 BRDFLutGenerator::RadicalInverse_VdC(u32 bits) {
    // Van der Corput sequence (base 2)
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<f32>(bits) * 2.3283064365386963e-10f; // / 0x100000000
}

} // namespace quantiloom
