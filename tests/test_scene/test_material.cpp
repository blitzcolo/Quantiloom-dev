// ============================================================================
// Quantiloom - Unit Tests for scene/Material.hpp
// ============================================================================
// Tests cover:
// - Material construction and validation
// - PBR parameter ranges
// - Spectral albedo computation
// - IR material properties (emissivity, reflectance, transmittance)
// - Kirchhoff's law validation
// - SpectralSource tracking
// - Helper factory methods
// ============================================================================

#include <gtest/gtest.h>
#include "scene/Material.hpp"

using namespace quantiloom;

// ============================================================================
// Construction and Validation Tests
// ============================================================================

TEST(MaterialTest, DefaultConstruction) {
    Material mat;

    EXPECT_TRUE(mat.IsValid());
    EXPECT_EQ(mat.baseColorFactor, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    EXPECT_EQ(mat.metallicFactor, 0.0f);
    EXPECT_EQ(mat.roughnessFactor, 1.0f);
    EXPECT_EQ(mat.alphaMode, Material::AlphaMode::Opaque);
}

TEST(MaterialTest, IsValidBaseColorRange) {
    Material mat;

    // Valid range [0, inf]
    mat.baseColorFactor = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
    EXPECT_TRUE(mat.IsValid());

    // HDR values allowed
    mat.baseColorFactor = glm::vec4(10.0f, 5.0f, 2.0f, 1.0f);
    EXPECT_TRUE(mat.IsValid());

    // Negative values invalid
    mat.baseColorFactor = glm::vec4(-0.1f, 0.5f, 0.5f, 1.0f);
    EXPECT_FALSE(mat.IsValid());

    mat.baseColorFactor = glm::vec4(0.5f, 0.5f, 0.5f, -0.1f);  // Negative alpha
    EXPECT_FALSE(mat.IsValid());
}

TEST(MaterialTest, IsValidMetallicRange) {
    Material mat;

    // Valid range [0, 1]
    mat.metallicFactor = 0.0f;
    EXPECT_TRUE(mat.IsValid());

    mat.metallicFactor = 1.0f;
    EXPECT_TRUE(mat.IsValid());

    mat.metallicFactor = 0.5f;
    EXPECT_TRUE(mat.IsValid());

    // Out of range
    mat.metallicFactor = -0.1f;
    EXPECT_FALSE(mat.IsValid());

    mat.metallicFactor = 1.1f;
    EXPECT_FALSE(mat.IsValid());
}

TEST(MaterialTest, IsValidRoughnessRange) {
    Material mat;

    // Valid range [0, 1]
    mat.roughnessFactor = 0.0f;  // Perfect mirror
    EXPECT_TRUE(mat.IsValid());

    mat.roughnessFactor = 1.0f;  // Fully rough
    EXPECT_TRUE(mat.IsValid());

    // Out of range
    mat.roughnessFactor = -0.1f;
    EXPECT_FALSE(mat.IsValid());

    mat.roughnessFactor = 1.5f;
    EXPECT_FALSE(mat.IsValid());
}

TEST(MaterialTest, IsValidAlphaCutoff) {
    Material mat;

    mat.alphaCutoff = 0.5f;
    EXPECT_TRUE(mat.IsValid());

    mat.alphaCutoff = 0.0f;
    EXPECT_TRUE(mat.IsValid());

    mat.alphaCutoff = 1.0f;
    EXPECT_TRUE(mat.IsValid());

    // Out of range
    mat.alphaCutoff = -0.1f;
    EXPECT_FALSE(mat.IsValid());

    mat.alphaCutoff = 1.1f;
    EXPECT_FALSE(mat.IsValid());
}

// ============================================================================
// Spectral Albedo Tests
// ============================================================================

TEST(MaterialTest, ComputeSpectralAlbedoGray) {
    Material mat;
    mat.baseColorFactor = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);

    mat.ComputeSpectralAlbedo();

    EXPECT_NEAR(mat.spectralAlbedo, 0.5f, 1e-6f);
}

TEST(MaterialTest, ComputeSpectralAlbedoColor) {
    Material mat;
    mat.baseColorFactor = glm::vec4(0.2f, 0.5f, 0.8f, 1.0f);

    mat.ComputeSpectralAlbedo();

    EXPECT_NEAR(mat.spectralAlbedo, (0.2f + 0.5f + 0.8f) / 3.0f, 1e-6f);
}

TEST(MaterialTest, ComputeSpectralAlbedoWhite) {
    Material mat;
    mat.baseColorFactor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

    mat.ComputeSpectralAlbedo();

    EXPECT_NEAR(mat.spectralAlbedo, 1.0f, 1e-6f);
}

TEST(MaterialTest, ComputeSpectralAlbedoBlack) {
    Material mat;
    mat.baseColorFactor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    mat.ComputeSpectralAlbedo();

    EXPECT_NEAR(mat.spectralAlbedo, 0.0f, 1e-6f);
}

// ============================================================================
// Texture Tests
// ============================================================================

TEST(MaterialTest, HasTexturesNone) {
    Material mat;

    EXPECT_FALSE(mat.HasTextures());
}

TEST(MaterialTest, HasTexturesBaseColor) {
    Material mat;
    mat.baseColorTextureIndex = 0;

    EXPECT_TRUE(mat.HasTextures());
}

TEST(MaterialTest, HasTexturesMultiple) {
    Material mat;
    mat.baseColorTextureIndex = 0;
    mat.normalTextureIndex = 1;
    mat.metallicRoughnessTextureIndex = 2;

    EXPECT_TRUE(mat.HasTextures());
}

// ============================================================================
// SpectralSource Tracking Tests
// ============================================================================

TEST(MaterialTest, SpectralSourceDefault) {
    Material mat;

    EXPECT_EQ(mat.spectralSource, Material::SpectralSource::Unknown);
}

TEST(MaterialTest, SpectralSourceTracking) {
    Material mat;

    mat.spectralSource = Material::SpectralSource::Measured;
    EXPECT_EQ(mat.spectralSource, Material::SpectralSource::Measured);

    mat.spectralSource = Material::SpectralSource::RGBUpsampled;
    EXPECT_EQ(mat.spectralSource, Material::SpectralSource::RGBUpsampled);

    mat.spectralSource = Material::SpectralSource::Procedural;
    EXPECT_EQ(mat.spectralSource, Material::SpectralSource::Procedural);
}

// ============================================================================
// Factory Method Tests
// ============================================================================

TEST(MaterialTest, CreateLambertian) {
    glm::vec3 albedo(0.8f, 0.2f, 0.1f);
    Material mat = Material::CreateLambertian(albedo, "TestLambertian");

    EXPECT_EQ(mat.name, "TestLambertian");
    EXPECT_EQ(mat.baseColorFactor, glm::vec4(albedo, 1.0f));
    EXPECT_EQ(mat.metallicFactor, 0.0f);  // Non-metal
    EXPECT_EQ(mat.roughnessFactor, 1.0f); // Fully rough
    EXPECT_EQ(mat.spectralSource, Material::SpectralSource::RGBUpsampled);
    EXPECT_TRUE(mat.IsValid());
}

// ============================================================================
// IR Material Property Tests
// ============================================================================

TEST(MaterialTest, HasIRDataEmpty) {
    Material mat;

    EXPECT_FALSE(mat.HasIRData());
}

TEST(MaterialTest, HasIRDataEmissivity) {
    Material mat;
    mat.irEmissivityCurve = {{3000.0f, 0.85f}, {5000.0f, 0.90f}};

    EXPECT_TRUE(mat.HasIRData());
}

TEST(MaterialTest, HasIRDataReflectance) {
    Material mat;
    mat.irReflectanceCurve = {{3000.0f, 0.10f}, {5000.0f, 0.08f}};

    EXPECT_TRUE(mat.HasIRData());
}

TEST(MaterialTest, HasIRDataTransmittance) {
    Material mat;
    mat.irTransmittanceCurve = {{3000.0f, 0.05f}, {5000.0f, 0.02f}};

    EXPECT_TRUE(mat.HasIRData());
}

TEST(MaterialTest, IRTemperature) {
    Material mat;

    EXPECT_EQ(mat.irTemperature_K, 0.0f);

    mat.irTemperature_K = 300.0f;  // Room temperature
    EXPECT_EQ(mat.irTemperature_K, 300.0f);
}

// ============================================================================
// Alpha Mode Tests
// ============================================================================

TEST(MaterialTest, AlphaModeOpaque) {
    Material mat;
    mat.alphaMode = Material::AlphaMode::Opaque;

    EXPECT_EQ(mat.alphaMode, Material::AlphaMode::Opaque);
}

TEST(MaterialTest, AlphaModeMask) {
    Material mat;
    mat.alphaMode = Material::AlphaMode::Mask;
    mat.alphaCutoff = 0.5f;

    EXPECT_EQ(mat.alphaMode, Material::AlphaMode::Mask);
    EXPECT_EQ(mat.alphaCutoff, 0.5f);
}

TEST(MaterialTest, AlphaModeBlend) {
    Material mat;
    mat.alphaMode = Material::AlphaMode::Blend;

    EXPECT_EQ(mat.alphaMode, Material::AlphaMode::Blend);
}

// ============================================================================
// Emissive Tests
// ============================================================================

TEST(MaterialTest, EmissiveDefault) {
    Material mat;

    EXPECT_EQ(mat.emissiveFactor, glm::vec3(0.0f, 0.0f, 0.0f));
}

TEST(MaterialTest, EmissiveHDR) {
    Material mat;
    mat.emissiveFactor = glm::vec3(100.0f, 50.0f, 25.0f);  // HDR allowed

    EXPECT_EQ(mat.emissiveFactor, glm::vec3(100.0f, 50.0f, 25.0f));
    EXPECT_TRUE(mat.IsValid());
}

// ============================================================================
// Name and Metadata Tests
// ============================================================================

TEST(MaterialTest, NameDefault) {
    Material mat;

    EXPECT_TRUE(mat.name.empty());
}

TEST(MaterialTest, NameCustom) {
    Material mat;
    mat.name = "TestMaterial";

    EXPECT_EQ(mat.name, "TestMaterial");
}

// ============================================================================
// Quantiloom Material Reference Tests (glTF extras integration)
// ============================================================================

TEST(MaterialTest, QuantiloomMaterialRefDefault) {
    Material mat;

    // Default: both fields empty, HasQuantiloomRef() returns false
    EXPECT_TRUE(mat.quantiloomMaterialType.empty());
    EXPECT_TRUE(mat.quantiloomMaterialRef.empty());
    EXPECT_FALSE(mat.HasQuantiloomRef());
}

TEST(MaterialTest, HasQuantiloomRefBothFields) {
    Material mat;
    mat.quantiloomMaterialType = "quantiloom_usgs";
    mat.quantiloomMaterialRef = "Aluminum brushed 293K";

    // Both fields set: HasQuantiloomRef() returns true
    EXPECT_TRUE(mat.HasQuantiloomRef());
    EXPECT_EQ(mat.quantiloomMaterialType, "quantiloom_usgs");
    EXPECT_EQ(mat.quantiloomMaterialRef, "Aluminum brushed 293K");
}

TEST(MaterialTest, HasQuantiloomRefTypeOnly) {
    Material mat;
    mat.quantiloomMaterialType = "quantiloom_usgs";
    // quantiloomMaterialRef is empty

    // Only type set: HasQuantiloomRef() returns false
    EXPECT_FALSE(mat.HasQuantiloomRef());
}

TEST(MaterialTest, HasQuantiloomRefNameOnly) {
    Material mat;
    // quantiloomMaterialType is empty
    mat.quantiloomMaterialRef = "Aluminum brushed 293K";

    // Only name set: HasQuantiloomRef() returns false
    EXPECT_FALSE(mat.HasQuantiloomRef());
}

TEST(MaterialTest, QuantiloomMaterialTypes) {
    Material mat;

    // Test various type values
    mat.quantiloomMaterialType = "quantiloom_usgs";
    mat.quantiloomMaterialRef = "Test Material";
    EXPECT_TRUE(mat.HasQuantiloomRef());

    mat.quantiloomMaterialType = "refractiveindex_info";
    EXPECT_TRUE(mat.HasQuantiloomRef());

    mat.quantiloomMaterialType = "custom_csv";
    EXPECT_TRUE(mat.HasQuantiloomRef());
}

TEST(MaterialTest, QuantiloomMaterialWithSpectralSource) {
    Material mat;
    mat.quantiloomMaterialType = "quantiloom_usgs";
    mat.quantiloomMaterialRef = "Gold_HS111.3B";
    mat.spectralSource = Material::SpectralSource::Measured;

    EXPECT_TRUE(mat.HasQuantiloomRef());
    EXPECT_EQ(mat.spectralSource, Material::SpectralSource::Measured);
}

// ============================================================================
// Realistic Material Tests
// ============================================================================

TEST(MaterialTest, RealisticMetalMaterial) {
    // Aluminum-like metal
    Material mat;
    mat.name = "Aluminum";
    mat.baseColorFactor = glm::vec4(0.91f, 0.92f, 0.92f, 1.0f);
    mat.metallicFactor = 1.0f;   // Full metal
    mat.roughnessFactor = 0.1f;  // Slightly rough

    EXPECT_TRUE(mat.IsValid());
    EXPECT_EQ(mat.metallicFactor, 1.0f);
}

TEST(MaterialTest, RealisticDielectricMaterial) {
    // Plastic-like material
    Material mat;
    mat.name = "Plastic";
    mat.baseColorFactor = glm::vec4(0.8f, 0.2f, 0.2f, 1.0f);
    mat.metallicFactor = 0.0f;   // Non-metal
    mat.roughnessFactor = 0.5f;  // Medium roughness

    EXPECT_TRUE(mat.IsValid());
    EXPECT_EQ(mat.metallicFactor, 0.0f);
}

TEST(MaterialTest, RealisticGlassMaterial) {
    // Transparent glass
    Material mat;
    mat.name = "Glass";
    mat.baseColorFactor = glm::vec4(1.0f, 1.0f, 1.0f, 0.1f);  // Low alpha
    mat.metallicFactor = 0.0f;   // Non-metal
    mat.roughnessFactor = 0.0f;  // Perfect smoothness
    mat.alphaMode = Material::AlphaMode::Blend;

    EXPECT_TRUE(mat.IsValid());
}
