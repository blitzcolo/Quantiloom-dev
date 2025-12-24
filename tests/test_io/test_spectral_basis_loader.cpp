// ============================================================================
// Quantiloom - Unit Tests for io/SpectralBasisLoader.hpp
// ============================================================================
// Tests cover:
// - Binary basis file loading (format v2)
// - JSON materials database loading
// - Spectral curve reconstruction from NMF weights
// - GPU format conversion
// - Material lookup (exact and partial match)
// - Error handling for missing/corrupt files
// ============================================================================

#include <gtest/gtest.h>
#include "io/SpectralBasisLoader.hpp"
#include "core/SpectralData.hpp"
#include <filesystem>
#include <fstream>
#include <cstring>

using namespace quantiloom;

// ============================================================================
// Test Fixture with Temporary File Management
// ============================================================================

class SpectralBasisLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        tempDir = std::filesystem::temp_directory_path() / "quantiloom_basis_tests";
        std::filesystem::create_directories(tempDir);
    }

    void TearDown() override {
        if (std::filesystem::exists(tempDir)) {
            std::filesystem::remove_all(tempDir);
        }
    }

    std::filesystem::path GetTempFilePath(const std::string& filename) {
        return tempDir / filename;
    }

    // Helper: Create a valid binary basis file (format v3)
    void CreateTestBasisFile(const std::filesystem::path& path,
                             u32 numBands = 5,
                             u32 numBasis = 4,
                             u32 numSamples = 10) {
        std::ofstream file(path, std::ios::binary);

        // Header (64 bytes)
        const char magic[4] = {'Q', 'B', 'A', 'S'};
        u32 version = 3;
        file.write(magic, 4);
        file.write(reinterpret_cast<const char*>(&version), sizeof(u32));
        file.write(reinterpret_cast<const char*>(&numBands), sizeof(u32));

        // Reserved bytes (52 bytes padding to reach 64)
        char reserved[52] = {0};
        file.write(reserved, 52);

        // Band data
        struct {
            const char* name;
            f32 startUm, endUm;
        } bands[5] = {
            {"VIS",  0.350f, 0.780f},
            {"NIR",  0.780f, 1.100f},
            {"SWIR", 1.100f, 2.500f},
            {"MWIR", 2.500f, 6.500f},
            {"LWIR", 6.500f, 15.000f}
        };

        for (u32 b = 0; b < numBands && b < 5; ++b) {
            // Band header (16 bytes)
            f32 startUm = bands[b].startUm;
            f32 endUm = bands[b].endUm;
            file.write(reinterpret_cast<const char*>(&startUm), sizeof(f32));
            file.write(reinterpret_cast<const char*>(&endUm), sizeof(f32));
            file.write(reinterpret_cast<const char*>(&numSamples), sizeof(u32));
            file.write(reinterpret_cast<const char*>(&numBasis), sizeof(u32));

            // Basis data (numBasis * numSamples floats)
            for (u32 i = 0; i < numBasis; ++i) {
                for (u32 j = 0; j < numSamples; ++j) {
                    // Create a simple pattern: basis[i][j] = (i+1) * 0.1 + j * 0.01
                    f32 value = static_cast<f32>(i + 1) * 0.1f + static_cast<f32>(j) * 0.01f;
                    file.write(reinterpret_cast<const char*>(&value), sizeof(f32));
                }
            }
        }

        file.close();
    }

    // Helper: Create a valid JSON materials file
    void CreateTestMaterialsJson(const std::filesystem::path& path,
                                 const std::vector<std::string>& materialNames,
                                 u32 numBasis = 4) {
        std::ofstream file(path);

        file << "{\n";
        file << "  \"metadata\": {\n";
        file << "    \"generator\": \"SpectralBaker Test\",\n";
        file << "    \"source_library\": \"Test Library\",\n";
        file << "    \"date_generated\": \"2025-01-01T00:00:00Z\",\n";
        file << "    \"num_materials\": " << materialNames.size() << "\n";
        file << "  },\n";
        file << "  \"materials\": {\n";

        for (size_t m = 0; m < materialNames.size(); ++m) {
            file << "    \"" << materialNames[m] << "\": {\n";
            file << "      \"source\": {\n";
            file << "        \"filename\": \"test_" << m << ".txt\",\n";
            file << "        \"record_id\": \"" << (10000 + m) << "\",\n";
            file << "        \"instrument\": \"ASDFRa\",\n";
            file << "        \"chapter\": \"TestChapter\"\n";
            file << "      },\n";
            file << "      \"bands\": {\n";

            const char* bandNames[] = {"VIS", "NIR", "SWIR", "MWIR", "LWIR"};
            for (int b = 0; b < 5; ++b) {
                file << "        \"" << bandNames[b] << "\": {\n";
                file << "          \"basis_weights\": [";

                // Generate weights that sum to approximately 1
                for (u32 i = 0; i < numBasis; ++i) {
                    f32 weight = 1.0f / static_cast<f32>(numBasis);
                    file << weight;
                    if (i < numBasis - 1) file << ", ";
                }

                file << "],\n";
                file << "          \"rmse\": 0.005,\n";
                file << "          \"explained_variance\": 0.98\n";
                file << "        }";
                if (b < 4) file << ",";
                file << "\n";
            }

            file << "      }\n";
            file << "    }";
            if (m < materialNames.size() - 1) file << ",";
            file << "\n";
        }

        file << "  }\n";
        file << "}\n";
        file.close();
    }

    std::filesystem::path tempDir;
};

// ============================================================================
// Binary Basis File Loading Tests
// ============================================================================

TEST_F(SpectralBasisLoaderTest, LoadBasisFileBasic) {
    auto basisPath = GetTempFilePath("test_basis.bin");
    CreateTestBasisFile(basisPath, 5, 4, 10);

    SpectralBasisLoader loader;
    bool success = loader.LoadBasis(basisPath);

    ASSERT_TRUE(success);
    EXPECT_TRUE(loader.HasBasis());
    EXPECT_EQ(loader.GetNumBands(), 5);
    EXPECT_EQ(loader.GetBasisVersion(), 3);
}

TEST_F(SpectralBasisLoaderTest, LoadBasisFileGetBasis) {
    auto basisPath = GetTempFilePath("test_basis_get.bin");
    CreateTestBasisFile(basisPath, 5, 8, 20);

    SpectralBasisLoader loader;
    loader.LoadBasis(basisPath);

    // Get VIS basis
    const BasisFunctions* visBasis = loader.GetBasis("VIS");
    ASSERT_NE(visBasis, nullptr);
    EXPECT_EQ(visBasis->name, "VIS");
    EXPECT_EQ(visBasis->numBasis, 8);
    EXPECT_EQ(visBasis->numSamples, 20);
    EXPECT_NEAR(visBasis->wavelengthStart_um, 0.350f, 1e-5f);
    EXPECT_NEAR(visBasis->wavelengthEnd_um, 0.780f, 1e-5f);
    EXPECT_TRUE(visBasis->IsValid());

    // Get NIR basis
    const BasisFunctions* nirBasis = loader.GetBasis("NIR");
    ASSERT_NE(nirBasis, nullptr);
    EXPECT_EQ(nirBasis->name, "NIR");

    // Get SWIR basis
    const BasisFunctions* swirBasis = loader.GetBasis("SWIR");
    ASSERT_NE(swirBasis, nullptr);
    EXPECT_EQ(swirBasis->name, "SWIR");

    // Get MWIR basis
    const BasisFunctions* mwirBasis = loader.GetBasis("MWIR");
    ASSERT_NE(mwirBasis, nullptr);
    EXPECT_EQ(mwirBasis->name, "MWIR");
    EXPECT_NEAR(mwirBasis->wavelengthStart_um, 2.500f, 1e-5f);
    EXPECT_NEAR(mwirBasis->wavelengthEnd_um, 6.500f, 1e-5f);

    // Get LWIR basis
    const BasisFunctions* lwirBasis = loader.GetBasis("LWIR");
    ASSERT_NE(lwirBasis, nullptr);
    EXPECT_EQ(lwirBasis->name, "LWIR");
    EXPECT_NEAR(lwirBasis->wavelengthStart_um, 6.500f, 1e-5f);
    EXPECT_NEAR(lwirBasis->wavelengthEnd_um, 15.000f, 1e-5f);

    // Get nonexistent band
    const BasisFunctions* invalid = loader.GetBasis("INVALID");
    EXPECT_EQ(invalid, nullptr);
}

TEST_F(SpectralBasisLoaderTest, LoadBasisFileNonexistent) {
    auto basisPath = GetTempFilePath("nonexistent.bin");

    SpectralBasisLoader loader;
    bool success = loader.LoadBasis(basisPath);

    EXPECT_FALSE(success);
    EXPECT_FALSE(loader.HasBasis());
}

TEST_F(SpectralBasisLoaderTest, LoadBasisFileInvalidMagic) {
    auto basisPath = GetTempFilePath("invalid_magic.bin");

    // Create file with wrong magic number
    std::ofstream file(basisPath, std::ios::binary);
    const char wrongMagic[4] = {'X', 'X', 'X', 'X'};
    file.write(wrongMagic, 4);
    file.close();

    SpectralBasisLoader loader;
    bool success = loader.LoadBasis(basisPath);

    EXPECT_FALSE(success);
}

TEST_F(SpectralBasisLoaderTest, LoadBasisFileInvalidVersion) {
    auto basisPath = GetTempFilePath("invalid_version.bin");

    // Create file with unsupported version
    std::ofstream file(basisPath, std::ios::binary);
    const char magic[4] = {'Q', 'B', 'A', 'S'};
    u32 version = 99;  // Invalid version
    u32 numBands = 1;
    file.write(magic, 4);
    file.write(reinterpret_cast<const char*>(&version), sizeof(u32));
    file.write(reinterpret_cast<const char*>(&numBands), sizeof(u32));
    file.close();

    SpectralBasisLoader loader;
    bool success = loader.LoadBasis(basisPath);

    EXPECT_FALSE(success);
}

TEST_F(SpectralBasisLoaderTest, LoadBasisFileTruncated) {
    auto basisPath = GetTempFilePath("truncated.bin");

    // Create truncated file (only header, no band data)
    std::ofstream file(basisPath, std::ios::binary);
    const char magic[4] = {'Q', 'B', 'A', 'S'};
    u32 version = 3;
    u32 numBands = 5;
    file.write(magic, 4);
    file.write(reinterpret_cast<const char*>(&version), sizeof(u32));
    file.write(reinterpret_cast<const char*>(&numBands), sizeof(u32));
    // Write partial reserved bytes (less than 52)
    char reserved[20] = {0};
    file.write(reserved, 20);
    file.close();

    SpectralBasisLoader loader;
    bool success = loader.LoadBasis(basisPath);

    EXPECT_FALSE(success);
}

// ============================================================================
// JSON Materials File Loading Tests
// ============================================================================

TEST_F(SpectralBasisLoaderTest, LoadMaterialsJsonBasic) {
    auto jsonPath = GetTempFilePath("test_materials.json");
    CreateTestMaterialsJson(jsonPath, {"Material_A", "Material_B", "Material_C"}, 4);

    SpectralBasisLoader loader;
    bool success = loader.LoadMaterials(jsonPath);

    ASSERT_TRUE(success);
    EXPECT_TRUE(loader.HasMaterials());
    EXPECT_EQ(loader.GetMaterialCount(), 3);
}

TEST_F(SpectralBasisLoaderTest, LoadMaterialsJsonMetadata) {
    auto jsonPath = GetTempFilePath("test_metadata.json");
    CreateTestMaterialsJson(jsonPath, {"Test_Material"});

    SpectralBasisLoader loader;
    loader.LoadMaterials(jsonPath);

    EXPECT_EQ(loader.GetGenerator(), "SpectralBaker Test");
    EXPECT_EQ(loader.GetSourceLibrary(), "Test Library");
    EXPECT_FALSE(loader.GetDateGenerated().empty());
}

TEST_F(SpectralBasisLoaderTest, LoadMaterialsJsonFindMaterial) {
    auto jsonPath = GetTempFilePath("test_find.json");
    CreateTestMaterialsJson(jsonPath, {"Gold_HS111", "Silver_AG001", "Copper_CU002"});

    SpectralBasisLoader loader;
    loader.LoadMaterials(jsonPath);

    // Exact match
    const MaterialSpectralData* gold = loader.FindMaterial("Gold_HS111");
    ASSERT_NE(gold, nullptr);
    EXPECT_EQ(gold->name, "Gold_HS111");
    EXPECT_EQ(gold->instrument, "ASDFRa");
    EXPECT_TRUE(gold->HasBand("VIS"));
    EXPECT_TRUE(gold->HasBand("NIR"));
    EXPECT_TRUE(gold->HasBand("SWIR"));
    EXPECT_TRUE(gold->HasBand("MWIR"));
    EXPECT_TRUE(gold->HasBand("LWIR"));

    // Not found
    const MaterialSpectralData* notFound = loader.FindMaterial("Nonexistent");
    EXPECT_EQ(notFound, nullptr);
}

TEST_F(SpectralBasisLoaderTest, LoadMaterialsJsonFindMaterialPartial) {
    auto jsonPath = GetTempFilePath("test_partial.json");
    CreateTestMaterialsJson(jsonPath, {"Gold_HS111.3B", "Silver_AG001", "GoldCoated_Surface"});

    SpectralBasisLoader loader;
    loader.LoadMaterials(jsonPath);

    // Partial match (case-insensitive)
    const MaterialSpectralData* gold = loader.FindMaterialPartial("gold");
    ASSERT_NE(gold, nullptr);
    // Should find first matching material
    EXPECT_TRUE(gold->name.find("Gold") != std::string::npos ||
                gold->name.find("gold") != std::string::npos);

    // Partial match with partial string
    const MaterialSpectralData* silver = loader.FindMaterialPartial("AG001");
    ASSERT_NE(silver, nullptr);
    EXPECT_EQ(silver->name, "Silver_AG001");
}

TEST_F(SpectralBasisLoaderTest, LoadMaterialsJsonGetMaterialNames) {
    auto jsonPath = GetTempFilePath("test_names.json");
    CreateTestMaterialsJson(jsonPath, {"Material_A", "Material_B", "Material_C"});

    SpectralBasisLoader loader;
    loader.LoadMaterials(jsonPath);

    auto names = loader.GetMaterialNames();
    EXPECT_EQ(names.size(), 3);

    // Check all names are present (order may vary due to unordered_map)
    bool hasA = false, hasB = false, hasC = false;
    for (const auto& name : names) {
        if (name == "Material_A") hasA = true;
        if (name == "Material_B") hasB = true;
        if (name == "Material_C") hasC = true;
    }
    EXPECT_TRUE(hasA && hasB && hasC);
}

TEST_F(SpectralBasisLoaderTest, LoadMaterialsJsonNonexistent) {
    auto jsonPath = GetTempFilePath("nonexistent.json");

    SpectralBasisLoader loader;
    bool success = loader.LoadMaterials(jsonPath);

    EXPECT_FALSE(success);
    EXPECT_FALSE(loader.HasMaterials());
}

TEST_F(SpectralBasisLoaderTest, LoadMaterialsJsonMalformed) {
    auto jsonPath = GetTempFilePath("malformed.json");

    std::ofstream file(jsonPath);
    file << "This is not valid JSON { broken: syntax }\n";
    file.close();

    SpectralBasisLoader loader;
    bool success = loader.LoadMaterials(jsonPath);

    // Should either fail or return empty materials
    if (success) {
        EXPECT_EQ(loader.GetMaterialCount(), 0);
    }
}

// ============================================================================
// Combined Loading Tests
// ============================================================================

TEST_F(SpectralBasisLoaderTest, LoadBothFiles) {
    auto basisPath = GetTempFilePath("combined_basis.bin");
    auto jsonPath = GetTempFilePath("combined_materials.json");

    CreateTestBasisFile(basisPath, 5, 4, 10);
    CreateTestMaterialsJson(jsonPath, {"Material_X", "Material_Y"}, 4);

    SpectralBasisLoader loader;
    bool success = loader.Load(basisPath, jsonPath);

    ASSERT_TRUE(success);
    EXPECT_TRUE(loader.HasBasis());
    EXPECT_TRUE(loader.HasMaterials());
    EXPECT_EQ(loader.GetNumBands(), 5);
    EXPECT_EQ(loader.GetMaterialCount(), 2);
}

// ============================================================================
// Spectral Curve Reconstruction Tests
// ============================================================================

TEST_F(SpectralBasisLoaderTest, ReconstructCurveBasic) {
    auto basisPath = GetTempFilePath("recon_basis.bin");
    auto jsonPath = GetTempFilePath("recon_materials.json");

    CreateTestBasisFile(basisPath, 3, 4, 10);
    CreateTestMaterialsJson(jsonPath, {"TestMaterial"}, 4);

    SpectralBasisLoader loader;
    loader.Load(basisPath, jsonPath);

    SpectralCurve curve = loader.ReconstructCurve("TestMaterial", "VIS");

    EXPECT_FALSE(curve.samples.empty());
    EXPECT_EQ(curve.samples.size(), 10);  // numSamples from basis

    // Verify wavelengths are in nm
    EXPECT_GT(curve.samples.front().first, 300.0f);  // Should be ~350nm
    EXPECT_LT(curve.samples.back().first, 800.0f);   // Should be ~780nm

    // All values should be non-negative and <= 1 (reflectance)
    for (const auto& sample : curve.samples) {
        EXPECT_GE(sample.second, 0.0f);
        EXPECT_LE(sample.second, 1.0f);
    }
}

TEST_F(SpectralBasisLoaderTest, ReconstructCurveGPU) {
    auto basisPath = GetTempFilePath("recon_gpu_basis.bin");
    auto jsonPath = GetTempFilePath("recon_gpu_materials.json");

    CreateTestBasisFile(basisPath, 3, 4, 50);
    CreateTestMaterialsJson(jsonPath, {"GPUTestMaterial"}, 4);

    SpectralBasisLoader loader;
    loader.Load(basisPath, jsonPath);

    SpectralCurveGPU gpuCurve = loader.ReconstructCurveGPU("GPUTestMaterial", "VIS");

    EXPECT_GT(gpuCurve.numSamples, 0);
    EXPECT_LE(gpuCurve.numSamples, MAX_SPECTRAL_SAMPLES);
    EXPECT_GT(gpuCurve.startWavelength_nm, 0.0f);
    EXPECT_GT(gpuCurve.stepSize_nm, 0.0f);

    // Verify GPU struct size
    EXPECT_EQ(sizeof(SpectralCurveGPU), 272);
}

TEST_F(SpectralBasisLoaderTest, ReconstructCurveMaterialNotFound) {
    auto basisPath = GetTempFilePath("recon_notfound_basis.bin");
    auto jsonPath = GetTempFilePath("recon_notfound_materials.json");

    CreateTestBasisFile(basisPath, 3, 4, 10);
    CreateTestMaterialsJson(jsonPath, {"ExistingMaterial"}, 4);

    SpectralBasisLoader loader;
    loader.Load(basisPath, jsonPath);

    SpectralCurve curve = loader.ReconstructCurve("NonexistentMaterial", "VIS");
    EXPECT_TRUE(curve.samples.empty());

    SpectralCurveGPU gpuCurve = loader.ReconstructCurveGPU("NonexistentMaterial", "VIS");
    EXPECT_EQ(gpuCurve.numSamples, 0);
}

TEST_F(SpectralBasisLoaderTest, ReconstructCurveBandNotFound) {
    auto basisPath = GetTempFilePath("recon_nobasis.bin");
    auto jsonPath = GetTempFilePath("recon_nobasis_materials.json");

    CreateTestBasisFile(basisPath, 1, 4, 10);  // Only 1 band (VIS)
    CreateTestMaterialsJson(jsonPath, {"TestMaterial"}, 4);

    SpectralBasisLoader loader;
    loader.Load(basisPath, jsonPath);

    // Should work for VIS
    SpectralCurve visCurve = loader.ReconstructCurve("TestMaterial", "VIS");
    EXPECT_FALSE(visCurve.samples.empty());

    // Should fail for NIR (no basis)
    SpectralCurve nirCurve = loader.ReconstructCurve("TestMaterial", "NIR");
    EXPECT_TRUE(nirCurve.samples.empty());
}

TEST_F(SpectralBasisLoaderTest, ReconstructFullSpectrum) {
    auto basisPath = GetTempFilePath("full_spectrum_basis.bin");
    auto jsonPath = GetTempFilePath("full_spectrum_materials.json");

    CreateTestBasisFile(basisPath, 5, 4, 20);
    CreateTestMaterialsJson(jsonPath, {"FullSpectrumMaterial"}, 4);

    SpectralBasisLoader loader;
    loader.Load(basisPath, jsonPath);

    SpectralCurve fullCurve = loader.ReconstructFullSpectrum("FullSpectrumMaterial");

    EXPECT_FALSE(fullCurve.samples.empty());
    // Should span VIS + NIR + SWIR + MWIR + LWIR range (5 bands)
    EXPECT_GT(fullCurve.samples.size(), 20);  // More than single band

    // Verify wavelength ordering (should be monotonically increasing)
    for (size_t i = 1; i < fullCurve.samples.size(); ++i) {
        EXPECT_GT(fullCurve.samples[i].first, fullCurve.samples[i-1].first);
    }
}

// ============================================================================
// BasisFunctions Helper Tests
// ============================================================================

TEST_F(SpectralBasisLoaderTest, BasisFunctionsGetWavelength) {
    auto basisPath = GetTempFilePath("wavelength_test.bin");
    CreateTestBasisFile(basisPath, 1, 4, 100);  // VIS band only

    SpectralBasisLoader loader;
    loader.LoadBasis(basisPath);

    const BasisFunctions* basis = loader.GetBasis("VIS");
    ASSERT_NE(basis, nullptr);

    // First wavelength should be start
    f32 firstWl = basis->GetWavelength_nm(0);
    EXPECT_NEAR(firstWl, 350.0f, 1.0f);

    // Last wavelength should be end
    f32 lastWl = basis->GetWavelength_nm(basis->numSamples - 1);
    EXPECT_NEAR(lastWl, 780.0f, 1.0f);

    // Midpoint wavelength
    f32 midWl = basis->GetWavelength_nm(basis->numSamples / 2);
    EXPECT_GT(midWl, firstWl);
    EXPECT_LT(midWl, lastWl);
}

TEST_F(SpectralBasisLoaderTest, BasisFunctionsGet) {
    auto basisPath = GetTempFilePath("get_test.bin");
    CreateTestBasisFile(basisPath, 1, 4, 10);

    SpectralBasisLoader loader;
    loader.LoadBasis(basisPath);

    const BasisFunctions* basis = loader.GetBasis("VIS");
    ASSERT_NE(basis, nullptr);

    // Test Get() accessor
    for (u32 b = 0; b < basis->numBasis; ++b) {
        for (u32 s = 0; s < basis->numSamples; ++s) {
            f32 value = basis->Get(b, s);
            // Based on our test pattern: (b+1) * 0.1 + s * 0.01
            f32 expected = static_cast<f32>(b + 1) * 0.1f + static_cast<f32>(s) * 0.01f;
            EXPECT_NEAR(value, expected, 1e-5f);
        }
    }
}

// ============================================================================
// MaterialSpectralData Tests
// ============================================================================

TEST_F(SpectralBasisLoaderTest, MaterialSpectralDataHasBand) {
    auto jsonPath = GetTempFilePath("hasband_test.json");
    CreateTestMaterialsJson(jsonPath, {"TestMaterial"}, 4);

    SpectralBasisLoader loader;
    loader.LoadMaterials(jsonPath);

    const MaterialSpectralData* mat = loader.FindMaterial("TestMaterial");
    ASSERT_NE(mat, nullptr);

    EXPECT_TRUE(mat->HasBand("VIS"));
    EXPECT_TRUE(mat->HasBand("NIR"));
    EXPECT_TRUE(mat->HasBand("SWIR"));
    EXPECT_TRUE(mat->HasBand("MWIR"));
    EXPECT_TRUE(mat->HasBand("LWIR"));
    EXPECT_FALSE(mat->HasBand("INVALID"));  // Invalid band name
}

TEST_F(SpectralBasisLoaderTest, MaterialSpectralDataWeights) {
    auto jsonPath = GetTempFilePath("weights_test.json");
    CreateTestMaterialsJson(jsonPath, {"WeightTestMaterial"}, 8);

    SpectralBasisLoader loader;
    loader.LoadMaterials(jsonPath);

    const MaterialSpectralData* mat = loader.FindMaterial("WeightTestMaterial");
    ASSERT_NE(mat, nullptr);

    // Check VIS band data
    auto visIt = mat->bands.find("VIS");
    ASSERT_NE(visIt, mat->bands.end());

    const auto& visBand = visIt->second;
    EXPECT_EQ(visBand.weights.size(), 8);  // numBasis = 8
    EXPECT_GT(visBand.rmse, 0.0f);
    EXPECT_GT(visBand.explainedVariance, 0.0f);
    EXPECT_LE(visBand.explainedVariance, 1.0f);
}

// ============================================================================
// NMFBandType Enum Tests
// ============================================================================

TEST_F(SpectralBasisLoaderTest, NMFBandTypeToString) {
    EXPECT_STREQ(NMFBandTypeToString(NMFBandType::VIS), "VIS");
    EXPECT_STREQ(NMFBandTypeToString(NMFBandType::NIR), "NIR");
    EXPECT_STREQ(NMFBandTypeToString(NMFBandType::SWIR), "SWIR");
    EXPECT_STREQ(NMFBandTypeToString(NMFBandType::MWIR), "MWIR");
    EXPECT_STREQ(NMFBandTypeToString(NMFBandType::LWIR), "LWIR");
}

TEST_F(SpectralBasisLoaderTest, GetBasisByEnum) {
    auto basisPath = GetTempFilePath("enum_test.bin");
    CreateTestBasisFile(basisPath, 5, 4, 10);

    SpectralBasisLoader loader;
    loader.LoadBasis(basisPath);

    // Test GetBasis with enum
    const BasisFunctions* visBasis = loader.GetBasis(NMFBandType::VIS);
    ASSERT_NE(visBasis, nullptr);
    EXPECT_EQ(visBasis->name, "VIS");

    const BasisFunctions* nirBasis = loader.GetBasis(NMFBandType::NIR);
    ASSERT_NE(nirBasis, nullptr);
    EXPECT_EQ(nirBasis->name, "NIR");

    const BasisFunctions* swirBasis = loader.GetBasis(NMFBandType::SWIR);
    ASSERT_NE(swirBasis, nullptr);
    EXPECT_EQ(swirBasis->name, "SWIR");

    const BasisFunctions* mwirBasis = loader.GetBasis(NMFBandType::MWIR);
    ASSERT_NE(mwirBasis, nullptr);
    EXPECT_EQ(mwirBasis->name, "MWIR");

    const BasisFunctions* lwirBasis = loader.GetBasis(NMFBandType::LWIR);
    ASSERT_NE(lwirBasis, nullptr);
    EXPECT_EQ(lwirBasis->name, "LWIR");
}

// ============================================================================
// Integration with Real Files (if available)
// ============================================================================

TEST_F(SpectralBasisLoaderTest, LoadRealFilesIfAvailable) {
    // Try to load actual SpectralBaker output files if they exist
    std::filesystem::path basisPath = "../../assets/spectral/quantiloom_basis_v1.bin";
    std::filesystem::path jsonPath = "../../assets/spectral/quantiloom_materials.json";

    if (!std::filesystem::exists(basisPath) || !std::filesystem::exists(jsonPath)) {
        GTEST_SKIP() << "Real SpectralBaker files not found, skipping integration test";
    }

    SpectralBasisLoader loader;
    bool success = loader.Load(basisPath, jsonPath);

    ASSERT_TRUE(success);
    EXPECT_TRUE(loader.HasBasis());
    EXPECT_TRUE(loader.HasMaterials());

    // Should have significant number of materials (SpectralBaker produces ~1374)
    EXPECT_GT(loader.GetMaterialCount(), 100);

    // Should have 5 bands
    EXPECT_EQ(loader.GetNumBands(), 5);

    // Try to reconstruct a known material
    auto names = loader.GetMaterialNames();
    if (!names.empty()) {
        SpectralCurve curve = loader.ReconstructCurve(names[0], "VIS");
        EXPECT_FALSE(curve.samples.empty());
        EXPECT_TRUE(curve.IsValid());
    }
}

// ============================================================================
// Performance Tests (optional)
// ============================================================================

TEST_F(SpectralBasisLoaderTest, LoadLargeMaterialsDatabase) {
    auto basisPath = GetTempFilePath("large_basis.bin");
    auto jsonPath = GetTempFilePath("large_materials.json");

    // Create basis with larger dimensions (5 bands for v3)
    CreateTestBasisFile(basisPath, 5, 32, 200);

    // Create JSON with many materials
    std::vector<std::string> materialNames;
    for (int i = 0; i < 100; ++i) {
        materialNames.push_back("Material_" + std::to_string(i));
    }
    CreateTestMaterialsJson(jsonPath, materialNames, 32);

    SpectralBasisLoader loader;
    bool success = loader.Load(basisPath, jsonPath);

    ASSERT_TRUE(success);
    EXPECT_EQ(loader.GetMaterialCount(), 100);

    // Verify reconstruction still works
    SpectralCurve curve = loader.ReconstructCurve("Material_50", "VIS");
    EXPECT_FALSE(curve.samples.empty());
}
