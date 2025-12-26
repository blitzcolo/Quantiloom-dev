// ============================================================================
// Quantiloom - Unit Tests for io/SpectralIO.hpp
// ============================================================================
// Tests cover:
// - CSV spectral curve loading
// - USGS Spectral Library loading
// - RefractiveIndex.INFO YAML loading
// - ASTM G-173 solar spectrum loading (sun/sky irradiance)
// - libRadtran uvspec atmospheric model output loading
// ============================================================================

#include <gtest/gtest.h>
#include "io/SpectralIO.hpp"
#include "core/SpectralData.hpp"
#include <filesystem>
#include <fstream>
#include <cstdio>

using namespace quantiloom;

// ============================================================================
// Test Fixture with Temporary File Management
// ============================================================================

class SpectralIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        tempDir = std::filesystem::temp_directory_path() / "quantiloom_spectral_tests";
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

    std::filesystem::path tempDir;
};

// ============================================================================
// CSV Spectral Curve Loading Tests
// ============================================================================

TEST_F(SpectralIOTest, LoadSpectralCurveCSVBasic) {
    auto filepath = GetTempFilePath("test_curve.csv");

    std::ofstream file(filepath);
    file << "400.0, 0.12\n";
    file << "450.0, 0.25\n";
    file << "500.0, 0.50\n";
    file << "550.0, 0.65\n";
    file << "600.0, 0.40\n";
    file.close();

    auto result = SpectralIO::LoadSpectralCurveCSV(filepath);
    ASSERT_TRUE(result.has_value());

    const auto& samples = result.value();
    EXPECT_EQ(samples.size(), 5);

    EXPECT_NEAR(samples[0].first, 400.0f, 1e-5f);
    EXPECT_NEAR(samples[0].second, 0.12f, 1e-5f);
    EXPECT_NEAR(samples[2].first, 500.0f, 1e-5f);
    EXPECT_NEAR(samples[2].second, 0.50f, 1e-5f);
    EXPECT_NEAR(samples[4].first, 600.0f, 1e-5f);
    EXPECT_NEAR(samples[4].second, 0.40f, 1e-5f);
}

TEST_F(SpectralIOTest, LoadSpectralCurveCSVWithComments) {
    auto filepath = GetTempFilePath("curve_comments.csv");

    std::ofstream file(filepath);
    file << "# This is a comment\n";
    file << "# wavelength_nm, reflectance (header as comment)\n";
    file << "400.0, 0.10\n";
    file << "500.0, 0.50\n";
    file << "# Another comment\n";
    file << "600.0, 0.30\n";
    file.close();

    auto result = SpectralIO::LoadSpectralCurveCSV(filepath);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result.value().size(), 3);
}

TEST_F(SpectralIOTest, LoadSpectralCurveCSVNonexistent) {
    auto filepath = GetTempFilePath("nonexistent_curve.csv");

    auto result = SpectralIO::LoadSpectralCurveCSV(filepath);
    EXPECT_FALSE(result.has_value());
}

TEST_F(SpectralIOTest, LoadSpectralCurveCSVEmpty) {
    auto filepath = GetTempFilePath("empty_curve.csv");

    std::ofstream file(filepath);
    file.close();

    auto result = SpectralIO::LoadSpectralCurveCSV(filepath);
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// USGS Spectral Library Loading Tests
// ============================================================================

TEST_F(SpectralIOTest, LoadUSGSBasic) {
    auto wavelengthPath = GetTempFilePath("wavelengths.txt");
    std::ofstream wvFile(wavelengthPath);
    wvFile << "Record=1 Wavelengths for BECK spectrometer\n";
    wvFile << "0.350000\n";
    wvFile << "0.400000\n";
    wvFile << "0.500000\n";
    wvFile << "0.600000\n";
    wvFile << "0.700000\n";
    wvFile.close();

    auto reflectancePath = GetTempFilePath("TestMaterial_AREF.txt");
    std::ofstream refFile(reflectancePath);
    refFile << "Record=1 TestMaterial DHR reflectance\n";
    refFile << "0.150000\n";
    refFile << "0.250000\n";
    refFile << "0.500000\n";
    refFile << "0.350000\n";
    refFile << "0.200000\n";
    refFile.close();

    auto result = SpectralIO::LoadUSGS(reflectancePath, wavelengthPath);
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_TRUE(curve.IsValid());
    EXPECT_EQ(curve.samples.size(), 5);

    EXPECT_NEAR(curve.samples[0].first, 350.0f, 1e-3f);
    EXPECT_NEAR(curve.samples[1].first, 400.0f, 1e-3f);
    EXPECT_NEAR(curve.samples[4].first, 700.0f, 1e-3f);

    EXPECT_NEAR(curve.samples[0].second, 0.15f, 1e-5f);
    EXPECT_NEAR(curve.samples[2].second, 0.50f, 1e-5f);
}

TEST_F(SpectralIOTest, LoadUSGSWithInvalidData) {
    auto wavelengthPath = GetTempFilePath("wavelengths_invalid.txt");
    std::ofstream wvFile(wavelengthPath);
    wvFile << "Record=1 Test wavelengths\n";
    wvFile << "0.400000\n";
    wvFile << "0.500000\n";
    wvFile << "0.600000\n";
    wvFile << "0.700000\n";
    wvFile.close();

    auto reflectancePath = GetTempFilePath("TestMaterial_Invalid_AREF.txt");
    std::ofstream refFile(reflectancePath);
    refFile << "Record=1 TestMaterial with invalid data\n";
    refFile << "0.250000\n";
    refFile << "-1.23e+034\n";
    refFile << "0.400000\n";
    refFile << "0.300000\n";
    refFile.close();

    auto result = SpectralIO::LoadUSGS(reflectancePath, wavelengthPath);
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_LT(curve.samples.size(), 4);
    EXPECT_TRUE(curve.IsValid());

    for (const auto& sample : curve.samples) {
        EXPECT_GE(sample.second, 0.0f);
        EXPECT_LE(sample.second, 1.0f);
    }
}

TEST_F(SpectralIOTest, LoadUSGSNonexistentFiles) {
    auto wavelengthPath = GetTempFilePath("nonexistent_wv.txt");
    auto reflectancePath = GetTempFilePath("nonexistent_ref.txt");

    auto result = SpectralIO::LoadUSGS(reflectancePath, wavelengthPath);
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// RefractiveIndex.INFO YAML Loading Tests
// ============================================================================

TEST_F(SpectralIOTest, LoadRefractiveIndexYAMLBasic) {
    auto filepath = GetTempFilePath("gold_test.yml");

    std::ofstream file(filepath);
    file << "# Test gold optical constants\n";
    file << "REFERENCES: \"Test reference\"\n";
    file << "COMMENTS: \"Test gold data\"\n";
    file << "DATA:\n";
    file << "  - type: tabulated nk\n";
    file << "    data: |\n";
    file << "        0.5000 0.9400 1.9600\n";
    file << "        0.5500 0.8800 2.2000\n";
    file << "        0.6000 0.2100 2.8700\n";
    file << "        0.6500 0.1400 3.1500\n";
    file << "        0.7000 0.1300 3.4200\n";
    file.close();

    auto result = SpectralIO::LoadRefractiveIndexYAML(filepath);
    ASSERT_TRUE(result.has_value());

    const ComplexRefractiveIndex& cri = result.value();
    EXPECT_TRUE(cri.IsValid());
    EXPECT_EQ(cri.wavelengths_nm.size(), 5);
    EXPECT_EQ(cri.n.size(), 5);
    EXPECT_EQ(cri.k.size(), 5);

    EXPECT_NEAR(cri.wavelengths_nm[0], 500.0f, 1e-3f);
    EXPECT_NEAR(cri.wavelengths_nm[4], 700.0f, 1e-3f);

    EXPECT_NEAR(cri.n[0], 0.94f, 1e-3f);
    EXPECT_NEAR(cri.n[2], 0.21f, 1e-3f);

    EXPECT_NEAR(cri.k[0], 1.96f, 1e-3f);
    EXPECT_NEAR(cri.k[4], 3.42f, 1e-3f);
}

TEST_F(SpectralIOTest, LoadRefractiveIndexYAMLNonexistent) {
    auto filepath = GetTempFilePath("nonexistent.yml");

    auto result = SpectralIO::LoadRefractiveIndexYAML(filepath);
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// ASTM G-173 Solar Spectrum Loading Tests
// ============================================================================

TEST_F(SpectralIOTest, LoadASTMG173Basic) {
    auto filepath = GetTempFilePath("astmg173_test.csv");

    std::ofstream file(filepath);
    file << "Wvlgth nm,Etr W*m-2*nm-1,Global tilt  W*m-2*nm-1,Direct+circumsolar W*m-2*nm-1\n";
    file << "280,8.2000E-02,4.7309E-23,2.5361E-26\n";
    file << "300,5.1400E-01,1.0230E-04,2.4980E-06\n";
    file << "400,1.5140E+00,1.2680E+00,1.1130E+00\n";
    file << "500,1.9170E+00,1.6750E+00,1.5290E+00\n";
    file << "600,1.8310E+00,1.6740E+00,1.5870E+00\n";
    file << "700,1.5050E+00,1.3820E+00,1.3140E+00\n";
    file << "1000,7.4900E-01,6.9660E-01,6.6810E-01\n";
    file << "2000,2.1630E-01,1.4440E-01,1.6200E-01\n";
    file << "4000,1.6170E-02,1.1010E-02,1.3080E-02\n";
    file.close();

    auto result = SpectralIO::LoadASTMG173(filepath);
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_TRUE(curve.IsValid());
    EXPECT_EQ(curve.samples.size(), 9);

    EXPECT_NEAR(curve.samples.front().first, 280.0f, 1e-3f);
    EXPECT_NEAR(curve.samples.back().first, 4000.0f, 1e-3f);
}

TEST_F(SpectralIOTest, LoadASTMG173Column2ETR) {
    auto filepath = GetTempFilePath("astmg173_etr.csv");

    std::ofstream file(filepath);
    file << "Wvlgth nm,Etr W*m-2*nm-1,Global tilt  W*m-2*nm-1,Direct+circumsolar W*m-2*nm-1\n";
    file << "400,1.5140E+00,1.2680E+00,1.1130E+00\n";
    file << "500,1.9170E+00,1.6750E+00,1.5290E+00\n";
    file << "600,1.8310E+00,1.6740E+00,1.5870E+00\n";
    file.close();

    auto result = SpectralIO::LoadASTMG173(filepath, 2);
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_EQ(curve.samples.size(), 3);
    EXPECT_NEAR(curve.Evaluate(500.0f), 1.917f, 0.01f);
}

TEST_F(SpectralIOTest, LoadASTMG173SunAndSkyBasic) {
    auto filepath = GetTempFilePath("astmg173_sun_sky.csv");

    std::ofstream file(filepath);
    file << "Wvlgth nm,Etr W*m-2*nm-1,Global tilt  W*m-2*nm-1,Direct+circumsolar W*m-2*nm-1\n";
    file << "400,1.5140E+00,1.2680E+00,1.1130E+00\n";
    file << "450,1.7610E+00,1.5510E+00,1.4000E+00\n";
    file << "500,1.9170E+00,1.6750E+00,1.5290E+00\n";
    file << "550,1.8690E+00,1.6620E+00,1.5510E+00\n";
    file << "600,1.8310E+00,1.6740E+00,1.5870E+00\n";
    file << "650,1.6760E+00,1.5460E+00,1.4720E+00\n";
    file << "700,1.5050E+00,1.3820E+00,1.3140E+00\n";
    file.close();

    auto result = SpectralIO::LoadASTMG173SunAndSky(filepath);
    ASSERT_TRUE(result.has_value());

    const auto& [sunCurve, skyCurve] = result.value();

    EXPECT_TRUE(sunCurve.IsValid());
    EXPECT_TRUE(skyCurve.IsValid());
    EXPECT_EQ(sunCurve.samples.size(), 7);
    EXPECT_EQ(skyCurve.samples.size(), 7);

    EXPECT_NEAR(sunCurve.Evaluate(500.0f), 1.5290f, 0.01f);
    EXPECT_NEAR(skyCurve.Evaluate(500.0f), 0.146f, 0.02f);
}

TEST_F(SpectralIOTest, LoadASTMG173Nonexistent) {
    auto filepath = GetTempFilePath("nonexistent_astm.csv");

    auto result = SpectralIO::LoadASTMG173(filepath);
    EXPECT_FALSE(result.has_value());

    auto result2 = SpectralIO::LoadASTMG173SunAndSky(filepath);
    EXPECT_FALSE(result2.has_value());
}

// ============================================================================
// libRadtran uvspec Output Loading Tests
// ============================================================================

TEST_F(SpectralIOTest, LoadLibRadtranUvspecBasic) {
    auto filepath = GetTempFilePath("uvspec_basic.txt");

    std::ofstream file(filepath);
    file << "# libRadtran uvspec output\n";
    file << "# wavelength(nm)  edir  edn  eup  uavg\n";
    file << "380.000  0.5591  0.1464  0.0000  0.0732\n";
    file << "400.000  1.1130  0.1550  0.0000  0.0775\n";
    file << "450.000  1.4000  0.1510  0.0000  0.0755\n";
    file << "500.000  1.5290  0.1460  0.0000  0.0730\n";
    file << "550.000  1.5510  0.1110  0.0000  0.0555\n";
    file << "600.000  1.5870  0.0870  0.0000  0.0435\n";
    file << "650.000  1.4720  0.0740  0.0000  0.0370\n";
    file << "700.000  1.3140  0.0680  0.0000  0.0340\n";
    file.close();

    auto result = SpectralIO::LoadLibRadtranUvspec(filepath, 2, "nm");
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_TRUE(curve.IsValid());
    EXPECT_EQ(curve.samples.size(), 8);

    EXPECT_NEAR(curve.samples.front().first, 380.0f, 1e-3f);
    EXPECT_NEAR(curve.samples.back().first, 700.0f, 1e-3f);
    EXPECT_NEAR(curve.Evaluate(500.0f), 1.529f, 0.01f);
}

TEST_F(SpectralIOTest, LoadLibRadtranUvspecMicrometerUnit) {
    auto filepath = GetTempFilePath("uvspec_um.txt");

    std::ofstream file(filepath);
    file << "0.400  1.1130  0.1550  0.0000  0.0775\n";
    file << "0.500  1.5290  0.1460  0.0000  0.0730\n";
    file << "0.600  1.5870  0.0870  0.0000  0.0435\n";
    file.close();

    auto result = SpectralIO::LoadLibRadtranUvspec(filepath, 2, "um");
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_NEAR(curve.samples.front().first, 400.0f, 1e-3f);
    EXPECT_NEAR(curve.samples.back().first, 600.0f, 1e-3f);
}

TEST_F(SpectralIOTest, LoadLibRadtranSunAndSkyBasic) {
    auto filepath = GetTempFilePath("uvspec_sun_sky.txt");

    std::ofstream file(filepath);
    file << "# libRadtran uvspec output for sun and sky\n";
    file << "380.000  0.5591  0.1464  0.0000  0.0732\n";
    file << "450.000  1.4000  0.1510  0.0000  0.0755\n";
    file << "500.000  1.5290  0.1460  0.0000  0.0730\n";
    file << "550.000  1.5510  0.1110  0.0000  0.0555\n";
    file << "600.000  1.5870  0.0870  0.0000  0.0435\n";
    file << "700.000  1.3140  0.0680  0.0000  0.0340\n";
    file.close();

    auto result = SpectralIO::LoadLibRadtranSunAndSky(filepath, "nm");
    ASSERT_TRUE(result.has_value());

    const auto& [sunCurve, skyCurve] = result.value();

    EXPECT_TRUE(sunCurve.IsValid());
    EXPECT_TRUE(skyCurve.IsValid());
    EXPECT_EQ(sunCurve.samples.size(), 6);
    EXPECT_EQ(skyCurve.samples.size(), 6);

    EXPECT_NEAR(sunCurve.Evaluate(500.0f), 1.529f, 0.01f);
    EXPECT_NEAR(skyCurve.Evaluate(500.0f), 0.146f, 0.01f);
}

TEST_F(SpectralIOTest, LoadLibRadtranUvspecNonexistent) {
    auto filepath = GetTempFilePath("nonexistent_uvspec.txt");

    auto result = SpectralIO::LoadLibRadtranUvspec(filepath, 2, "nm");
    EXPECT_FALSE(result.has_value());

    auto result2 = SpectralIO::LoadLibRadtranSunAndSky(filepath, "nm");
    EXPECT_FALSE(result2.has_value());
}

TEST_F(SpectralIOTest, LoadLibRadtranUvspecInvalidColumn) {
    auto filepath = GetTempFilePath("uvspec_inv_col.txt");

    std::ofstream file(filepath);
    file << "400.000  1.1130  0.1550  0.0000  0.0775\n";
    file.close();

    auto result = SpectralIO::LoadLibRadtranUvspec(filepath, 11, "nm");
    EXPECT_FALSE(result.has_value());
}

TEST_F(SpectralIOTest, LoadLibRadtranUvspecInvalidUnit) {
    auto filepath = GetTempFilePath("uvspec_inv_unit.txt");

    std::ofstream file(filepath);
    file << "400.000  1.1130  0.1550  0.0000  0.0775\n";
    file.close();

    auto result = SpectralIO::LoadLibRadtranUvspec(filepath, 2, "invalid_unit");
    EXPECT_FALSE(result.has_value());
}
