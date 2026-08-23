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
#include "io/SpectralBasisLoader.hpp"
#include "core/SpectralData.hpp"
#include "core/Blackbody.hpp"
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cmath>

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

// ============================================================================
// ReconstructBasisCurve
// ============================================================================
// The host-facing half of the NMF database: a material browser needs a curve
// without loading a scene, and must get the same curve the renderer would.

TEST_F(SpectralIOTest, ReconstructBasisCurveMatchesTheLoaderItWraps) {
    const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
    const auto basis = root / "assets" / "spectral" / "quantiloom_basis_v3_usgs.qlbin";
    const auto json = root / "assets" / "spectral" / "quantiloom_materials_usgs.json";
    ASSERT_TRUE(std::filesystem::exists(basis)) << basis.string();
    ASSERT_TRUE(std::filesystem::exists(json)) << json.string();

    SpectralBasisLoader reference;
    ASSERT_TRUE(reference.Load(basis, json));
    const auto names = reference.GetMaterialNames();
    ASSERT_FALSE(names.empty());
    const SpectralCurve expected = reference.ReconstructCurve(names[0], "VIS");
    ASSERT_FALSE(expected.samples.empty());

    auto result = SpectralIO::ReconstructBasisCurve(basis, json, names[0], "VIS");
    ASSERT_TRUE(result.has_value()) << result.error();

    // Same curve, sample for sample -- a browser that previewed something
    // other than what renders would be worse than no preview.
    ASSERT_EQ(result.value().samples.size(), expected.samples.size());
    for (size_t i = 0; i < expected.samples.size(); ++i) {
        EXPECT_FLOAT_EQ(result.value().samples[i].first, expected.samples[i].first);
        EXPECT_FLOAT_EQ(result.value().samples[i].second, expected.samples[i].second);
    }
}

TEST_F(SpectralIOTest, ReconstructBasisCurveFallsBackToPartialName) {
    const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
    const auto basis = root / "assets" / "spectral" / "quantiloom_basis_v3_usgs.qlbin";
    const auto json = root / "assets" / "spectral" / "quantiloom_materials_usgs.json";
    ASSERT_TRUE(std::filesystem::exists(basis));

    SpectralBasisLoader reference;
    ASSERT_TRUE(reference.Load(basis, json));
    const auto names = reference.GetMaterialNames();
    ASSERT_FALSE(names.empty());

    // A substring of a real name must resolve, the same way a scene's
    // quantiloom_material_ref does -- exact first, then substring.
    const String fragment = names[0].substr(0, names[0].size() / 2);
    ASSERT_FALSE(fragment.empty());
    auto result = SpectralIO::ReconstructBasisCurve(basis, json, fragment, "VIS");
    EXPECT_TRUE(result.has_value()) << result.error();
}

TEST_F(SpectralIOTest, ReconstructBasisCurveReportsWhatWentWrong) {
    const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
    const auto basis = root / "assets" / "spectral" / "quantiloom_basis_v3_usgs.qlbin";
    const auto json = root / "assets" / "spectral" / "quantiloom_materials_usgs.json";
    ASSERT_TRUE(std::filesystem::exists(basis));

    // A name nothing matches, a band this database does not carry, and a
    // missing file are three different failures and each says which.
    EXPECT_FALSE(
        SpectralIO::ReconstructBasisCurve(basis, json, "no such material at all", "VIS")
            .has_value());
    // USGS stops at 2.5 um, so LWIR is absent by construction.
    SpectralBasisLoader reference;
    ASSERT_TRUE(reference.Load(basis, json));
    const auto names = reference.GetMaterialNames();
    ASSERT_FALSE(names.empty());
    EXPECT_FALSE(
        SpectralIO::ReconstructBasisCurve(basis, json, names[0], "LWIR").has_value());
    EXPECT_FALSE(
        SpectralIO::ReconstructBasisCurve(root / "nope.qlbin", json, names[0], "VIS")
            .has_value());
    EXPECT_FALSE(SpectralIO::ReconstructBasisCurve(basis, json, "", "VIS").has_value());
}

// ============================================================================
// Emission spectra
// ============================================================================
// The point of these is not that a table loads. It is that the two properties
// the emission path depends on hold: a built-in token always resolves to the
// same lamp, and a lamp's spectrum stops where its measurement stops.

TEST_F(SpectralIOTest, BuiltinEmissionTokensAllResolve) {
    // Every token the listing advertises must load, or a UI populated from that
    // listing offers the user a choice that fails.
    for (const auto& info : SpectralIO::BuiltinEmissionSpectra()) {
        const String token =
            (info.token == "blackbody_<T>k") ? "blackbody_3000k" : info.token;
        auto curve = SpectralIO::LoadEmissionSpectrum(token, {});
        ASSERT_TRUE(curve.has_value()) << token << ": " << curve.error();
        ASSERT_GE(curve.value().samples.size(), 2u) << token;
        EXPECT_NEAR(curve.value().samples.front().first, info.lambdaMinNm, 1.0f) << token;
        EXPECT_NEAR(curve.value().samples.back().first, info.lambdaMaxNm, 1.0f) << token;
        for (const auto& [lambda, value] : curve.value().samples) {
            EXPECT_GE(value, 0.0f) << token << " at " << lambda << " nm";
        }
    }
}

TEST_F(SpectralIOTest, EmissionTokensAreCaseInsensitiveAndAliased) {
    auto lower = SpectralIO::LoadEmissionSpectrum("cie_f7", {});
    auto upper = SpectralIO::LoadEmissionSpectrum("CIE_F7", {});
    ASSERT_TRUE(lower.has_value());
    ASSERT_TRUE(upper.has_value());
    EXPECT_EQ(lower.value().samples, upper.value().samples);

    // halogen is documented as an alias, so it must not drift from what it
    // aliases -- a lamp that changes when you spell it differently is worse
    // than no lamp at all.
    auto halogen = SpectralIO::LoadEmissionSpectrum("halogen", {});
    auto planck = SpectralIO::LoadEmissionSpectrum("blackbody_3000k", {});
    ASSERT_TRUE(halogen.has_value());
    ASSERT_TRUE(planck.has_value());
    EXPECT_EQ(halogen.value().samples, planck.value().samples);
}

TEST_F(SpectralIOTest, FluorescentSpectraCarryTheMercuryLines) {
    // The whole reason a fluorescent lamp cannot be an RGB triple. If these
    // spikes are ever smoothed away by a resampling change, this fails.
    // Mercury emits at 405, 436 and 546 nm; CIE tabulates on a 5 nm grid, so
    // the lines land in the 405, 435 and 545 nm bins. Compared against the mean
    // of the two adjacent bins, which is the phosphor continuum under them.
    // Measured margins across these four lamps are 2.9-8.8x at 405, 3.3-4.2x at
    // 435 and 1.9-2.0x at 545; 1.5x is below all of them and well above 1.
    for (const char* lamp : {"cie_f1", "cie_f2", "cie_f7", "cie_f11"}) {
        auto fl = SpectralIO::LoadEmissionSpectrum(lamp, {});
        ASSERT_TRUE(fl.has_value()) << lamp;
        const auto& c = fl.value();
        for (const f32 line : {405.0f, 435.0f, 545.0f}) {
            const f32 continuum = 0.5f * (c.Evaluate(line - 5.0f) + c.Evaluate(line + 5.0f));
            EXPECT_GT(c.Evaluate(line), 1.5f * continuum)
                << lamp << " has no mercury line at " << line << " nm";
        }
    }
}

TEST_F(SpectralIOTest, IlluminantAMatchesItsDefiningEquation) {
    // CIE 015:2018 defines illuminant A by an equation, so this is the one
    // built-in that can be checked against its own standard rather than against
    // a copy of the table it came from.
    auto a = SpectralIO::LoadEmissionSpectrum("illuminant_a", {});
    ASSERT_TRUE(a.has_value());
    EXPECT_NEAR(a.value().Evaluate(560.0f), 100.0f, 0.05f);  // normalisation point
    for (const f32 lambda : {300.0f, 400.0f, 560.0f, 700.0f, 830.0f}) {
        const f64 shape = std::pow(560.0 / lambda, 5.0);
        const f64 expected = 100.0 * shape *
            ((std::exp(1.435e7 / (2848.0 * 560.0)) - 1.0) /
             (std::exp(1.435e7 / (2848.0 * lambda)) - 1.0));
        EXPECT_NEAR(a.value().Evaluate(lambda), static_cast<f32>(expected),
                    static_cast<f32>(expected) * 1e-4f) << lambda << " nm";
    }
}

TEST_F(SpectralIOTest, BlackbodyEmissionIsAbsoluteAndPeaksWhereWienSaysItShould) {
    // Absolute, unlike every other built-in: this one is Planck's law, not a
    // relative distribution, so the magnitude is a claim about W/m2/sr/nm.
    auto bb = SpectralIO::LoadEmissionSpectrum("blackbody_3000k", {});
    ASSERT_TRUE(bb.has_value());
    const auto& c = bb.value();

    const f32 peakNm = 2.897771955e6f / 3000.0f;  // Wien, ~966 nm
    EXPECT_GT(c.Evaluate(peakNm), c.Evaluate(peakNm * 0.5f));
    EXPECT_GT(c.Evaluate(peakNm), c.Evaluate(peakNm * 2.0f));
    EXPECT_NEAR(c.Evaluate(peakNm),
                static_cast<f32>(blackbody::SpectralRadiancePerNm(peakNm, 3000.0)),
                static_cast<f32>(blackbody::SpectralRadiancePerNm(peakNm, 3000.0)) * 1e-3f);

    // A tungsten filament's peak is in the NIR band, which is the reason NIR
    // reads a bound emission curve at all.
    EXPECT_GT(peakNm, 930.0f);
    EXPECT_LT(peakNm, 1200.0f);
}

TEST_F(SpectralIOTest, EmissionSpectraStopWhereTheirMeasurementStops) {
    // Not a limitation to be worked around -- it is what the shader's
    // zero-outside rule is defined against. A fluorescent table has nothing to
    // say about SWIR, and the span is how the renderer knows that.
    auto fl = SpectralIO::LoadEmissionSpectrum("cie_f7", {});
    ASSERT_TRUE(fl.has_value());
    EXPECT_LE(fl.value().samples.back().first, 780.0f);

    auto d65 = SpectralIO::LoadEmissionSpectrum("d65", {});
    ASSERT_TRUE(d65.has_value());
    EXPECT_LE(d65.value().samples.back().first, 780.0f);

    // The blackbody family is the deliberate exception: it is a formula, valid
    // wherever it is evaluated, so it spans every band the renderer has.
    auto bb = SpectralIO::LoadEmissionSpectrum("blackbody_2856k", {});
    ASSERT_TRUE(bb.has_value());
    EXPECT_LE(bb.value().samples.front().first, 400.0f);
    EXPECT_GE(bb.value().samples.back().first, 12000.0f);
}

TEST_F(SpectralIOTest, UnknownEmissionTokenSaysWhatTheTokensAre) {
    // A misspelt token would otherwise fail as "file not found", which sends
    // the reader looking for a file they never meant to write.
    auto bad = SpectralIO::LoadEmissionSpectrum("cie_f99", {});
    ASSERT_FALSE(bad.has_value());
    EXPECT_NE(bad.error().find("built-in"), String::npos) << bad.error();
    EXPECT_NE(bad.error().find("blackbody_<T>k"), String::npos) << bad.error();

    EXPECT_FALSE(SpectralIO::LoadEmissionSpectrum("blackbody_0k", {}).has_value());
    EXPECT_FALSE(SpectralIO::LoadEmissionSpectrum("blackbody_abck", {}).has_value());
}

TEST_F(SpectralIOTest, EmissionSpectrumLoadsFromAFileWithAChosenColumn) {
    // The path a user with a calibrated lamp measurement takes.
    const auto path = tempDir / "lamp.csv";
    std::ofstream(path) << "400 1.0 7.0\n500 2.0 8.0\n600 3.0 9.0\n";

    auto second = SpectralIO::LoadEmissionSpectrum("lamp.csv", tempDir, 2);
    ASSERT_TRUE(second.has_value()) << second.error();
    EXPECT_FLOAT_EQ(second.value().Evaluate(500.0f), 2.0f);

    auto third = SpectralIO::LoadEmissionSpectrum("lamp.csv", tempDir, 3);
    ASSERT_TRUE(third.has_value()) << third.error();
    EXPECT_FLOAT_EQ(third.value().Evaluate(500.0f), 8.0f);
}
