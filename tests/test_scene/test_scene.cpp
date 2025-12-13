// ============================================================================
// Quantiloom - Unit Tests for scene/Scene.hpp
// ============================================================================
// Tests cover:
// - Scene::FromConfig TOML parsing
// - Camera configuration loading
// - Spectral band parsing (MS-RT mode)
// - Wavelength range parsing (HS-OFF mode)
// - Resolution configuration
// - Scene validation
// - Metadata parsing
// ============================================================================

#include <gtest/gtest.h>
#include "scene/Scene.hpp"
#include "core/Config.hpp"
#include <filesystem>
#include <fstream>

using namespace quantiloom;

// ============================================================================
// Test Fixture with Temporary File Management
// ============================================================================

class SceneTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for test files
        tempDir = std::filesystem::temp_directory_path() / "quantiloom_scene_tests";
        std::filesystem::create_directories(tempDir);
    }

    void TearDown() override {
        // Clean up temporary files
        if (std::filesystem::exists(tempDir)) {
            std::filesystem::remove_all(tempDir);
        }
    }

    std::filesystem::path GetTempFilePath(const std::string& filename) {
        return tempDir / filename;
    }

    // Helper: Create a TOML config file
    void CreateConfigFile(const std::filesystem::path& filepath, const std::string& content) {
        std::ofstream file(filepath);
        file << content;
        file.close();
    }

    std::filesystem::path tempDir;
};

// ============================================================================
// Basic Scene Configuration Tests
// ============================================================================

TEST_F(SceneTest, MinimalValidConfig) {
    std::string configContent = R"(
[renderer]
resolution = [640, 480]

[camera]
position = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
fov = 45.0

[spectral]
range_nm = [400.0, 700.0]
step_nm = 5.0

[scene]
name = "Test Scene"
)";

    auto filepath = GetTempFilePath("minimal_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    EXPECT_TRUE(scene.IsValid());
    EXPECT_EQ(scene.width, 640);
    EXPECT_EQ(scene.height, 480);
    EXPECT_EQ(scene.name, "Test Scene");
    EXPECT_NEAR(scene.lambda_min, 400.0f, 1e-5f);
    EXPECT_NEAR(scene.lambda_max, 700.0f, 1e-5f);
    EXPECT_NEAR(scene.delta_lambda, 5.0f, 1e-5f);
}

TEST_F(SceneTest, CameraConfiguration) {
    std::string configContent = R"(
[renderer]
resolution = [800, 600]

[camera]
position = [10.0, 5.0, 15.0]
look_at = [0.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
fov = 60.0

[spectral]
range_nm = [400.0, 700.0]
step_nm = 10.0
)";

    auto filepath = GetTempFilePath("camera_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    glm::vec3 position = scene.camera.GetPosition();
    glm::vec3 lookAt = scene.camera.GetLookAt();
    glm::vec3 up = scene.camera.GetUp();

    EXPECT_NEAR(position.x, 10.0f, 1e-5f);
    EXPECT_NEAR(position.y, 5.0f, 1e-5f);
    EXPECT_NEAR(position.z, 15.0f, 1e-5f);

    EXPECT_NEAR(lookAt.x, 0.0f, 1e-5f);
    EXPECT_NEAR(lookAt.y, 0.0f, 1e-5f);
    EXPECT_NEAR(lookAt.z, 0.0f, 1e-5f);

    // Camera orthonormalizes the up vector, so it may not be exactly (0,1,0)
    // Just check that the up vector is reasonable (unit length, mostly upward)
    f32 upLength = glm::length(up);
    EXPECT_NEAR(upLength, 1.0f, 1e-5f) << "Up vector should be normalized";
    EXPECT_GT(up.y, 0.5f) << "Up vector should point mostly upward";
}

TEST_F(SceneTest, ResolutionConfiguration) {
    std::string configContent = R"(
[renderer]
resolution = [1920, 1080]

[camera]
position = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
fov = 45.0

[spectral]
range_nm = [400.0, 700.0]
step_nm = 5.0
)";

    auto filepath = GetTempFilePath("resolution_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    EXPECT_EQ(scene.width, 1920);
    EXPECT_EQ(scene.height, 1080);

    // Aspect ratio should be updated
    f32 expectedAspect = 1920.0f / 1080.0f;
    EXPECT_NEAR(scene.camera.GetAspectRatio(), expectedAspect, 1e-5f);
}

// ============================================================================
// Spectral Configuration Tests
// ============================================================================

TEST_F(SceneTest, HSOffModeConfiguration) {
    std::string configContent = R"(
[renderer]
resolution = [640, 480]

[camera]
position = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
fov = 45.0

[spectral]
range_nm = [380.0, 760.0]
step_nm = 5.0
)";

    auto filepath = GetTempFilePath("hsoff_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    EXPECT_NEAR(scene.lambda_min, 380.0f, 1e-5f);
    EXPECT_NEAR(scene.lambda_max, 760.0f, 1e-5f);
    EXPECT_NEAR(scene.delta_lambda, 5.0f, 1e-5f);
    EXPECT_TRUE(scene.bands.empty());  // No bands in HS-OFF mode
}

TEST_F(SceneTest, MSRTModeBandsConfiguration) {
    std::string configContent = R"(
[renderer]
resolution = [640, 480]

[camera]
position = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
fov = 45.0

[[spectral.bands]]
name = "Blue"
center_nm = 470.0
fwhm_nm = 50.0

[[spectral.bands]]
name = "Green"
center_nm = 550.0
fwhm_nm = 50.0

[[spectral.bands]]
name = "Red"
center_nm = 650.0
fwhm_nm = 50.0
)";

    auto filepath = GetTempFilePath("msrt_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    EXPECT_EQ(scene.bands.size(), 3);

    EXPECT_EQ(scene.bands[0].name, "Blue");
    EXPECT_NEAR(scene.bands[0].center_nm, 470.0f, 1e-5f);
    EXPECT_NEAR(scene.bands[0].fwhm_nm, 50.0f, 1e-5f);

    EXPECT_EQ(scene.bands[1].name, "Green");
    EXPECT_NEAR(scene.bands[1].center_nm, 550.0f, 1e-5f);

    EXPECT_EQ(scene.bands[2].name, "Red");
    EXPECT_NEAR(scene.bands[2].center_nm, 650.0f, 1e-5f);
}

TEST_F(SceneTest, MSRTModeMultipleBands) {
    std::string configContent = R"(
[renderer]
resolution = [640, 480]

[camera]
position = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
fov = 45.0

[[spectral.bands]]
name = "VIS_450"
center_nm = 450.0
fwhm_nm = 40.0

[[spectral.bands]]
name = "VIS_550"
center_nm = 550.0
fwhm_nm = 40.0

[[spectral.bands]]
name = "NIR_850"
center_nm = 850.0
fwhm_nm = 60.0

[[spectral.bands]]
name = "SWIR_1600"
center_nm = 1600.0
fwhm_nm = 100.0
)";

    auto filepath = GetTempFilePath("multispectral_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    EXPECT_EQ(scene.bands.size(), 4);

    // Verify all bands are valid
    for (const auto& band : scene.bands) {
        EXPECT_TRUE(band.IsValid());
        EXPECT_GT(band.center_nm, 0.0f);
        EXPECT_GT(band.fwhm_nm, 0.0f);
    }
}

// ============================================================================
// Scene Metadata Tests
// ============================================================================

TEST_F(SceneTest, SceneNameAndDescription) {
    std::string configContent = R"(
[renderer]
resolution = [640, 480]

[camera]
position = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
fov = 45.0

[spectral]
range_nm = [400.0, 700.0]
step_nm = 5.0

[scene]
name = "Cornell Box Test"
description = "Test scene for path tracing validation"
)";

    auto filepath = GetTempFilePath("metadata_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    EXPECT_EQ(scene.name, "Cornell Box Test");
    EXPECT_EQ(scene.description, "Test scene for path tracing validation");
}

TEST_F(SceneTest, DefaultSceneName) {
    std::string configContent = R"(
[renderer]
resolution = [640, 480]

[camera]
position = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
fov = 45.0

[spectral]
range_nm = [400.0, 700.0]
step_nm = 5.0
)";

    auto filepath = GetTempFilePath("default_name_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    EXPECT_EQ(scene.name, "Untitled Scene");
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST_F(SceneTest, ValidationSuccess) {
    std::string configContent = R"(
[renderer]
resolution = [640, 480]

[camera]
position = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
fov = 45.0

[spectral]
range_nm = [400.0, 700.0]
step_nm = 5.0
)";

    auto filepath = GetTempFilePath("valid_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    EXPECT_TRUE(scene.IsValid());
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SceneTest, SmallResolution) {
    std::string configContent = R"(
[renderer]
resolution = [1, 1]

[camera]
position = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
fov = 45.0

[spectral]
range_nm = [400.0, 700.0]
step_nm = 5.0
)";

    auto filepath = GetTempFilePath("small_res_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    EXPECT_EQ(scene.width, 1);
    EXPECT_EQ(scene.height, 1);
    EXPECT_TRUE(scene.IsValid());
}

TEST_F(SceneTest, LargeResolution) {
    std::string configContent = R"(
[renderer]
resolution = [7680, 4320]

[camera]
position = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
fov = 45.0

[spectral]
range_nm = [400.0, 700.0]
step_nm = 5.0
)";

    auto filepath = GetTempFilePath("large_res_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    EXPECT_EQ(scene.width, 7680);
    EXPECT_EQ(scene.height, 4320);
}

TEST_F(SceneTest, WideSpectralRange) {
    std::string configContent = R"(
[renderer]
resolution = [640, 480]

[camera]
position = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
fov = 45.0

[spectral]
range_nm = [300.0, 2500.0]
step_nm = 10.0
)";

    auto filepath = GetTempFilePath("wide_spectral_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    EXPECT_NEAR(scene.lambda_min, 300.0f, 1e-5f);
    EXPECT_NEAR(scene.lambda_max, 2500.0f, 1e-5f);
}

TEST_F(SceneTest, FineSpectralResolution) {
    std::string configContent = R"(
[renderer]
resolution = [640, 480]

[camera]
position = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
fov = 45.0

[spectral]
range_nm = [400.0, 700.0]
step_nm = 1.0
)";

    auto filepath = GetTempFilePath("fine_spectral_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    EXPECT_NEAR(scene.delta_lambda, 1.0f, 1e-5f);
}

// ============================================================================
// Default Values Tests
// ============================================================================

TEST_F(SceneTest, DefaultCameraUp) {
    std::string configContent = R"(
[renderer]
resolution = [640, 480]

[camera]
position = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
fov = 45.0

[spectral]
range_nm = [400.0, 700.0]
step_nm = 5.0
)";

    auto filepath = GetTempFilePath("default_up_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    glm::vec3 up = scene.camera.GetUp();

    // Default up vector should be (0, 1, 0)
    EXPECT_NEAR(up.x, 0.0f, 1e-5f);
    EXPECT_NEAR(up.y, 1.0f, 1e-5f);
    EXPECT_NEAR(up.z, 0.0f, 1e-5f);
}

TEST_F(SceneTest, DefaultSpectralStep) {
    std::string configContent = R"(
[renderer]
resolution = [640, 480]

[camera]
position = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
fov = 45.0

[spectral]
range_nm = [400.0, 700.0]
)";

    auto filepath = GetTempFilePath("default_step_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    // Default step should be 5.0nm based on Scene.cpp
    EXPECT_NEAR(scene.delta_lambda, 5.0f, 1e-5f);
}

// ============================================================================
// Realistic Configuration Tests
// ============================================================================

TEST_F(SceneTest, RealisticCornellBoxConfig) {
    std::string configContent = R"(
[renderer]
resolution = [1024, 1024]

[camera]
position = [0.0, 1.0, 3.9]
look_at = [0.0, 1.0, 0.0]
up = [0.0, 1.0, 0.0]
fov = 40.0

[spectral]
range_nm = [380.0, 760.0]
step_nm = 5.0

[scene]
name = "Cornell Box"
description = "Classic path tracing validation scene"
)";

    auto filepath = GetTempFilePath("cornell_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    EXPECT_TRUE(scene.IsValid());
    EXPECT_EQ(scene.name, "Cornell Box");
}

TEST_F(SceneTest, RealisticMultispectralSatelliteConfig) {
    std::string configContent = R"(
[renderer]
resolution = [4096, 4096]

[camera]
position = [0.0, 600000.0, 0.0]
look_at = [0.0, 0.0, 0.0]
fov = 10.0

[[spectral.bands]]
name = "Coastal_Aerosol"
center_nm = 443.0
fwhm_nm = 20.0

[[spectral.bands]]
name = "Blue"
center_nm = 482.0
fwhm_nm = 60.0

[[spectral.bands]]
name = "Green"
center_nm = 561.0
fwhm_nm = 60.0

[[spectral.bands]]
name = "Red"
center_nm = 655.0
fwhm_nm = 30.0

[[spectral.bands]]
name = "NIR"
center_nm = 865.0
fwhm_nm = 30.0

[[spectral.bands]]
name = "SWIR1"
center_nm = 1610.0
fwhm_nm = 80.0

[[spectral.bands]]
name = "SWIR2"
center_nm = 2190.0
fwhm_nm = 180.0

[scene]
name = "Satellite Observation"
description = "7-band multispectral satellite configuration"
)";

    auto filepath = GetTempFilePath("satellite_config.toml");
    CreateConfigFile(filepath, configContent);

    auto config = Config::Load(filepath.string());
    ASSERT_TRUE(config.has_value());

    auto scene_result = Scene::FromConfig(*config);
    ASSERT_TRUE(scene_result.has_value());

    Scene& scene = *scene_result;
    EXPECT_TRUE(scene.IsValid());
    EXPECT_EQ(scene.bands.size(), 7);
    EXPECT_EQ(scene.name, "Satellite Observation");
}
