// ============================================================================
// Quantiloom - Resource locator tests
// ============================================================================
// The lookup is what makes an installed SDK runnable from any working
// directory, and what refuses a stale share/quantiloom instead of rendering
// with it. These drive the pure resolution over temporary directories, so no
// process state (environment, module path, working directory) is involved.
// ============================================================================

#include <gtest/gtest.h>

#include "core/ResourceLocator.hpp"
#include "core/Resources.hpp"

#include <filesystem>
#include <fstream>

namespace {

namespace fs = std::filesystem;
using quantiloom::String;
using quantiloom::core::ResourceCandidates;
using quantiloom::core::ResourceOrigin;
using quantiloom::core::ResolvedResources;

constexpr const char* kDigest = "0123456789abcdef";

class ResourceLocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        static int counter = 0;
        dir = fs::temp_directory_path() /
              ("quantiloom_resource_locator_" + std::to_string(++counter));
        fs::remove_all(dir);
        fs::create_directories(dir);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    /// A root in the installed layout: shaders/spirv with one shader and the
    /// SHADERSET stamp, plus an assets directory.
    fs::path MakeRoot(const char* name, const String& stamp = kDigest) {
        const fs::path root = dir / name;
        fs::create_directories(root / "shaders" / "spirv");
        fs::create_directories(root / "assets");
        std::ofstream(root / "shaders" / "spirv" / "raygen.spv", std::ios::binary)
            << "spirv";
        std::ofstream(root / "shaders" / "spirv" / "SHADERSET") << stamp << "\n";
        return root;
    }

    ResourceCandidates Candidates() {
        ResourceCandidates candidates;
        candidates.shaderSetDigest = kDigest;
        candidates.devAssetDir = "assets";
        return candidates;
    }

    fs::path dir;
};

TEST_F(ResourceLocatorTest, HostRootWinsOverEveryOtherCandidate) {
    const fs::path host = MakeRoot("host");
    const fs::path env = MakeRoot("env");
    const fs::path library = MakeRoot("library");

    auto candidates = Candidates();
    candidates.hostRoot = host;
    candidates.envRoot = env;
    candidates.libraryRoots = {library};

    const ResolvedResources resolved = quantiloom::core::ResolveResourceCandidates(candidates);
    ASSERT_TRUE(resolved.Found()) << resolved.error;
    EXPECT_EQ(resolved.origin, ResourceOrigin::Host);
    EXPECT_EQ(resolved.root, host);
    EXPECT_EQ(resolved.shaderDir, host / "shaders" / "spirv");
    EXPECT_EQ(resolved.assetDir, host / "assets");
}

TEST_F(ResourceLocatorTest, EnvironmentRootWinsOverTheLibraryRoots) {
    const fs::path env = MakeRoot("env");
    const fs::path library = MakeRoot("library");

    auto candidates = Candidates();
    candidates.envRoot = env;
    candidates.libraryRoots = {library};

    const ResolvedResources resolved = quantiloom::core::ResolveResourceCandidates(candidates);
    ASSERT_TRUE(resolved.Found()) << resolved.error;
    EXPECT_EQ(resolved.origin, ResourceOrigin::Environment);
    EXPECT_EQ(resolved.root, env);
}

TEST_F(ResourceLocatorTest, HostRootIsExclusiveWhenItCannotBeUsed) {
    const fs::path library = MakeRoot("library");

    auto candidates = Candidates();
    candidates.hostRoot = dir / "host_missing";
    candidates.envRoot = library;
    candidates.libraryRoots = {library};

    const ResolvedResources resolved = quantiloom::core::ResolveResourceCandidates(candidates);
    EXPECT_FALSE(resolved.Found());
    EXPECT_NE(resolved.error.find("no Quantiloom resources found; probed:"), String::npos)
        << resolved.error;
    EXPECT_NE(resolved.error.find("SetResourceRoot"), String::npos) << resolved.error;
    // The valid env and library roots must not have been picked up silently.
    EXPECT_EQ(resolved.error.find(library.string()), String::npos) << resolved.error;
}

TEST_F(ResourceLocatorTest, StampMismatchNamesBothDigests) {
    const fs::path root = MakeRoot("stale", "deadbeef");

    auto candidates = Candidates();
    candidates.hostRoot = root;

    quantiloom::Vector<String> log;
    const ResolvedResources resolved = quantiloom::core::ResolveResourceCandidates(candidates, &log);
    EXPECT_FALSE(resolved.Found());
    EXPECT_NE(resolved.error.find("SHADERSET mismatch"), String::npos) << resolved.error;
    EXPECT_NE(resolved.error.find("deadbeef"), String::npos) << resolved.error;
    EXPECT_NE(resolved.error.find(kDigest), String::npos) << resolved.error;
}

TEST_F(ResourceLocatorTest, MissingStampIsRejected) {
    const fs::path root = MakeRoot("unstamped");
    fs::remove(root / "shaders" / "spirv" / "SHADERSET");

    auto candidates = Candidates();
    candidates.hostRoot = root;

    const ResolvedResources resolved = quantiloom::core::ResolveResourceCandidates(candidates);
    EXPECT_FALSE(resolved.Found());
    EXPECT_NE(resolved.error.find("SHADERSET is missing"), String::npos) << resolved.error;
}

TEST_F(ResourceLocatorTest, StaleLibraryRootDoesNotFallBackToTheWorkingDirectory) {
    const fs::path stale = MakeRoot("stale", "deadbeef");
    const fs::path devShaders = dir / "src" / "shaders";
    fs::create_directories(devShaders);
    std::ofstream(devShaders / "raygen.spv", std::ios::binary) << "spirv";

    auto candidates = Candidates();
    candidates.libraryRoots = {stale};
    candidates.devShaderDirs = {devShaders};

    const ResolvedResources resolved = quantiloom::core::ResolveResourceCandidates(candidates);
    EXPECT_FALSE(resolved.Found());
    EXPECT_NE(resolved.error.find("SHADERSET mismatch"), String::npos) << resolved.error;
}

TEST_F(ResourceLocatorTest, FirstMatchingLibraryRootWins) {
    const fs::path first = MakeRoot("first");
    const fs::path second = MakeRoot("second");

    auto candidates = Candidates();
    candidates.libraryRoots = {dir / "absent", first, second};

    const ResolvedResources resolved = quantiloom::core::ResolveResourceCandidates(candidates);
    ASSERT_TRUE(resolved.Found()) << resolved.error;
    EXPECT_EQ(resolved.origin, ResourceOrigin::Library);
    EXPECT_EQ(resolved.root, first);
}

TEST_F(ResourceLocatorTest, DevelopmentFallbackWhenNoRootExists) {
    const fs::path devShaders = dir / "src" / "shaders";
    fs::create_directories(devShaders);
    std::ofstream(devShaders / "raygen.spv", std::ios::binary) << "spirv";

    auto candidates = Candidates();
    candidates.libraryRoots = {dir / "absent"};
    candidates.devShaderDirs = {dir / "nothing_here", devShaders};

    const ResolvedResources resolved = quantiloom::core::ResolveResourceCandidates(candidates);
    ASSERT_TRUE(resolved.Found()) << resolved.error;
    EXPECT_EQ(resolved.origin, ResourceOrigin::Development);
    EXPECT_TRUE(resolved.root.empty());
    EXPECT_EQ(resolved.shaderDir, devShaders);
    EXPECT_EQ(resolved.assetDir, fs::path("assets"));
    EXPECT_EQ(resolved.ShaderPath("raygen.spv"), devShaders / "raygen.spv");
}

TEST_F(ResourceLocatorTest, NothingFoundListsTheProbedRoots) {
    auto candidates = Candidates();
    candidates.libraryRoots = {dir / "absent"};
    candidates.devShaderDirs = {dir / "also_absent"};

    const ResolvedResources resolved = quantiloom::core::ResolveResourceCandidates(candidates);
    EXPECT_FALSE(resolved.Found());
    EXPECT_NE(resolved.error.find("no Quantiloom resources found; probed:"), String::npos)
        << resolved.error;
    EXPECT_NE(resolved.error.find("absent"), String::npos) << resolved.error;
}

TEST_F(ResourceLocatorTest, PublicRootOverrideRoundTrip) {
    const fs::path root = MakeRoot("public");

    quantiloom::SetResourceRoot(root.string());
    const String resolved = quantiloom::GetResourceRoot();
    // Restore before asserting: a failed expectation must not leave an
    // override behind for the rest of the suite.
    quantiloom::SetResourceRoot("");

    EXPECT_EQ(fs::path(resolved), root);
    EXPECT_TRUE(quantiloom::core::GetResolvedResources().origin != ResourceOrigin::Host);
}

}  // namespace
