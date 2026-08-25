/**
 * @file test_thermal_solve_cache.cpp
 * @brief The solve cache: round-trip fidelity, key coverage, and refusing junk
 *
 * Two properties matter more than the rest and drive most of what is below.
 *
 * A key that misses an input is the dangerous failure: the render exits 0 with
 * another scene's temperature field. So every class of input gets a case that
 * flips exactly one thing and demands the key move -- and the fields that are
 * deliberately NOT in the key get cases demanding it does not.
 *
 * An empty sun array is a different claim from an array of zeros: empty means
 * the tangent was never carried (sun_correction off), zeros mean it was carried
 * and came out flat. The shader reads those differently, so the round trip has
 * to preserve the distinction rather than normalise it.
 */

#include <gtest/gtest.h>

#include "thermal/ThermalSolveCache.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

using namespace quantiloom;
using namespace quantiloom::thermal;

namespace {

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A scratch directory that cleans itself up, so a failing case cannot leave an
/// entry behind for the next one to find.
class ThermalSolveCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("ql_thermal_cache_test_" + std::to_string(::testing::UnitTest::GetInstance()
                                                              ->current_test_info()
                                                              ->line()));
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
        std::filesystem::create_directories(dir_, ec);
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    [[nodiscard]] std::filesystem::path Entry(const char* name = "entry.qltc") const {
        return dir_ / name;
    }

    std::filesystem::path dir_;
};

/// A result with every field distinct, so a serializer that swaps two of them
/// fails rather than passing by symmetry.
ThermalResult MakeResult() {
    ThermalResult result;
    result.surfaceTemperature_K = {301.5f, 288.25f, 315.0f, 274.125f};
    result.instanceElementBase = {0u, 2u};
    result.sunSensitivity_K = {12.5f, 0.0f, 27.75f, -3.25f};
    result.sunVisibility = {1.0f, 0.0f, 0.5f, 0.25f};
    result.sunDirection = glm::vec3(0.25f, 0.75f, -0.5f);
    result.elementCount = 4;
    result.participatingElements = 3;
    result.exchangeNonZeros = 137;
    result.stepsTaken = 720;
    result.minTemperature_K = 274.125;
    result.maxTemperature_K = 315.0;
    result.meanTemperature_K = 294.71875;
    return result;
}

struct KeyFixture {
    ThermalMesh mesh;
    Vector<ThermalMaterial> materials;
    ThermalConfig config;
    glm::vec3 exchangeSun{0.3f, 0.8f, 0.1f};
    String gpu = "TestGPU|4318|8712|123456";
    String stepper = "CPU Crank-Nicolson";
    String version = "0.2.5";

    KeyFixture() {
        ThermalElement a;
        a.centroid = glm::vec3(1.0f, 2.0f, 3.0f);
        a.area_m2 = 0.5f;
        a.normal = glm::vec3(0.0f, 1.0f, 0.0f);
        a.materialId = 0;
        ThermalElement b = a;
        b.centroid = glm::vec3(4.0f, 5.0f, 6.0f);
        b.materialId = 1;
        mesh.elements = {a, b};
        mesh.instanceElementBase = {0u, 1u};

        ThermalMaterial sand;
        sand.conductivity_W_mK = 0.3f;
        sand.longwaveEmissivity = 0.92f;
        ThermalMaterial metal;
        metal.conductivity_W_mK = 45.0f;
        metal.longwaveEmissivity = 0.15f;
        materials = {sand, metal};

        config.enabled = true;
        config.time_h = 11.0;
        config.timestep_s = 60.0;
        config.nodeCount = 10;
    }

    [[nodiscard]] String Key() const {
        ThermalSolveCacheKeyInputs inputs;
        inputs.mesh = &mesh;
        inputs.solvedMaterials = &materials;
        inputs.config = &config;
        inputs.exchangeSunDirection = exchangeSun;
        inputs.gpuIdentity = gpu;
        inputs.stepperName = stepper;
        inputs.libVersion = version;
        return ComputeThermalSolveCacheKey(inputs);
    }
};

/// Truncate a file to the first n bytes.
void TruncateTo(const std::filesystem::path& file, usize bytes) {
    std::string data;
    {
        std::ifstream in(file, std::ios::binary);
        data.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out.write(data.data(), static_cast<std::streamsize>(std::min(bytes, data.size())));
}

void FlipByteAt(const std::filesystem::path& file, usize offset) {
    std::fstream io(file, std::ios::binary | std::ios::in | std::ios::out);
    io.seekg(static_cast<std::streamoff>(offset));
    char byte = 0;
    io.read(&byte, 1);
    byte = static_cast<char>(byte ^ 0xFF);
    io.seekp(static_cast<std::streamoff>(offset));
    io.write(&byte, 1);
}

constexpr const char* kKeyA =
    "0000000000000000000000000000000000000000000000000000000000000001";
constexpr const char* kKeyB =
    "0000000000000000000000000000000000000000000000000000000000000002";

}  // namespace

// ===========================================================================
// Round trip
// ===========================================================================

TEST_F(ThermalSolveCacheTest, RoundTripPreservesEveryField) {
    const ThermalResult original = MakeResult();
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, original));

    const auto loaded = LoadThermalSolveCache(Entry(), kKeyA);
    ASSERT_TRUE(loaded.has_value());

    EXPECT_EQ(loaded->surfaceTemperature_K, original.surfaceTemperature_K);
    EXPECT_EQ(loaded->instanceElementBase, original.instanceElementBase);
    EXPECT_EQ(loaded->sunSensitivity_K, original.sunSensitivity_K);
    EXPECT_EQ(loaded->sunVisibility, original.sunVisibility);
    EXPECT_EQ(loaded->sunDirection, original.sunDirection);
    EXPECT_EQ(loaded->elementCount, original.elementCount);
    EXPECT_EQ(loaded->participatingElements, original.participatingElements);
    EXPECT_EQ(loaded->exchangeNonZeros, original.exchangeNonZeros);
    EXPECT_EQ(loaded->stepsTaken, original.stepsTaken);
    // Exact, not near: these print into the gate line, which must match a
    // solved render character for character.
    EXPECT_DOUBLE_EQ(loaded->minTemperature_K, original.minTemperature_K);
    EXPECT_DOUBLE_EQ(loaded->maxTemperature_K, original.maxTemperature_K);
    EXPECT_DOUBLE_EQ(loaded->meanTemperature_K, original.meanTemperature_K);
    EXPECT_TRUE(loaded->error.empty());
}

TEST_F(ThermalSolveCacheTest, EmptySunArraysStayEmpty) {
    // sun_correction = false: the tangent was never carried.
    ThermalResult noTangent = MakeResult();
    noTangent.sunSensitivity_K.clear();
    noTangent.sunVisibility.clear();
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, noTangent));

    const auto loaded = LoadThermalSolveCache(Entry(), kKeyA);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_TRUE(loaded->sunSensitivity_K.empty())
        << "empty must not come back as a run of zeros -- the shader reads them differently";
    EXPECT_TRUE(loaded->sunVisibility.empty());
    EXPECT_EQ(loaded->surfaceTemperature_K.size(), 4u);
}

TEST_F(ThermalSolveCacheTest, ZeroFilledSunArraysStaySized) {
    ThermalResult flatTangent = MakeResult();
    flatTangent.sunSensitivity_K.assign(4, 0.0f);
    flatTangent.sunVisibility.assign(4, 0.0f);
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, flatTangent));

    const auto loaded = LoadThermalSolveCache(Entry(), kKeyA);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->sunSensitivity_K.size(), 4u);
    EXPECT_EQ(loaded->sunVisibility.size(), 4u);
}

TEST_F(ThermalSolveCacheTest, EmptyAndZeroFilledEntriesDifferOnDisk) {
    ThermalResult empty = MakeResult();
    empty.sunSensitivity_K.clear();
    empty.sunVisibility.clear();
    ThermalResult zeros = MakeResult();
    zeros.sunSensitivity_K.assign(4, 0.0f);
    zeros.sunVisibility.assign(4, 0.0f);

    ASSERT_TRUE(StoreThermalSolveCache(Entry("empty.qltc"), kKeyA, empty));
    ASSERT_TRUE(StoreThermalSolveCache(Entry("zeros.qltc"), kKeyA, zeros));
    EXPECT_NE(std::filesystem::file_size(Entry("empty.qltc")),
              std::filesystem::file_size(Entry("zeros.qltc")));
}

TEST_F(ThermalSolveCacheTest, StoreRefusesAnEmptySolve) {
    // A failed solve has no temperatures. Nothing should ever cache one.
    ThermalResult failed;
    failed.error = "no material in the scene has thermal properties";
    EXPECT_FALSE(StoreThermalSolveCache(Entry(), kKeyA, failed));
    EXPECT_FALSE(std::filesystem::exists(Entry()));
}

TEST_F(ThermalSolveCacheTest, StoreCreatesMissingDirectories) {
    const auto nested = dir_ / "a" / "b" / "entry.qltc";
    EXPECT_TRUE(StoreThermalSolveCache(nested, kKeyA, MakeResult()));
    EXPECT_TRUE(std::filesystem::exists(nested));
}

TEST_F(ThermalSolveCacheTest, StoringTwiceLeavesAReadableEntry) {
    // Stands in for two batch processes solving the same scene: the second
    // write must not leave a torn file behind.
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, MakeResult()));
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, MakeResult()));
    EXPECT_TRUE(LoadThermalSolveCache(Entry(), kKeyA).has_value());

    // And no temporaries survive.
    for (const auto& e : std::filesystem::directory_iterator(dir_)) {
        EXPECT_EQ(e.path().extension().string(), ".qltc") << e.path().string();
    }
}

// ===========================================================================
// Refusing what it should not read
// ===========================================================================

TEST_F(ThermalSolveCacheTest, MissingFileIsAQuietMiss) {
    EXPECT_FALSE(LoadThermalSolveCache(Entry("absent.qltc"), kKeyA).has_value());
}

TEST_F(ThermalSolveCacheTest, WrongKeyIsRefused) {
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, MakeResult()));
    EXPECT_FALSE(LoadThermalSolveCache(Entry(), kKeyB).has_value());
}

TEST_F(ThermalSolveCacheTest, ForeignFileIsRefused) {
    {
        std::ofstream out(Entry(), std::ios::binary);
        const std::string junk(512, 'x');
        out.write(junk.data(), static_cast<std::streamsize>(junk.size()));
    }
    EXPECT_FALSE(LoadThermalSolveCache(Entry(), kKeyA).has_value());
}

TEST_F(ThermalSolveCacheTest, EmptyFileIsRefused) {
    { std::ofstream out(Entry(), std::ios::binary); }
    EXPECT_FALSE(LoadThermalSolveCache(Entry(), kKeyA).has_value());
}

TEST_F(ThermalSolveCacheTest, TruncatedHeaderIsRefused) {
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, MakeResult()));
    TruncateTo(Entry(), 20);
    EXPECT_FALSE(LoadThermalSolveCache(Entry(), kKeyA).has_value());
}

TEST_F(ThermalSolveCacheTest, TruncatedPayloadIsRefused) {
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, MakeResult()));
    const auto full = std::filesystem::file_size(Entry());
    // Past the 64-byte trailer and into the arrays, so this exercises the
    // array reads rather than the digest read a shallower cut would hit. The
    // header still promises four elements, so a loader that trusted the counts
    // would read past the end of the file.
    TruncateTo(Entry(), static_cast<usize>(full) - 80);
    EXPECT_FALSE(LoadThermalSolveCache(Entry(), kKeyA).has_value());
}

TEST_F(ThermalSolveCacheTest, MissingDigestIsRefused) {
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, MakeResult()));
    const auto full = std::filesystem::file_size(Entry());
    TruncateTo(Entry(), static_cast<usize>(full) - 64);  // exactly the trailer
    EXPECT_FALSE(LoadThermalSolveCache(Entry(), kKeyA).has_value());
}

TEST_F(ThermalSolveCacheTest, BadMagicIsRefused) {
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, MakeResult()));
    FlipByteAt(Entry(), 0);
    EXPECT_FALSE(LoadThermalSolveCache(Entry(), kKeyA).has_value());
}

TEST_F(ThermalSolveCacheTest, BadFormatVersionIsRefused) {
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, MakeResult()));
    FlipByteAt(Entry(), 4);  // formatVersion follows magic
    EXPECT_FALSE(LoadThermalSolveCache(Entry(), kKeyA).has_value());
}

TEST_F(ThermalSolveCacheTest, CorruptedPayloadIsCaughtByTheDigest) {
    // A bit-rot flip deep in the temperatures, where nothing but the digest
    // would notice: the counts and the key still agree.
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, MakeResult()));
    const auto full = std::filesystem::file_size(Entry());
    FlipByteAt(Entry(), static_cast<usize>(full) - 70);
    EXPECT_FALSE(LoadThermalSolveCache(Entry(), kKeyA).has_value());
}

TEST_F(ThermalSolveCacheTest, CorruptedStatsAreCaughtByTheDigest) {
    // The stats are not pixels -- they are the summary line the downstream
    // gates parse to notice a scene whose subject fell out of the solve. A
    // flipped participatingElements would be read as a measurement, so the
    // digest has to cover the header and not just the arrays.
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, MakeResult()));

    // participatingElements sits after magic(4) + formatVersion(4) +
    // keyHex(64) + four counts(32) + sunDirection(12) + elementCount(4).
    constexpr usize kParticipatingOffset = 4 + 4 + 64 + 32 + 12 + 4;
    FlipByteAt(Entry(), kParticipatingOffset);
    EXPECT_FALSE(LoadThermalSolveCache(Entry(), kKeyA).has_value());
}

TEST_F(ThermalSolveCacheTest, CorruptedTemperatureRangeIsCaughtByTheDigest) {
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, MakeResult()));
    // meanTemperature_K: the last f64 before the arrays begin.
    constexpr usize kMeanOffset = 4 + 4 + 64 + 32 + 12 + 16 + 8 + 8;
    FlipByteAt(Entry(), kMeanOffset);
    EXPECT_FALSE(LoadThermalSolveCache(Entry(), kKeyA).has_value());
}

TEST_F(ThermalSolveCacheTest, RejectedEntryCanBeOverwritten) {
    // The whole recovery story: a bad entry is a miss, the caller re-solves,
    // and the store replaces it.
    {
        std::ofstream out(Entry(), std::ios::binary);
        const std::string junk(64, 'q');
        out.write(junk.data(), static_cast<std::streamsize>(junk.size()));
    }
    ASSERT_FALSE(LoadThermalSolveCache(Entry(), kKeyA).has_value());
    ASSERT_TRUE(StoreThermalSolveCache(Entry(), kKeyA, MakeResult()));
    EXPECT_TRUE(LoadThermalSolveCache(Entry(), kKeyA).has_value());
}

// ===========================================================================
// The key: what must change it
// ===========================================================================

TEST(ThermalSolveCacheKey, IsDeterministicAndWellFormed) {
    const KeyFixture fixture;
    const String key = fixture.Key();
    EXPECT_EQ(key.size(), 64u);
    EXPECT_EQ(key, KeyFixture().Key());
    EXPECT_EQ(key.find_first_not_of("0123456789abcdef"), String::npos);
}

TEST(ThermalSolveCacheKey, GeometryChangesIt) {
    KeyFixture fixture;
    const String before = fixture.Key();

    fixture.mesh.elements[1].centroid.x += 0.001f;
    EXPECT_NE(fixture.Key(), before) << "a moved element must not reuse an entry";

    fixture = KeyFixture();
    fixture.mesh.elements[0].area_m2 = 0.6f;
    EXPECT_NE(fixture.Key(), before);

    fixture = KeyFixture();
    fixture.mesh.elements[0].normal = glm::vec3(1.0f, 0.0f, 0.0f);
    EXPECT_NE(fixture.Key(), before);

    fixture = KeyFixture();
    fixture.mesh.elements[0].materialId = 1;
    EXPECT_NE(fixture.Key(), before);

    fixture = KeyFixture();
    fixture.mesh.elements.pop_back();
    EXPECT_NE(fixture.Key(), before);
}

TEST(ThermalSolveCacheKey, InstancingChangesIt) {
    KeyFixture fixture;
    const String before = fixture.Key();
    fixture.mesh.instanceElementBase = {0u, 0u};
    EXPECT_NE(fixture.Key(), before)
        << "instance bases decide which triangle reads which temperature";
}

TEST(ThermalSolveCacheKey, EveryMaterialFieldChangesIt) {
    KeyFixture base;
    const String before = base.Key();

    const std::vector<std::function<void(ThermalMaterial&)>> mutations = {
        [](ThermalMaterial& m) { m.conductivity_W_mK += 0.1f; },
        [](ThermalMaterial& m) { m.density_kg_m3 += 1.0f; },
        [](ThermalMaterial& m) { m.specificHeat_J_kgK += 1.0f; },
        [](ThermalMaterial& m) { m.thickness_m += 0.01f; },
        [](ThermalMaterial& m) { m.convection_W_m2K += 1.0f; },
        [](ThermalMaterial& m) { m.shortwaveAbsorptivity += 0.01f; },
        // The one a config-text key would miss: it comes from the material's
        // measured IR curve, not from the TOML, and is worth 0.4 K.
        [](ThermalMaterial& m) { m.longwaveEmissivity += 0.01f; },
        [](ThermalMaterial& m) { m.wetnessFactor += 0.1f; },
        [](ThermalMaterial& m) { m.interiorBoundary = InteriorBoundary::FixedTemperature; },
        [](ThermalMaterial& m) { m.interiorTemperature_K += 1.0f; },
    };

    for (usize i = 0; i < mutations.size(); ++i) {
        KeyFixture fixture;
        mutations[i](fixture.materials[0]);
        EXPECT_NE(fixture.Key(), before) << "material mutation " << i << " left the key alone";
    }
}

TEST(ThermalSolveCacheKey, EveryConfigScalarChangesIt) {
    KeyFixture base;
    const String before = base.Key();

    const std::vector<std::function<void(ThermalConfig&)>> mutations = {
        [](ThermalConfig& c) { c.time_h = 12.0; },
        [](ThermalConfig& c) { c.startTime_h = 1.0; },
        [](ThermalConfig& c) { c.timestep_s = 30.0; },
        [](ThermalConfig& c) { c.nodeCount = 12; },
        [](ThermalConfig& c) { c.initial = InitialCondition::Uniform; },
        [](ThermalConfig& c) { c.initialTemperature_K = 290.0; },
        [](ThermalConfig& c) { c.checkpointStride_h = 2.0; },
        [](ThermalConfig& c) { c.exchangeRays = 512; },
        [](ThermalConfig& c) { c.exchangeTopK = 64; },
        [](ThermalConfig& c) { c.airTemperature_K = 300.0; },
        [](ThermalConfig& c) { c.sunIrradiance_W_m2 = 900.0; },
        [](ThermalConfig& c) { c.diffuseIrradiance_W_m2 = 100.0; },
        [](ThermalConfig& c) { c.sunDirection = glm::vec3(1.0f, 0.0f, 0.0f); },
        [](ThermalConfig& c) { c.skyTemperature_K = 260.0; },
        [](ThermalConfig& c) { c.relativeHumidity = 80.0; },
        [](ThermalConfig& c) { c.sunCorrection = false; },
    };

    for (usize i = 0; i < mutations.size(); ++i) {
        KeyFixture fixture;
        mutations[i](fixture.config);
        EXPECT_NE(fixture.Key(), before) << "config mutation " << i << " left the key alone";
    }
}

TEST(ThermalSolveCacheKey, LightingSunDirectionChangesIt) {
    // Distinct from config.sunDirection: this is what the exchange precompute
    // traces sun visibility against.
    KeyFixture fixture;
    const String before = fixture.Key();
    fixture.exchangeSun = glm::vec3(-0.3f, 0.8f, 0.1f);
    EXPECT_NE(fixture.Key(), before);
}

TEST(ThermalSolveCacheKey, ProvenanceChangesIt) {
    KeyFixture fixture;
    const String before = fixture.Key();

    fixture.gpu = "OtherGPU|4318|8712|123456";
    EXPECT_NE(fixture.Key(), before) << "the exchange precompute is not driver-portable";

    fixture = KeyFixture();
    fixture.stepper = "GPU Crank-Nicolson f32";
    EXPECT_NE(fixture.Key(), before) << "f32 and f64 steppers must not share an entry";

    fixture = KeyFixture();
    fixture.version = "0.2.6";
    EXPECT_NE(fixture.Key(), before);
}

TEST(ThermalSolveCacheKey, GpuIdentityStopsAtTheFirstNul) {
    // VkPhysicalDeviceProperties::deviceName is 256 fixed bytes; the tail past
    // the name is unspecified, and the key must not depend on it.
    const String a = MakeGpuIdentity(StringView("RTX 4090\0junk", 13), 1u, 2u, 3u);
    const String b = MakeGpuIdentity(StringView("RTX 4090\0other", 14), 1u, 2u, 3u);
    EXPECT_EQ(a, b);
    EXPECT_EQ(a, "RTX 4090|1|2|3");
}

TEST(ThermalSolveCacheKey, GpuIdentityFieldsAreDistinguished) {
    EXPECT_NE(MakeGpuIdentity("dev", 1u, 2u, 3u), MakeGpuIdentity("dev", 1u, 2u, 4u));
    EXPECT_NE(MakeGpuIdentity("dev", 1u, 2u, 3u), MakeGpuIdentity("dev", 2u, 2u, 3u));
}

// ---------------------------------------------------------------------------
// The key: what must NOT change it
// ---------------------------------------------------------------------------

TEST(ThermalSolveCacheKey, OutputAndGatingFieldsAreExcluded) {
    KeyFixture fixture;
    const String before = fixture.Key();

    // An output path is not an input. (The caller bypasses the cache entirely
    // when this is set, but the key itself must not depend on it.)
    fixture.config.dumpElementsFile = "dump.csv";
    EXPECT_EQ(fixture.Key(), before);

    // We are only ever called with this true.
    fixture = KeyFixture();
    fixture.config.enabled = false;
    EXPECT_EQ(fixture.Key(), before);
}

TEST(ThermalSolveCacheKey, TwoRunsDifferingOnlyInTheSensorCollide) {
    // The reason the cache exists: the whole downstream sweep varies sensor and
    // optics, and none of it reaches the solver. Nothing sensor-shaped is in
    // ThermalSolveCacheKeyInputs at all, so two such runs are the same call.
    EXPECT_EQ(KeyFixture().Key(), KeyFixture().Key());
}

TEST(ThermalSolveCacheKey, IsEmptyWhenTheForcingFileCannotBeRead) {
    KeyFixture fixture;
    fixture.config.forcingFile = "definitely/not/here/forcing.csv";
    EXPECT_TRUE(fixture.Key().empty())
        << "an unreadable forcing file must bypass the cache, not be guessed at";
}

TEST(ThermalSolveCacheKey, ForcingContentsAreHashedNotThePath) {
    const auto dir = std::filesystem::temp_directory_path() / "ql_thermal_cache_forcing";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);

    const auto writeCsv = [&dir](const char* name, const char* body) {
        const auto path = dir / name;
        std::ofstream out(path);
        out << body;
        return path.string();
    };

    const char* seriesA = "time_h,air_k,dni,az,el,sky_k\n0,288,0,0,0,268\n12,300,900,180,60,270\n";
    const char* seriesB = "time_h,air_k,dni,az,el,sky_k\n0,288,0,0,0,268\n12,310,900,180,60,270\n";

    KeyFixture one;
    one.config.forcingFile = writeCsv("a.csv", seriesA);
    KeyFixture copyUnderAnotherName;
    copyUnderAnotherName.config.forcingFile = writeCsv("b.csv", seriesA);
    KeyFixture different;
    different.config.forcingFile = writeCsv("c.csv", seriesB);

    // Same bytes under a different name: one entry, which is what lets a batch
    // that copies its forcing file per job still share a solve.
    EXPECT_EQ(one.Key(), copyUnderAnotherName.Key());
    EXPECT_NE(one.Key(), different.Key());

    // An empty CSV is not the same as no CSV at all.
    KeyFixture emptyCsv;
    emptyCsv.config.forcingFile = writeCsv("empty.csv", "");
    KeyFixture noCsv;
    EXPECT_NE(emptyCsv.Key(), noCsv.Key());

    std::filesystem::remove_all(dir, ec);
}

// ===========================================================================
// Settings
// ===========================================================================

namespace {

void SetEnv(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

}  // namespace

TEST(ThermalSolveCacheSettingsTest, DefaultsToEnabledUnderThePlatformCacheDir) {
    SetEnv("QUANTILOOM_THERMAL_CACHE", nullptr);
    SetEnv("QUANTILOOM_THERMAL_CACHE_DIR", nullptr);

    const auto settings = ResolveThermalSolveCacheSettings();
    EXPECT_TRUE(settings.enabled);
    EXPECT_EQ(settings.directory.filename().string(), "thermal");
}

TEST(ThermalSolveCacheSettingsTest, RecognisesTheDisablingSpellings) {
    SetEnv("QUANTILOOM_THERMAL_CACHE_DIR", nullptr);
    for (const char* off : {"0", "off", "OFF", "false", "False", "no"}) {
        SetEnv("QUANTILOOM_THERMAL_CACHE", off);
        EXPECT_FALSE(ResolveThermalSolveCacheSettings().enabled) << off;
    }
    for (const char* on : {"1", "on", "true", "yes", ""}) {
        SetEnv("QUANTILOOM_THERMAL_CACHE", on);
        EXPECT_TRUE(ResolveThermalSolveCacheSettings().enabled) << "'" << on << "'";
    }
    SetEnv("QUANTILOOM_THERMAL_CACHE", nullptr);
}

TEST(ThermalSolveCacheSettingsTest, DirectoryOverrideIsTakenAsWritten) {
    SetEnv("QUANTILOOM_THERMAL_CACHE", nullptr);
    const auto scratch = (std::filesystem::temp_directory_path() / "ql_cache_override").string();
    SetEnv("QUANTILOOM_THERMAL_CACHE_DIR", scratch.c_str());

    const auto settings = ResolveThermalSolveCacheSettings();
    EXPECT_EQ(settings.directory, std::filesystem::path(scratch))
        << "no 'thermal' suffix is appended to an explicit directory";

    SetEnv("QUANTILOOM_THERMAL_CACHE_DIR", nullptr);
}
