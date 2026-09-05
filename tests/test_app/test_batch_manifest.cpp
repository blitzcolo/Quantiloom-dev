// ============================================================================
// Quantiloom - Unit Tests for src/app/BatchManifest.hpp
// ============================================================================
// A manifest line is the only thing standing between "render this sequence"
// and a directory of frames that are all subtly the same. The failure worth
// testing for is not a crash: it is a typo that layers nothing, renders the
// unmodified scene, and produces a sequence whose frames differ by less than
// anyone checks.
// ============================================================================

#include <gtest/gtest.h>

#include "BatchManifest.hpp"

#include <filesystem>
#include <fstream>

using namespace quantiloom;
using namespace quantiloom::app;

namespace {

class BatchManifestTest : public ::testing::Test {
protected:
    void SetUp() override {
        testDir = std::filesystem::temp_directory_path() / "quantiloom_batch_manifest";
        std::filesystem::create_directories(testDir);
    }

    void TearDown() override {
        if (std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
    }

    std::filesystem::path Write(const std::string& name, const std::string& contents) {
        const auto path = testDir / name;
        std::ofstream file(path);
        file << contents;
        return path;
    }

    /// A config with something in it for an override to layer onto.
    Config BaseConfig() {
        auto parsed = Config::Parse("[renderer]\nspp = 4\noutput = \"base.exr\"\n"
                                    "[material_overrides.Plate]\nir_temperature_k = 300.0\n");
        EXPECT_TRUE(parsed.has_value());
        return parsed.value();
    }

    std::filesystem::path testDir;
};

}  // namespace

// ============================================================================
// The old format still reads
// ============================================================================

TEST_F(BatchManifestTest, BarePathsReadAsTheyAlwaysDid) {
    const auto manifest = Write("m.txt", "# a comment\n\na.toml\n  b.toml  \n");
    auto entries = ParseManifest(manifest);
    ASSERT_TRUE(entries.has_value()) << entries.error();
    ASSERT_EQ(entries.value().size(), 2u);

    EXPECT_EQ(entries.value()[0].configPath, (testDir / "a.toml").lexically_normal());
    EXPECT_EQ(entries.value()[1].configPath, (testDir / "b.toml").lexically_normal());
    EXPECT_TRUE(entries.value()[0].sidecars.empty());
    EXPECT_TRUE(entries.value()[0].inlineOverrides.empty());
}

TEST_F(BatchManifestTest, RelativePathsResolveAgainstTheManifestNotTheCwd) {
    // So a list can sit beside the configs it names and travel with them.
    const auto manifest = Write("m.txt", "scenes/a.toml | @over.toml\n");
    auto entries = ParseManifest(manifest);
    ASSERT_TRUE(entries.has_value()) << entries.error();

    EXPECT_EQ(entries.value()[0].configPath,
              (testDir / "scenes" / "a.toml").lexically_normal());
    ASSERT_EQ(entries.value()[0].sidecars.size(), 1u);
    EXPECT_EQ(entries.value()[0].sidecars[0], (testDir / "over.toml").lexically_normal());
}

TEST_F(BatchManifestTest, AnEmptyManifestIsAnError) {
    const auto manifest = Write("m.txt", "# nothing but comments\n\n");
    EXPECT_FALSE(ParseManifest(manifest).has_value());
}

// ============================================================================
// Inline overrides
// ============================================================================

TEST_F(BatchManifestTest, InlinePairsBecomeATomlDocument) {
    const auto manifest =
        Write("m.txt", "a.toml | renderer.spp=64 renderer.output=\"frame.exr\"\n");
    auto entries = ParseManifest(manifest);
    ASSERT_TRUE(entries.has_value()) << entries.error();

    auto merged = ApplyEntryOverrides(BaseConfig(), entries.value()[0]);
    ASSERT_TRUE(merged.has_value()) << merged.error();
    EXPECT_EQ(merged.value().Get<u32>("renderer.spp", 0), 64u);
    EXPECT_EQ(merged.value().GetString("renderer.output", ""), "frame.exr");
}

TEST_F(BatchManifestTest, QuotedValuesMayContainSpaces) {
    const auto manifest = Write("m.txt", "a.toml | renderer.output=\"my frame.exr\"\n");
    auto entries = ParseManifest(manifest);
    ASSERT_TRUE(entries.has_value()) << entries.error();

    auto merged = ApplyEntryOverrides(BaseConfig(), entries.value()[0]);
    ASSERT_TRUE(merged.has_value()) << merged.error();
    EXPECT_EQ(merged.value().GetString("renderer.output", ""), "my frame.exr");
}

TEST_F(BatchManifestTest, ATypoIsRefusedRatherThanRenderingTheUnmodifiedScene) {
    // The failure this test exists for: a line that layers nothing produces a
    // frame identical to its neighbours, and nothing about the output says so.
    for (const char* bad : {"a.toml | renderer.spp\n",         // no value
                            "a.toml | =64\n",                  // no key
                            "a.toml | renderer.spp=not-toml\n" // not a TOML value
                           }) {
        const auto manifest = Write("m.txt", bad);
        EXPECT_FALSE(ParseManifest(manifest).has_value()) << "accepted: " << bad;
    }
}

TEST_F(BatchManifestTest, OverridesWithoutAConfigAreRefused) {
    const auto manifest = Write("m.txt", "| renderer.spp=64\n");
    EXPECT_FALSE(ParseManifest(manifest).has_value());
}

// ============================================================================
// Layering
// ============================================================================

TEST_F(BatchManifestTest, InlinePairsWinOverSidecars) {
    // Most specific last: the line's own text beats a file it names.
    Write("over.toml", "[renderer]\nspp = 16\n");
    const auto manifest = Write("m.txt", "a.toml | @over.toml renderer.spp=99\n");
    auto entries = ParseManifest(manifest);
    ASSERT_TRUE(entries.has_value()) << entries.error();

    auto merged = ApplyEntryOverrides(BaseConfig(), entries.value()[0]);
    ASSERT_TRUE(merged.has_value()) << merged.error();
    EXPECT_EQ(merged.value().Get<u32>("renderer.spp", 0), 99u);
}

TEST_F(BatchManifestTest, LaterSidecarsWinOverEarlierOnes) {
    Write("first.toml", "[renderer]\nspp = 16\n");
    Write("second.toml", "[renderer]\nspp = 32\n");
    const auto manifest = Write("m.txt", "a.toml | @first.toml @second.toml\n");
    auto entries = ParseManifest(manifest);
    ASSERT_TRUE(entries.has_value()) << entries.error();

    auto merged = ApplyEntryOverrides(BaseConfig(), entries.value()[0]);
    ASSERT_TRUE(merged.has_value()) << merged.error();
    EXPECT_EQ(merged.value().Get<u32>("renderer.spp", 0), 32u);
}

TEST_F(BatchManifestTest, AnAbsentKeyIsLeftAlone) {
    const auto manifest = Write("m.txt", "a.toml | renderer.spp=64\n");
    auto entries = ParseManifest(manifest);
    ASSERT_TRUE(entries.has_value()) << entries.error();

    auto merged = ApplyEntryOverrides(BaseConfig(), entries.value()[0]);
    ASSERT_TRUE(merged.has_value()) << merged.error();
    EXPECT_EQ(merged.value().GetString("renderer.output", ""), "base.exr");
}

TEST_F(BatchManifestTest, AMissingSidecarIsReportedRatherThanIgnored) {
    const auto manifest = Write("m.txt", "a.toml | @no_such_file.toml\n");
    auto entries = ParseManifest(manifest);
    ASSERT_TRUE(entries.has_value()) << entries.error();

    // Parsing does not touch the file; applying does, and that is where a
    // missing one has to be an error rather than a silently unmodified scene.
    EXPECT_FALSE(ApplyEntryOverrides(BaseConfig(), entries.value()[0]).has_value());
}

// ============================================================================
// What a temperature sequence actually relies on
// ============================================================================

TEST_F(BatchManifestTest, MaterialOverridesMergeByNameWhileMaterialsArraysReplace) {
    // The reason [material_overrides] exists. Config::MergedWith replaces
    // arrays whole -- array elements carry no identity to pair them by -- so a
    // per-frame [[materials]] override would delete every other material the
    // scene names. The table form merges key by key, which is what a sequence
    // needs.
    auto base = Config::Parse("[[materials]]\nname = \"A\"\nir_temperature_k = 300.0\n"
                              "[[materials]]\nname = \"B\"\nir_temperature_k = 310.0\n"
                              "[material_overrides.A]\nir_emissivity = 0.9\n");
    ASSERT_TRUE(base.has_value());

    const auto manifest =
        Write("m.txt", "a.toml | material_overrides.A.ir_temperature_k=320.0\n");
    auto entries = ParseManifest(manifest);
    ASSERT_TRUE(entries.has_value()) << entries.error();

    auto merged = ApplyEntryOverrides(base.value(), entries.value()[0]);
    ASSERT_TRUE(merged.has_value()) << merged.error();

    // The override took, the sibling key beside it survived, and neither
    // [[materials]] entry was touched.
    EXPECT_FLOAT_EQ(merged.value().GetFloat("material_overrides.A.ir_temperature_k", 0.0f),
                    320.0f);
    EXPECT_FLOAT_EQ(merged.value().GetFloat("material_overrides.A.ir_emissivity", 0.0f), 0.9f);
    EXPECT_EQ(merged.value().GetTableArray("materials").size(), 2u);
}

TEST_F(BatchManifestTest, AQuotedKeyReachesAMaterialAMergeRenamed) {
    // Two [[models]] that bring a material of the same name leave the second
    // one called "<model>/<name>", and TOML spells a key with a slash in it
    // between quotes. The tokeniser keeps the quotes and hands the whole token
    // to the TOML parser, so the dotted form works with no special case here
    // -- which is what this pins, since the rename rule would be useless from
    // a manifest otherwise.
    auto base = Config::Parse("[material_overrides.\"block/Material\"]\n"
                              "ir_temperature_k = 300.0\nir_emissivity = 0.85\n");
    ASSERT_TRUE(base.has_value());

    const auto manifest = Write(
        "m.txt", "a.toml | material_overrides.\"block/Material\".ir_temperature_k=320.0\n");
    auto entries = ParseManifest(manifest);
    ASSERT_TRUE(entries.has_value()) << entries.error();

    auto merged = ApplyEntryOverrides(base.value(), entries.value()[0]);
    ASSERT_TRUE(merged.has_value()) << merged.error();

    // Config's own paths split on dots only, so the slash needs no quoting
    // on the way back out.
    EXPECT_FLOAT_EQ(
        merged.value().GetFloat("material_overrides.block/Material.ir_temperature_k", 0.0f),
        320.0f);
    EXPECT_FLOAT_EQ(
        merged.value().GetFloat("material_overrides.block/Material.ir_emissivity", 0.0f),
        0.85f);
}

TEST_F(BatchManifestTest, TheSameConfigMayAppearOnEveryLine) {
    // Which is what a sequence is: one scene, one line per frame.
    const auto manifest = Write("m.txt",
                                "a.toml | material_overrides.P.ir_temperature_k=280.0\n"
                                "a.toml | material_overrides.P.ir_temperature_k=300.0\n"
                                "a.toml | material_overrides.P.ir_temperature_k=320.0\n");
    auto entries = ParseManifest(manifest);
    ASSERT_TRUE(entries.has_value()) << entries.error();
    ASSERT_EQ(entries.value().size(), 3u);

    for (const auto& entry : entries.value()) {
        EXPECT_EQ(entry.configPath, (testDir / "a.toml").lexically_normal());
    }
    EXPECT_NE(entries.value()[0].inlineOverrides, entries.value()[1].inlineOverrides);
}

TEST_F(BatchManifestTest, TheSummarySaysWhatTheLineChanged) {
    // The dry run's only account of a line, so it has to name both halves.
    Write("noon.toml", "[renderer]\nspp = 8\n");
    const auto manifest = Write("m.txt", "a.toml | @noon.toml renderer.spp=64\n");
    auto entries = ParseManifest(manifest);
    ASSERT_TRUE(entries.has_value()) << entries.error();

    const String summary = DescribeEntryOverrides(entries.value()[0]);
    EXPECT_NE(summary.find("noon.toml"), String::npos);
    EXPECT_NE(summary.find("renderer.spp"), String::npos);

    // A bare path has nothing to say, and says nothing.
    const auto plain = Write("plain.txt", "a.toml\n");
    auto plainEntries = ParseManifest(plain);
    ASSERT_TRUE(plainEntries.has_value());
    EXPECT_TRUE(DescribeEntryOverrides(plainEntries.value()[0]).empty());
}

TEST_F(BatchManifestTest, AnErrorNamesTheLineItIsOn) {
    const auto manifest = Write("m.txt", "good.toml\ngood.toml\nbad.toml | oops\n");
    auto entries = ParseManifest(manifest);
    ASSERT_FALSE(entries.has_value());
    EXPECT_NE(entries.error().find("line 3"), String::npos) << entries.error();
}
