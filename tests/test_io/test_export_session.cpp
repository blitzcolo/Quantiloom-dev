#include <gtest/gtest.h>
#include "dataset/ExportSession.hpp"
#include "io/ImageIO.hpp"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace quantiloom;
using dataset::ExportSession;
namespace fs = std::filesystem;
using Json = nlohmann::json;

class ExportSessionTest : public ::testing::Test {
protected:
    fs::path root = fs::temp_directory_path() / "quantiloom_export_session_test";
    Config config;
    Image image{2, 2, 1};
    void SetUp() override {
        fs::remove_all(root);
        fs::create_directories(root);
        image.data = {1, 2, 3, 4};
        image.channelNames = {"Y"};
    }
    void TearDown() override { fs::remove_all(root); }
    auto Create() { return ExportSession::Create((root / "frame.exr").string(), config, {"{}"}); }
    Json Record() { return Json::parse(std::ifstream(root / "frame.metadata.json")); }
    bool Valid() { return ExportSession::Verify((root / "frame.metadata.json").string()).valid; }
    void Publish() {
        auto session = Create();
        ASSERT_TRUE(session) << session.error();
        ASSERT_TRUE(session.value()->WriteImage("frame.exr", "radiance", image, "{}"));
        ASSERT_TRUE(session.value()->Commit());
    }
};
TEST_F(ExportSessionTest, StagesBeforePublishingAndPreservesPixelsAndMetadata) {
    image.metadata["units"] = "W/m^2/sr/nm";
    auto session = Create();
    ASSERT_TRUE(session);
    ASSERT_TRUE(session.value()->WriteImage("frame.exr", "radiance", image, "{}"));
    EXPECT_FALSE(fs::exists(root / "frame.exr"));
    ASSERT_TRUE(session.value()->Commit());
    EXPECT_TRUE(Valid());
    auto loaded = ImageIO::ReadEXR((root / "frame.exr").string());
    ASSERT_TRUE(loaded);
    EXPECT_EQ(loaded->data, image.data);
    EXPECT_EQ(loaded->metadata["units"], image.metadata["units"]);
    EXPECT_EQ(loaded->metadata["quantiloom_record_id"], session.value()->RecordId());
    EXPECT_EQ(loaded->metadata["quantiloom_product_id"], "radiance");
    EXPECT_EQ(loaded->metadata["quantiloom_sidecar"], "frame.metadata.json");
    const auto completed = Record();
    EXPECT_FALSE(session.value()->Commit());
    EXPECT_EQ(Record(), completed);
    EXPECT_TRUE(Valid());
}
TEST_F(ExportSessionTest, RejectsConcurrentWriterAndReleasesLock) {
    { auto first = Create(); ASSERT_TRUE(first); EXPECT_FALSE(Create()); }
    EXPECT_TRUE(Create());
}
TEST_F(ExportSessionTest, AbandonedStagingPreservesPreviousCompleteSample) {
    Publish();
    const auto previous = Record();
    { auto next = Create(); ASSERT_TRUE(next);
      ASSERT_TRUE(next.value()->WriteImage("frame.exr", "radiance", image, "{}")); }
    EXPECT_EQ(Record(), previous);
    EXPECT_TRUE(Valid());
}
TEST_F(ExportSessionTest, ReplacementGetsNewIdentityAndOldUnlistedFilesAreNotProducts) {
    Publish();
    const auto id = Record()["record_id"];
    std::ofstream(root / "historical.exr") << "old";
    Publish();
    EXPECT_NE(Record()["record_id"], id);
    EXPECT_EQ(Record()["products"].size(), 1u);
    EXPECT_TRUE(Valid());
}
TEST_F(ExportSessionTest, DetectsChangedProductAndReplay) {
    Publish();
    std::ofstream(root / "frame.exr", std::ios::app) << "changed";
    EXPECT_FALSE(Valid());
    Publish();
    std::ofstream(root / "frame.replay.toml", std::ios::app) << "# changed";
    EXPECT_FALSE(Valid());
}
TEST_F(ExportSessionTest, FailedPublicationInvalidatesOldSample) {
    Publish();
    auto session = Create(); ASSERT_TRUE(session);
    fs::remove(root / "frame.exr");
    fs::create_directory(root / "frame.exr");
    std::ofstream(root / "frame.exr" / "blocker") << "x";
    ASSERT_TRUE(session.value()->WriteImage("frame.exr", "radiance", image, "{}"));
    EXPECT_FALSE(session.value()->Commit());
    EXPECT_EQ(Record()["state"], "failed");
    EXPECT_FALSE(Valid());
}
TEST_F(ExportSessionTest, RejectsTraversalDuplicateAndReservedNames) {
    auto session = Create(); ASSERT_TRUE(session);
    for (const String name : {"../out.exr", "sub/../out.exr", "C:out.exr", "sub\\out.exr",
                              "frame.metadata.json", "FRAME.REPLAY.TOML", "record.tmp"})
        EXPECT_FALSE(session.value()->StagingPath(name)) << name;
    ASSERT_TRUE(session.value()->WriteImage("frame.exr", "radiance", image, "{}"));
    EXPECT_FALSE(session.value()->WriteImage("FRAME.EXR", "second", image, "{}"));
    EXPECT_FALSE(session.value()->WriteImage("other.exr", "radiance", image, "{}"));
}
TEST_F(ExportSessionTest, ModifiedStagingDoesNotInvalidatePreviousSample) {
    Publish();
    const auto previous = Record();
    auto session = Create(); ASSERT_TRUE(session);
    auto staged = session.value()->StagingPath("frame.exr"); ASSERT_TRUE(staged);
    ASSERT_TRUE(session.value()->WriteImage("frame.exr", "radiance", image, "{}"));
    std::ofstream(staged.value(), std::ios::app) << "changed";
    EXPECT_FALSE(session.value()->Commit());
    EXPECT_EQ(Record(), previous);
    EXPECT_TRUE(Valid());
}

TEST_F(ExportSessionTest, PublishesNestedBandsWithRelativeSummary) {
    auto session = Create(); ASSERT_TRUE(session);
    ASSERT_TRUE(session.value()->WriteImage("frame_bands/band.exr", "band", image, "{}"));
    ASSERT_TRUE(session.value()->Commit());
    EXPECT_TRUE(Valid());
    auto loaded = ImageIO::ReadEXR((root / "frame_bands/band.exr").string());
    ASSERT_TRUE(loaded);
    EXPECT_EQ(loaded->metadata["quantiloom_sidecar"], "../frame.metadata.json");
}

TEST_F(ExportSessionTest, RejectsNewRecordIdentityClaimingOldImageBytes) {
    Publish();
    auto record = Record();
    record["record_id"] = String(64, 'a');
    std::ofstream(root / "frame.metadata.json") << record.dump();
    EXPECT_FALSE(Valid());
}
TEST_F(ExportSessionTest, PngSummaryCarriesTheTransactionIdentity) {
    auto session = Create(); ASSERT_TRUE(session);
    ASSERT_TRUE(session.value()->WriteImage("frame.png", "preview", image, "{}"));
    ASSERT_TRUE(session.value()->Commit());
    EXPECT_TRUE(Valid());
    auto record = Record();
    record["products"][0]["product_id"] = "different_preview";
    std::ofstream(root / "frame.metadata.json") << record.dump();
    EXPECT_FALSE(Valid());
}

TEST_F(ExportSessionTest, RequiresDeclaredSchemaFieldsAndRejectsUnknownRootFields) {
    Publish();
    const auto good = Record();
    for (const String key : {"pairing_status", "schema_version", "replay", "provenance"}) {
        auto record = good;
        record.erase(key);
        { std::ofstream(root / "frame.metadata.json") << record.dump(); }
        EXPECT_FALSE(Valid()) << key;
    }
    for (const String key : {"product_id", "size_bytes", "description"}) {
        auto record = good;
        record["products"][0].erase(key);
        { std::ofstream(root / "frame.metadata.json") << record.dump(); }
        EXPECT_FALSE(Valid()) << key;
    }
    auto record = good;
    record["unexpected"] = 1;
    { std::ofstream(root / "frame.metadata.json") << record.dump(); }
    EXPECT_FALSE(Valid());
}
TEST_F(ExportSessionTest, RejectsDuplicateJsonKeysInsteadOfTakingTheLastValue) {
    Publish();
    const String good = Record().dump();
    { std::ofstream(root / "frame.metadata.json") << "{\"state\":\"failed\"," << good.substr(1); }
    EXPECT_FALSE(Valid());
}
TEST_F(ExportSessionTest, ReservedInternalDirectoriesAreCaseInsensitive) {
    auto session = Create(); ASSERT_TRUE(session);
    EXPECT_FALSE(session.value()->StagingPath(".INTERNAL/file.exr"));
    EXPECT_FALSE(session.value()->StagingPath(".QUANTILOOM-EXPORT.LOCK/file.exr"));
}

TEST_F(ExportSessionTest, RefusesToPublishImageWithMissingOrForeignRecordSummary) {
    Publish();
    const auto previous = Record();
    auto session = Create(); ASSERT_TRUE(session);
    const auto path = session.value()->StagingPath("foreign.exr"); ASSERT_TRUE(path);
    ASSERT_TRUE(ImageIO::WriteEXR(path.value(), image));
    ASSERT_TRUE(session.value()->RegisterFile("foreign.exr", "foreign", R"({"width":2,"height":2,"channels":1})"));
    EXPECT_FALSE(session.value()->Commit());
    EXPECT_EQ(Record(), previous);
    EXPECT_TRUE(Valid());
}

TEST_F(ExportSessionTest, RejectsDescriptionDimensionsThatDisagreeWithTheImage) {
    Publish();
    auto record = Record();
    record["products"][0]["description"]["width"] = 3;
    { std::ofstream(root / "frame.metadata.json") << record.dump(); }
    EXPECT_FALSE(Valid());
}

TEST_F(ExportSessionTest, DisjointOutputSetsShareADirectory) {
    auto first = Create(); ASSERT_TRUE(first);
    auto second = ExportSession::Create((root / "other.exr").string(), config, {"{}"});
    ASSERT_TRUE(second) << second.error();
    ASSERT_TRUE(first.value()->WriteImage("frame.exr", "first", image, "{}"));
    ASSERT_TRUE(second.value()->WriteImage("other.exr", "second", image, "{}"));
    ASSERT_TRUE(second.value()->Commit());
    ASSERT_TRUE(first.value()->Commit());
    EXPECT_TRUE(Valid());
    EXPECT_TRUE(ExportSession::Verify((root / "other.metadata.json").string()).valid);
}
TEST_F(ExportSessionTest, NestedWritersCannotClaimTheSameArtifact) {
    auto first = Create(); ASSERT_TRUE(first);
    ASSERT_TRUE(first.value()->StagingPath("bands/shared.exr"));
    auto second = ExportSession::Create((root / "bands/other.exr").string(), config, {"{}"});
    ASSERT_TRUE(second) << second.error();
    EXPECT_FALSE(second.value()->StagingPath("shared.exr"));
    EXPECT_FALSE(ExportSession::Create((root / "bands/shared.exr").string(), config, {"{}"}));
    first.value().reset();
    EXPECT_TRUE(second.value()->StagingPath("shared.exr"));
}
TEST_F(ExportSessionTest, RejectsFileAndDescendantCollisionInBothOrders) {
    auto first = Create(); ASSERT_TRUE(first);
    auto second = ExportSession::Create((root / "other.exr").string(), config, {"{}"});
    ASSERT_TRUE(second);
    ASSERT_TRUE(first.value()->StagingPath("group"));
    EXPECT_FALSE(second.value()->StagingPath("group/band.exr"));
    ASSERT_TRUE(second.value()->StagingPath("bands/a.exr"));
    EXPECT_FALSE(first.value()->StagingPath("bands"));
    EXPECT_TRUE(first.value()->StagingPath("bands/b.exr"));
}
TEST_F(ExportSessionTest, PreservesForeignAndLegacyStaleClaims) {
    fs::create_directory(root / "frame.exr.quantiloom-export.lock");
    EXPECT_FALSE(Create());
    EXPECT_TRUE(fs::exists(root / "frame.exr.quantiloom-export.lock"));
    fs::remove(root / "frame.exr.quantiloom-export.lock");
    fs::create_directory(root / ".quantiloom-export.lock");
    EXPECT_FALSE(Create());
    EXPECT_TRUE(fs::exists(root / ".quantiloom-export.lock"));
}
TEST_F(ExportSessionTest, FailedPartialReservationReleasesOnlyOwnedClaims) {
    fs::create_directory(root / "frame.replay.toml.quantiloom-export.lock");
    EXPECT_FALSE(Create());
    EXPECT_FALSE(fs::exists(root / "frame.exr.quantiloom-export.lock"));
    EXPECT_FALSE(fs::exists(root / "frame.metadata.json.quantiloom-export.lock"));
    EXPECT_TRUE(fs::exists(root / "frame.replay.toml.quantiloom-export.lock"));
}
TEST_F(ExportSessionTest, ArtifactNamesCannotEnterReservationDirectories) {
    auto session = Create(); ASSERT_TRUE(session);
    EXPECT_FALSE(session.value()->StagingPath("other.exr.quantiloom-export.lock/file.exr"));
    EXPECT_FALSE(session.value()->StagingPath("other.EXR.QUANTILOOM-EXPORT.LOCK"));
}

TEST_F(ExportSessionTest, RejectedNestedCreateCannotTurnClaimedFileIntoDirectory) {
    auto first = ExportSession::Create((root / "destination").string(), config, {"{}"});
    ASSERT_TRUE(first);
    EXPECT_FALSE(ExportSession::Create((root / "destination/child.exr").string(), config, {"{}"}));
    EXPECT_FALSE(fs::exists(root / "destination"));
    ASSERT_TRUE(first.value()->WriteImage("frame.exr", "frame", image, "{}"));
    EXPECT_TRUE(first.value()->Commit());
}

TEST_F(ExportSessionTest, NestedDestinationCannotBeRedirectedAfterReservation) {
    const auto foreign = root / "foreign";
    fs::create_directory(foreign);
    std::ofstream(foreign / "band.exr") << "untouched";
    auto session = Create(); ASSERT_TRUE(session);
    ASSERT_TRUE(session.value()->WriteImage("bands/band.exr", "band", image, "{}"));
    std::error_code error;
    fs::rename(root / "bands", root / "retired", error);
#ifdef _WIN32
    // A junction cannot be inserted: the existing directory cannot be moved.
    EXPECT_TRUE(error);
    const auto writableDirectory = CreateFileW((root / "bands").c_str(), GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    EXPECT_EQ(writableDirectory, INVALID_HANDLE_VALUE);
    if (writableDirectory != INVALID_HANDLE_VALUE) CloseHandle(writableDirectory);
    ASSERT_TRUE(session.value()->Commit());
    EXPECT_TRUE(Valid());
#else
    ASSERT_FALSE(error) << error.message();
    fs::create_directory_symlink(foreign, root / "bands");
    EXPECT_FALSE(session.value()->Commit());
#endif
    std::string sentinel;
    std::ifstream(foreign / "band.exr") >> sentinel;
    EXPECT_EQ(sentinel, "untouched");
    session.value().reset();
    EXPECT_TRUE(fs::exists(foreign / "band.exr"));
}

TEST_F(ExportSessionTest, RejectsExistingLinkedNestedParent) {
    const auto foreign = root / "foreign";
    fs::create_directory(foreign);
    std::error_code error;
    fs::create_directory_symlink(foreign, root / "bands", error);
    if (error) GTEST_SKIP() << "Directory symlink creation is unavailable";
    auto session = Create(); ASSERT_TRUE(session);
    EXPECT_FALSE(session.value()->StagingPath("bands/band.exr"));
    EXPECT_TRUE(fs::is_empty(foreign));
}

TEST_F(ExportSessionTest, KeepsRootAndStagingParentsPinnedForStreamingWriter) {
    auto session = Create(); ASSERT_TRUE(session);
    const auto staged = session.value()->StagingPath("deep/nested/data.bin"); ASSERT_TRUE(staged);
    std::error_code error;
#ifdef _WIN32
    fs::rename(fs::path(staged.value()).parent_path(), root / "stolen", error);
    EXPECT_TRUE(error);
    fs::rename(root, fs::path(root.string() + "_moved"), error);
    EXPECT_TRUE(error);
#else
    // The directory descriptor path remains usable even if an attacker renames
    // its original directory. No pathname traversal can redirect this writer.
    const auto original = root / "frame.exr.quantiloom-export.lock" / session.value()->RecordId() / "deep/nested";
    fs::rename(original, original.parent_path() / "moved", error);
    ASSERT_FALSE(error);
    fs::create_directory_symlink(root, original);
#endif
    { std::ofstream(staged.value()) << "private staged bytes"; }
    EXPECT_FALSE(fs::exists(root / "data.bin"));
#ifdef _WIN32
    ASSERT_TRUE(session.value()->RegisterFile("deep/nested/data.bin", "data", "{}"));
    ASSERT_TRUE(session.value()->Commit());
    EXPECT_TRUE(Valid());
#else
    EXPECT_FALSE(session.value()->RegisterFile("deep/nested/data.bin", "data", "{}"));
#endif
}

TEST_F(ExportSessionTest, PublishesWhenPrivateStagingExceedsLegacyWindowsPathLimit) {
    // The visible output remains below MAX_PATH, while the claim and random
    // staging ID push its private paths beyond the old 248-character limit.
    const auto longRoot = root / std::string(150, 'x');
    fs::create_directory(longRoot);
    auto session = ExportSession::Create((longRoot / "frame.exr").string(), config, {"{}"});
    ASSERT_TRUE(session) << session.error();
    const auto staged = session.value()->StagingPath("bands/band.exr");
    ASSERT_TRUE(staged) << staged.error();
#ifdef _WIN32
    EXPECT_GT(staged.value().size(), 260u);
    EXPECT_TRUE(staged.value().starts_with("\\\\?\\"));
#endif
    ASSERT_TRUE(session.value()->WriteImage("bands/band.exr", "band", image, "{}"));
    ASSERT_TRUE(session.value()->Commit());
    EXPECT_TRUE(ExportSession::Verify((longRoot / "frame.metadata.json").string()).valid);
    session.value().reset();
    EXPECT_FALSE(fs::exists(longRoot / "frame.exr.quantiloom-export.lock"));
}
