#include <gtest/gtest.h>

#include "atmos/AtmosNet.hpp"
#include "atmos/Safetensors.hpp"

#include <nlohmann/json.hpp>
#include <cstring>
#include <fstream>
#include <limits>

using namespace quantiloom;
using nlohmann::json;

namespace {
class AtmosNetInputTest : public ::testing::Test {
protected:
    std::filesystem::path file;
    json metadata;
    void SetUp() override {
        file = std::filesystem::temp_directory_path() /
               ("ql_atmos_input_" + std::to_string(::testing::UnitTest::GetInstance()
                    ->current_test_info()->line()) + ".safetensors");
        metadata = {
            {"format_version", "1"}, {"net", "tau"}, {"path_type", "horizontal"},
            {"opaque_delta", "7"}, {"delta_clamp", "20"}, {"sampled_json", "{}"},
            {"band_json", json({{"name", "vis"}, {"v1_cm", 100.0}, {"v2_cm", 100.0},
                {"dv_cm", 1.0}, {"K", 1}, {"thermal", false}}).dump()},
            {"model_json", json({{"d_in", 2}, {"d_out", 1}, {"width", 1}, {"blocks", 0},
                {"pca_mode", "none"}, {"n_pc", 0}}).dump()},
            {"feature_names_json", json::array({"atmos_model", "h1_km", "h2_km",
                "cos_view_zenith", "range_km"}).dump()},
            {"input_spec_json", json({{"entries", json::array({{
                {"kind", "onehot"}, {"name", "atmos_model"}, {"col", 0},
                {"values", json::array({2.0, 3.0})}}})}}).dump()},
            {"targets_json", json({{"K", 1}, {"rows", json::array({{
                {"block", "tau"}, {"column", "LOG_TOTAL"}, {"kind", "delta"}}})}}).dump()}
        };
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove(file, ec);
    }
    template<class F>
    void Edit(const char* key, F edit) {
        json value = json::parse(metadata.at(key).get<std::string>());
        edit(value);
        metadata[key] = value.dump();
    }
    void Write(json extra = json::object(), bool alignedHeader = true, bool prefixU8 = false) {
        json header = {{"__metadata__", metadata}};
        std::vector<uint8_t> data(prefixU8 ? 1 : 0, 0);
        if (prefixU8)
            header["prefix"] = {{"dtype", "U8"}, {"shape", {1}}, {"data_offsets", {0, 1}}};
        auto floats = [&](const char* name, json shape, std::vector<float> values) {
            const size_t begin = data.size();
            data.resize(begin + values.size() * sizeof(float));
            std::memcpy(data.data() + begin, values.data(), values.size() * sizeof(float));
            header[name] = {{"dtype", "F32"}, {"shape", shape},
                {"data_offsets", json::array({begin, data.size()})}};
        };
        floats("stem.weight", {1, 2}, {0, 0});
        floats("stem.bias", {1}, {0});
        floats("head.weight", {1, 1}, {0});
        floats("head.bias", {1}, {0.5f});
        floats("norm.0.mean", {1}, {0});
        floats("norm.0.std", {1}, {1});
        floats("norm.0.log_eps", {1}, {0});
        const size_t offset = data.size();
        data.push_back(0);
        header["norm.0.log_mask"] = {{"dtype", "U8"}, {"shape", {1}},
            {"data_offsets", {offset, offset + 1}}};
        for (const auto& [name, info] : extra.items()) header[name] = info;
        std::string encoded = header.dump();
        encoded.append(alignedHeader ? (8 - encoded.size() % 8) % 8
                                     : (5 - encoded.size() % 4) % 4, ' ');
        const uint64_t length = encoded.size();
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&length), sizeof(length));
        out.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
};

TEST_F(AtmosNetInputTest, ValidPackPreservesInferenceAndEmptyBatch) {
    Write();
    AtmosNet net(file);
    const double row[] = {2, 0, 0, 0, 1};
    double result = -1;
    net.Infer(row, 1, &result);
    EXPECT_DOUBLE_EQ(result, 0.5);
    EXPECT_NO_THROW(net.Infer(nullptr, 0, nullptr));
    EXPECT_THROW(net.Infer(row, std::numeric_limits<size_t>::max(), &result), std::runtime_error);
}

TEST_F(AtmosNetInputTest, LegalUnalignedHeaderAndU8PrefixPreserveInference) {
    const double row[] = {2, 0, 0, 0, 1};
    for (bool prefix : {false, true}) {
        Write(json::object(), prefix, prefix);
        // prefix=true exercises a U8 tensor before F32 in an aligned blob;
        // the other branch deliberately uses a header length not divisible by four.
        AtmosNet net(file);
        double result = -1;
        net.Infer(row, 1, &result);
        EXPECT_DOUBLE_EQ(result, 0.5);
        TensorView copied = [&] {
            SafetensorsFile pack(file);
            return pack.Get("head.bias");
        }();
        EXPECT_FLOAT_EQ(copied.F32Data()[0], 0.5f);
        TensorView moved = std::move(copied);
        EXPECT_FLOAT_EQ(moved.F32Data()[0], 0.5f);
    }
}

TEST_F(AtmosNetInputTest, RejectsShortOrReorderedGeometryBlocks) {
    for (size_t count = 0; count < 4; ++count) {
        metadata["feature_names_json"] = json(std::vector<std::string>(count, "h1_km")).dump();
        Write();
        EXPECT_THROW(AtmosNet net(file), std::runtime_error);
    }
    metadata["feature_names_json"] = json::array({"atmos_model", "h2_km", "h1_km",
        "cos_view_zenith", "range_km"}).dump();
    Write();
    EXPECT_THROW(AtmosNet net(file), std::runtime_error);
}

TEST_F(AtmosNetInputTest, RejectsEmptyOnehotAndBadColumn) {
    const auto original = metadata["input_spec_json"];
    Edit("input_spec_json", [](json& j) { j["entries"][0]["values"] = json::array(); });
    Write();
    EXPECT_THROW(AtmosNet net(file), std::runtime_error);
    metadata["input_spec_json"] = original;
    Edit("input_spec_json", [](json& j) { j["entries"][0]["col"] = -1; });
    Write();
    EXPECT_THROW(AtmosNet net(file), std::runtime_error);
}

TEST_F(AtmosNetInputTest, RejectsInvalidDimensionsAndExpandedCounts) {
    const auto original = metadata["model_json"];
    for (const char* key : {"d_in", "d_out", "width"}) {
        metadata["model_json"] = original;
        Edit("model_json", [key](json& j) { j[key] = 0; });
        Write();
        EXPECT_THROW(AtmosNet net(file), std::runtime_error);
    }
    metadata["model_json"] = original;
    Edit("model_json", [](json& j) { j["blocks"] = -1; });
    Write();
    EXPECT_THROW(AtmosNet net(file), std::runtime_error);
    metadata["model_json"] = original;
    Edit("model_json", [](json& j) { j["d_in"] = 1; });
    Write();
    EXPECT_THROW(AtmosNet net(file), std::runtime_error);
    metadata["model_json"] = original;
    Edit("model_json", [](json& j) { j["d_in"] = (uint64_t{1} << 32) + 2; });
    Write();
    EXPECT_THROW(AtmosNet net(file), std::runtime_error);
    metadata["model_json"] = original;
    Edit("model_json", [](json& j) { j["blocks"] = std::numeric_limits<int>::max(); });
    Write();
    EXPECT_THROW(AtmosNet net(file), std::runtime_error);
}

TEST_F(AtmosNetInputTest, RejectsZeroSpectralGridAndEmptyTargets) {
    const auto original = metadata["band_json"];
    Edit("band_json", [](json& j) { j["K"] = 0; });
    Write();
    EXPECT_THROW(AtmosNet net(file), std::runtime_error);
    metadata["band_json"] = original;
    Edit("targets_json", [](json& j) { j["rows"] = json::array(); });
    Write();
    EXPECT_THROW(AtmosNet net(file), std::runtime_error);
}

TEST_F(AtmosNetInputTest, RejectsWrappedTensorShapeAndWrongDtype) {
    Write({{"stem.weight", {{"dtype", "F32"},
        {"shape", {int64_t{1} << 32, int64_t{1} << 32}}, {"data_offsets", {0, 0}}}}});
    EXPECT_THROW(SafetensorsFile pack(file), std::runtime_error);
    Write({{"norm.0.mean", {{"dtype", "U8"}, {"shape", {1}}, {"data_offsets", {0, 1}}}}});
    EXPECT_THROW(AtmosNet net(file), std::runtime_error);
}

TEST_F(AtmosNetInputTest, ExistingPacksKeepTheirDuplicateSampledAndGeometryNames) {
    const auto dir = std::filesystem::path(QUANTILOOM_SOURCE_ROOT) / "assets" / "atmos_models";
    ASSERT_TRUE(std::filesystem::is_directory(dir));
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() != ".safetensors") continue;
        SCOPED_TRACE(entry.path().filename().string());
        EXPECT_NO_THROW(AtmosNet net(entry.path()));
    }
}
}  // namespace
