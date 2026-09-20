#include <gtest/gtest.h>

#include "postprocess/CameraDemosaic.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace quantiloom;
using namespace quantiloom::camera;

namespace {

// Reads a `static const float kName[25] = { ... };` table out of the HLSL
// include and returns the 25 values in order.
std::vector<f64> ParseHlslKernel(const std::filesystem::path& path,
                                 const std::string& name) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    std::ostringstream text;
    text << input.rdbuf();
    const std::string source = text.str();
    const size_t declAt = source.find(name);
    if (declAt == std::string::npos)
        throw std::runtime_error(name + " not found in " + path.string());
    const size_t openAt = source.find('{', declAt);
    const size_t closeAt = source.find("};", openAt);
    if (openAt == std::string::npos || closeAt == std::string::npos)
        throw std::runtime_error(name + " has no initializer in " + path.string());
    const std::string body = source.substr(openAt + 1, closeAt - openAt - 1);
    std::vector<f64> values;
    static const std::regex number(R"([-0-9.eE+]+)");
    for (auto it = std::sregex_iterator(body.begin(), body.end(), number);
         it != std::sregex_iterator(); ++it)
        values.push_back(std::stod(it->str()));
    return values;
}

void ExpectKernelMatches(const char* hlslName,
                         const std::array<std::array<f64, 5>, 5>& cpp) {
    const std::filesystem::path path =
        std::filesystem::path(QUANTILOOM_SOURCE_ROOT) / "src" / "shaders" /
        "camera_isp.hlsli";
    if (!std::filesystem::exists(path))
        GTEST_SKIP() << "camera_isp.hlsli not present at " << path;
    std::vector<f64> parsed;
    try {
        parsed = ParseHlslKernel(path, hlslName);
    } catch (const std::exception& error) {
        FAIL() << error.what();
    }
    ASSERT_EQ(parsed.size(), 25u) << hlslName << " must hold 25 coefficients";
    for (u32 row = 0; row < 5; ++row)
        for (u32 col = 0; col < 5; ++col)
            // Both sides are exact binary fractions; this must be bit-equal.
            EXPECT_DOUBLE_EQ(parsed[row * 5 + col], cpp[row][col])
                << hlslName << " coefficient [" << row << "][" << col << "]";
}

} // namespace

// The HLSL copies of the MHC kernels must equal the C++ constexpr tables to
// the bit, so the GPU and CPU chains can never drift apart silently.
TEST(IspCoefficientsTest, GreenAtRedBlueMatchesHlsl) {
    ExpectKernelMatches("kMhcGreenAtRedBlue", mhc::kGreenAtRedBlue);
}

TEST(IspCoefficientsTest, ColorAtGreenHorizontalMatchesHlsl) {
    ExpectKernelMatches("kMhcColorAtGreenHorizontal", mhc::kColorAtGreenHorizontal);
}

TEST(IspCoefficientsTest, ColorAtGreenVerticalMatchesHlsl) {
    ExpectKernelMatches("kMhcColorAtGreenVertical", mhc::kColorAtGreenVertical);
}

TEST(IspCoefficientsTest, ColorAtOppositeMatchesHlsl) {
    ExpectKernelMatches("kMhcColorAtOpposite", mhc::kColorAtOpposite);
}

// Every kernel must be DC-preserving: a constant CFA reproduces the constant.
TEST(IspCoefficientsTest, KernelsSumToOne) {
    const auto sumOf = [](const std::array<std::array<f64, 5>, 5>& kernel) {
        f64 sum = 0.0;
        for (const auto& row : kernel)
            for (const f64 weight : row) sum += weight;
        return sum;
    };
    EXPECT_DOUBLE_EQ(sumOf(mhc::kGreenAtRedBlue), 1.0);
    EXPECT_DOUBLE_EQ(sumOf(mhc::kColorAtGreenHorizontal), 1.0);
    EXPECT_DOUBLE_EQ(sumOf(mhc::kColorAtGreenVertical), 1.0);
    EXPECT_DOUBLE_EQ(sumOf(mhc::kColorAtOpposite), 1.0);
}
