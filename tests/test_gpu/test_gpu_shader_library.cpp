#include <gtest/gtest.h>

#include "gpu/ShaderLibrary.hpp"

#include <filesystem>
#include <fstream>
#include <string>

using namespace quantiloom;
using namespace quantiloom::gpu;

namespace {

class ShaderDirectory {
public:
    ShaderDirectory() {
        path_ = std::filesystem::temp_directory_path() / "ql_gpu_shader_library_test";
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~ShaderDirectory() { std::filesystem::remove_all(path_); }
    void Write(const std::string& file, size_t bytes) const {
        std::ofstream out(path_ / file, std::ios::binary);
        const std::string content(bytes, 'x');
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
    }
    [[nodiscard]] const std::filesystem::path& Path() const { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace

TEST(GpuShaderLibraryTest, LoadsAWholeSetFromOneDirectory) {
    ShaderDirectory dir;
    dir.Write("a.spv", 8);
    dir.Write("b.spv", 12);
    const ShaderLibrary library(ShaderFormat::SpirV);
    const StringView names[] = {"a", "b"};
    const auto loaded = library.LoadFrom(dir.Path(), names);
    ASSERT_TRUE(loaded.has_value()) << loaded.error();
    ASSERT_EQ(loaded.value().size(), 2u);
    EXPECT_EQ(loaded.value()[0].name, "a");
    EXPECT_EQ(loaded.value()[1].bytes.size(), 12u);
}

TEST(GpuShaderLibraryTest, OneMissingNameFailsTheWholeSet) {
    ShaderDirectory dir;
    dir.Write("a.spv", 8);
    const ShaderLibrary library(ShaderFormat::SpirV);
    const StringView names[] = {"a", "b"};
    const auto loaded = library.LoadFrom(dir.Path(), names);
    ASSERT_FALSE(loaded.has_value());
    EXPECT_NE(loaded.error().find("b.spv"), std::string::npos);
}

TEST(GpuShaderLibraryTest, SpirvThatIsNotWholeWordsIsRejected) {
    ShaderDirectory dir;
    dir.Write("odd.spv", 7);
    const ShaderLibrary library(ShaderFormat::SpirV);
    const StringView names[] = {"odd"};
    EXPECT_FALSE(library.LoadFrom(dir.Path(), names).has_value());
}

TEST(GpuShaderLibraryTest, MetalLibUsesItsOwnExtension) {
    ShaderDirectory dir;
    dir.Write("k.metallib", 7);
    const ShaderLibrary library(ShaderFormat::MetalLib);
    const StringView names[] = {"k"};
    EXPECT_TRUE(library.LoadFrom(dir.Path(), names).has_value());
}
