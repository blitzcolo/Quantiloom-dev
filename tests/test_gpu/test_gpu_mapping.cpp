#include <gtest/gtest.h>

#include "gpu/ShaderLibrary.hpp"
#include "gpu/vulkan/VulkanMapping.hpp"
#include "gpu/vulkan/VulkanResources.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <string>

using namespace quantiloom;
using namespace quantiloom::gpu;
using namespace quantiloom::gpu::vulkan;

TEST(GpuMappingTest, ComputeAndTransferScopes) {
    const AccessScope computeRw = ScopeOf(Access::ComputeRead | Access::ComputeWrite, false);
    EXPECT_EQ(computeRw.stages, static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT));
    EXPECT_EQ(computeRw.access, static_cast<VkAccessFlags>(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));

    const AccessScope transferRead = ScopeOf(Access::TransferRead, false);
    EXPECT_EQ(transferRead.stages, static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_TRANSFER_BIT));
    EXPECT_EQ(transferRead.access, static_cast<VkAccessFlags>(VK_ACCESS_TRANSFER_READ_BIT));

    const AccessScope host = ScopeOf(Access::HostRead, false);
    EXPECT_EQ(host.stages, static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_HOST_BIT));
    EXPECT_EQ(host.access, static_cast<VkAccessFlags>(VK_ACCESS_HOST_READ_BIT));
}

TEST(GpuMappingTest, RayTraceAndAnyScopes) {
    const AccessScope rt = ScopeOf(Access::RayTraceWrite, true);
    EXPECT_EQ(rt.stages, static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR));
    EXPECT_EQ(rt.access, static_cast<VkAccessFlags>(VK_ACCESS_SHADER_WRITE_BIT));

    const AccessScope any = ScopeOf(Access::AnyWrite, true);
    EXPECT_EQ(any.stages, static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT));
    EXPECT_EQ(any.access, static_cast<VkAccessFlags>(VK_ACCESS_MEMORY_WRITE_BIT));
}

TEST(GpuMappingTest, NoneIsTopOrBottomOfPipe) {
    const AccessScope src = ScopeOf(Access::None, true);
    EXPECT_EQ(src.stages, static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT));
    EXPECT_EQ(src.access, 0u);
    const AccessScope dst = ScopeOf(Access::None, false);
    EXPECT_EQ(dst.stages, static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT));
    EXPECT_EQ(dst.access, 0u);
}

TEST(GpuMappingTest, CombinedScopeIsTheUnionOfItsBits) {
    const AccessScope restore = ScopeOf(Access::ComputeRead | Access::ComputeWrite | Access::TransferWrite, true);
    EXPECT_EQ(restore.stages,
              static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT));
}

TEST(GpuMappingTest, GeneralPolicyIsGeneralUnlessDiscarded) {
    EXPECT_EQ(LayoutOf(TextureLayoutPolicy::General, Access::TransferRead, false), VK_IMAGE_LAYOUT_GENERAL);
    EXPECT_EQ(LayoutOf(TextureLayoutPolicy::General, Access::None, true), VK_IMAGE_LAYOUT_UNDEFINED);
}

TEST(GpuMappingTest, OptimalPolicyFollowsTheAccess) {
    EXPECT_EQ(LayoutOf(TextureLayoutPolicy::Optimal, Access::ComputeRead, false),
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    EXPECT_EQ(LayoutOf(TextureLayoutPolicy::Optimal, Access::TransferRead, false), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    EXPECT_EQ(LayoutOf(TextureLayoutPolicy::Optimal, Access::TransferWrite, false),
              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    EXPECT_EQ(LayoutOf(TextureLayoutPolicy::Optimal, Access::ComputeWrite, false), VK_IMAGE_LAYOUT_GENERAL);
}

TEST(GpuMappingTest, DescriptorTypesFollowTheHlslClass) {
    EXPECT_EQ(ToDescriptorType(BindingKind::ReadOnlyBuffer), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    EXPECT_EQ(ToDescriptorType(BindingKind::ReadWriteBuffer), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    EXPECT_EQ(ToDescriptorType(BindingKind::StorageTexture), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    EXPECT_EQ(ToDescriptorType(BindingKind::SampledTexture), VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
    EXPECT_EQ(ToDescriptorType(BindingKind::Sampler), VK_DESCRIPTOR_TYPE_SAMPLER);
    EXPECT_EQ(ToDescriptorType(BindingKind::ConstantBuffer), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
}

TEST(GpuMappingTest, FormatsRoundTrip) {
    for (const Format f : {Format::R32Float, Format::RGBA32Float, Format::RGBA32Uint, Format::RGBA8Unorm,
                           Format::RGBA8Srgb, Format::BGRA8Srgb}) {
        EXPECT_EQ(FromVk(ToVk(f)), f);
    }
}

TEST(GpuMappingTest, HandlesRoundTrip) {
    const VkBuffer buffer = FromHandle<VkBuffer>(0x1234u);
    EXPECT_EQ(ToHandle(buffer), 0x1234u);
}

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

TEST(GpuBindingLayoutTest, RepeatedSlotsAreRejectedBeforeAnyDeviceCall) {
    const std::array<BindingSlot, 2> slots{BindingSlot{3, BindingKind::ReadOnlyBuffer, 1, false},
                                           BindingSlot{3, BindingKind::StorageTexture, 1, false}};
    const auto layout = VulkanBindingLayout::Create(VK_NULL_HANDLE, BindingLayoutDesc{0, slots});
    ASSERT_FALSE(layout.has_value());
    EXPECT_NE(layout.error().find("slot 3"), std::string::npos);
}
