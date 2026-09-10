// ============================================================================
// Quantiloom - Unit Tests for scene/TangentGenerator.hpp
// ============================================================================
// A tangent frame is only as good as its direction. A normal map is defined
// relative to whatever frame it is given, so almost anything continuous works;
// an anisotropic highlight points ALONG the tangent, so a frame that is merely
// continuous puts the streak somewhere the author did not choose. These pin the
// direction and the handedness, which are the two things that can be silently
// wrong.
// ============================================================================

#include <gtest/gtest.h>

#include "scene/TangentGenerator.hpp"

#include <cmath>

using namespace quantiloom;

namespace {

/// A quad in the XY plane, facing +Z, with the UVs the caller supplies.
GeometryPrimitive Quad(const std::vector<glm::vec2>& uvs) {
    GeometryPrimitive primitive;
    primitive.positions = {
        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    primitive.normals.assign(4, glm::vec3(0.0f, 0.0f, 1.0f));
    primitive.uvs = uvs;
    primitive.indices = {0, 1, 2, 0, 2, 3};
    return primitive;
}

}  // namespace

TEST(TangentGeneratorTest, AQuadWithAlignedUvsGetsTheUDirection) {
    // u runs with +X, so the tangent must be +X: this is the direction an
    // anisotropic streak will follow.
    GeometryPrimitive primitive = Quad({{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}});

    ASSERT_TRUE(TangentGenerator::FromUv(primitive));
    ASSERT_EQ(primitive.tangents.size(), primitive.positions.size());

    for (const glm::vec4& tangent : primitive.tangents) {
        EXPECT_NEAR(tangent.x, 1.0f, 1e-5f);
        EXPECT_NEAR(tangent.y, 0.0f, 1e-5f);
        EXPECT_NEAR(tangent.z, 0.0f, 1e-5f);
        EXPECT_NEAR(tangent.w, 1.0f, 1e-5f);
    }
}

TEST(TangentGeneratorTest, RotatedUvsRotateTheTangent) {
    // u now runs with +Y. A frame that only had to be continuous would not
    // notice; the highlight would be 90 degrees out.
    GeometryPrimitive primitive = Quad({{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}});

    ASSERT_TRUE(TangentGenerator::FromUv(primitive));
    for (const glm::vec4& tangent : primitive.tangents) {
        EXPECT_NEAR(tangent.x, 0.0f, 1e-5f);
        EXPECT_NEAR(tangent.y, 1.0f, 1e-5f);
    }
}

TEST(TangentGeneratorTest, MirroredUvsFlipHandedness) {
    GeometryPrimitive normal =
        Quad({{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}});
    GeometryPrimitive mirrored =
        Quad({{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, -1.0f}, {0.0f, -1.0f}});

    ASSERT_TRUE(TangentGenerator::FromUv(normal));
    ASSERT_TRUE(TangentGenerator::FromUv(mirrored));

    // Same tangent direction, opposite bitangent: w is what carries that, and
    // without it a mirrored UV island lights from the wrong side.
    EXPECT_NEAR(normal.tangents[0].x, 1.0f, 1e-5f);
    EXPECT_NEAR(mirrored.tangents[0].x, 1.0f, 1e-5f);
    EXPECT_NEAR(normal.tangents[0].w, 1.0f, 1e-5f);
    EXPECT_NEAR(mirrored.tangents[0].w, -1.0f, 1e-5f);
}

TEST(TangentGeneratorTest, TangentsAreOrthogonalToTheVertexNormal) {
    GeometryPrimitive primitive = Quad({{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}});
    // A shading normal that is not the geometric one, which is the usual case
    // on a smoothed mesh.
    primitive.normals.assign(4, glm::normalize(glm::vec3(0.3f, 0.2f, 1.0f)));

    ASSERT_TRUE(TangentGenerator::FromUv(primitive));
    for (usize i = 0; i < primitive.tangents.size(); ++i) {
        const glm::vec3 tangent(primitive.tangents[i]);
        EXPECT_NEAR(glm::length(tangent), 1.0f, 1e-5f);
        EXPECT_NEAR(glm::dot(tangent, primitive.normals[i]), 0.0f, 1e-5f);
    }
}

TEST(TangentGeneratorTest, DegenerateUvsFallBackToAFrameRatherThanANaN) {
    // Every UV identical: the texture-space triangle has no area, so nothing can
    // be derived. A NaN here would propagate into the shading frame.
    GeometryPrimitive primitive = Quad({{0.5f, 0.5f}, {0.5f, 0.5f}, {0.5f, 0.5f}, {0.5f, 0.5f}});

    ASSERT_TRUE(TangentGenerator::FromUv(primitive));
    for (usize i = 0; i < primitive.tangents.size(); ++i) {
        const glm::vec3 tangent(primitive.tangents[i]);
        EXPECT_TRUE(std::isfinite(tangent.x));
        EXPECT_NEAR(glm::length(tangent), 1.0f, 1e-5f);
        EXPECT_NEAR(glm::dot(tangent, primitive.normals[i]), 0.0f, 1e-5f);
    }
}

TEST(TangentGeneratorTest, RefusesAPrimitiveWithNothingToDeriveFrom) {
    GeometryPrimitive noUvs = Quad({});
    EXPECT_FALSE(TangentGenerator::FromUv(noUvs));
    EXPECT_TRUE(noUvs.tangents.empty());

    GeometryPrimitive noNormals =
        Quad({{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}});
    noNormals.normals.clear();
    EXPECT_FALSE(TangentGenerator::FromUv(noNormals));
    EXPECT_TRUE(noNormals.tangents.empty());
}
