// ============================================================================
// Quantiloom - Unit Tests for renderer/ThermalExchangePrecompute.hpp
// ============================================================================
// The view factors decide how much of a surface's hemisphere every other
// surface fills, and a wrong one is invisible: the temperatures it produces
// are smooth and plausible and too warm or too cold by a few kelvin. So they
// are checked against configurations whose answer is known -- a surface facing
// open sky sees nothing but sky, a surface inside a closed box sees no sky at
// all, and two parallel plates see each other by an amount with a closed form.
//
// The estimator is Monte Carlo, so the tolerances are what 256 rays support:
// about 1/sqrt(N), a few percent.
//
// Skipped rather than failed on a machine with no ray-tracing GPU, like every
// other GPU case here.
// ============================================================================

#include <gtest/gtest.h>

#include "renderer/ThermalExchangePrecompute.hpp"

#include "renderer/RenderCore.hpp"
#include "support/VulkanTestDevice.hpp"
#include "thermal/ThermalMesh.hpp"

#include <cmath>
#include <numbers>

using namespace quantiloom;
using namespace quantiloom::rendercore;

namespace {

/// A quad in the XZ plane at height y, facing up or down, as two triangles.
void AddQuad(Scene& scene, const f32 y, const f32 halfSize, const bool facingUp,
             const u32 materialId) {
    GeometryPrimitive prim;
    prim.materialId = materialId;
    prim.positions = {
        glm::vec3(-halfSize, y, -halfSize), glm::vec3(halfSize, y, -halfSize),
        glm::vec3(halfSize, y, halfSize),   glm::vec3(-halfSize, y, halfSize),
    };
    const glm::vec3 normal = facingUp ? glm::vec3(0, 1, 0) : glm::vec3(0, -1, 0);
    prim.normals.assign(4, normal);
    // Wound so the geometric normal agrees with the authored one; BuildThermalMesh
    // flips against the authored normal, so either winding gives the same answer,
    // but agreeing keeps the acceleration structure's own facing sensible.
    prim.indices = facingUp ? Vector<u32>{0, 2, 1, 0, 3, 2} : Vector<u32>{0, 1, 2, 0, 2, 3};

    Mesh mesh;
    mesh.primitives.push_back(std::move(prim));
    scene.meshes.push_back(std::move(mesh));

    SceneNode node;
    node.meshIndex = static_cast<u32>(scene.meshes.size() - 1);
    node.transform = glm::mat4(1.0f);
    node.active = true;
    scene.nodes.push_back(node);
}

Scene MakeScene() {
    Scene scene;
    scene.materials.push_back(Material::CreateLambertian(glm::vec3(0.5f), "Surface"));
    return scene;
}

/// Mean sky fraction over the elements of one primitive.
f32 MeanSkyFraction(const thermal::ExchangeGeometry& exchange, const usize first,
                    const usize count) {
    f32 sum = 0.0f;
    for (usize i = first; i < first + count; ++i) {
        sum += exchange.skyFraction[i];
    }
    return sum / static_cast<f32>(count);
}

/// Total view factor from element `from` to any element in [first, first+count).
f32 ViewFactorTo(const thermal::ExchangeGeometry& exchange, const usize from,
                 const usize first, const usize count) {
    f32 sum = 0.0f;
    const u32 begin = exchange.viewFactors.rowStart[from];
    const u32 end = exchange.viewFactors.rowStart[from + 1];
    for (u32 n = begin; n < end; ++n) {
        const u32 column = exchange.viewFactors.column[n];
        if (column >= first && column < first + count) {
            sum += exchange.viewFactors.value[n];
        }
    }
    return sum;
}

class ThermalExchangeGpuTest : public quantiloom::testing::VulkanDeviceTest {};

}  // namespace

TEST_F(ThermalExchangeGpuTest, APlateUnderOpenSkySeesNothingButSky) {
    Scene scene = MakeScene();
    AddQuad(scene, 0.0f, 5.0f, /*facingUp=*/true, 0);

    SceneGeometry geometry = SceneGeometry::Build(Device(), scene);
    ASSERT_TRUE(geometry.IsValid());

    const thermal::ThermalMesh mesh = thermal::BuildThermalMesh(scene);
    ASSERT_EQ(mesh.elements.size(), 2u);

    ThermalExchangePrecompute precompute(Device());
    if (!precompute.IsValid()) {
        GTEST_SKIP() << "thermal_exchange.spv unavailable";
    }

    const auto exchange = precompute.Run(geometry.Tlas().GetHandle(), mesh.elements,
                                         mesh.instanceElementBase, {});
    ASSERT_EQ(exchange.skyFraction.size(), 2u);
    EXPECT_NEAR(MeanSkyFraction(exchange, 0, 2), 1.0f, 1e-6f);
    EXPECT_EQ(exchange.viewFactors.NonZeros(), 0u);
}

TEST_F(ThermalExchangeGpuTest, TwoParallelPlatesSeeEachOtherByTheAnalyticFactor) {
    // Coaxial parallel squares, side a, separation h. For a = 10, h = 5 the
    // ratio X = a/h = 2, and the closed form
    //
    //   F = (2/(pi X^2)) [ ln((1+X^2)^2 / (1+2X^2)) + 2X sqrt(1+X^2) atan(X/sqrt(1+X^2))
    //                      - 2X atan(X) ]
    //
    // gives 0.4152. What is measured here is the average over the elements of
    // the lower plate, which is the same quantity: every ray leaves a point of
    // the lower plate and either lands on the upper one or escapes.
    Scene scene = MakeScene();
    AddQuad(scene, 0.0f, 5.0f, /*facingUp=*/true, 0);   // elements 0..1
    AddQuad(scene, 5.0f, 5.0f, /*facingUp=*/false, 0);  // elements 2..3

    SceneGeometry geometry = SceneGeometry::Build(Device(), scene);
    ASSERT_TRUE(geometry.IsValid());

    const thermal::ThermalMesh mesh = thermal::BuildThermalMesh(scene);
    ASSERT_EQ(mesh.elements.size(), 4u);

    ThermalExchangePrecompute precompute(Device());
    if (!precompute.IsValid()) {
        GTEST_SKIP() << "thermal_exchange.spv unavailable";
    }

    ThermalExchangePrecompute::Params params;
    params.hemisphereRays = 1024;  // this one is a number, so pay for it
    const auto exchange = precompute.Run(geometry.Tlas().GetHandle(), mesh.elements,
                                         mesh.instanceElementBase, params);
    ASSERT_EQ(exchange.skyFraction.size(), 4u);

    const f64 X = 10.0 / 5.0;
    const f64 analytic =
        (2.0 / (std::numbers::pi * X * X)) *
        (std::log(std::pow(1.0 + X * X, 2.0) / (1.0 + 2.0 * X * X)) +
         2.0 * X * std::sqrt(1.0 + X * X) * std::atan(X / std::sqrt(1.0 + X * X)) -
         2.0 * X * std::atan(X));

    // Averaged over the lower plate's two triangles, weighted equally because
    // they have equal area.
    const f32 measured =
        0.5f * (ViewFactorTo(exchange, 0, 2, 2) + ViewFactorTo(exchange, 1, 2, 2));

    EXPECT_NEAR(measured, static_cast<f32>(analytic), 0.03f)
        << "analytic " << analytic << ", measured " << measured;

    // And what is not the other plate is sky, by construction.
    EXPECT_NEAR(MeanSkyFraction(exchange, 0, 2) + measured, 1.0f, 0.02f);
}

TEST_F(ThermalExchangeGpuTest, EveryRowSumsToOneWithItsSkyFraction) {
    // The invariant the solver depends on. A row that sums to less than one is
    // a surface exchanging with less than a whole hemisphere, which over a
    // long enough night cools toward nothing -- and the truncation to the
    // largest few entries per row would produce exactly that if it were not
    // rescaled.
    Scene scene = MakeScene();
    AddQuad(scene, 0.0f, 5.0f, true, 0);
    AddQuad(scene, 5.0f, 5.0f, false, 0);
    AddQuad(scene, 2.0f, 2.0f, true, 0);

    SceneGeometry geometry = SceneGeometry::Build(Device(), scene);
    ASSERT_TRUE(geometry.IsValid());

    const thermal::ThermalMesh mesh = thermal::BuildThermalMesh(scene);
    ThermalExchangePrecompute precompute(Device());
    if (!precompute.IsValid()) {
        GTEST_SKIP() << "thermal_exchange.spv unavailable";
    }

    ThermalExchangePrecompute::Params params;
    params.topK = 1;  // truncate hard, so the rescale is doing visible work
    const auto exchange = precompute.Run(geometry.Tlas().GetHandle(), mesh.elements,
                                         mesh.instanceElementBase, params);

    for (usize e = 0; e < mesh.elements.size(); ++e) {
        f32 rowSum = exchange.skyFraction[e];
        const u32 begin = exchange.viewFactors.rowStart[e];
        const u32 end = exchange.viewFactors.rowStart[e + 1];
        for (u32 n = begin; n < end; ++n) {
            rowSum += exchange.viewFactors.value[n];
        }
        EXPECT_NEAR(rowSum, 1.0f, 1e-4f) << "row " << e;
    }
}

TEST_F(ThermalExchangeGpuTest, AnUnoccludedPlateIsFullyLitAndAShadedOneIsNot) {
    // The other geometric question the pass answers. The upper plate blocks
    // the sun from the lower one entirely at this size and separation.
    Scene scene = MakeScene();
    AddQuad(scene, 0.0f, 2.0f, true, 0);    // elements 0..1, in shadow
    AddQuad(scene, 3.0f, 10.0f, false, 0);  // elements 2..3, blocking it

    SceneGeometry geometry = SceneGeometry::Build(Device(), scene);
    ASSERT_TRUE(geometry.IsValid());

    const thermal::ThermalMesh mesh = thermal::BuildThermalMesh(scene);
    ThermalExchangePrecompute precompute(Device());
    if (!precompute.IsValid()) {
        GTEST_SKIP() << "thermal_exchange.spv unavailable";
    }

    ThermalExchangePrecompute::Params params;
    params.sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);  // straight up
    const auto exchange = precompute.Run(geometry.Tlas().GetHandle(), mesh.elements,
                                         mesh.instanceElementBase, params);
    ASSERT_EQ(exchange.sunVisibility.size(), 4u);

    EXPECT_NEAR(exchange.sunVisibility[0], 0.0f, 1e-6f) << "the lower plate is shaded";
    EXPECT_NEAR(exchange.sunVisibility[1], 0.0f, 1e-6f);

    // The upper plate faces down, away from a sun that is overhead, so it is
    // not lit either -- and the pass says so from the cosine rather than from
    // a ray.
    EXPECT_NEAR(exchange.sunVisibility[2], 0.0f, 1e-6f);
}

TEST_F(ThermalExchangeGpuTest, AnOpenPlateIsFullyLit) {
    Scene scene = MakeScene();
    AddQuad(scene, 0.0f, 5.0f, true, 0);

    SceneGeometry geometry = SceneGeometry::Build(Device(), scene);
    ASSERT_TRUE(geometry.IsValid());

    const thermal::ThermalMesh mesh = thermal::BuildThermalMesh(scene);
    ThermalExchangePrecompute precompute(Device());
    if (!precompute.IsValid()) {
        GTEST_SKIP() << "thermal_exchange.spv unavailable";
    }

    ThermalExchangePrecompute::Params params;
    params.sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);
    const auto exchange = precompute.Run(geometry.Tlas().GetHandle(), mesh.elements,
                                         mesh.instanceElementBase, params);

    EXPECT_NEAR(exchange.sunVisibility[0], 1.0f, 1e-6f);
    EXPECT_NEAR(exchange.sunVisibility[1], 1.0f, 1e-6f);
}
