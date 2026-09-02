// ============================================================================
// Quantiloom - lateral conduction between elements
// ============================================================================
// The solver gives every triangle its own one-dimensional column, so the
// temperature field it produces can only have edges where the mesh has edges.
// For dry sand at an hour's timescale that is very nearly true -- heat
// diffuses about three centimetres in an hour, well inside a 0.6 m triangle --
// and for a metal panel it is not true at any timescale.
//
// What lateral conduction adds is one explicit term coupling elements that
// share an edge. Two things have to hold for it to be worth having: the
// adjacency has to be the mesh's own, and the term has to diffuse at the rate
// the material says. Both are checked here against answers that do not come
// from this solver.
// ============================================================================

#include <gtest/gtest.h>

#include "thermal/CpuCrankNicolsonStepper.hpp"
#include "thermal/ThermalMesh.hpp"
#include "thermal/ThermalSolver.hpp"  // MakeOpenSkyExchange

#include <cmath>

using namespace quantiloom;
using namespace quantiloom::thermal;

namespace {

/// A quad in the XZ plane at height y, as two triangles sharing one diagonal.
void AddQuad(Scene& scene, const f32 y, const f32 halfSize, const u32 materialId,
             const glm::vec3& offset = glm::vec3(0.0f)) {
    GeometryPrimitive prim;
    prim.materialId = materialId;
    prim.positions = {
        glm::vec3(-halfSize, y, -halfSize), glm::vec3(halfSize, y, -halfSize),
        glm::vec3(halfSize, y, halfSize),   glm::vec3(-halfSize, y, halfSize),
    };
    prim.normals.assign(4, glm::vec3(0, 1, 0));
    prim.indices = {0, 2, 1, 0, 3, 2};

    Mesh mesh;
    mesh.primitives.push_back(std::move(prim));
    scene.meshes.push_back(std::move(mesh));

    SceneNode node;
    node.meshIndex = static_cast<u32>(scene.meshes.size() - 1);
    node.transform = glm::translate(glm::mat4(1.0f), offset);
    node.active = true;
    scene.nodes.push_back(node);
}

Scene MakeScene() {
    Scene scene;
    scene.materials.push_back(Material::CreateLambertian(glm::vec3(0.5f), "Surface"));
    return scene;
}

ThermalMaterial Conductor() {
    ThermalMaterial material;
    material.conductivity_W_mK = 1.0f;
    material.density_kg_m3 = 2000.0f;
    material.specificHeat_J_kgK = 900.0f;
    material.thickness_m = 0.05f;
    material.convection_W_m2K = 0.0f;   // nothing but the sideways term
    material.shortwaveAbsorptivity = 0.0f;
    material.longwaveEmissivity = 0.0f;
    material.interiorBoundary = InteriorBoundary::Adiabatic;
    return material;
}

}  // namespace

// ============================================================================
// The adjacency is the mesh's own
// ============================================================================

TEST(ThermalLateralTest, ContactsAreOffUnlessAskedFor) {
    Scene scene = MakeScene();
    AddQuad(scene, 0.0f, 1.0f, 0);

    EXPECT_TRUE(BuildThermalMesh(scene).contacts.empty());
    EXPECT_EQ(BuildThermalMesh(scene, {.contacts = true}).contacts.size(), 1u)
        << "two triangles of a quad share their diagonal";
}

TEST(ThermalLateralTest, TheSharedEdgeAndTheCentroidGapAreMeasured) {
    // A unit quad's diagonal is sqrt(2) long, and the two centroids sit at
    // (+1/3, -1/3) and (-1/3, +1/3) of the half size either side of it.
    Scene scene = MakeScene();
    AddQuad(scene, 0.0f, 1.0f, 0);

    const ThermalMesh mesh = BuildThermalMesh(scene, {.contacts = true});
    ASSERT_EQ(mesh.contacts.size(), 1u);

    EXPECT_NEAR(mesh.contacts[0].sharedEdge_m, 2.0f * std::sqrt(2.0f), 1e-4f);
    const glm::vec3 gap =
        mesh.elements[mesh.contacts[0].a].centroid - mesh.elements[mesh.contacts[0].b].centroid;
    EXPECT_NEAR(mesh.contacts[0].centroidDistance_m, glm::length(gap), 1e-5f);
}

TEST(ThermalLateralTest, TwoObjectsThatTouchAreNotJoined) {
    // Heat does cross a contact, but through a contact conductance nobody has
    // supplied. Joining a box to the ground because they touch would be
    // inventing one, so adjacency stops at the object.
    Scene scene = MakeScene();
    AddQuad(scene, 0.0f, 1.0f, 0);
    AddQuad(scene, 0.0f, 1.0f, 0, glm::vec3(2.0f, 0.0f, 0.0f));  // shares an edge

    const ThermalMesh mesh = BuildThermalMesh(scene, {.contacts = true});
    ASSERT_EQ(mesh.elements.size(), 4u);
    EXPECT_EQ(mesh.contacts.size(), 2u) << "one diagonal each, and nothing across";
}

TEST(ThermalLateralTest, AnInsulatorIsNotJoinedToItsNeighbour) {
    // A conductance is only as good as the worse of the two sides, and a
    // material that does not take part in the solve has no temperature to
    // conduct with.
    Scene scene = MakeScene();
    AddQuad(scene, 0.0f, 1.0f, 0);
    const ThermalMesh mesh = BuildThermalMesh(scene, {.contacts = true});

    ThermalMaterial inert;
    inert.conductivity_W_mK = 0.0f;
    EXPECT_EQ(BuildLateralConduction(mesh, {inert}).NonZeros(), 0u);
    EXPECT_EQ(BuildLateralConduction(mesh, {Conductor()}).NonZeros(), 2u)
        << "one join, both ways round";
}

TEST(ThermalLateralTest, TheConductanceIsTheHarmonicMeanOverTheGap) {
    Scene scene = MakeScene();
    AddQuad(scene, 0.0f, 1.0f, 0);
    const ThermalMesh mesh = BuildThermalMesh(scene, {.contacts = true});

    ThermalMaterial material = Conductor();
    material.conductivity_W_mK = 4.0f;
    const CsrMatrix lateral = BuildLateralConduction(mesh, {material});
    ASSERT_EQ(lateral.NonZeros(), 2u);

    const f64 expected = 4.0 * mesh.contacts[0].sharedEdge_m /
                         mesh.contacts[0].centroidDistance_m;
    EXPECT_NEAR(lateral.value[0], expected, expected * 1e-5);
    EXPECT_NEAR(lateral.value[1], expected, expected * 1e-5)
        << "the matrix is symmetric, since the join is";
}

// ============================================================================
// The term diffuses at the rate the material says
// ============================================================================

TEST(ThermalLateralTest, AStepAlongAChainDiffusesAsTheErrorFunctionSays) {
    // A row of square elements joined edge to edge is the one-dimensional
    // diffusion equation with spacing L, and a step in an infinite medium has
    // a closed form:
    //
    //     T(x, t) = T_cold + (T_hot - T_cold)/2 erfc(x / (2 sqrt(alpha t)))
    //
    // with alpha = k / (rho c) and x measured from the interface. Nothing else
    // is on: no sun, no sky, no convection, an insulated back, so the only
    // thing that can move a column is its neighbours.
    //
    // The chain is built by hand rather than from a scene, because what is
    // being checked is the rate rather than the adjacency -- the tests above
    // are the ones that check the adjacency.
    constexpr usize kCells = 80;
    constexpr f64 kSide_m = 0.05;
    constexpr f64 kHot = 400.0;
    constexpr f64 kCold = 300.0;

    const ThermalMaterial material = Conductor();
    const f64 alpha = material.conductivity_W_mK /
                      (static_cast<f64>(material.density_kg_m3) * material.specificHeat_J_kgK);

    Vector<ThermalElement> elements(kCells);
    for (usize e = 0; e < kCells; ++e) {
        elements[e].centroid = glm::vec3(static_cast<f32>(e) * kSide_m, 0.0f, 0.0f);
        elements[e].normal = glm::vec3(0.0f, 1.0f, 0.0f);
        elements[e].area_m2 = static_cast<f32>(kSide_m * kSide_m);
        elements[e].materialId = 0;
    }

    // g = k w / d with w = d = L, so g = k and the chain's discrete Laplacian
    // is the continuous one to second order in L.
    ExchangeGeometry exchange = MakeOpenSkyExchange(kCells);
    exchange.lateral.rowStart.assign(kCells + 1, 0u);
    for (usize e = 0; e < kCells; ++e) {
        const u32 neighbours = (e == 0 || e + 1 == kCells) ? 1u : 2u;
        exchange.lateral.rowStart[e + 1] = exchange.lateral.rowStart[e] + neighbours;
    }
    for (usize e = 0; e < kCells; ++e) {
        u32 at = exchange.lateral.rowStart[e];
        if (e > 0) {
            exchange.lateral.column.push_back(static_cast<u32>(e - 1));
            exchange.lateral.value.push_back(material.conductivity_W_mK);
            ++at;
        }
        if (e + 1 < kCells) {
            exchange.lateral.column.push_back(static_cast<u32>(e + 1));
            exchange.lateral.value.push_back(material.conductivity_W_mK);
        }
    }

    ThermalState state;
    state.nodeCount = 4;
    state.temperature_K.resize(kCells * state.nodeCount);
    for (usize e = 0; e < kCells; ++e) {
        const f64 T = e < kCells / 2 ? kHot : kCold;
        for (u32 i = 0; i < state.nodeCount; ++i) {
            state.temperature_K[e * state.nodeCount + i] = T;
        }
    }

    ThermalForcing forcing;
    forcing.airTemperature_K = 300.0;
    forcing.skyTemperature_K = 300.0;

    // dt against the shortest lateral time constant, rho c A / sum g = 2250 s
    // here: the term is explicit, so this stays well inside twice it.
    constexpr f64 kStep_s = 500.0;
    constexpr usize kSteps = 144;
    CpuCrankNicolsonStepper stepper;
    for (usize i = 0; i < kSteps; ++i) {
        stepper.Step(state, elements, {material}, exchange, forcing, kStep_s,
                     {exchange.sunVisibility});
    }

    const f64 t = kStep_s * static_cast<f64>(kSteps);
    const f64 diffusionLength = 2.0 * std::sqrt(alpha * t);
    ASSERT_GT(diffusionLength, 4.0 * kSide_m) << "the front has to span several cells";

    // The interface lies between cell 39 and cell 40, so a cell centre sits
    // half a spacing either side of it.
    f64 worst = 0.0;
    for (usize e = kCells / 2 - 8; e < kCells / 2 + 8; ++e) {
        const f64 x = (static_cast<f64>(e) - (static_cast<f64>(kCells) / 2.0 - 0.5)) * kSide_m;
        const f64 analytic = kCold + 0.5 * (kHot - kCold) * std::erfc(x / diffusionLength);
        worst = std::max(worst, std::abs(state.Surface(e) - analytic));
    }
    EXPECT_LT(worst, 1.0) << "a 100 K step, so this is one percent of the amplitude";

    // The column stays isothermal through its depth: every node of an element
    // gains at the same rate, which is what makes the term one per element
    // rather than one per node.
    for (u32 i = 1; i < state.nodeCount; ++i) {
        EXPECT_NEAR(state.temperature_K[(kCells / 2) * state.nodeCount + i],
                    state.Surface(kCells / 2), 1e-9);
    }
}

TEST(ThermalLateralTest, NoJoinsIsTheSolverThatWasThereBefore) {
    // The default has to leave every existing scene exactly where it was.
    const ThermalMaterial material = Conductor();
    Vector<ThermalElement> elements(2);
    for (usize e = 0; e < elements.size(); ++e) {
        elements[e].centroid = glm::vec3(static_cast<f32>(e), 0.0f, 0.0f);
        elements[e].normal = glm::vec3(0.0f, 1.0f, 0.0f);
        elements[e].area_m2 = 1.0f;
    }

    const auto bare = MakeOpenSkyExchange(2);
    ExchangeGeometry empty = MakeOpenSkyExchange(2);
    empty.lateral.rowStart.assign(3, 0u);  // rows, but no entries

    ThermalForcing forcing;
    forcing.airTemperature_K = 300.0;

    const auto run = [&](const ExchangeGeometry& exchange) {
        ThermalState state;
        state.nodeCount = 6;
        state.temperature_K.assign(2 * state.nodeCount, 0.0);
        for (u32 i = 0; i < state.nodeCount; ++i) {
            state.temperature_K[i] = 350.0;
            state.temperature_K[state.nodeCount + i] = 280.0;
        }
        CpuCrankNicolsonStepper stepper;
        for (usize s = 0; s < 20; ++s) {
            stepper.Step(state, elements, {material}, exchange, forcing, 60.0,
                         {exchange.sunVisibility});
        }
        return state.temperature_K;
    };

    const Vector<f64> a = run(bare);
    const Vector<f64> b = run(empty);
    for (usize i = 0; i < a.size(); ++i) {
        EXPECT_DOUBLE_EQ(a[i], b[i]) << "node " << i;
        EXPECT_DOUBLE_EQ(a[i], i < 6 ? 350.0 : 280.0)
            << "with nothing else on and nothing to conduct to, nothing moves";
    }
}
