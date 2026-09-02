// ============================================================================
// Quantiloom - a thin shell is one slab exposed on both sides
// ============================================================================
// The solver gives every triangle a column with an exposed front and something
// behind it. A car panel, a road sign, a tent and an aircraft skin have nothing
// behind them: both faces see the sky, and an asset models them as two sheets
// of triangles.
//
// Solved naively that is two independent slabs, each insulated against a wall
// that is not there. A panel in the sun then comes out as hot as if it were
// bolted to masonry, and the face in shadow sits wherever the initial condition
// left it, because nothing in the model can reach it.
//
// With shell = true the pair shares one column: a full surface balance at each
// end, one thickness between them. Two things have to hold for that to be worth
// having -- the pairing has to find the faces of a shell and nothing else, and
// the shared column has to be symmetric when the forcing is.
// ============================================================================

#include <gtest/gtest.h>

#include "thermal/CpuCrankNicolsonStepper.hpp"
#include "thermal/ThermalMesh.hpp"
#include "thermal/ThermalSolver.hpp"  // MakeOpenSkyExchange

#include <cmath>

using namespace quantiloom;
using namespace quantiloom::thermal;

namespace {

/// A quad in the XZ plane at height y, facing up or down, as two triangles.
/// Two of these at the same height facing opposite ways is a shell.
void AddQuad(Scene& scene, const f32 y, const f32 halfSize, const u32 materialId,
             const bool faceUp) {
    GeometryPrimitive prim;
    prim.materialId = materialId;
    prim.positions = {
        glm::vec3(-halfSize, y, -halfSize), glm::vec3(halfSize, y, -halfSize),
        glm::vec3(halfSize, y, halfSize),   glm::vec3(-halfSize, y, halfSize),
    };
    prim.normals.assign(4, glm::vec3(0.0f, faceUp ? 1.0f : -1.0f, 0.0f));
    prim.indices = {0, 2, 1, 0, 3, 2};

    Mesh mesh;
    mesh.primitives.push_back(std::move(prim));
    scene.meshes.push_back(std::move(mesh));

    SceneNode node;
    node.meshIndex = static_cast<u32>(scene.meshes.size() - 1);
    node.transform = glm::mat4(1.0f);
    node.active = true;
    scene.nodes.push_back(node);
}

/// Both faces of one shell in ONE primitive, which is what an exported panel
/// looks like: the pairing does not cross a primitive, so a fixture that puts
/// them in two would pair nothing however close they were.
void AddShellPanel(Scene& scene, const f32 gap, const u32 materialId) {
    GeometryPrimitive prim;
    prim.materialId = materialId;
    const f32 half = 1.0f;
    prim.positions = {
        // The up-facing sheet.
        glm::vec3(-half, gap, -half), glm::vec3(half, gap, -half),
        glm::vec3(half, gap, half),   glm::vec3(-half, gap, half),
        // The down-facing sheet, directly beneath it.
        glm::vec3(-half, 0.0f, -half), glm::vec3(half, 0.0f, -half),
        glm::vec3(half, 0.0f, half),   glm::vec3(-half, 0.0f, half),
    };
    prim.normals = {
        glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0),
        glm::vec3(0, -1, 0), glm::vec3(0, -1, 0), glm::vec3(0, -1, 0), glm::vec3(0, -1, 0),
    };
    prim.indices = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7};

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
    scene.materials.push_back(Material::CreateLambertian(glm::vec3(0.5f), "Panel"));
    return scene;
}

ThermalMaterial Panel(const bool shell, const f32 thickness = 0.01f) {
    ThermalMaterial material;
    material.conductivity_W_mK = 150.0f;   // aluminium: a shell that conducts
    material.density_kg_m3 = 2700.0f;
    material.specificHeat_J_kgK = 900.0f;
    material.thickness_m = thickness;
    material.convection_W_m2K = 10.0f;
    material.shortwaveAbsorptivity = 0.7f;
    material.longwaveEmissivity = 0.9f;
    material.isShell = shell;
    return material;
}

}  // namespace

// ============================================================================
// The pairing finds the faces of a shell, and nothing else
// ============================================================================

TEST(ThermalShellTest, PairingIsOffUnlessAMaterialAsksForIt) {
    Scene scene = MakeScene();
    AddShellPanel(scene, 0.01f, 0);

    EXPECT_TRUE(BuildThermalMesh(scene).shellPartner.empty())
        << "a mesh built with no options pairs nothing";

    const ThermalMesh plain =
        BuildThermalMesh(scene, MeshOptionsFor({Panel(false)}, false));
    EXPECT_TRUE(plain.shellPartner.empty()) << "shell = false pairs nothing";
    EXPECT_EQ(plain.shellPairCount, 0u);
}

TEST(ThermalShellTest, TwoSheetsOfOnePanelPairUp) {
    Scene scene = MakeScene();
    AddShellPanel(scene, 0.01f, 0);

    const ThermalMesh mesh = BuildThermalMesh(scene, MeshOptionsFor({Panel(true)}, false));
    ASSERT_EQ(mesh.elements.size(), 4u);
    EXPECT_EQ(mesh.shellPairCount, 2u) << "two triangles a side, two pairs";
    EXPECT_EQ(mesh.shellUnpairedCount, 0u);

    // Each partner is on the other sheet, and the pairing is symmetric.
    for (u32 e = 0; e < 4; ++e) {
        const u32 partner = mesh.shellPartner[e];
        ASSERT_NE(partner, ThermalMesh::kNoShellPartner) << "element " << e;
        EXPECT_EQ(mesh.shellPartner[partner], e) << "element " << e;
        EXPECT_LT(glm::dot(mesh.elements[e].normal, mesh.elements[partner].normal), -0.5f)
            << "a shell's two faces point opposite ways";
    }
}

TEST(ThermalShellTest, SheetsTooFarApartAreNotOneShell) {
    // The rule is a small multiple of the material's own thickness. Two
    // triangles a metre apart are two surfaces of a box, and pairing them would
    // put a slab of aluminium where a room is.
    Scene scene = MakeScene();
    AddShellPanel(scene, 1.0f, 0);   // a metre apart, thickness 0.01

    const ThermalMesh mesh = BuildThermalMesh(scene, MeshOptionsFor({Panel(true)}, false));
    EXPECT_EQ(mesh.shellPairCount, 0u);
    EXPECT_EQ(mesh.shellUnpairedCount, 4u)
        << "and every one of them is reported rather than quietly one-sided";
}

TEST(ThermalShellTest, TwoSheetsFacingTheSameWayAreNotOneShell) {
    // Both up: a floor and a ceiling, not a panel. The distance test alone
    // would pair them.
    Scene scene = MakeScene();
    AddQuad(scene, 0.0f, 1.0f, 0, true);
    AddQuad(scene, 0.005f, 1.0f, 0, true);

    const ThermalMesh mesh = BuildThermalMesh(scene, MeshOptionsFor({Panel(true)}, false));
    EXPECT_EQ(mesh.shellPairCount, 0u);
}

TEST(ThermalShellTest, PairingDoesNotCrossAnObject) {
    // Two separate panels, each modelled as one sheet, a few millimetres apart.
    // Geometrically indistinguishable from one shell; different objects, and a
    // conductance between them is one nobody supplied.
    Scene scene = MakeScene();
    AddQuad(scene, 0.0f, 1.0f, 0, true);
    AddQuad(scene, 0.005f, 1.0f, 0, false);

    const ThermalMesh mesh = BuildThermalMesh(scene, MeshOptionsFor({Panel(true)}, false));
    EXPECT_EQ(mesh.shellPairCount, 0u) << "the pairing stops at the primitive";
    EXPECT_EQ(mesh.shellUnpairedCount, 4u);
}

// ============================================================================
// The shared column
// ============================================================================

namespace {

/// Step a two-element shell to something like equilibrium and report the two
/// face temperatures. Both faces see the same sky and the same air; whether
/// they see the same sun is the caller's business.
struct ShellRun {
    f64 front = 0.0;
    f64 back = 0.0;
};

ShellRun StepShell(const ThermalMaterial& material, const f32 frontVisibility,
                   const f32 backVisibility, const glm::vec3& sunDirection,
                   const bool paired) {
    // Two elements facing opposite ways, a millimetre apart.
    Vector<ThermalElement> elements(2);
    elements[0].centroid = glm::vec3(0.0f, material.thickness_m, 0.0f);
    elements[0].normal = glm::vec3(0.0f, 1.0f, 0.0f);
    elements[0].area_m2 = 1.0f;
    elements[1].centroid = glm::vec3(0.0f, 0.0f, 0.0f);
    elements[1].normal = glm::vec3(0.0f, -1.0f, 0.0f);
    elements[1].area_m2 = 1.0f;

    ExchangeGeometry exchange = MakeOpenSkyExchange(2);

    ThermalForcing forcing;
    forcing.airTemperature_K = 290.0;
    forcing.skyTemperature_K = 250.0;
    forcing.sunIrradiance_W_m2 = 900.0;
    forcing.sunDirection = sunDirection;

    CpuCrankNicolsonStepper stepper{ConvectionLaw{}};
    if (paired) stepper.SetShellPartners({1u, 0u});

    ThermalState state;
    state.nodeCount = 6;
    state.temperature_K.assign(2 * state.nodeCount, 290.0);

    const Vector<f32> visibility = {frontVisibility, backVisibility};
    ShortwaveSample shortwave;
    shortwave.sunVisibility = visibility;

    // Long enough for a thin aluminium sheet to settle: its time constant is
    // seconds, so an hour of minutes is equilibrium many times over.
    for (int step = 0; step < 60; ++step) {
        stepper.Step(state, elements, {material}, exchange, forcing, 60.0, shortwave);
    }
    return {state.Surface(0), state.Surface(1)};
}

}  // namespace

TEST(ThermalShellTest, ASymmetricShellUnderSymmetricForcingHasNoGradient) {
    // The closed form this has: with both faces seeing the same sun, the same
    // sky and the same air, nothing distinguishes them, so the slab is
    // isothermal whatever its conductivity. A back row that was still a
    // boundary condition rather than a second exposed face would show a
    // gradient instead.
    //
    // The sun is edge-on to both faces, which is the only way both can see the
    // same short-wave input: cos(theta) is zero on each.
    const ShellRun run = StepShell(Panel(true), 1.0f, 1.0f,
                                   glm::vec3(1.0f, 0.0f, 0.0f), true);

    EXPECT_GT(run.front, 100.0);
    EXPECT_NEAR(run.front, run.back, 1e-6)
        << "front " << run.front << " K, back " << run.back << " K";
}

TEST(ThermalShellTest, TheBackFaceOfAShellIsSolvedRatherThanLeftWhereItStarted) {
    // What the whole thing is for. Unpaired, the second element is its own
    // insulated slab and the two answers have nothing to do with each other;
    // paired, the sunlit face conducts into the shaded one and both move.
    const glm::vec3 overhead(0.0f, 1.0f, 0.0f);   // only the up-facing sheet is lit

    const ShellRun paired = StepShell(Panel(true), 1.0f, 1.0f, overhead, true);
    const ShellRun apart = StepShell(Panel(true), 1.0f, 1.0f, overhead, false);

    // Aluminium a centimetre thick: the two faces of one slab end up within a
    // fraction of a degree of each other, because the slab conducts far faster
    // than its surfaces exchange.
    EXPECT_NEAR(paired.front, paired.back, 0.5)
        << "front " << paired.front << " K, back " << paired.back << " K";

    // Solved apart, the shaded sheet never sees the sun at all and settles far
    // colder -- which is the wrong answer this exists to replace.
    EXPECT_GT(paired.back - apart.back, 5.0)
        << "paired back " << paired.back << " K, unpaired back " << apart.back << " K";
}

TEST(ThermalShellTest, AShellIsCoolerInTheSunThanASlabAgainstAWall) {
    // The physical claim. One face absorbs; a shell sheds through both faces
    // and an insulated slab sheds through one, so the shell runs cooler under
    // the same sun. If the back row were still adiabatic this difference would
    // be zero.
    const glm::vec3 overhead(0.0f, 1.0f, 0.0f);

    const ShellRun shell = StepShell(Panel(true), 1.0f, 1.0f, overhead, true);
    const ShellRun insulated = StepShell(Panel(false), 1.0f, 1.0f, overhead, false);

    EXPECT_LT(shell.front, insulated.front - 2.0)
        << "shell " << shell.front << " K, insulated " << insulated.front << " K";
}
