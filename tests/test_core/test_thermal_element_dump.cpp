// ============================================================================
// Quantiloom - the per-element dump is one format, written from two paths
// ============================================================================
// thermal.dump_elements is how a mesh-refinement study gets at the temperature
// field underneath the image, and the offline solve is no longer its only
// writer: a viewport can now be asked for the same file through
// ExternalRenderContext::DumpThermalElements. Two writers is exactly how a file
// format acquires a second dialect, so there is one function and both call it.
//
// What the cases below pin is the part of the contract a reader depends on and
// no compiler checks: which column means what, and the two places where an
// absent number is deliberately not a zero.
// ============================================================================

#include <gtest/gtest.h>

#include "thermal/ThermalSolver.hpp"  // DumpThermalElements, MakeOpenSkyExchange

#include <filesystem>
#include <fstream>
#include <sstream>

using namespace quantiloom;
using namespace quantiloom::thermal;

namespace {

Vector<ThermalElement> TwoElements() {
    ThermalElement first;
    first.centroid = glm::vec3(1.0f, 2.0f, 3.0f);
    first.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    first.area_m2 = 0.5f;
    first.materialId = 0;

    ThermalElement second = first;
    second.centroid = glm::vec3(4.0f, 5.0f, 6.0f);
    second.materialId = 1;
    return {first, second};
}

/// Two materials that differ in every field the header block prints, so a
/// column written from the wrong one is visible rather than plausible.
Vector<ThermalMaterial> TwoMaterials() {
    ThermalMaterial solving;
    solving.conductivity_W_mK = 0.5f;
    solving.density_kg_m3 = 1600.0f;
    solving.specificHeat_J_kgK = 875.0f;
    solving.thickness_m = 0.5f;
    solving.convection_W_m2K = 10.0f;
    solving.shortwaveAbsorptivity = 0.79f;
    solving.longwaveEmissivity = 0.96f;
    solving.interiorBoundary = InteriorBoundary::Adiabatic;

    // Conductivity zero is how a config declines to solve a material.
    ThermalMaterial inert;
    inert.conductivity_W_mK = 0.0f;
    return {solving, inert};
}

/// The data rows, in order, each split on commas.
Vector<Vector<String>> ReadRows(const std::filesystem::path& path) {
    Vector<Vector<String>> rows;
    std::ifstream in(path);
    String line;
    bool seenHeader = false;
    while (std::getline(in, line)) {
        if (line.empty() || line.front() == '#') continue;
        if (!seenHeader) { seenHeader = true; continue; }
        Vector<String> fields;
        std::stringstream stream(line);
        String field;
        while (std::getline(stream, field, ',')) fields.push_back(field);
        rows.push_back(fields);
    }
    return rows;
}

class ThermalElementDumpTest : public ::testing::Test {
protected:
    void SetUp() override {
        dir = std::filesystem::temp_directory_path() / "ql_thermal_dump_test";
        std::filesystem::create_directories(dir);
        path = dir / "elements.csv";
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
    std::filesystem::path dir;
    std::filesystem::path path;
};

}  // namespace

TEST_F(ThermalElementDumpTest, AnUnsolvedElementIsMarkedRatherThanOmitted) {
    // A row per element in mesh order is the whole point: a study joins this
    // file against its own traversal of the same mesh, and a file that dropped
    // the elements it had nothing to say about would silently shift every
    // index after the first one.
    const auto elements = TwoElements();
    const auto materials = TwoMaterials();
    const auto exchange = MakeOpenSkyExchange(elements.size());

    // The second element's material does not participate, which the solve
    // reports as a temperature of zero.
    const Vector<f32> temperature{301.5f, 0.0f};
    const Vector<f32> sensitivity{2.25f, 0.0f};
    const Vector<f32> visibility{0.75f, 0.0f};

    DumpThermalElements(path.string(), elements, materials, exchange, temperature,
                        sensitivity, visibility);
    ASSERT_TRUE(std::filesystem::exists(path));

    const auto rows = ReadRows(path);
    ASSERT_EQ(rows.size(), 2u);

    // element,cx,cy,cz,nx,ny,nz,area,material_id,solved,T_K,dTdv_K,v,sky
    EXPECT_EQ(rows[0][0], "0");
    EXPECT_EQ(rows[0][8], "0") << "material id, not element index";
    EXPECT_EQ(rows[0][9], "1") << "a positive temperature is a solved element";
    EXPECT_FLOAT_EQ(std::stof(rows[0][10]), 301.5f);
    EXPECT_FLOAT_EQ(std::stof(rows[0][11]), 2.25f);
    EXPECT_FLOAT_EQ(std::stof(rows[0][12]), 0.75f);
    EXPECT_FLOAT_EQ(std::stof(rows[0][13]), 1.0f) << "open sky is a sky fraction of 1";

    EXPECT_EQ(rows[1][0], "1");
    EXPECT_EQ(rows[1][8], "1");
    EXPECT_EQ(rows[1][9], "0") << "an element the solve skipped says so";
}

TEST_F(ThermalElementDumpTest, NoTangentLeavesTheColumnEmptyRatherThanZero) {
    // sun_correction = false sizes the tangent out of the solve, so there is no
    // dT/dv to report. Reporting 0.0 would say the temperature does not move
    // with the sun, which is a measurement -- and a wrong one. The column is
    // left blank, which a reader parses as missing.
    //
    // v_element is not blank in the same case, and the asymmetry is the point:
    // the visibility is a property of the geometry and the sun that an element
    // has whether or not anyone asked how its temperature responds to it. Both
    // writers pass it either way.
    const auto elements = TwoElements();
    const auto materials = TwoMaterials();
    const auto exchange = MakeOpenSkyExchange(elements.size());
    const Vector<f32> temperature{301.5f, 299.0f};
    const Vector<f32> visibility{0.75f, 0.25f};

    DumpThermalElements(path.string(), elements, materials, exchange, temperature,
                        /*sunSensitivity_K=*/{}, visibility);

    const auto rows = ReadRows(path);
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_TRUE(rows[0][11].empty()) << "dTdv_K must be blank, not 0";
    EXPECT_FLOAT_EQ(std::stof(rows[0][12]), 0.75f) << "v_element still lands";
    EXPECT_FLOAT_EQ(std::stof(rows[0][10]), 301.5f) << "the temperature still lands";
}

TEST_F(ThermalElementDumpTest, TheHeaderCarriesThePropertiesTheSolveUsed) {
    // The reason this block exists at all: a material bound to a measured
    // spectrum is solved at the Planck-weighted band average of that curve
    // rather than at the emissivity its config typed, so reproducing an
    // element's trajectory from the config is an error that looks like a
    // result. Whatever the caller passed as the material table is what has to
    // appear, which is why the value below is not any config's default.
    auto materials = TwoMaterials();
    materials[0].longwaveEmissivity = 0.9329f;

    const auto elements = TwoElements();
    const auto exchange = MakeOpenSkyExchange(elements.size());
    DumpThermalElements(path.string(), elements, materials, exchange, {300.0f, 300.0f},
                        {}, {});

    std::ifstream in(path);
    String line;
    bool found = false;
    while (std::getline(in, line)) {
        if (line.rfind("# material 0:", 0) != 0) continue;
        found = true;
        EXPECT_NE(line.find("eps_lw=0.9329"), String::npos) << line;
        EXPECT_NE(line.find("solves=1"), String::npos) << line;
    }
    EXPECT_TRUE(found) << "no material block was written";

    // And the material that does not solve says so, since an element pointing
    // at it is skipped for that reason rather than for its own geometry.
    in.clear();
    in.seekg(0);
    while (std::getline(in, line)) {
        if (line.rfind("# material 1:", 0) != 0) continue;
        EXPECT_NE(line.find("solves=0"), String::npos) << line;
    }
}
