// ============================================================================
// Quantiloom M1 - Scene Builder Utility
// ============================================================================
// Provides helper functions to generate test scenes for M1 validation
// ============================================================================

#pragma once

#include "scene/Mesh.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>  // For glm::pi<T>()
#include <vector>
#include <map>  // For icosphere midpoint cache

namespace quantiloom {

// ============================================================================
// Scene Configuration Structures
// ============================================================================

struct CameraConfig {
    glm::vec3 position;      // Camera eye position (world space)
    glm::vec3 lookAt;        // Point camera is looking at
    glm::vec3 up;            // Up vector (usually (0,1,0))
    float fovYDegrees;       // Vertical field-of-view (degrees)

    // Presets
    static CameraConfig DefaultOverview() {
        return CameraConfig{
            .position = glm::vec3(0.0f, 2.0f, -8.0f),
            .lookAt = glm::vec3(0.0f, 1.0f, 0.0f),
            .up = glm::vec3(0.0f, 1.0f, 0.0f),
            .fovYDegrees = 60.0f
        };
    }

    static CameraConfig GroundLevel() {
        return CameraConfig{
            .position = glm::vec3(0.0f, 0.5f, -5.0f),
            .lookAt = glm::vec3(0.0f, 0.5f, 0.0f),
            .up = glm::vec3(0.0f, 1.0f, 0.0f),
            .fovYDegrees = 70.0f
        };
    }

    static CameraConfig TopDown() {
        return CameraConfig{
            .position = glm::vec3(0.0f, 10.0f, 0.1f),
            .lookAt = glm::vec3(0.0f, 0.0f, 0.0f),
            .up = glm::vec3(0.0f, 0.0f, 1.0f),
            .fovYDegrees = 45.0f
        };
    }
};

struct LightingConfig {
    glm::vec3 sunDirection;   // FROM surface TO sun (normalized)
    glm::vec3 sunRadiance;    // Sun color/intensity
    glm::vec3 skyRadiance;    // Sky background color

    // Presets
    static LightingConfig Standard3Point() {
        return LightingConfig{
            .sunDirection = glm::normalize(glm::vec3(-0.5f, 0.8f, -0.3f)),
            .sunRadiance = glm::vec3(3.0f, 3.0f, 3.0f),
            .skyRadiance = glm::vec3(0.3f, 0.5f, 0.8f)
        };
    }

    static LightingConfig MorningLight() {
        return LightingConfig{
            .sunDirection = glm::normalize(glm::vec3(0.7f, 0.3f, -0.2f)),
            .sunRadiance = glm::vec3(4.0f, 3.5f, 2.8f),
            .skyRadiance = glm::vec3(0.8f, 0.6f, 0.4f)
        };
    }

    static LightingConfig NoonOverhead() {
        return LightingConfig{
            .sunDirection = glm::normalize(glm::vec3(0.0f, 1.0f, 0.1f)),
            .sunRadiance = glm::vec3(5.0f, 5.0f, 5.0f),
            .skyRadiance = glm::vec3(0.4f, 0.6f, 1.0f)
        };
    }

    static LightingConfig Backlight() {
        return LightingConfig{
            .sunDirection = glm::normalize(glm::vec3(0.0f, 0.5f, 1.0f)),
            .sunRadiance = glm::vec3(6.0f, 6.0f, 6.0f),
            .skyRadiance = glm::vec3(0.2f, 0.3f, 0.5f)
        };
    }
};

// ============================================================================
// Primitive Mesh Generators
// ============================================================================

class SceneBuilder {
public:
    // ========================================================================
    // Ground Plane
    // ========================================================================

    // Create horizontal ground plane (Y=0)
    // size: side length in meters (e.g., 10.0 = 10x10 m)
    // y: vertical offset (default 0.0)
    // materialId: index into Scene::materials
    static Mesh CreateGroundPlane(float size = 10.0f, float y = 0.0f, u32 materialId = 0) {
        Mesh mesh;
        mesh.name = "ground_plane";

        GeometryPrimitive primitive;
        primitive.materialId = materialId;

        float half = size * 0.5f;

        // 4 vertices forming a quad
        primitive.positions = {
            {-half, y, -half},  // 0: Far-left
            { half, y, -half},  // 1: Far-right
            { half, y,  half},  // 2: Near-right
            {-half, y,  half},  // 3: Near-left
        };

        // 2 triangles (CCW winding for upward normal)
        // When viewed from above (Y+), vertices go counter-clockwise
        primitive.indices = {
            0, 2, 1,  // First triangle: CCW from above
            0, 3, 2   // Second triangle: CCW from above
        };

        mesh.primitives.push_back(std::move(primitive));
        return mesh;
    }

    // ========================================================================
    // Box / Cube
    // ========================================================================

    // Create axis-aligned box
    // size: dimensions (width, height, depth)
    // center: box center position
    // materialId: index into Scene::materials
    static Mesh CreateBox(glm::vec3 size, glm::vec3 center = glm::vec3(0.0f), u32 materialId = 0) {
        Mesh mesh;
        mesh.name = "box";

        GeometryPrimitive primitive;
        primitive.materialId = materialId;

        glm::vec3 half = size * 0.5f;
        glm::vec3 min = center - half;
        glm::vec3 max = center + half;

        // 8 vertices of box
        primitive.positions = {
            // Bottom face (Y = min.y)
            {min.x, min.y, min.z},  // 0
            {max.x, min.y, min.z},  // 1
            {max.x, min.y, max.z},  // 2
            {min.x, min.y, max.z},  // 3

            // Top face (Y = max.y)
            {min.x, max.y, min.z},  // 4
            {max.x, max.y, min.z},  // 5
            {max.x, max.y, max.z},  // 6
            {min.x, max.y, max.z},  // 7
        };

        // 12 triangles (6 faces, 2 triangles each, CCW winding from outside)
        primitive.indices = {
            // Bottom face (Y = min, normal = -Y, looking from below)
            0, 1, 2,  0, 2, 3,

            // Top face (Y = max, normal = +Y, looking from above)
            4, 6, 5,  4, 7, 6,

            // Front face (Z = max, normal = +Z, looking from front)
            3, 2, 6,  3, 6, 7,

            // Back face (Z = min, normal = -Z, looking from back)
            0, 5, 1,  0, 4, 5,

            // Left face (X = min, normal = -X, looking from left)
            0, 3, 7,  0, 7, 4,

            // Right face (X = max, normal = +X, looking from right)
            1, 6, 2,  1, 5, 6,
        };

        mesh.primitives.push_back(std::move(primitive));
        return mesh;
    }

    // Convenience: Create unit cube
    static Mesh CreateCube(float size, glm::vec3 center = glm::vec3(0.0f), u32 materialId = 0) {
        return CreateBox(glm::vec3(size), center, materialId);
    }

    // ========================================================================
    // Sphere (Icosphere - No Degenerate Triangles)
    // ========================================================================

    // Create icosphere (subdivided icosahedron)
    // radius: sphere radius
    // center: sphere center position
    // subdivisions: number of subdivisions (0 = 20 triangles, 1 = 80, 2 = 320, etc.)
    // materialId: index into Scene::materials
    static Mesh CreateSphere(float radius, glm::vec3 center = glm::vec3(0.0f),
                             u32 subdivisions = 2, u32 materialId = 0) {
        Mesh mesh;
        mesh.name = "icosphere";

        GeometryPrimitive primitive;
        primitive.materialId = materialId;

        // Golden ratio constant
        constexpr float phi = 1.618033988749895f;  // (1 + sqrt(5)) / 2

        // Step 1: Create base icosahedron (12 vertices, 20 triangles)
        // Vertices arranged on 3 perpendicular golden rectangles
        const float a = 1.0f;
        const float b = 1.0f / phi;

        std::vector<glm::vec3> baseVertices = {
            {-b,  a,  0}, { b,  a,  0}, {-b, -a,  0}, { b, -a,  0},  // Rectangle in XY plane
            { 0, -b,  a}, { 0,  b,  a}, { 0, -b, -a}, { 0,  b, -a},  // Rectangle in YZ plane
            { a,  0, -b}, { a,  0,  b}, {-a,  0, -b}, {-a,  0,  b}   // Rectangle in XZ plane
        };

        // Normalize base vertices to unit sphere
        for (auto& v : baseVertices) {
            v = glm::normalize(v);
        }

        // Base icosahedron faces (20 triangles, CCW from outside)
        std::vector<u32> baseIndices = {
            // 5 faces around point 0
            0, 11, 5,   0, 5, 1,   0, 1, 7,   0, 7, 10,   0, 10, 11,
            // 5 adjacent faces
            1, 5, 9,   5, 11, 4,   11, 10, 2,   10, 7, 6,   7, 1, 8,
            // 5 faces around point 3
            3, 9, 4,   3, 4, 2,   3, 2, 6,   3, 6, 8,   3, 8, 9,
            // 5 adjacent faces
            4, 9, 5,   2, 4, 11,   6, 2, 10,   8, 6, 7,   9, 8, 1
        };

        // Step 2: Subdivide triangles
        primitive.positions = baseVertices;
        primitive.indices = baseIndices;

        for (u32 sub = 0; sub < subdivisions; ++sub) {
            std::vector<u32> newIndices;
            newIndices.reserve(primitive.indices.size() * 4);

            // Cache for midpoint vertices to avoid duplicates
            std::map<std::pair<u32, u32>, u32> midpointCache;

            auto GetMidpoint = [&](u32 i0, u32 i1) -> u32 {
                // Ensure consistent ordering (smaller index first)
                if (i0 > i1) std::swap(i0, i1);

                auto key = std::make_pair(i0, i1);
                auto it = midpointCache.find(key);
                if (it != midpointCache.end()) {
                    return it->second;  // Midpoint already exists
                }

                // Create new midpoint vertex
                glm::vec3 v0 = primitive.positions[i0];
                glm::vec3 v1 = primitive.positions[i1];
                glm::vec3 midpoint = glm::normalize((v0 + v1) * 0.5f);  // Project to sphere

                u32 newIndex = static_cast<u32>(primitive.positions.size());
                primitive.positions.push_back(midpoint);
                midpointCache[key] = newIndex;

                return newIndex;
            };

            // Subdivide each triangle into 4 smaller triangles
            for (size_t i = 0; i < primitive.indices.size(); i += 3) {
                u32 v0 = primitive.indices[i + 0];
                u32 v1 = primitive.indices[i + 1];
                u32 v2 = primitive.indices[i + 2];

                // Get midpoints of edges
                u32 m01 = GetMidpoint(v0, v1);
                u32 m12 = GetMidpoint(v1, v2);
                u32 m20 = GetMidpoint(v2, v0);

                // Create 4 new triangles (CCW order preserved)
                //       v0
                //      /  \
                //    m01--m20
                //    / \  / \
                //  v1--m12--v2

                newIndices.push_back(v0);   newIndices.push_back(m01); newIndices.push_back(m20);
                newIndices.push_back(v1);   newIndices.push_back(m12); newIndices.push_back(m01);
                newIndices.push_back(v2);   newIndices.push_back(m20); newIndices.push_back(m12);
                newIndices.push_back(m01);  newIndices.push_back(m12); newIndices.push_back(m20);
            }

            primitive.indices = std::move(newIndices);
        }

        // Step 3: Scale to desired radius and translate to center
        for (auto& v : primitive.positions) {
            v = center + v * radius;
        }

        mesh.primitives.push_back(std::move(primitive));
        return mesh;
    }

    // ========================================================================
    // Multi-Mesh Scene Composition
    // ========================================================================

    // Merge multiple meshes into one (for compatibility with M1 tests)
    // PROPERLY merges all vertices and indices into a single primitive
    static Mesh MergeMeshes(const std::vector<Mesh>& meshes) {
        Mesh merged;
        merged.name = "merged_scene";

        GeometryPrimitive mergedPrim;
        mergedPrim.materialId = 0;  // Use default material

        // Merge all vertices and indices from all primitives
        for (const auto& mesh : meshes) {
            for (const auto& prim : mesh.primitives) {
                u32 baseVertex = static_cast<u32>(mergedPrim.positions.size());

                // Append vertices
                mergedPrim.positions.insert(
                    mergedPrim.positions.end(),
                    prim.positions.begin(),
                    prim.positions.end()
                );

                // Append indices (offset by baseVertex)
                for (u32 idx : prim.indices) {
                    mergedPrim.indices.push_back(idx + baseVertex);
                }
            }
        }

        merged.primitives.push_back(std::move(mergedPrim));
        return merged;
    }
};

// ============================================================================
// Predefined Test Scenes
// ============================================================================

class TestScenes {
public:
    // Scene 1: Minimal Cornell Box Style
    static Mesh CreateCornellBoxScene() {
        std::vector<Mesh> objects;

        // Ground
        objects.push_back(SceneBuilder::CreateGroundPlane(10.0f));

        // Center cube (0.5x0.5x0.5 at origin)
        objects.push_back(SceneBuilder::CreateCube(0.5f, glm::vec3(0.0f, 0.25f, 0.0f)));

        return SceneBuilder::MergeMeshes(objects);
    }

    // Scene 2: Multi-Object Layout
    static Mesh CreateMultiObjectScene() {
        std::vector<Mesh> objects;

        // Ground
        objects.push_back(SceneBuilder::CreateGroundPlane(15.0f));

        // Tall box (left)
        objects.push_back(SceneBuilder::CreateBox(
            glm::vec3(1.0f, 2.0f, 1.0f),
            glm::vec3(-2.5f, 1.0f, 0.0f)
        ));

        // Cube (center)
        objects.push_back(SceneBuilder::CreateCube(1.0f, glm::vec3(0.0f, 0.5f, 0.0f)));

        // Sphere (right)
        objects.push_back(SceneBuilder::CreateSphere(0.5f, glm::vec3(2.5f, 0.5f, 0.0f)));

        return SceneBuilder::MergeMeshes(objects);
    }

    // Scene 3: Lighting Test (Row of cubes)
    static Mesh CreateLightingTestScene() {
        std::vector<Mesh> objects;

        // Ground
        objects.push_back(SceneBuilder::CreateGroundPlane(20.0f));

        // 5 cubes in a row
        for (int i = -2; i <= 2; ++i) {
            objects.push_back(SceneBuilder::CreateCube(
                0.8f,
                glm::vec3(static_cast<float>(i) * 2.0f, 0.4f, 0.0f)
            ));
        }

        return SceneBuilder::MergeMeshes(objects);
    }
};

} // namespace quantiloom
