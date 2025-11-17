// ============================================================================
// Quantiloom M1 - Scene Builder Utility
// ============================================================================
// Provides helper functions to generate test scenes for M1 validation
// ============================================================================

#pragma once

#include "scene/Mesh.hpp"
#include <glm/glm.hpp>
#include <vector>

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
    static Mesh CreateGroundPlane(float size = 10.0f, float y = 0.0f) {
        Mesh mesh;
        mesh.name = "ground_plane";

        float half = size * 0.5f;

        // 4 vertices forming a quad
        mesh.positions = {
            {-half, y, -half},  // 0: Far-left
            { half, y, -half},  // 1: Far-right
            { half, y,  half},  // 2: Near-right
            {-half, y,  half},  // 3: Near-left
        };

        // 2 triangles (CCW winding for upward normal)
        mesh.indices = {
            0, 1, 2,  // First triangle
            0, 2, 3   // Second triangle
        };

        return mesh;
    }

    // ========================================================================
    // Box / Cube
    // ========================================================================

    // Create axis-aligned box
    // size: dimensions (width, height, depth)
    // center: box center position
    static Mesh CreateBox(glm::vec3 size, glm::vec3 center = glm::vec3(0.0f)) {
        Mesh mesh;
        mesh.name = "box";

        glm::vec3 half = size * 0.5f;
        glm::vec3 min = center - half;
        glm::vec3 max = center + half;

        // 8 vertices of box
        mesh.positions = {
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

        // 12 triangles (6 faces, 2 triangles each, CCW winding)
        mesh.indices = {
            // Bottom face (Y = min, normal = -Y)
            0, 2, 1,  0, 3, 2,

            // Top face (Y = max, normal = +Y)
            4, 5, 6,  4, 6, 7,

            // Front face (Z = max, normal = +Z)
            3, 6, 2,  3, 7, 6,

            // Back face (Z = min, normal = -Z)
            0, 1, 5,  0, 5, 4,

            // Left face (X = min, normal = -X)
            0, 7, 3,  0, 4, 7,

            // Right face (X = max, normal = +X)
            1, 2, 6,  1, 6, 5,
        };

        return mesh;
    }

    // Convenience: Create unit cube
    static Mesh CreateCube(float size, glm::vec3 center = glm::vec3(0.0f)) {
        return CreateBox(glm::vec3(size), center);
    }

    // ========================================================================
    // Sphere (Icosphere Approximation)
    // ========================================================================

    // Create UV sphere (latitude/longitude grid)
    // radius: sphere radius
    // center: sphere center position
    // subdivisions: resolution (16 = reasonable quality)
    static Mesh CreateSphere(float radius, glm::vec3 center = glm::vec3(0.0f),
                             u32 latSegments = 16, u32 lonSegments = 32) {
        Mesh mesh;
        mesh.name = "sphere";

        // Generate vertices
        for (u32 lat = 0; lat <= latSegments; ++lat) {
            float theta = static_cast<float>(lat) * glm::pi<float>() / static_cast<float>(latSegments);
            float sinTheta = glm::sin(theta);
            float cosTheta = glm::cos(theta);

            for (u32 lon = 0; lon <= lonSegments; ++lon) {
                float phi = static_cast<float>(lon) * 2.0f * glm::pi<float>() / static_cast<float>(lonSegments);
                float sinPhi = glm::sin(phi);
                float cosPhi = glm::cos(phi);

                glm::vec3 position(
                    radius * sinTheta * cosPhi,
                    radius * cosTheta,
                    radius * sinTheta * sinPhi
                );

                mesh.positions.push_back(center + position);
            }
        }

        // Generate indices (quads as 2 triangles)
        for (u32 lat = 0; lat < latSegments; ++lat) {
            for (u32 lon = 0; lon < lonSegments; ++lon) {
                u32 first = lat * (lonSegments + 1) + lon;
                u32 second = first + lonSegments + 1;

                // First triangle (CCW)
                mesh.indices.push_back(first);
                mesh.indices.push_back(second);
                mesh.indices.push_back(first + 1);

                // Second triangle (CCW)
                mesh.indices.push_back(second);
                mesh.indices.push_back(second + 1);
                mesh.indices.push_back(first + 1);
            }
        }

        return mesh;
    }

    // ========================================================================
    // Multi-Mesh Scene Composition
    // ========================================================================

    // Merge multiple meshes into one (for single BLAS)
    static Mesh MergeMeshes(const std::vector<Mesh>& meshes) {
        Mesh merged;
        merged.name = "merged_scene";

        for (const auto& mesh : meshes) {
            u32 indexOffset = static_cast<u32>(merged.positions.size());

            // Append positions
            merged.positions.insert(merged.positions.end(),
                                    mesh.positions.begin(), mesh.positions.end());

            // Append indices (with offset)
            for (u32 idx : mesh.indices) {
                merged.indices.push_back(idx + indexOffset);
            }
        }

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
