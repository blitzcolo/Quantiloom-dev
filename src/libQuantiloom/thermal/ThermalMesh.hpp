/**
 * @file ThermalMesh.hpp
 * @brief The scene's triangles, as things that exchange heat
 */

#pragma once

#include "scene/Scene.hpp"
#include "thermal/ThermalTypes.hpp"

namespace quantiloom::thermal {

/**
 * @brief Every triangle of the scene, in world space, with where it came from
 *
 * The element list and the map back into the render's instances. The shader
 * finds an element by adding its instance's base to PrimitiveIndex(), so the
 * order here has to be the order SceneGeometry builds instances in: node by
 * node, primitive by primitive, triangle by triangle. That is why this walks
 * the scene the same way rather than sorting or filtering.
 */
struct ThermalMesh {
    Vector<ThermalElement> elements;
    /// First element index of each (node, primitive) instance, in the same
    /// order SceneGeometry enumerates them. The shader adds PrimitiveIndex()
    /// to this, so it is the whole of the mapping.
    Vector<u32> instanceElementBase;

    /// Elements whose area is zero: degenerate triangles, which the solver
    /// skips. Counted rather than removed, because the shader indexes by
    /// PrimitiveIndex() and dropping one would shift every element after it in
    /// the same primitive onto the wrong slab.
    u32 degenerateCount = 0;
};

/// Build the element list from a scene, one element per triangle, in the order
/// SceneGeometry enumerates instances.
[[nodiscard]] ThermalMesh BuildThermalMesh(const Scene& scene);

}  // namespace quantiloom::thermal
