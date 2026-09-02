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
/**
 * @brief Two elements that share an edge, and the geometry of the join
 *
 * Pure geometry: which two, how long the edge they share is, and how far apart
 * their centroids are. What that is worth as a conductance depends on the
 * materials, and is BuildLateralConduction's business rather than the mesh's.
 */
struct ThermalContact {
    u32 a = 0;
    u32 b = 0;
    f32 sharedEdge_m = 0.0f;
    f32 centroidDistance_m = 0.0f;
};

struct ThermalMesh {
    Vector<ThermalElement> elements;
    /// First element index of each (node, primitive) instance, in the same
    /// order SceneGeometry enumerates them. The shader adds PrimitiveIndex()
    /// to this, so it is the whole of the mapping.
    Vector<u32> instanceElementBase;

    /// Which elements share an edge, when the caller asked for them. Built
    /// while the indices are still in hand -- they are the whole of the
    /// adjacency and this is the only place that has them.
    Vector<ThermalContact> contacts;

    /// Elements whose area is zero: degenerate triangles, which the solver
    /// skips. Counted rather than removed, because the shader indexes by
    /// PrimitiveIndex() and dropping one would shift every element after it in
    /// the same primitive onto the wrong slab.
    u32 degenerateCount = 0;
    /// Edges presented by more than two triangles of one object. Joined to the
    /// first two and reported: a T-junction is a modelling decision, not a
    /// conduction path this can resolve.
    u32 nonManifoldEdgeCount = 0;
};

struct ThermalMeshOptions {
    /// Find which elements share an edge. Off by default because it costs a
    /// hash insert per triangle edge over the whole scene, and nothing reads
    /// it unless lateral conduction is on.
    bool contacts = false;
};

/// Build the element list from a scene, one element per triangle, in the order
/// SceneGeometry enumerates instances.
[[nodiscard]] ThermalMesh BuildThermalMesh(const Scene& scene,
                                           const ThermalMeshOptions& options = {});

/**
 * @brief Turn shared edges into conductances
 *
 * g_ij = k_harmonic w_ij / d_ij, in W/(m K) -- a conductance per metre of slab
 * depth, so that a node's own cell height cancels out of the rate and every
 * node of a column conducts sideways at the same rate. The harmonic mean is
 * what two conductors in series have; a pair where either side does not take
 * part in the solve is dropped rather than given a zero.
 *
 * @return a symmetric CSR over @p elements, or an empty one when there is
 *         nothing to join
 */
[[nodiscard]] CsrMatrix BuildLateralConduction(const ThermalMesh& mesh,
                                               const Vector<ThermalMaterial>& materials);

/// The shortest lateral time constant, rho c A / sum_j g_ij, in seconds. The
/// lateral term is explicit, so a timestep much past this one oscillates --
/// reported the way the through-thickness time constant is, and for the same
/// reason. Infinite when nothing is joined.
[[nodiscard]] f64 LateralTimeConstantSeconds(const CsrMatrix& lateral,
                                             const Vector<ThermalElement>& elements,
                                             const Vector<ThermalMaterial>& materials);

}  // namespace quantiloom::thermal
