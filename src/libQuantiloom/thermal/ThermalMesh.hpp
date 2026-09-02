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

    /// For each element, the triangle on the other side of the same thin
    /// shell, or kNoShellPartner. Built only for materials that declared
    /// themselves a shell.
    ///
    /// A shell is one slab exposed on both sides: a car panel, a road sign, a
    /// tent. Modelled as two sheets of triangles, which is what every asset
    /// does, it would otherwise be solved as two independent slabs that each
    /// insulate against nothing -- so a panel in the sun would be as hot as if
    /// its back face were against a wall, and the back face itself would sit at
    /// whatever the initial condition left it.
    ///
    /// The pair shares ONE column. The lower index owns it and is stepped; the
    /// higher index is not stepped at all and reads its temperature off the
    /// owner's back node, which the stepper writes into its surface slot so
    /// that every consumer -- the radiative exchange, the render's temperature
    /// buffer, a dump, a probe -- sees it without knowing a shell is involved.
    Vector<u32> shellPartner;
    static constexpr u32 kNoShellPartner = 0xFFFFFFFFu;

    /// Triangles of a shell material that found no partner, and are therefore
    /// solved one-sided. Reported rather than hidden: the pairing is a
    /// heuristic over an asset nobody wrote for it, and a material that
    /// declared itself a shell and paired a tenth of its triangles is a
    /// modelling problem rather than a solver one.
    u32 shellUnpairedCount = 0;
    u32 shellPairCount = 0;

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

    /// Per material, whether its triangles are two sides of one thin shell.
    /// Indexed by ThermalElement::materialId; an empty vector is no shells,
    /// which is every scene that does not ask.
    Vector<u8> shellMaterials;

    /// How far apart two triangles' centroids may be, as a multiple of the
    /// material's own thickness, and still be the two faces of one shell.
    ///
    /// Two rather than one because the thickness is the slab's and the
    /// centroids sit on its faces, so the distance IS the thickness for a
    /// flat pair -- and a curved shell, or one whose triangles do not line up
    /// exactly, wants room. Larger than about three starts pairing a shell
    /// with the wall behind it.
    f32 shellThicknessTolerance = 2.0f;

    /// Per material, the thickness the tolerance is a multiple of. Indexed
    /// like shellMaterials.
    Vector<f32> materialThickness_m;
};

/// The mesh options a solved material table implies.
///
/// Shell pairing needs to know which materials are shells and how thick they
/// are, and both live on the materials -- so the mesh cannot be built before
/// they are. Three call sites want the same derivation, and a fourth reading
/// of "which materials are shells" is a fourth chance to disagree.
[[nodiscard]] ThermalMeshOptions MeshOptionsFor(const Vector<ThermalMaterial>& materials,
                                                bool contacts);

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
