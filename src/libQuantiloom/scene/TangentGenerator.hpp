/**
 * @file TangentGenerator.hpp
 * @brief Per-vertex tangent frames derived from the UV parameterisation
 *
 * A normal map and an anisotropic highlight are both read in the surface's own
 * texture frame, so both need a tangent. glTF and USD may supply one; most
 * assets do not, and the renderer then synthesises a frame per vertex that is
 * continuous but arbitrary in its rotation within the surface. That is enough
 * for a normal map, whose perturbation is defined relative to whatever frame it
 * is given, and wrong for anisotropy, whose highlight points along the tangent
 * -- so the streak lands somewhere the author did not choose, and nothing
 * downstream can tell.
 *
 * Deriving the frame from the UVs is what the author did choose: the direction
 * of increasing u across the surface is the same direction the map was painted
 * along.
 *
 * @author blitzcolo
 */

#pragma once

#include "scene/Mesh.hpp"

namespace quantiloom {

class TangentGenerator {
public:
    /**
     * @brief Derive tangents from positions, normals, UVs and topology
     *
     * Accumulates dP/du over the triangles sharing each vertex, orthogonalises
     * it against the vertex normal (Gram-Schmidt), and records the handedness
     * in w as the sign of dot(cross(n, t), dP/dv) -- the convention
     * `Mesh::tangents` documents and the shaders read.
     *
     * Must run before deduplication, which keys on the tangent, and after
     * normal generation, which duplicates vertices at hard edges.
     *
     * @param primitive Modified in place; existing tangents are replaced.
     * @return false, leaving `tangents` untouched, when the primitive has no
     *         UVs, no normals or no triangles -- there is nothing to derive a
     *         frame from and a fabricated one would be worse than none.
     *
     * @note A triangle whose UVs are degenerate (zero area in texture space)
     *       contributes nothing; a vertex reached only by such triangles falls
     *       back to an arbitrary frame orthogonal to its normal, which is what
     *       the renderer would have synthesised anyway.
     */
    static bool FromUv(GeometryPrimitive& primitive);
};

}  // namespace quantiloom
