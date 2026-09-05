/**
 * @file SceneMerge.hpp
 * @brief Putting several loaded files into one Scene
 *
 * A Scene is four parallel arrays that index each other -- nodes into meshes,
 * primitives into materials, materials into textures -- so concatenating two
 * of them is not concatenating four vectors. Every index in the second one has
 * to move by the size of the first, and there is exactly one list of the
 * places an index lives. That list is ForEachTextureIndex below, and the
 * reason it is a loop over a callback rather than eighteen lines of `+=` is
 * that Material grows a texture slot every few months.
 *
 * ## Names
 *
 * Nodes get their model's name and a slash in front: `car/Wheel_FL`. Two files
 * that both call something "Body" would otherwise be indistinguishable to
 * `[[nodes]]`, `[[duplicates]]` and `scene.removed_nodes`, all of which match
 * by name.
 *
 * Materials keep their bare names. Prefixing them all would break every
 * `[[materials]]` and `[material_overrides]` entry ever written, for a
 * collision that mostly does not happen -- so only a name that is *already*
 * taken gets the prefix, and the caller is told how many did.
 */

#pragma once

#include "scene/Scene.hpp"

#include <glm/glm.hpp>

namespace quantiloom::scene {

/// What one AppendScene call added, so a caller can say which nodes belong to
/// which model without re-deriving it from the names.
struct AppendReport {
    u32 firstNode = 0;
    u32 nodeCount = 0;
    u32 firstMaterial = 0;
    u32 materialCount = 0;
    u32 renamedMaterials = 0;
};

/**
 * @brief Append @p part to @p into, offsetting every index it carries
 *
 * @param into       grown in place; its own indices are untouched
 * @param part       consumed
 * @param modelName  the `[[models]]` name; prefixes node names
 * @param rest       the model's rest pose, pre-multiplied onto every node
 *
 * When @p into is empty the camera, the spectral band definitions and the
 * scene name come across too -- the first model in a config is the one that
 * gets to say what the scene is.
 */
AppendReport AppendScene(Scene& into, Scene&& part, StringView modelName, const glm::mat4& rest);

}  // namespace quantiloom::scene
