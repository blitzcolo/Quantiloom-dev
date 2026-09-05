#include "scene/SceneMerge.hpp"

#include "core/Log.hpp"

#include <string>
#include <unordered_set>
#include <utility>

namespace quantiloom::scene {

namespace {

/**
 * @brief Every field of a Material that indexes Scene::textures
 *
 * The single list. Adding a texture slot to Material means adding one line
 * here, and forgetting to means a second model's materials sample the first
 * model's images -- which renders, and renders wrong, and looks like a
 * material bug rather than an indexing one.
 *
 * Curve indices are deliberately absent: they point into the resolved spectral
 * curves, which do not exist yet when scenes are merged (ResolveMaterialSpectra
 * runs later, against the merged scene, and fills them by name).
 */
template <typename F>
void ForEachTextureIndex(Material& material, F&& visit) {
    visit(material.baseColorTextureIndex);
    visit(material.metallicRoughnessTextureIndex);
    visit(material.normalTextureIndex);
    visit(material.emissiveTextureIndex);
    visit(material.temperatureTextureIndex);
    visit(material.transmissionTextureIndex);
    visit(material.thicknessTextureIndex);
    visit(material.weightTextureIndex);
    visit(material.sheenColorTextureIndex);
    visit(material.sheenRoughnessTextureIndex);
    visit(material.specularTextureIndex);
    visit(material.specularColorTextureIndex);
    visit(material.anisotropyTextureIndex);
    visit(material.clearcoatTextureIndex);
    visit(material.clearcoatRoughnessTextureIndex);
    visit(material.clearcoatNormalTextureIndex);
    visit(material.diffuseTransmissionTextureIndex);
    visit(material.diffuseTransmissionColorTextureIndex);
}

}  // namespace

AppendReport AppendScene(Scene& into, Scene&& part, StringView modelName, const glm::mat4& rest) {
    AppendReport report;

    const bool intoWasEmpty =
        into.meshes.empty() && into.nodes.empty() && into.materials.empty();

    const u32 meshBase = static_cast<u32>(into.meshes.size());
    const u32 materialBase = static_cast<u32>(into.materials.size());
    const u32 textureBase = static_cast<u32>(into.textures.size());
    report.firstNode = static_cast<u32>(into.nodes.size());
    report.firstMaterial = materialBase;

    const String prefix = String(modelName) + "/";

    // Textures first: nothing indexes them from the other direction.
    into.textures.reserve(into.textures.size() + part.textures.size());
    for (auto& texture : part.textures) into.textures.push_back(std::move(texture));

    std::unordered_set<String> takenNames;
    takenNames.reserve(into.materials.size() + part.materials.size());
    for (const Material& material : into.materials) takenNames.insert(material.name);

    into.materials.reserve(into.materials.size() + part.materials.size());
    for (Material& material : part.materials) {
        ForEachTextureIndex(material, [textureBase](i32& index) {
            if (index >= 0) index += static_cast<i32>(textureBase);
        });
        if (!material.name.empty() && takenNames.count(material.name) > 0) {
            material.name = prefix + material.name;
            ++report.renamedMaterials;
        }
        takenNames.insert(material.name);
        into.materials.push_back(std::move(material));
    }
    report.materialCount = static_cast<u32>(part.materials.size());

    into.meshes.reserve(into.meshes.size() + part.meshes.size());
    for (Mesh& mesh : part.meshes) {
        for (GeometryPrimitive& primitive : mesh.primitives) {
            primitive.materialId += materialBase;
        }
        into.meshes.push_back(std::move(mesh));
    }

    into.nodes.reserve(into.nodes.size() + part.nodes.size());
    for (usize i = 0; i < part.nodes.size(); ++i) {
        SceneNode& node = part.nodes[i];
        node.meshIndex += meshBase;
        node.name = node.name.empty() ? (prefix + "node_" + std::to_string(i))
                                      : (prefix + node.name);
        node.transform = rest * node.transform;
        into.nodes.push_back(std::move(node));
    }
    report.nodeCount = static_cast<u32>(part.nodes.size());

    // The first model in a config is the one that says what the scene is. A
    // later one bringing its own camera is not a conflict worth reporting --
    // an exporter writes one into every file, and a config that wanted a
    // particular camera says so in [camera].
    if (intoWasEmpty) {
        into.camera = part.camera;
        into.bands = std::move(part.bands);
        into.lambda_min = part.lambda_min;
        into.lambda_max = part.lambda_max;
        into.delta_lambda = part.delta_lambda;
        into.name = part.name;
        into.description = part.description;
        if (part.atmosphereLUT.has_value()) {
            into.atmosphereLUT = std::move(part.atmosphereLUT);
        }
    }

    if (report.renamedMaterials > 0) {
        QL_LOG_INFO("  Model '{}': {} material name(s) collided and were prefixed with '{}'",
                    modelName, report.renamedMaterials, prefix);
    }

    return report;
}

}  // namespace quantiloom::scene
