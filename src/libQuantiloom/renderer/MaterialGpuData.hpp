#pragma once
// MaterialDataCPU — the host's name for MaterialData, which the shaders and the
// host share as one definition with every offset asserted:
// src/shaders/ql_shared_scene.hlsli. Every host file that fills the material
// buffer reaches it through this header; no local copies.

#include "core/Types.hpp"
#include <glm/glm.hpp>
#include <cstddef>

// By a path relative to this file, as RtBindings.hpp includes its manifest.
#include "../../shaders/ql_shared_scene.hlsli"

namespace quantiloom {

using MaterialDataCPU = shader::MaterialData;  // 656 bytes

// UV transform slots. The weight texture deliberately has no slot of its own:
// it is unmixed from the base-colour texture's texels, so it must be sampled
// with the base colour's transform or the mixture reads the wrong texels.
// The temperature texture has none either -- it is a Quantiloom-authored slot
// with no glTF textureInfo to carry the extension, so it is always identity.
enum : i32 {
    UV_SLOT_BASE_COLOR = 0,
    UV_SLOT_METALLIC_ROUGHNESS = 1,
    UV_SLOT_NORMAL = 2,
    UV_SLOT_EMISSIVE = 3,
    UV_SLOT_SHEEN_COLOR = 4,
    UV_SLOT_SHEEN_ROUGHNESS = 5,
    UV_SLOT_SPECULAR = 6,
    UV_SLOT_SPECULAR_COLOR = 7,
    UV_SLOT_ANISOTROPY = 8,
    UV_SLOT_CLEARCOAT = 9,
    UV_SLOT_CLEARCOAT_ROUGHNESS = 10,
    UV_SLOT_CLEARCOAT_NORMAL = 11,
    UV_SLOT_DIFFUSE_TRANSMISSION = 12,
    UV_SLOT_DIFFUSE_TRANSMISSION_COLOR = 13,
    UV_SLOT_COUNT = 14,
};

} // namespace quantiloom
