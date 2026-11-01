/**
 * @file RtScene.hpp
 * @brief Acceleration structures of the RHI: build inputs, build sizes, instances, owners (internal)
 *
 * Only the objects and the one recording call live here. Which barriers surround a build and
 * when a refit replaces a rebuild stay with the caller (renderer/AccelerationStructure), so a
 * command-stream diff still sees every native call.
 *
 * Metal: an RtAccelerationStructure is an MTL::AccelerationStructure from
 * newAccelerationStructureWithSize. A Bottom geometry is an
 * MTLPrimitiveAccelerationStructureDescriptor holding one triangle geometry, a Top geometry an
 * MTLInstanceAccelerationStructureDescriptor. Builds and refits run on an
 * MTLAccelerationStructureCommandEncoder (buildAccelerationStructure / refitAccelerationStructure),
 * and every Bottom level must be resident (useResource or a residency set) while a trace reads
 * the Top level.
 */

#pragma once

#include "gpu/GpuTypes.hpp"

#include <span>
#include <type_traits>

namespace quantiloom::gpu {

enum class RtLevel : u8 { Bottom, Top };

/// Vulkan build flags. Metal: AllowUpdate is MTLAccelerationStructureUsageRefit; PreferFastTrace
/// is Metal's default usage (it only has a PreferFastBuild bit to opt out).
enum class RtBuildFlags : u32 { None = 0, PreferFastTrace = 1u << 0, AllowUpdate = 1u << 1 };

constexpr RtBuildFlags operator|(RtBuildFlags a, RtBuildFlags b) {
    using U = std::underlying_type_t<RtBuildFlags>;
    return static_cast<RtBuildFlags>(static_cast<U>(a) | static_cast<U>(b));
}
constexpr bool HasAny(RtBuildFlags value, RtBuildFlags bits) {
    using U = std::underlying_type_t<RtBuildFlags>;
    return (static_cast<U>(value) & static_cast<U>(bits)) != 0;
}

/// Build: a new structure into RtBuild::dst. Update: refit RtBuild::src into dst (the same
/// structure for an in-place refit); src must have been built with AllowUpdate.
/// Metal: buildAccelerationStructure | refitAccelerationStructure.
enum class RtBuildMode : u8 { Build, Update };

/// The one triangle geometry of a Bottom level: RGB32Float positions with a 12-byte stride,
/// u32 indices, no transform. Offsets are in bytes from the start of each buffer.
/// Metal: MTLAccelerationStructureTriangleGeometryDescriptor (vertexBuffer + vertexBufferOffset,
/// indexBuffer + indexBufferOffset, opaque).
struct RtTriangles {
    BufferRef vertices;
    u64 vertexOffset = 0;  // bytes
    u32 vertexCount = 0;
    BufferRef indices;
    u64 indexOffset = 0;  // bytes
    u32 triangleCount = 0;
    bool opaque = true;  // false lets the any-hit coverage test run on this geometry
};

/// The instance records of a Top level: `count` records of Device::RtInstanceStride() bytes,
/// written by Device::PackRtInstances, from `offset` bytes into `buffer` (16-byte aligned on
/// Vulkan).
struct RtInstances {
    BufferRef buffer;
    u64 offset = 0;  // bytes
    u32 count = 0;
    /// Metal 3 names an instance's Bottom level by its index in this array
    /// (instancedAccelerationStructures). Vulkan names it by device address and ignores it.
    std::span<const RtSceneRef> referencedBottomLevels{};
};

/// What a build reads: `triangles` for a Bottom level, `instances` for a Top level.
struct RtGeometry {
    RtLevel level{};
    RtBuildFlags flags{};
    RtTriangles triangles{};
    RtInstances instances{};
};

/// Device::GetRtBuildSizes. Metal: MTLAccelerationStructureSizes (accelerationStructureSize,
/// buildScratchBufferSize, refitScratchBufferSize).
struct RtBuildSizes {
    u64 structure = 0;
    u64 buildScratch = 0;
    u64 updateScratch = 0;
};

/// One CommandList::BuildRtAccelerationStructure: `geometry` into `dst`, with `scratch` used
/// from its start. `src` is read by an Update only and ignored by a Build.
struct RtBuild {
    RtGeometry geometry;
    RtBuildMode mode{};
    RtSceneRef dst;
    RtSceneRef src;
    BufferRef scratch;
};

/// One Top-level instance, backend-neutral; Device::PackRtInstances writes the native record.
/// Vulkan: a 64-byte instance record naming the Bottom level by device address. Metal:
/// MTLAccelerationStructureUserIDInstanceDescriptor, naming it by its index in
/// RtInstances::referencedBottomLevels, with the transform column-major (MTLPackedFloat4x3).
struct RtInstance {
    f32 transform[3][4]{};    // row-major 3x4: the top three rows of the 4x4 object-to-world matrix
    u32 userId = 0;           // InstanceID() in HLSL; 24 bits on Vulkan
    u8 mask = 0xFF;           // visible to every ray
    bool cullDisable = true;  // back faces are hit (double-sided)
    RtSceneRef bottomLevel;
};

/// An acceleration structure together with its storage; destroying it frees both.
/// Metal: MTL::AccelerationStructure (its storage is internal to it).
class RtAccelerationStructure {
public:
    virtual ~RtAccelerationStructure() = default;
    /// The handle, and for a Bottom level the device address an instance names (0 for Top).
    [[nodiscard]] virtual RtSceneRef Ref() const = 0;
    [[nodiscard]] virtual RtLevel Level() const = 0;
};

}  // namespace quantiloom::gpu
