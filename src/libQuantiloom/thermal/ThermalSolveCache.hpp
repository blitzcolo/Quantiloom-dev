/**
 * @file ThermalSolveCache.hpp
 * @brief A solved temperature field, on disk, addressed by its inputs
 *
 * Internal. On by default, and invisible from a config: nothing in a TOML file
 * names it and no host has to ask for it.
 *
 * WHAT IT IS FOR. A batch render varies the sensor -- resolution, PSF, bit
 * depth -- and none of those reach the energy balance, so a list of 1290 LWIR
 * renders contains about 15 distinct thermal solves and computes each of them
 * eighty-odd times. At 165 s a solve on a 1.5M-element scene that is 61% of the
 * wall clock of a full rebuild.
 *
 * THE BOUNDARY IS THE WHOLE SOLVE, NOT THE STEPPING. An entry stands for the
 * GPU view-factor precompute *and* the Crank-Nicolson trajectory -- 15 s plus
 * 150 s. Caching only the stepping would leave the precompute to run on every
 * hit and give back a tenth of the win.
 *
 * A MISS IS NEVER AN ERROR. Every failure here -- unreadable directory, foreign
 * file, truncated write, a driver that has since been updated -- resolves to
 * "solve it again". Nothing in this file can fail a render, and nothing throws.
 *
 * WHAT THE KEY HAS TO COVER, AND WHY IT IS A SHA-256. The key is a digest of
 * the *resolved* inputs, never of the config text: two configs differing only
 * in `sensor.*` must land on one entry, which is the entire point. So it hashes
 * the built mesh, the merged materials, the config scalars, the forcing CSV's
 * contents, the lighting sun direction, the stepper, the library version and
 * the GPU. Miss a field and the cache silently serves another scene's
 * temperature field while the render exits 0 -- the reason for 256 bits rather
 * than a cheap 64-bit mix.
 *
 * The one thing deliberately approximated: the geometry enters as the built
 * ThermalMesh -- per-triangle centroid, area, normal, material -- rather than
 * as vertex data. That is what the solve and the exchange rays consume, and it
 * folds transforms, winding and degenerate handling in for free. Two meshes
 * agreeing on every triangle's centroid, area and normal but differing in
 * vertices would collide; that requires a whole-mesh coincidence, and is
 * accepted.
 *
 * WHY THE GPU IS IN THE KEY. The view-factor precompute is a ray-traced GPU
 * pass. It is deterministic for a fixed binary on a fixed driver -- no RNG, no
 * atomics, a Hammersley sequence and a stateless coverage hash -- but BVH
 * construction and intersection are a vendor's business, so results are not
 * portable across GPUs or driver versions. Keying on the device identity means
 * a driver update reads as a miss rather than as a wrong answer.
 *
 * MANAGEMENT IS `rm -rf`. There is no eviction. Entries are a few hundred kB to
 * a few tens of MB and the workload that motivated this produces about fifteen;
 * stale ones from an older library version or driver are unreachable rather
 * than dangerous. Point QUANTILOOM_THERMAL_CACHE_DIR at a scratch directory to
 * keep a run's entries separate, or QUANTILOOM_THERMAL_CACHE=0 to switch the
 * whole thing off.
 */

#pragma once

#include "core/Types.hpp"
#include "thermal/ThermalMesh.hpp"
#include "thermal/ThermalSolver.hpp"

#include <filesystem>
#include <optional>

namespace quantiloom::thermal {

/// Where entries live and whether they are consulted at all.
struct ThermalSolveCacheSettings {
    /// QUANTILOOM_THERMAL_CACHE: 0/off/false/no disables. Anything else, or
    /// unset, leaves it on.
    bool enabled = true;
    /// QUANTILOOM_THERMAL_CACHE_DIR, else <platform cache dir>/thermal.
    /// Taken as written -- it is an output location, and this repo resolves
    /// only inputs against a config's directory.
    std::filesystem::path directory;
};

[[nodiscard]] ThermalSolveCacheSettings ResolveThermalSolveCacheSettings();

/**
 * @brief Everything the solve depends on, gathered for hashing
 *
 * Pointers rather than values because the caller already owns all of it and
 * these can be large; none are copied, and all must outlive the call.
 */
struct ThermalSolveCacheKeyInputs {
    /// The built mesh: elements and instance bases. Covers geometry, world
    /// transforms, per-element material assignment and instancing.
    const ThermalMesh* mesh = nullptr;
    /// From BuildSolvedMaterials -- what the solve will run on, including the
    /// band-averaged emissivity, which is not the config's number.
    const Vector<ThermalMaterial>* solvedMaterials = nullptr;
    /// Scalars only; dumpElementsFile, enabled and the materials map are
    /// excluded (the first two are not solve inputs, the third arrives already
    /// merged in solvedMaterials).
    const ThermalConfig* config = nullptr;
    /// [lighting] sun_direction, which is what the exchange precompute traces
    /// sun visibility against. Not the same field as config->sunDirection and
    /// not redundant with it.
    glm::vec3 exchangeSunDirection{0.0f, 1.0f, 0.0f};
    /// Device name, vendor, device id and driver version, however the caller
    /// wants to spell them -- see MakeGpuIdentity.
    StringView gpuIdentity;
    /// IThermalStepper::Name() of the stepper that will run. CPU and GPU
    /// results differ in the last bits, so they must not share an entry.
    StringView stepperName;
    /// quantiloom::version::LibVersionString.
    StringView libVersion;
};

/**
 * @brief The entry's name: 64 lowercase hex characters
 *
 * Returns an empty string when the config names a forcing file that cannot be
 * read. The key cannot describe that run's inputs, and guessing would risk
 * serving an entry built from a file that has since changed, so the caller
 * treats an empty key as "do not use the cache this time".
 */
[[nodiscard]] String ComputeThermalSolveCacheKey(const ThermalSolveCacheKeyInputs& inputs);

/// The device identity string for the key, from the fields of
/// VkPhysicalDeviceProperties. Takes them loose so this file needs no Vulkan.
[[nodiscard]] String MakeGpuIdentity(StringView deviceName, u32 vendorId, u32 deviceId,
                                     u32 driverVersion);

/**
 * @brief Read an entry back, if it is there and it is ours
 *
 * Checks magic, format version, the key it was written under, the array counts
 * against each other, and a digest of the payload. Any disagreement warns and
 * returns nothing -- the caller re-solves and overwrites.
 */
[[nodiscard]] std::optional<ThermalResult> LoadThermalSolveCache(
    const std::filesystem::path& file, StringView expectedKeyHex);

/**
 * @brief Write an entry, atomically
 *
 * Through a temporary in the same directory, then a rename, so a reader never
 * sees a half-written file and two batch processes solving the same scene
 * produce one winner rather than a torn entry. Best effort: warns and returns
 * false rather than failing anything.
 */
bool StoreThermalSolveCache(const std::filesystem::path& file, StringView keyHex,
                            const ThermalResult& result);

}  // namespace quantiloom::thermal
