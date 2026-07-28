#pragma once

#include "atmos/AtmosNet.hpp"
#include "core/Platform.hpp"

#include <array>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace quantiloom {

// ============================================================================
// AtmosModelPack - directory of <band>_<geom>_<net>.safetensors files
// ============================================================================
// band ∈ {lwir, mwir, nir, swir, vis}, geom ∈ {ground, slant, sky},
// net ∈ {tau, lpath, ldown}. Networks are loaded lazily on first use and
// cached. Missing required files are a hard error naming the exact file --
// no silent fallback.

class AtmosModelPack {
public:
    explicit AtmosModelPack(const std::filesystem::path& dir);

    // Non-copyable (owns loaded networks); required explicitly because
    // dllexport would otherwise instantiate the deleted map copy-ctor
    AtmosModelPack(const AtmosModelPack&) = delete;
    AtmosModelPack& operator=(const AtmosModelPack&) = delete;
    AtmosModelPack(AtmosModelPack&&) = default;
    AtmosModelPack& operator=(AtmosModelPack&&) = default;

    const std::filesystem::path& Dir() const { return dir_; }

    // True if the pack directory contains the file for (band, geom, net).
    bool Has(const std::string& band, const std::string& geom,
             const std::string& net) const;

    // Loads (cached). Throws std::runtime_error naming the file if absent
    // or malformed.
    const AtmosNet& Get(const std::string& band, const std::string& geom,
                        const std::string& net);

    // Verifies every (band, geom, net) triple exists up front; throws one
    // error listing all missing file names otherwise.
    void RequireNets(const std::vector<std::array<std::string, 3>>& triples) const;

    static std::string FileName(const std::string& band, const std::string& geom,
                                const std::string& net) {
        return band + "_" + geom + "_" + net + ".safetensors";
    }

private:
    std::filesystem::path dir_;
    std::map<std::string, std::unique_ptr<AtmosNet>> cache_;
};

}  // namespace quantiloom
