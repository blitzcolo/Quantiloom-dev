#include "atmos/AtmosModelPack.hpp"

#include "core/Log.hpp"

#include <array>
#include <stdexcept>

namespace quantiloom {

AtmosModelPack::AtmosModelPack(const std::filesystem::path& dir) : dir_(dir) {
    if (!std::filesystem::is_directory(dir_))
        throw std::runtime_error("AtmosModelPack: model pack directory does not "
                                 "exist: " + dir_.string());
}

bool AtmosModelPack::Has(const std::string& band, const std::string& geom,
                         const std::string& net) const {
    return std::filesystem::exists(dir_ / FileName(band, geom, net));
}

const AtmosNet& AtmosModelPack::Get(const std::string& band,
                                    const std::string& geom,
                                    const std::string& net) {
    const std::string key = FileName(band, geom, net);
    auto it = cache_.find(key);
    if (it != cache_.end()) return *it->second;

    const std::filesystem::path file = dir_ / key;
    if (!std::filesystem::exists(file))
        throw std::runtime_error("AtmosModelPack: missing network file: " +
                                 file.string());
    QL_LOG_INFO("AtmosModelPack: loading {}", file.string());
    auto net_ptr = std::make_unique<AtmosNet>(file);
    const AtmosNet& ref = *net_ptr;
    cache_.emplace(key, std::move(net_ptr));
    return ref;
}

void AtmosModelPack::RequireNets(
    const std::vector<std::array<std::string, 3>>& triples) const {
    std::string missing;
    for (const auto& t : triples) {
        if (!Has(t[0], t[1], t[2])) {
            if (!missing.empty()) missing += ", ";
            missing += (dir_ / FileName(t[0], t[1], t[2])).string();
        }
    }
    if (!missing.empty())
        throw std::runtime_error("AtmosModelPack: missing required network "
                                 "files: " + missing);
}

}  // namespace quantiloom
