#include "thermal/ThermalSolveCache.hpp"

#include "core/CacheDirectory.hpp"
#include "core/Log.hpp"
#include "core/Sha256.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#if defined(_WIN32)
    #include <process.h>
    #define QL_GETPID _getpid
#else
    #include <unistd.h>
    #define QL_GETPID getpid
#endif

namespace quantiloom::thermal {

namespace {

/// Bumped when the *list* of hashed inputs changes -- adding a field would
/// otherwise leave every existing entry addressable under a new meaning.
constexpr u32 kKeySchemaVersion = 2u;

/// "QLTC", little-endian.
constexpr u32 kCacheMagic = 0x43544C51u;
/// Bumped when the file layout below changes. Independent of the key schema:
/// one describes what an entry means, the other how it is written down.
constexpr u32 kCacheFormatVersion = 2u;

constexpr usize kKeyHexLength = 64;

/// Header, read and written field by field rather than as a struct -- a packed
/// POD across a file boundary is a padding question on every compiler.
struct CacheHeader {
    u32 magic = 0;
    u32 formatVersion = 0;
    char keyHex[kKeyHexLength] = {};
    u64 temperatureCount = 0;
    u64 instanceBaseCount = 0;
    u64 sensitivityCount = 0;
    u64 visibilityCount = 0;
    /// Per-column tangents: how many columns, and the two slot-major arrays'
    /// lengths. Zero slots is an entry from a run that carried none.
    u64 lagSlots = 0;
    u64 lagSensitivityCount = 0;
    u64 lagVisibilityCount = 0;
    u64 lagDirectionCount = 0;
    f32 sunDirection[3] = {0.0f, 0.0f, 0.0f};
    u32 elementCount = 0;
    u32 participatingElements = 0;
    u32 exchangeNonZeros = 0;
    u32 stepsTaken = 0;
    f64 minTemperature_K = 0.0;
    f64 maxTemperature_K = 0.0;
    f64 meanTemperature_K = 0.0;
};

bool IsDisablingValue(String value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value == "0" || value == "off" || value == "false" || value == "no";
}

template <typename T>
void WritePod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
bool ReadPod(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<usize>(in.gcount()) == sizeof(T);
}

/// The entry's digest: header first, then the arrays -- everything the file
/// holds except the digest itself. The header is in there because it carries
/// the counts the gate line is printed from, and a temperature range flipped
/// by bit rot would otherwise be reported as a measurement.
String DigestEntry(const CacheHeader& header, const ThermalResult& result) {
    core::Sha256 hasher;

    hasher.UpdateU32(header.magic);
    hasher.UpdateU32(header.formatVersion);
    hasher.Update(header.keyHex, kKeyHexLength);
    hasher.UpdateU64(header.temperatureCount);
    hasher.UpdateU64(header.instanceBaseCount);
    hasher.UpdateU64(header.sensitivityCount);
    hasher.UpdateU64(header.visibilityCount);
    hasher.UpdateU64(header.lagSlots);
    hasher.UpdateU64(header.lagSensitivityCount);
    hasher.UpdateU64(header.lagVisibilityCount);
    hasher.UpdateU64(header.lagDirectionCount);
    hasher.UpdateF32(header.sunDirection[0]);
    hasher.UpdateF32(header.sunDirection[1]);
    hasher.UpdateF32(header.sunDirection[2]);
    hasher.UpdateU32(header.elementCount);
    hasher.UpdateU32(header.participatingElements);
    hasher.UpdateU32(header.exchangeNonZeros);
    hasher.UpdateU32(header.stepsTaken);
    hasher.UpdateF64(header.minTemperature_K);
    hasher.UpdateF64(header.maxTemperature_K);
    hasher.UpdateF64(header.meanTemperature_K);

    const auto feed = [&hasher](const auto& vec) {
        if (!vec.empty()) {
            hasher.Update(vec.data(), vec.size() * sizeof(typename std::decay_t<decltype(vec)>::value_type));
        }
    };
    feed(result.surfaceTemperature_K);
    feed(result.instanceElementBase);
    feed(result.sunSensitivity_K);
    feed(result.sunVisibility);
    feed(result.lagSensitivity_K);
    feed(result.lagVisibility);
    feed(result.lagDirection);
    return hasher.FinalizeHex();
}

}  // namespace

ThermalSolveCacheSettings ResolveThermalSolveCacheSettings() {
    ThermalSolveCacheSettings settings;

    if (const char* toggle = std::getenv("QUANTILOOM_THERMAL_CACHE");
        toggle != nullptr && IsDisablingValue(toggle)) {
        settings.enabled = false;
    }

    if (const char* dir = std::getenv("QUANTILOOM_THERMAL_CACHE_DIR");
        dir != nullptr && dir[0] != '\0') {
        settings.directory = std::filesystem::path(dir);
    } else {
        settings.directory = std::filesystem::path(core::GetDefaultCacheDirectory()) / "thermal";
    }

    return settings;
}

String MakeGpuIdentity(StringView deviceName, u32 vendorId, u32 deviceId, u32 driverVersion) {
    // Trim at the first NUL: VkPhysicalDeviceProperties::deviceName is a fixed
    // 256-byte array, and the tail after the name is unspecified padding that
    // would make the key depend on uninitialised bytes.
    const usize end = deviceName.find('\0');
    if (end != StringView::npos) {
        deviceName = deviceName.substr(0, end);
    }
    return String(deviceName) + "|" + std::to_string(vendorId) + "|" + std::to_string(deviceId) +
           "|" + std::to_string(driverVersion);
}

String ComputeThermalSolveCacheKey(const ThermalSolveCacheKeyInputs& inputs) {
    if (inputs.mesh == nullptr || inputs.solvedMaterials == nullptr || inputs.config == nullptr) {
        return {};
    }
    const ThermalConfig& config = *inputs.config;

    // Read the forcing file before hashing anything: a named-but-unreadable
    // file means we cannot describe this run's inputs, and an entry keyed on a
    // guess would be served to runs that are not the same.
    String forcingContents;
    if (!config.forcingFile.empty()) {
        std::ifstream forcing(config.forcingFile, std::ios::binary);
        if (!forcing) {
            return {};
        }
        forcingContents.assign(std::istreambuf_iterator<char>(forcing),
                               std::istreambuf_iterator<char>());
        if (forcing.bad()) {
            return {};
        }
    }

    core::Sha256 hasher;

    // 1. Who wrote this key.
    hasher.UpdateString("QLTC-KEY");
    hasher.UpdateU32(kKeySchemaVersion);
    hasher.UpdateString(inputs.libVersion);

    // 2. What ran it. The exchange precompute is GPU work and the stepper
    //    decides f64 versus f32, so both change the answer's last bits.
    hasher.UpdateString(inputs.gpuIdentity);
    hasher.UpdateString(inputs.stepperName);

    // 3. The geometry, as the solve sees it. ThermalElement is a packed
    //    32-byte POD (vec3 + f32 + vec3 + u32) with no padding, so its bytes
    //    are the whole of its value.
    static_assert(sizeof(ThermalElement) == 32, "ThermalElement gained padding or a field; "
                                                "hash it field-by-field or bump the key schema");
    const auto& elements = inputs.mesh->elements;
    hasher.UpdateSized(elements.data(), elements.size() * sizeof(ThermalElement));
    const auto& bases = inputs.mesh->instanceElementBase;
    hasher.UpdateSized(bases.data(), bases.size() * sizeof(u32));

    // 4. The materials the solve will use, field by field: ThermalMaterial
    //    has padding after its u8 enum, and hashing that would make the key
    //    depend on bytes nobody wrote.
    hasher.UpdateU64(inputs.solvedMaterials->size());
    for (const ThermalMaterial& material : *inputs.solvedMaterials) {
        hasher.UpdateF32(material.conductivity_W_mK);
        hasher.UpdateF32(material.density_kg_m3);
        hasher.UpdateF32(material.specificHeat_J_kgK);
        hasher.UpdateF32(material.thickness_m);
        hasher.UpdateF32(material.convection_W_m2K);
        hasher.UpdateF32(material.shortwaveAbsorptivity);
        hasher.UpdateF32(material.longwaveEmissivity);
        hasher.UpdateF32(material.wetnessFactor);
        hasher.UpdateF32(material.internalHeat_W_m2);
        hasher.UpdateU8(static_cast<u8>(material.interiorBoundary));
        hasher.UpdateF32(material.interiorTemperature_K);
        hasher.UpdateF32(material.interiorConvection_W_m2K);
    }

    // 5. The [thermal] scalars. Not dumpElementsFile (an output), not enabled
    //    (we are only here because it is true), not forcingFile as a path
    //    (its contents go in below), not materials (merged in above).
    hasher.UpdateF64(config.time_h);
    hasher.UpdateF64(config.startTime_h);
    hasher.UpdateF64(config.timestep_s);
    hasher.UpdateU32(config.nodeCount);
    hasher.UpdateU8(static_cast<u8>(config.initial));
    hasher.UpdateF64(config.initialTemperature_K);
    hasher.UpdateF64(config.checkpointStride_h);
    hasher.UpdateU32(config.exchangeRays);
    hasher.UpdateU32(config.exchangeTopK);
    hasher.UpdateF64(config.airTemperature_K);
    hasher.UpdateF64(config.sunIrradiance_W_m2);
    hasher.UpdateF64(config.diffuseIrradiance_W_m2);
    hasher.UpdateF32(config.sunDirection.x);
    hasher.UpdateF32(config.sunDirection.y);
    hasher.UpdateF32(config.sunDirection.z);
    hasher.UpdateF64(config.skyTemperature_K);
    hasher.UpdateF64(config.relativeHumidity);
    hasher.UpdateBool(config.sunCorrection);
    hasher.UpdateU8(static_cast<u8>(config.convection.model));
    hasher.UpdateF64(config.convection.windIntercept_W_m2K);
    hasher.UpdateF64(config.convection.windSlope_W_s_m3K);
    hasher.UpdateF64(config.convection.freeCoefficient);
    hasher.UpdateF64(config.convection.referenceHeight_m);
    hasher.UpdateF64(config.convection.stableDamping);
    // The flag alone: the conductances it produces are a deterministic
    // function of the element bytes and the material table, and both are
    // already in this key above.
    hasher.UpdateBool(config.lateralConduction);
    hasher.UpdateU32(config.sunMemoryLags);

    // 6. The lighting sun direction, which is what the exchange traces sun
    //    visibility against -- a separate field from config.sunDirection.
    hasher.UpdateF32(inputs.exchangeSunDirection.x);
    hasher.UpdateF32(inputs.exchangeSunDirection.y);
    hasher.UpdateF32(inputs.exchangeSunDirection.z);

    // 7. The forcing series, by content. A named file and an absent one are
    //    distinguished by the tag, so an empty CSV is not the same as no CSV.
    hasher.UpdateU8(config.forcingFile.empty() ? 0u : 1u);
    hasher.UpdateString(forcingContents);

    return hasher.FinalizeHex();
}

std::optional<ThermalResult> LoadThermalSolveCache(const std::filesystem::path& file,
                                                   StringView expectedKeyHex) {
    std::error_code ec;
    if (!std::filesystem::exists(file, ec) || ec) {
        return std::nullopt;  // the ordinary miss: not a warning
    }

    std::ifstream in(file, std::ios::binary);
    if (!in) {
        QL_LOG_WARN("  Thermal cache: {} cannot be opened; solving instead", file.string());
        return std::nullopt;
    }

    CacheHeader header;
    const bool headerRead =
        ReadPod(in, header.magic) && ReadPod(in, header.formatVersion) &&
        (in.read(header.keyHex, kKeyHexLength),
         static_cast<usize>(in.gcount()) == kKeyHexLength) &&
        ReadPod(in, header.temperatureCount) && ReadPod(in, header.instanceBaseCount) &&
        ReadPod(in, header.sensitivityCount) && ReadPod(in, header.visibilityCount) &&
        ReadPod(in, header.lagSlots) && ReadPod(in, header.lagSensitivityCount) &&
        ReadPod(in, header.lagVisibilityCount) && ReadPod(in, header.lagDirectionCount) &&
        ReadPod(in, header.sunDirection) && ReadPod(in, header.elementCount) &&
        ReadPod(in, header.participatingElements) && ReadPod(in, header.exchangeNonZeros) &&
        ReadPod(in, header.stepsTaken) && ReadPod(in, header.minTemperature_K) &&
        ReadPod(in, header.maxTemperature_K) && ReadPod(in, header.meanTemperature_K);
    if (!headerRead) {
        QL_LOG_WARN("  Thermal cache: {} is truncated; solving instead", file.string());
        return std::nullopt;
    }

    if (header.magic != kCacheMagic) {
        QL_LOG_WARN("  Thermal cache: {} is not a thermal cache entry; solving instead",
                    file.string());
        return std::nullopt;
    }
    if (header.formatVersion != kCacheFormatVersion) {
        QL_LOG_WARN("  Thermal cache: {} was written in format {} and this build reads {}; "
                    "solving instead",
                    file.string(), header.formatVersion, kCacheFormatVersion);
        return std::nullopt;
    }
    if (StringView(header.keyHex, kKeyHexLength) != expectedKeyHex) {
        // Only reachable if the file was renamed or the hash collided; either
        // way this entry is not the one the caller asked for.
        QL_LOG_WARN("  Thermal cache: {} holds a different key than its name claims; "
                    "solving instead",
                    file.string());
        return std::nullopt;
    }

    // Counts have to agree with each other before they size any allocation.
    // An empty temperature array means the entry recorded a failed solve,
    // which is not a thing we ever want to serve.
    const u64 elements = header.elementCount;
    const bool countsSane =
        elements > 0 && header.temperatureCount == elements &&
        (header.sensitivityCount == 0 || header.sensitivityCount == elements) &&
        (header.visibilityCount == 0 || header.visibilityCount == elements) &&
        header.lagSensitivityCount == header.lagSlots * elements &&
        header.lagVisibilityCount == header.lagSlots * elements &&
        header.lagDirectionCount == header.lagSlots;
    if (!countsSane) {
        QL_LOG_WARN("  Thermal cache: {} has inconsistent element counts; solving instead",
                    file.string());
        return std::nullopt;
    }

    ThermalResult result;
    // resize to exactly the stored count: zero stays empty, which is a
    // different claim from an array of zeros and is why the counts are stored
    // separately rather than inferred from elementCount.
    result.surfaceTemperature_K.resize(static_cast<usize>(header.temperatureCount));
    result.instanceElementBase.resize(static_cast<usize>(header.instanceBaseCount));
    result.sunSensitivity_K.resize(static_cast<usize>(header.sensitivityCount));
    result.sunVisibility.resize(static_cast<usize>(header.visibilityCount));
    result.lagSlots = static_cast<u32>(header.lagSlots);
    result.lagSensitivity_K.resize(static_cast<usize>(header.lagSensitivityCount));
    result.lagVisibility.resize(static_cast<usize>(header.lagVisibilityCount));
    result.lagDirection.resize(static_cast<usize>(header.lagDirectionCount));

    const auto readArray = [&in](auto& vec) {
        if (vec.empty()) {
            return true;
        }
        const std::streamsize bytes =
            static_cast<std::streamsize>(vec.size() * sizeof(typename std::decay_t<decltype(vec)>::value_type));
        in.read(reinterpret_cast<char*>(vec.data()), bytes);
        return in.gcount() == bytes;
    };
    if (!readArray(result.surfaceTemperature_K) || !readArray(result.instanceElementBase) ||
        !readArray(result.sunSensitivity_K) || !readArray(result.sunVisibility) ||
        !readArray(result.lagSensitivity_K) || !readArray(result.lagVisibility) ||
        !readArray(result.lagDirection)) {
        QL_LOG_WARN("  Thermal cache: {} is truncated; solving instead", file.string());
        return std::nullopt;
    }

    char storedDigest[kKeyHexLength] = {};
    in.read(storedDigest, kKeyHexLength);
    if (static_cast<usize>(in.gcount()) != kKeyHexLength) {
        QL_LOG_WARN("  Thermal cache: {} has no payload digest; solving instead", file.string());
        return std::nullopt;
    }

    result.sunDirection =
        glm::vec3(header.sunDirection[0], header.sunDirection[1], header.sunDirection[2]);
    result.elementCount = header.elementCount;
    result.participatingElements = header.participatingElements;
    result.exchangeNonZeros = header.exchangeNonZeros;
    result.stepsTaken = header.stepsTaken;
    result.minTemperature_K = header.minTemperature_K;
    result.maxTemperature_K = header.maxTemperature_K;
    result.meanTemperature_K = header.meanTemperature_K;

    if (DigestEntry(header, result) != StringView(storedDigest, kKeyHexLength)) {
        QL_LOG_WARN("  Thermal cache: {} failed its digest; solving instead", file.string());
        return std::nullopt;
    }

    return result;
}

bool StoreThermalSolveCache(const std::filesystem::path& file, StringView keyHex,
                            const ThermalResult& result) {
    if (keyHex.size() != kKeyHexLength || result.surfaceTemperature_K.empty()) {
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(file.parent_path(), ec);
    if (ec) {
        QL_LOG_WARN("  Thermal cache: cannot create {}: {}", file.parent_path().string(),
                    ec.message());
        return false;
    }

    // A temporary in the same directory, so the rename below is within one
    // filesystem and therefore atomic. The pid keeps two processes solving the
    // same scene from writing each other's temporary.
    std::filesystem::path temp = file;
    temp += ".tmp." + std::to_string(QL_GETPID());

    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) {
            QL_LOG_WARN("  Thermal cache: cannot write {}", temp.string());
            return false;
        }

        CacheHeader header;
        header.magic = kCacheMagic;
        header.formatVersion = kCacheFormatVersion;
        std::memcpy(header.keyHex, keyHex.data(), kKeyHexLength);
        header.temperatureCount = result.surfaceTemperature_K.size();
        header.instanceBaseCount = result.instanceElementBase.size();
        header.sensitivityCount = result.sunSensitivity_K.size();
        header.visibilityCount = result.sunVisibility.size();
        header.lagSlots = result.lagSlots;
        header.lagSensitivityCount = result.lagSensitivity_K.size();
        header.lagVisibilityCount = result.lagVisibility.size();
        header.lagDirectionCount = result.lagDirection.size();
        header.sunDirection[0] = result.sunDirection.x;
        header.sunDirection[1] = result.sunDirection.y;
        header.sunDirection[2] = result.sunDirection.z;
        header.elementCount = result.elementCount;
        header.participatingElements = result.participatingElements;
        header.exchangeNonZeros = result.exchangeNonZeros;
        header.stepsTaken = result.stepsTaken;
        header.minTemperature_K = result.minTemperature_K;
        header.maxTemperature_K = result.maxTemperature_K;
        header.meanTemperature_K = result.meanTemperature_K;

        WritePod(out, header.magic);
        WritePod(out, header.formatVersion);
        out.write(header.keyHex, kKeyHexLength);
        WritePod(out, header.temperatureCount);
        WritePod(out, header.instanceBaseCount);
        WritePod(out, header.sensitivityCount);
        WritePod(out, header.visibilityCount);
        WritePod(out, header.lagSlots);
        WritePod(out, header.lagSensitivityCount);
        WritePod(out, header.lagVisibilityCount);
        WritePod(out, header.lagDirectionCount);
        WritePod(out, header.sunDirection);
        WritePod(out, header.elementCount);
        WritePod(out, header.participatingElements);
        WritePod(out, header.exchangeNonZeros);
        WritePod(out, header.stepsTaken);
        WritePod(out, header.minTemperature_K);
        WritePod(out, header.maxTemperature_K);
        WritePod(out, header.meanTemperature_K);

        const auto writeArray = [&out](const auto& vec) {
            if (!vec.empty()) {
                out.write(reinterpret_cast<const char*>(vec.data()),
                          static_cast<std::streamsize>(
                              vec.size() *
                              sizeof(typename std::decay_t<decltype(vec)>::value_type)));
            }
        };
        writeArray(result.surfaceTemperature_K);
        writeArray(result.instanceElementBase);
        writeArray(result.sunSensitivity_K);
        writeArray(result.sunVisibility);
        writeArray(result.lagSensitivity_K);
        writeArray(result.lagVisibility);
        writeArray(result.lagDirection);

        const String digest = DigestEntry(header, result);
        out.write(digest.data(), kKeyHexLength);

        out.flush();
        if (!out.good()) {
            out.close();
            std::filesystem::remove(temp, ec);
            QL_LOG_WARN("  Thermal cache: writing {} failed; the entry is not kept",
                        temp.string());
            return false;
        }
    }

    std::filesystem::rename(temp, file, ec);
    if (ec) {
        // Windows will not rename onto an existing file. If the destination is
        // there now, another process solved the same scene and won -- its entry
        // is as good as ours, so this is a success with nothing left to do.
        std::error_code existsEc;
        const bool winner = std::filesystem::exists(file, existsEc) && !existsEc;
        std::error_code removeEc;
        std::filesystem::remove(temp, removeEc);
        if (winner) {
            return true;
        }
        QL_LOG_WARN("  Thermal cache: cannot place {}: {}", file.string(), ec.message());
        return false;
    }

    return true;
}

}  // namespace quantiloom::thermal
