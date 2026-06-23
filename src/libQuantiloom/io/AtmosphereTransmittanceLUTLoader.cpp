#include "AtmosphereTransmittanceLUTLoader.hpp"
#include "core/Log.hpp"

#include <fstream>
#include <sstream>
#include <cstring>

// tomlplusplus for TOML parsing
#include <toml++/toml.hpp>

namespace quantiloom {

// ============================================================================
// Load Implementation
// ============================================================================

auto AtmosphereTransmittanceLUTLoader::Load(const fs::path& filepath)
    -> Optional<AtmosphereTransmittanceLUT>
{
    // Check file exists
    if (!fs::exists(filepath)) {
        Log::Error("AtmosphereTransmittanceLUTLoader: File not found: {}", filepath.string());
        return std::nullopt;
    }

    // Check file extension
    if (filepath.extension() != FILE_EXTENSION) {
        Log::Warn("AtmosphereTransmittanceLUTLoader: Unexpected extension '{}', expected '{}'",
                  filepath.extension().string(), FILE_EXTENSION);
    }

    // Open file
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        Log::Error("AtmosphereTransmittanceLUTLoader: Failed to open file: {}", filepath.string());
        return std::nullopt;
    }

    // Read TOML header (fixed 1024 bytes)
    String header(HEADER_SIZE, '\0');
    file.read(header.data(), static_cast<std::streamsize>(HEADER_SIZE));
    if (!file) {
        Log::Error("AtmosphereTransmittanceLUTLoader: Failed to read header from: {}",
                   filepath.string());
        return std::nullopt;
    }

    // Trim null padding from header
    auto null_pos = header.find('\0');
    if (null_pos != String::npos) {
        header.resize(null_pos);
    }

    // Parse header
    AtmosphereTransmittanceLUT lut;
    if (!ParseHeader(header, lut)) {
        Log::Error("AtmosphereTransmittanceLUTLoader: Failed to parse header in: {}",
                   filepath.string());
        return std::nullopt;
    }

    // Calculate expected data sizes
    const usize total_elements = lut.TotalDataSize();
    const usize transmittance_bytes = total_elements * sizeof(f32);

    // Read transmittance data
    lut.transmittance.resize(total_elements);
    file.read(reinterpret_cast<char*>(lut.transmittance.data()),
              static_cast<std::streamsize>(transmittance_bytes));

    if (!file) {
        Log::Error("AtmosphereTransmittanceLUTLoader: Failed to read transmittance data from: {}",
                   filepath.string());
        return std::nullopt;
    }

    // Try to read path radiance (optional)
    // Check if there's more data in the file
    auto current_pos = file.tellg();
    file.seekg(0, std::ios::end);
    auto file_size = file.tellg();
    file.seekg(current_pos);

    usize remaining_bytes = static_cast<usize>(file_size - current_pos);
    if (remaining_bytes >= transmittance_bytes) {
        lut.path_radiance.resize(total_elements);
        file.read(reinterpret_cast<char*>(lut.path_radiance.data()),
                  static_cast<std::streamsize>(transmittance_bytes));

        if (!file) {
            Log::Warn("AtmosphereTransmittanceLUTLoader: Failed to read path radiance, "
                      "continuing without it");
            lut.path_radiance.clear();
        }
    }

    // Validate loaded data
    if (!lut.IsValid()) {
        Log::Error("AtmosphereTransmittanceLUTLoader: Loaded LUT failed validation: {}",
                   filepath.string());
        return std::nullopt;
    }

    Log::Info("AtmosphereTransmittanceLUTLoader: Loaded '{}' from {} "
              "(λ: {:.0f}-{:.0f}nm, alt: 0-{:.0f}m, {} elements)",
              lut.name, filepath.string(),
              lut.wavelength.start, lut.wavelength.stop,
              lut.altitude.stop, total_elements);

    return lut;
}

// ============================================================================
// Save Implementation
// ============================================================================

auto AtmosphereTransmittanceLUTLoader::Save(const fs::path& filepath,
                                             const AtmosphereTransmittanceLUT& lut) -> bool
{
    // Validate LUT before saving
    if (!lut.IsValid()) {
        Log::Error("AtmosphereTransmittanceLUTLoader: Cannot save invalid LUT");
        return false;
    }

    // Generate TOML header
    String header = GenerateHeader(lut);

    // Ensure header fits in HEADER_SIZE
    if (header.size() > HEADER_SIZE) {
        Log::Error("AtmosphereTransmittanceLUTLoader: Header exceeds {} bytes ({} bytes)",
                   HEADER_SIZE, header.size());
        return false;
    }

    // Pad header to HEADER_SIZE with null bytes
    header.resize(HEADER_SIZE, '\0');

    // Open file for writing
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        Log::Error("AtmosphereTransmittanceLUTLoader: Failed to create file: {}",
                   filepath.string());
        return false;
    }

    // Write header
    file.write(header.data(), static_cast<std::streamsize>(HEADER_SIZE));

    // Write transmittance data
    const usize data_bytes = lut.transmittance.size() * sizeof(f32);
    file.write(reinterpret_cast<const char*>(lut.transmittance.data()),
               static_cast<std::streamsize>(data_bytes));

    // Write path radiance if available
    if (!lut.path_radiance.empty()) {
        file.write(reinterpret_cast<const char*>(lut.path_radiance.data()),
                   static_cast<std::streamsize>(data_bytes));
    }

    if (!file) {
        Log::Error("AtmosphereTransmittanceLUTLoader: Failed to write data to: {}",
                   filepath.string());
        return false;
    }

    Log::Info("AtmosphereTransmittanceLUTLoader: Saved '{}' to {} ({} bytes)",
              lut.name, filepath.string(), HEADER_SIZE + data_bytes * (lut.path_radiance.empty() ? 1 : 2));

    return true;
}

// ============================================================================
// PeekInfo Implementation
// ============================================================================

auto AtmosphereTransmittanceLUTLoader::PeekInfo(const fs::path& filepath)
    -> Optional<LUTInfo>
{
    if (!fs::exists(filepath)) {
        return std::nullopt;
    }

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }

    // Read only the header
    String header(HEADER_SIZE, '\0');
    file.read(header.data(), static_cast<std::streamsize>(HEADER_SIZE));
    if (!file) {
        return std::nullopt;
    }

    // Trim null padding
    auto null_pos = header.find('\0');
    if (null_pos != String::npos) {
        header.resize(null_pos);
    }

    // Parse TOML
    try {
        toml::table config = toml::parse(header);

        LUTInfo info;
        info.name = config["metadata"]["name"].value_or("Unknown");
        info.source = config["metadata"]["source"].value_or("Unknown");
        info.created = config["metadata"]["created"].value_or("Unknown");
        info.atmospheric_model = config["metadata"]["atmospheric_model"].value_or("Unknown");

        info.wavelength_min = config["grid"]["wavelength"]["start"].value_or(0.0f);
        info.wavelength_max = config["grid"]["wavelength"]["stop"].value_or(0.0f);
        info.altitude_max = config["grid"]["altitude"]["stop"].value_or(0.0f);

        // Calculate data size
        u32 n_wave = config["grid"]["wavelength"]["count"].value_or(0u);
        u32 n_alt = config["grid"]["altitude"]["count"].value_or(0u);
        u32 n_zen = 0;

        if (auto zenith_arr = config["grid"]["zenith"]["values"].as_array()) {
            n_zen = static_cast<u32>(zenith_arr->size());
        }

        info.data_size_bytes = static_cast<usize>(n_wave) * n_alt * n_zen * sizeof(f32);

        return info;

    } catch (const toml::parse_error& e) {
        Log::Error("AtmosphereTransmittanceLUTLoader::PeekInfo: TOML parse error: {}", e.description());
        return std::nullopt;
    }
}

// ============================================================================
// ParseHeader Implementation
// ============================================================================

auto AtmosphereTransmittanceLUTLoader::ParseHeader(const String& header,
                                                    AtmosphereTransmittanceLUT& lut) -> bool
{
    try {
        toml::table config = toml::parse(header);

        // Parse metadata
        lut.name = config["metadata"]["name"].value_or("Unknown");
        lut.source = config["metadata"]["source"].value_or("Unknown");
        lut.created = config["metadata"]["created"].value_or("");
        lut.atmospheric_model = config["metadata"]["atmospheric_model"].value_or("Unknown");
        lut.ihaze   = config["metadata"]["ihaze"].value_or(4);
        lut.weather = config["metadata"]["weather"].value_or(0);

        // Parse wavelength axis (uniform)
        if (auto wave = config["grid"]["wavelength"].as_table()) {
            lut.wavelength.start = (*wave)["start"].value_or(0.0f);
            lut.wavelength.stop = (*wave)["stop"].value_or(0.0f);
            lut.wavelength.step = (*wave)["step"].value_or(1.0f);
            lut.wavelength.count = (*wave)["count"].value_or(0u);
        } else {
            Log::Error("ParseHeader: Missing [grid.wavelength] section");
            return false;
        }

        // Parse altitude axis (uniform)
        if (auto alt = config["grid"]["altitude"].as_table()) {
            lut.altitude.start = (*alt)["start"].value_or(0.0f);
            lut.altitude.stop = (*alt)["stop"].value_or(0.0f);
            lut.altitude.step = (*alt)["step"].value_or(1.0f);
            lut.altitude.count = (*alt)["count"].value_or(0u);
        } else {
            Log::Error("ParseHeader: Missing [grid.altitude] section");
            return false;
        }

        // Parse zenith axis (non-uniform array)
        if (auto zenith_arr = config["grid"]["zenith"]["values"].as_array()) {
            lut.zenith.values.clear();
            lut.zenith.values.reserve(zenith_arr->size());
            for (const auto& val : *zenith_arr) {
                if (auto fval = val.value<f64>()) {
                    lut.zenith.values.push_back(static_cast<f32>(*fval));
                }
            }
        } else {
            Log::Error("ParseHeader: Missing [grid.zenith.values] array");
            return false;
        }

        // Validate axes
        if (!lut.wavelength.IsValid()) {
            Log::Error("ParseHeader: Invalid wavelength axis");
            return false;
        }
        if (!lut.altitude.IsValid()) {
            Log::Error("ParseHeader: Invalid altitude axis");
            return false;
        }
        if (!lut.zenith.IsValid()) {
            Log::Error("ParseHeader: Invalid zenith axis");
            return false;
        }

        return true;

    } catch (const toml::parse_error& e) {
        Log::Error("ParseHeader: TOML parse error at line {}: {}",
                   e.source().begin.line, e.description());
        return false;
    }
}

// ============================================================================
// GenerateHeader Implementation
// ============================================================================

auto AtmosphereTransmittanceLUTLoader::GenerateHeader(const AtmosphereTransmittanceLUT& lut)
    -> String
{
    std::ostringstream ss;

    // Metadata section
    ss << "[metadata]\n";
    ss << "name = \"" << lut.name << "\"\n";
    ss << "source = \"" << lut.source << "\"\n";
    ss << "created = \"" << lut.created << "\"\n";
    ss << "atmospheric_model = \"" << lut.atmospheric_model << "\"\n";
    ss << "ihaze = " << lut.ihaze << "\n";
    ss << "weather = " << lut.weather << "\n";
    ss << "format_version = " << FORMAT_VERSION << "\n";
    ss << "\n";

    // Wavelength grid
    ss << "[grid.wavelength]\n";
    ss << "start = " << lut.wavelength.start << "\n";
    ss << "stop = " << lut.wavelength.stop << "\n";
    ss << "step = " << lut.wavelength.step << "\n";
    ss << "count = " << lut.wavelength.count << "\n";
    ss << "\n";

    // Altitude grid
    ss << "[grid.altitude]\n";
    ss << "start = " << lut.altitude.start << "\n";
    ss << "stop = " << lut.altitude.stop << "\n";
    ss << "step = " << lut.altitude.step << "\n";
    ss << "count = " << lut.altitude.count << "\n";
    ss << "\n";

    // Zenith angles (non-uniform)
    ss << "[grid.zenith]\n";
    ss << "values = [";
    for (usize i = 0; i < lut.zenith.values.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << lut.zenith.values[i];
    }
    ss << "]\n";
    ss << "\n";

    // Data section (documentation only, actual data follows header)
    const usize total_elements = lut.TotalDataSize();
    const usize transmittance_bytes = total_elements * sizeof(f32);

    ss << "[data]\n";
    ss << "header_size = " << HEADER_SIZE << "\n";
    ss << "transmittance_offset = " << HEADER_SIZE << "\n";
    ss << "transmittance_count = " << total_elements << "\n";

    if (!lut.path_radiance.empty()) {
        ss << "path_radiance_offset = " << (HEADER_SIZE + transmittance_bytes) << "\n";
        ss << "path_radiance_count = " << total_elements << "\n";
    }

    ss << "dtype = \"float32\"\n";
    ss << "byte_order = \"little\"\n";
    ss << "layout = \"wavelength_altitude_zenith\"\n";

    return ss.str();
}

} // namespace quantiloom
