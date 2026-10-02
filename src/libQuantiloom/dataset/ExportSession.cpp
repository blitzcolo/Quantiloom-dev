#include "dataset/ExportSession.hpp"
#include "core/Sha256.hpp"
#include "io/ImageIO.hpp"

#include <nlohmann/json.hpp>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfChannelList.h>
#include <cmath>
#include <map>
#include <OpenEXR/ImfStringAttribute.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace quantiloom::dataset {
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
using Status = Result<void, String>;
constexpr int schemaVersion = 1;
constexpr const char* lockSuffix = ".quantiloom-export.lock";

// Serialize only reservation changes. Rendering and writing disjoint output
// sets never hold this gate. Persistent per-artifact claims survive a crash.
class ReservationGate {
public:
    ReservationGate() {
#ifdef _WIN32
        handle = CreateMutexW(nullptr, FALSE, L"Local\\QuantiloomExportReservationsV1");
        if (!handle) throw std::runtime_error("cannot create export reservation gate");
        const auto status = WaitForSingleObject(handle, INFINITE);
        if (status != WAIT_OBJECT_0 && status != WAIT_ABANDONED) {
            CloseHandle(handle);
            throw std::runtime_error("cannot acquire export reservation gate");
        }
#else
        const auto path = fs::temp_directory_path() / "quantiloom-export-reservations-v1";
        handle = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (handle < 0) throw std::runtime_error("cannot open export reservation gate");
        if (flock(handle, LOCK_EX) != 0) {
            ::close(handle);
            throw std::runtime_error("cannot acquire export reservation gate");
        }
#endif
    }
    ~ReservationGate() {
#ifdef _WIN32
        ReleaseMutex(handle);
        CloseHandle(handle);
#else
        flock(handle, LOCK_UN);
        ::close(handle);
#endif
    }
    ReservationGate(const ReservationGate&) = delete;
    ReservationGate& operator=(const ReservationGate&) = delete;
private:
#ifdef _WIN32
    HANDLE handle = nullptr;
#else
    int handle = -1;
#endif
};

fs::path ClaimPath(const fs::path& path) {
    auto result = path;
    result += lockSuffix;
    return result;
}

String Fold(String name) {
    for (auto& c : name) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
    return name;
}
bool Basename(const String& name) {
    if (name.empty()) return false;
    for (unsigned char c : name) if (c < 32) return false;
    auto device = name.substr(0, name.find('.'));
    for (auto& c : device) if (c >= 'a' && c <= 'z') c = static_cast<char>(c - ('a' - 'A'));
    if (device == "CON" || device == "PRN" || device == "AUX" || device == "NUL" ||
        (device.size() == 4 && (device.starts_with("COM") || device.starts_with("LPT")) &&
         device[3] >= '1' && device[3] <= '9')) return false;
    return name != "." && name != ".." && name.find_first_of("<>\"|?*") == String::npos &&
           name.find_first_of("/\\:") == String::npos &&
           name.back() != '.' && name.back() != ' ';
}
bool RelativeArtifact(const String& name) {
    if (name.empty() || name.front() == '/' || name.find('\\') != String::npos) return false;
    if (fs::path(name).lexically_normal().generic_string() != name) return false;
    for (const auto& part : fs::path(name)) {
        if (!Basename(part.string()) || Fold(part.string()) == ".internal" || Fold(part.string()).ends_with(lockSuffix)) return false;
    }
    return true;
}
void CheckParents(const fs::path& root, const fs::path& name) {
    fs::path current = root;
    for (const auto& part : name.parent_path()) {
        current /= part;
        if (fs::is_symlink(fs::symlink_status(current)))
            throw std::runtime_error("artifact parent is a symbolic link: " + current.string());
    }
}
String Digest(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read " + path.string());
    core::Sha256 hash;
    std::array<char, 65536> buffer;
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        hash.Update(buffer.data(), static_cast<usize>(file.gcount()));
    }
    if (!file.eof()) throw std::runtime_error("read failed: " + path.string());
    return hash.FinalizeHex();
}
void WriteText(const fs::path& path, const String& text) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("cannot write " + path.string());
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    file.close();
    if (!file) throw std::runtime_error("write failed: " + path.string());
}
void Replace(const fs::path& from, const fs::path& to) {
#ifdef _WIN32
    if (!MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("atomic replacement failed: " + to.string() +
                                 " (Windows error " + std::to_string(GetLastError()) + ")");
#else
    fs::rename(from, to);
#endif
}
String NewId() {
    std::random_device random;
    core::Sha256 hash;
    for (int i = 0; i < 8; ++i) hash.UpdateU32(random());
    hash.UpdateU64(static_cast<u64>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    return hash.FinalizeHex();
}
std::map<String, String> ImageSummary(const fs::path& path, Json& dimensions) {
    std::map<String, String> summary;
    const auto extension = Fold(path.extension().string());
    if (extension == ".exr") {
        Imf::InputFile file(path.string().c_str());
        const auto window = file.header().dataWindow();
        u32 channels = 0;
        for (auto it = file.header().channels().begin(); it != file.header().channels().end(); ++it) ++channels;
        dimensions = {{"width", window.max.x-window.min.x+1}, {"height", window.max.y-window.min.y+1}, {"channels", channels}};
        for (const char* key : {"quantiloom_record_id", "quantiloom_product_id", "quantiloom_sidecar"}) {
            const auto* attribute = file.header().findTypedAttribute<Imf::StringAttribute>(key);
            if (attribute) summary[key] = attribute->value();
        }
    } else if (extension == ".png") {
        std::ifstream input(path, std::ios::binary);
        char signature[8]{};
        input.read(signature, 8);
        if (!input || String(signature, 8) != String("\x89PNG\r\n\x1a\n", 8))
            throw std::runtime_error("invalid PNG signature");
        bool ended = false;
        while (input && !ended) {
            unsigned char header[8]{};
            input.read(reinterpret_cast<char*>(header), 8);
            if (!input) break;
            const u32 size = (u32(header[0]) << 24) | (u32(header[1]) << 16) |
                             (u32(header[2]) << 8) | u32(header[3]);
            const String type(reinterpret_cast<char*>(header + 4), 4);
            if (type == "IHDR") {
                if (size != 13 || !dimensions.is_null()) throw std::runtime_error("invalid PNG image header");
                unsigned char data[13]{};
                input.read(reinterpret_cast<char*>(data), 13);
                const auto u32be = [](const unsigned char* p) { return (u32(p[0])<<24)|(u32(p[1])<<16)|(u32(p[2])<<8)|u32(p[3]); };
                const u32 channels = data[9] == 0 || data[9] == 3 ? 1 : data[9] == 2 ? 3 : data[9] == 4 ? 2 : data[9] == 6 ? 4 : 0;
                dimensions = {{"width", u32be(data)}, {"height", u32be(data+4)}, {"channels", channels}};
            } else if (type == "tEXt" || type == "iTXt") {
                if (size > 1024 * 1024) throw std::runtime_error("oversized PNG summary");
                String data(size, '\0');
                input.read(data.data(), size);
                const auto separator = data.find('\0');
                if (separator == String::npos) throw std::runtime_error("invalid PNG text");
                const String key = data.substr(0, separator);
                usize start = separator + 1;
                if (type == "iTXt") {
                    if (start + 2 > data.size() || data[start] != 0 || data[start + 1] != 0)
                        throw std::runtime_error("unsupported PNG compressed summary");
                    start += 2;
                    for (int field = 0; field < 2; ++field) {
                        const auto end = data.find('\0', start);
                        if (end == String::npos) throw std::runtime_error("invalid PNG international text");
                        start = end + 1;
                    }
                }
                if (!summary.emplace(key, data.substr(start)).second)
                    throw std::runtime_error("duplicate PNG summary key");
            } else input.seekg(size, std::ios::cur);
            char crc[4]{}; input.read(crc, 4); // CRC bytes are covered by the managed file hash.
            ended = type == "IEND";
        }
        if (!input || !ended) throw std::runtime_error("truncated PNG");
    }
    return summary;
}
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
bool NonnegativeInteger(const Json& value) {
    return value.is_number_unsigned() || (value.is_number_integer() && value.get<i64>() >= 0);
}
Json::parser_callback_t UniqueKeys() {
    return [objects = Vector<std::set<String>>{}](int, Json::parse_event_t event, Json& value) mutable {
        if (event == Json::parse_event_t::object_start) objects.emplace_back();
        else if (event == Json::parse_event_t::key) {
            if (objects.empty() || !objects.back().insert(value.get<String>()).second)
                throw std::runtime_error("duplicate JSON object key");
        } else if (event == Json::parse_event_t::object_end) objects.pop_back();
        return true;
    };
}
void ValidateRecordShape(const Json& record) {
    Require(record.is_object(), "record must be an object");
    const std::set<String> required{"schema", "schema_version", "record_id", "state",
        "capture_status", "pairing_status", "provenance", "replay", "products"};
    for (const auto& key : required) Require(record.contains(key), "missing required record field");
    for (auto it = record.begin(); it != record.end(); ++it)
        Require(required.contains(it.key()) || it.key() == "error", "unknown record field");
    if (record.contains("error")) Require(record.at("error").is_string(), "error must be a string");
    Require(record.at("provenance").is_object(), "provenance must be an object");
    const auto& pairing = record.at("pairing_status");
    Require(pairing == "not_requested" || pairing == "pending" || pairing == "complete" || pairing == "failed",
            "invalid pairing status");
    Require(record.at("replay").is_object(), "replay must be an object");
    Require(record.at("products").is_array() && !record.at("products").empty(), "record has no products");
    for (const auto& product : record.at("products")) {
        Require(product.is_object() && product.contains("product_id") && product.at("product_id").is_string() &&
                !product.at("product_id").get<String>().empty(), "missing or invalid product ID");
        Require(product.contains("size_bytes") && NonnegativeInteger(product.at("size_bytes")),
                "missing or invalid product size");
        Require(product.contains("description") && product.at("description").is_object(),
                "missing or invalid product description");
    }
}
void ValidateGeometry(const Json& description) {
    if (!description.contains("provenance")) return;
    const auto& provenance = description.at("provenance");
    Require(provenance.is_object(), "product provenance must be an object");
    if (!provenance.contains("geometry")) return;
    const auto& g = provenance.at("geometry");
    Require(g.is_object() && g.at("version") == 1, "invalid geometry version");
    for (const char* key : {"width", "height"}) {
        Require(NonnegativeInteger(g.at(key)) && g.at(key).get<u64>() > 0 && g.at(key) == description.at(key),
                "geometry and product grid disagree");
    }
    Require(g.at("kind") == "instantaneous_geometry" && g.at("camera_axes") == "right_down_forward" &&
            g.at("pixel_origin") == "top_left" && g.at("matrix_order") == "row_major" &&
            g.at("pixel_center_offset") == Json::array({0.5, 0.5}), "invalid geometry convention");
    const auto scalar = [](const Json& j) {
        Require(j.is_number() && std::isfinite(j.get<f64>()), "non-finite geometry value");
        return j.get<f64>();
    };
    (void)scalar(g.at("reference_time_s"));
    Require(scalar(g.at("world_units_to_meters")) > 0, "invalid world scale");
    const auto array = [&](const Json& j, usize count) {
        Require(j.is_array() && j.size() == count, "invalid geometry array");
        Vector<f64> out; out.reserve(count);
        for (const auto& v : j) out.push_back(scalar(v));
        return out;
    };
    const auto w = array(g.at("world_to_camera"), 16), c = array(g.at("camera_to_world"), 16);
    for (usize i = 0; i < 4; ++i) for (usize j = 0; j < 4; ++j) {
        f64 value = 0;
        for (usize k = 0; k < 4; ++k) value += w[i*4+k] * c[k*4+j];
        Require(std::abs(value - (i == j ? 1.0 : 0.0)) < 1e-5, "camera matrices are not inverses");
    }
    if (g.at("projection") == "perspective") {
        const auto k = array(g.at("intrinsics"), 9);
        Require(k[0] > 0 && k[4] > 0 && k[8] == 1 && k[1] == 0 && k[3] == 0 && k[6] == 0 && k[7] == 0 &&
                k[2] == scalar(g.at("width"))/2 && k[5] == scalar(g.at("height"))/2, "invalid intrinsics");
    } else {
        Require(g.at("projection") == "orthographic" && g.at("intrinsics").is_null() &&
                scalar(g.at("film_height_world_units")) > 0 && scalar(g.at("film_width_world_units")) > 0,
                "invalid orthographic projection");
    }
}
void CheckProductImage(const fs::path& artifact, const fs::path& relativeName,
                       const fs::path& sidecarName, const String& recordId, const Json& product) {
    const auto extension = Fold(relativeName.extension().string());
    if (extension != ".exr" && extension != ".png") return;
    Json dimensions;
    const auto summary = ImageSummary(artifact, dimensions);
    for (const char* key : {"width", "height", "channels"})
        Require(dimensions.contains(key) && dimensions.at(key).is_number_integer() &&
            dimensions.at(key).get<i64>() > 0 && product.at("description").at(key) == dimensions.at(key),
            "image dimensions/channels disagree with product description");
    const auto expectedSidecar = sidecarName.lexically_relative(
        relativeName.parent_path().empty() ? fs::path(".") : relativeName.parent_path()).generic_string();
    const auto matches = [&](const char* key, const String& expected) {
        const auto found = summary.find(key);
        if (found == summary.end() || found->second != expected)
            throw std::runtime_error(String("image summary mismatch: ") + relativeName.generic_string() + ": " + key);
    };
    matches("quantiloom_record_id", recordId);
    matches("quantiloom_product_id", product.at("product_id").get<String>());
    matches("quantiloom_sidecar", expectedSidecar);
}
Json ReadJson(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    return Json::parse(input, UniqueKeys());
}
}

struct ExportSession::Impl {
    fs::path directory, staging;
    Vector<fs::path> claims;
    String id, sidecar, replay;
    Json record;
    std::set<String> names, products;
    bool publishing = false;
    bool complete = false;

    void Reserve(const fs::path& name) {
        const auto destination = directory / name;
        const auto claim = ClaimPath(destination);
        const ReservationGate gate;
        if (std::find(claims.begin(), claims.end(), claim) != claims.end()) return;
        CheckParents(directory, name);
        // Creating parent directories under the gate makes a file-versus-
        // descendant collision symmetric: either the ancestor is claimed or
        // it already exists as a directory and cannot become a product file.
        for (auto parent = destination.parent_path(); !parent.empty();) {
            if (fs::exists(ClaimPath(parent)) || fs::exists(parent / lockSuffix))
                throw std::runtime_error("output ancestor is locked: " + parent.string());
            const auto next = parent.parent_path();
            if (next == parent) break;
            parent = next;
        }
        const auto status = fs::symlink_status(destination);
        if (fs::is_symlink(status) || fs::is_directory(status))
            throw std::runtime_error("output is not a regular file: " + destination.string());
        fs::create_directories(destination.parent_path());
        // Allocate ownership bookkeeping before creating the on-disk claim.
        claims.push_back(claim);
        try {
            if (!fs::create_directory(claim))
                throw std::runtime_error("output is locked: " + destination.string());
        } catch (...) {
            claims.pop_back();
            throw;
        }
    }

    void PublishRecord() {
        const auto temporary = staging / ".internal" / "record.tmp";
        WriteText(temporary, record.dump(2) + "\n");
        Replace(temporary, directory / sidecar);
    }
    void RequireOpen() const {
        if (publishing || complete) throw std::runtime_error("export session is no longer writable");
    }
    ~Impl() {
        std::error_code ec;
        if (!staging.empty()) fs::remove_all(staging, ec);
        for (auto it = claims.rbegin(); it != claims.rend(); ++it) fs::remove(*it, ec);
    }
};
ExportSession::ExportSession() : m_impl(std::make_unique<Impl>()) {}
ExportSession::~ExportSession() = default;

Result<std::unique_ptr<ExportSession>, String> ExportSession::Create(
    const String& outputPath, const Config& replayConfig,
    const RenderProvenance& provenance) {
    try {
        auto session = std::unique_ptr<ExportSession>(new ExportSession);
        auto& impl = *session->m_impl;
        const fs::path output = fs::absolute(fs::path(outputPath));
        if (!Basename(output.filename().string()) || output.stem().empty())
            throw std::runtime_error("invalid export basename");
        impl.directory = fs::weakly_canonical(output.parent_path());
        impl.sidecar = output.stem().string() + ".metadata.json";
        impl.replay = output.stem().string() + ".replay.toml";
        if (!RelativeArtifact(output.filename().string())) throw std::runtime_error("reserved export basename");
        impl.Reserve(output.filename());
        impl.Reserve(impl.sidecar);
        impl.Reserve(impl.replay);
        impl.id = NewId();
        impl.staging = impl.claims.front() / impl.id;
        fs::create_directory(impl.staging);
        fs::create_directory(impl.staging / ".internal");
        impl.names.insert(".internal");
        const auto frozen = Json::parse(provenance.json, UniqueKeys());
        if (!frozen.is_object()) throw std::runtime_error("provenance must be an object");
        impl.names.insert("record.tmp");
        impl.names.insert(Fold(impl.sidecar));
        impl.names.insert(Fold(impl.replay));
        WriteText(impl.staging / impl.replay, replayConfig.ToToml());
        impl.record = {
            {"schema", "quantiloom.dataset.export"}, {"schema_version", schemaVersion},
            {"record_id", impl.id}, {"state", "staging"},
            {"capture_status", "complete"}, {"pairing_status", "not_requested"},
            {"provenance", frozen}, {"products", Json::array()},
            {"replay", {{"path", impl.replay}, {"sha256", Digest(impl.staging / impl.replay)},
                        {"replayable", false}, {"reason", "Resource and execution history verification is not implemented"}}}
        };
        return Result<std::unique_ptr<ExportSession>, String>(std::move(session));
    } catch (const std::exception& e) {
        return Result<std::unique_ptr<ExportSession>, String>::Err(e.what());
    }
}
Result<String, String> ExportSession::StagingPath(const String& name) const {
    try {
        m_impl->RequireOpen();
        if (!RelativeArtifact(name) || m_impl->names.contains(Fold(name)))
            throw std::runtime_error("invalid or duplicate product filename: " + name);
        m_impl->Reserve(fs::path(name));
        CheckParents(m_impl->staging, fs::path(name));
        fs::create_directories((m_impl->staging / name).parent_path());
        return Result<String, String>((m_impl->staging / name).string());
    } catch (const std::exception& e) { return Result<String, String>::Err(e.what()); }
}
Status ExportSession::RegisterFile(const String& name, const String& productId,
                                   const String& descriptionJson) {
    try {
        const auto path = StagingPath(name);
        if (!path) return Status::Err(path.error());
        if (productId.empty() || m_impl->products.contains(productId))
            throw std::runtime_error("empty or duplicate product ID");
        const auto description = Json::parse(descriptionJson, UniqueKeys());
        if (!description.is_object()) throw std::runtime_error("product description must be an object");
        const fs::path staged(path.value());
        if (!fs::is_regular_file(fs::symlink_status(staged)))
            throw std::runtime_error("product is not a regular file: " + name);
        m_impl->record["products"].push_back({
            {"product_id", productId}, {"path", name},
            {"sha256", Digest(staged)}, {"size_bytes", fs::file_size(staged)},
            {"description", description}
        });
        m_impl->names.insert(Fold(name));
        m_impl->products.insert(productId);
        return Status::Ok();
    } catch (const std::exception& e) { return Status::Err(e.what()); }
}
Status ExportSession::WriteImage(const String& name, const String& productId,
                                 const Image& image, const String& descriptionJson) {
    try {
        const auto path = StagingPath(name);
        if (!path) return Status::Err(path.error());
        if (!image.IsValid()) return Status::Err("invalid image");
        Image product = image;
        product.metadata["quantiloom_record_id"] = m_impl->id;
        product.metadata["quantiloom_product_id"] = productId;
        product.metadata["quantiloom_sidecar"] =
            fs::path(m_impl->sidecar).lexically_relative(fs::path(name).parent_path().empty() ?
                fs::path(".") : fs::path(name).parent_path()).generic_string();
        const auto extension = Fold(fs::path(name).extension().string());
        bool written = false;
        if (extension == ".exr") written = ImageIO::WriteEXR(path.value(), product);
        else if (extension == ".png") written = ImageIO::WritePNG(path.value(), product);
        else return Status::Err("unsupported image extension: " + extension);
        if (!written) return Status::Err("failed to write " + name);
        auto description = Json::parse(descriptionJson, UniqueKeys());
        description["width"] = image.width;
        description["height"] = image.height;
        description["channels"] = image.channels;
        description["channel_names"] = image.channelNames;
        description["image_metadata"] = image.metadata;
        if (const auto found = image.metadata.find("quantiloom_provenance"); found != image.metadata.end()) {
            description["provenance"] = Json::parse(found->second, UniqueKeys());
            description["image_metadata"].erase("quantiloom_provenance");
        }
        return RegisterFile(name, productId, description.dump());
    } catch (const std::exception& e) { return Status::Err(e.what()); }
}
Status ExportSession::Commit() {
    auto& impl = *m_impl;
    if (impl.complete || impl.publishing)
        return Status::Err("export session has already finished or started publication");
    try {
        impl.RequireOpen();
        if (impl.products.empty()) throw std::runtime_error("cannot publish an empty export");
        ValidateRecordShape(impl.record);
        // Check staged bytes before invalidating a previous complete export.
        for (const auto& product : impl.record["products"]) {
            ValidateGeometry(product.at("description"));
            const fs::path stagedName(product.at("path").get<String>());
            CheckParents(impl.staging, stagedName);
            if (!fs::is_regular_file(fs::symlink_status(impl.staging / stagedName)))
                throw std::runtime_error("staged product is no longer a regular file");
            if (Digest(impl.staging / product.at("path").get<String>()) != product.at("sha256").get<String>())
                throw std::runtime_error("staged product changed before publication");
            CheckProductImage(impl.staging / stagedName, stagedName, fs::path(impl.sidecar), impl.id, product);
        }
        if (Digest(impl.staging / impl.replay) != impl.record["replay"]["sha256"].get<String>())
            throw std::runtime_error("staged replay configuration changed");
        impl.record["state"] = "publishing";
        impl.PublishRecord();
        impl.publishing = true;
        Replace(impl.staging / impl.replay, impl.directory / impl.replay);
        for (const auto& product : impl.record["products"]) {
            const auto name = product.at("path").get<String>();
            CheckParents(impl.directory, fs::path(name));
            fs::create_directories((impl.directory / name).parent_path());
            Replace(impl.staging / name, impl.directory / name);
            if (Digest(impl.directory / name) != product.at("sha256").get<String>())
                throw std::runtime_error("published product hash mismatch: " + name);
        }
        if (Digest(impl.directory / impl.replay) != impl.record["replay"]["sha256"].get<String>())
            throw std::runtime_error("published replay hash mismatch");
        impl.record["state"] = "complete";
        impl.PublishRecord();
        impl.complete = true;
        return Status::Ok();
    } catch (const std::exception& e) {
        if (impl.publishing) {
            impl.record["state"] = "failed";
            impl.record["error"] = e.what();
            try { impl.PublishRecord(); } catch (...) { /* publishing remains invalid */ }
        }
        return Status::Err(e.what());
    }
}
const String& ExportSession::RecordId() const { return m_impl->id; }
const String& ExportSession::SidecarName() const { return m_impl->sidecar; }
VerificationReport ExportSession::Verify(const String& recordPath) {
    Json errors = Json::array();
    Json provenance = nullptr;
    try {
        const fs::path path(recordPath);
        const auto record = ReadJson(path);
        ValidateRecordShape(record);
        if (record.at("schema") != "quantiloom.dataset.export" ||
            !record.at("schema_version").is_number_integer() || record.at("schema_version") != schemaVersion)
            throw std::runtime_error("unsupported record schema");
        if (record.at("state") != "complete") throw std::runtime_error("record is not complete");
        const auto id = record.at("record_id").get<String>();
        if (id.size() != 64 || id.find_first_not_of("0123456789abcdef") != String::npos)
            throw std::runtime_error("invalid record ID");
        if (record.at("capture_status") != "complete") throw std::runtime_error("capture is not complete");
        if (!record.at("products").is_array() || record.at("products").empty())
            throw std::runtime_error("record has no products");
        std::set<String> names{Fold(path.filename().string())}, ids;
        const auto check = [&](const Json& entry) {
            const auto name = entry.at("path").get<String>();
            if (!RelativeArtifact(name) || !names.insert(Fold(name)).second)
                throw std::runtime_error("invalid or duplicate artifact path");
            const auto hash = entry.at("sha256").get<String>();
            if (hash.size() != 64 || hash.find_first_not_of("0123456789abcdef") != String::npos)
                throw std::runtime_error("invalid SHA-256");
            CheckParents(path.parent_path(), fs::path(name));
            const auto artifact = path.parent_path() / name;
            if (!fs::is_regular_file(fs::symlink_status(artifact)))
                throw std::runtime_error("missing or non-regular artifact: " + name);
            if (Digest(artifact) != hash) errors.push_back("hash mismatch: " + name);
            if (entry.contains("size_bytes") && entry.at("size_bytes").get<u64>() != fs::file_size(artifact))
                errors.push_back("size mismatch: " + name);
        };
        check(record.at("replay"));
        for (const auto& product : record.at("products")) {
            const auto productId = product.at("product_id").get<String>();
            if (productId.empty() || !ids.insert(productId).second)
                throw std::runtime_error("empty or duplicate product ID");
            if (!product.at("description").is_object()) throw std::runtime_error("invalid product description");
            ValidateGeometry(product.at("description"));
            check(product);
            CheckProductImage(path.parent_path() / product.at("path").get<String>(),
                fs::path(product.at("path").get<String>()), path.filename(), id, product);

        }
        provenance = record.at("provenance");
        if (!provenance.is_object()) throw std::runtime_error("invalid provenance");
    } catch (const std::exception& e) { errors.push_back(e.what()); }
    const bool valid = errors.empty();
    return {valid, Json{{"valid", valid}, {"checks", "export_integrity"},
        {"errors", errors}, {"reproducibility_verified", false}, {"provenance", provenance}}.dump(2)};
}
} // namespace quantiloom::dataset
