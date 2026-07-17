#include "atmos/Safetensors.hpp"

#include <nlohmann/json.hpp>

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace quantiloom {

namespace {

using nlohmann::json;

[[noreturn]] void Fail(const std::filesystem::path& path, const std::string& what) {
    throw std::runtime_error("safetensors: " + path.string() + ": " + what);
}

size_t DtypeSize(SafetensorsDtype dt) {
    switch (dt) {
        case SafetensorsDtype::F32: return 4;
        case SafetensorsDtype::U8: return 1;
    }
    return 0;
}

}  // namespace

SafetensorsFile::SafetensorsFile(const std::filesystem::path& path) : path_(path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) Fail(path, "cannot open file");
    const std::streamoff fileSize = f.tellg();
    if (fileSize < 8) Fail(path, "file too small for header length");
    f.seekg(0);
    storage_.resize(static_cast<size_t>(fileSize));
    if (!f.read(reinterpret_cast<char*>(storage_.data()), fileSize))
        Fail(path, "read failed");

    uint64_t headerLen = 0;
    std::memcpy(&headerLen, storage_.data(), 8);  // File format is little-endian
    if (headerLen > storage_.size() - 8)
        Fail(path, "header length " + std::to_string(headerLen) + " exceeds file size");

    json header;
    try {
        header = json::parse(storage_.begin() + 8, storage_.begin() + 8 + headerLen);
    } catch (const json::exception& e) {
        Fail(path, std::string("malformed JSON header: ") + e.what());
    }
    if (!header.is_object()) Fail(path, "header is not a JSON object");

    const uint8_t* blob = storage_.data() + 8 + headerLen;
    const size_t blobSize = storage_.size() - 8 - headerLen;

    for (auto& [name, info] : header.items()) {
        if (name == "__metadata__") {
            if (!info.is_object()) Fail(path, "__metadata__ is not an object");
            for (auto& [k, v] : info.items()) {
                if (!v.is_string()) Fail(path, "__metadata__." + k + " is not a string");
                metadata_[k] = v.get<std::string>();
            }
            continue;
        }
        if (!info.is_object()) Fail(path, "tensor '" + name + "' entry is not an object");

        TensorView t;
        const std::string dtype = info.value("dtype", "");
        if (dtype == "F32") t.dtype = SafetensorsDtype::F32;
        else if (dtype == "U8") t.dtype = SafetensorsDtype::U8;
        else Fail(path, "tensor '" + name + "' has unsupported dtype '" + dtype + "'");

        if (!info.contains("shape") || !info["shape"].is_array())
            Fail(path, "tensor '" + name + "' missing shape");
        int64_t numel = 1;
        for (auto& d : info["shape"]) {
            if (!d.is_number_unsigned() && !d.is_number_integer())
                Fail(path, "tensor '" + name + "' has non-integer shape");
            const int64_t dim = d.get<int64_t>();
            if (dim < 0) Fail(path, "tensor '" + name + "' has negative dimension");
            t.shape.push_back(dim);
            numel *= dim;
        }

        if (!info.contains("data_offsets") || !info["data_offsets"].is_array() ||
            info["data_offsets"].size() != 2)
            Fail(path, "tensor '" + name + "' missing data_offsets");
        const uint64_t o0 = info["data_offsets"][0].get<uint64_t>();
        const uint64_t o1 = info["data_offsets"][1].get<uint64_t>();
        if (o1 < o0 || o1 > blobSize)
            Fail(path, "tensor '" + name + "' data_offsets out of bounds");
        if (o1 - o0 != static_cast<uint64_t>(numel) * DtypeSize(t.dtype))
            Fail(path, "tensor '" + name + "' byte size does not match shape");

        t.data = blob + o0;
        t.byteSize = static_cast<size_t>(o1 - o0);
        if (!tensors_.emplace(name, std::move(t)).second)
            Fail(path, "duplicate tensor name '" + name + "'");
    }
}

const TensorView& SafetensorsFile::Get(const std::string& name) const {
    const TensorView* t = Find(name);
    if (!t) Fail(path_, "missing tensor '" + name + "'");
    return *t;
}

const TensorView* SafetensorsFile::Find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const std::string& SafetensorsFile::Metadata(const std::string& key) const {
    const std::string* v = FindMetadata(key);
    if (!v) Fail(path_, "missing metadata key '" + key + "'");
    return *v;
}

const std::string* SafetensorsFile::FindMetadata(const std::string& key) const {
    auto it = metadata_.find(key);
    return it == metadata_.end() ? nullptr : &it->second;
}

std::vector<std::string> SafetensorsFile::TensorNames() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (auto& [k, v] : tensors_) names.push_back(k);
    return names;
}

}  // namespace quantiloom
