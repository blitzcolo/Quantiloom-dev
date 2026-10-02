#pragma once

#include "core/Platform.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include <limits>
#include <stdexcept>
#include <memory>

namespace quantiloom {

// ============================================================================
// Safetensors - minimal read-only safetensors container parser
// ============================================================================
// File layout: u64 LE header length | JSON header | raw tensor blob.
// Header maps tensor name -> {dtype, shape, data_offsets}, plus an optional
// "__metadata__" object of string key/value pairs.
//
// Only the dtypes the atmosphere model packs use are supported (F32, U8).
// All offsets are validated against the blob size; malformed files throw
// std::runtime_error with the file path in the message.

enum class SafetensorsDtype : uint8_t {
    F32,
    U8,
};

struct TensorView {
    SafetensorsDtype dtype = SafetensorsDtype::F32;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;  // File storage, or alignedStorage when needed
    size_t byteSize = 0;
    // Only needed for a legal F32 tensor whose file offset is not aligned.
    // Shared ownership keeps data stable when a view is copied or moved.
    std::shared_ptr<std::vector<float>> alignedStorage;

    int64_t NumElements() const {
        int64_t n = 1;
        for (int64_t d : shape) {
            if (d < 0 || (d != 0 && n > std::numeric_limits<int64_t>::max() / d))
                throw std::runtime_error("safetensors: tensor shape product out of range");
            n *= d;
        }
        return n;
    }
    const float* F32Data() const {
        if (dtype != SafetensorsDtype::F32)
            throw std::runtime_error("safetensors: tensor is not F32 data");
        return reinterpret_cast<const float*>(data);
    }
    const uint8_t* U8Data() const {
        if (dtype != SafetensorsDtype::U8)
            throw std::runtime_error("safetensors: tensor is not U8 data");
        return data;
    }
};

class SafetensorsFile {
public:
    // Reads the whole file into memory and parses/validates the header.
    // Throws std::runtime_error on any malformed content.
    explicit SafetensorsFile(const std::filesystem::path& path);

    // Tensor access. Get() throws if the name is absent; Find() returns
    // nullptr instead.
    const TensorView& Get(const std::string& name) const;
    const TensorView* Find(const std::string& name) const;
    bool Has(const std::string& name) const { return Find(name) != nullptr; }

    // __metadata__ access. Metadata() throws if the key is absent.
    const std::string& Metadata(const std::string& key) const;
    const std::string* FindMetadata(const std::string& key) const;

    const std::filesystem::path& Path() const { return path_; }
    std::vector<std::string> TensorNames() const;

private:
    std::filesystem::path path_;
    std::vector<uint8_t> storage_;
    std::unordered_map<std::string, TensorView> tensors_;
    std::unordered_map<std::string, std::string> metadata_;
};

}  // namespace quantiloom
