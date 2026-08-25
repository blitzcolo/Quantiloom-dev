/**
 * @file Sha256.hpp
 * @brief SHA-256, streaming, stdlib only -- the digest behind the cache keys
 *
 * Internal. FIPS 180-4, about a hundred lines, no dependency: the repo had no
 * hash of any kind before this and pulling one in for a cache key would have
 * been the larger change.
 *
 * WHY A CRYPTOGRAPHIC DIGEST FOR A CACHE KEY. A thermal solve cache entry is
 * addressed by a hash of its inputs. A collision does not corrupt anything
 * visibly -- it serves *another scene's* temperature field, and the render
 * exits 0 with a plausible-looking thermogram. That is the worst failure shape
 * there is, so the key is 256 bits rather than the 64 a `std::hash` mix would
 * give. The cost is nothing next to what it protects: hashing a 1.5M-element
 * mesh is milliseconds against a 165 s solve.
 *
 * FRAMING IS PART OF THE HASH. `Update` on two adjacent strings cannot
 * distinguish ("ab","c") from ("a","bc"), which for a key built by
 * concatenating heterogeneous fields is a collision waiting to happen. Use
 * `UpdateSized`/`UpdateString` for anything variable-length -- they prefix the
 * byte count -- and the typed `UpdateU32`/`UpdateF64`/... for scalars, which
 * are fixed-width and self-delimiting. The typed helpers also pin the byte
 * order and width at the call site, so a key does not change meaning when a
 * field's type does.
 *
 * Scalars are hashed little-endian regardless of host, so a key computed on one
 * machine names the same entry on another. Floats go in by their bit pattern:
 * NaN hashes deterministically, and -0.0 differs from +0.0, which can only ever
 * cost a spurious miss.
 */

#pragma once

#include "core/Types.hpp"

#include <array>
#include <cstring>

namespace quantiloom::core {

/// Streaming SHA-256. Feed with Update*, then Finalize once -- the object is
/// spent afterwards and must not be fed again.
class Sha256 {
public:
    static constexpr usize kDigestBytes = 32;
    using Digest = std::array<u8, kDigestBytes>;

    Sha256() = default;

    /// Raw bytes, no framing. For fixed-width or already-delimited data only.
    void Update(const void* data, usize bytes);

    /// Length-prefixed bytes: hashes the count as u64, then the payload. Two
    /// adjacent calls can never be confused for one longer one.
    void UpdateSized(const void* data, usize bytes);
    void UpdateString(StringView text);

    // Fixed-width scalars, little-endian. Self-delimiting, so no length prefix.
    void UpdateU8(u8 value);
    void UpdateU32(u32 value);
    void UpdateU64(u64 value);
    void UpdateF32(f32 value);
    void UpdateF64(f64 value);
    void UpdateBool(bool value) { UpdateU8(value ? 1u : 0u); }

    /// Pads, appends the length, and returns the digest. Call once.
    [[nodiscard]] Digest Finalize();

    /// The same digest as 64 lowercase hex characters -- a usable filename.
    [[nodiscard]] String FinalizeHex();

private:
    void CompressBlock(const u8* block);

    std::array<u32, 8> state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::array<u8, 64> buffer_{};
    usize bufferLength_ = 0;
    u64 totalBits_ = 0;
};

/// One-shot convenience for the common "digest this blob" case.
[[nodiscard]] String Sha256Hex(const void* data, usize bytes);

}  // namespace quantiloom::core
