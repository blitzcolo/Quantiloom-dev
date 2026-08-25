#include "Sha256.hpp"

#include <bit>

namespace quantiloom::core {

namespace {

// FIPS 180-4 section 4.2.2: the first 32 bits of the fractional parts of the
// cube roots of the first 64 primes.
constexpr std::array<u32, 64> kRoundConstants = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

constexpr u32 RotateRight(u32 value, int bits) {
    return std::rotr(value, bits);
}

}  // namespace

void Sha256::CompressBlock(const u8* block) {
    std::array<u32, 64> w{};
    for (usize i = 0; i < 16; ++i) {
        // Big-endian, as the standard specifies; the host's own order is
        // irrelevant because the bytes are assembled by hand.
        w[i] = (static_cast<u32>(block[i * 4 + 0]) << 24) |
               (static_cast<u32>(block[i * 4 + 1]) << 16) |
               (static_cast<u32>(block[i * 4 + 2]) << 8) |
               (static_cast<u32>(block[i * 4 + 3]));
    }
    for (usize i = 16; i < 64; ++i) {
        const u32 s0 = RotateRight(w[i - 15], 7) ^ RotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const u32 s1 = RotateRight(w[i - 2], 17) ^ RotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    u32 a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    u32 e = state_[4], f = state_[5], g = state_[6], h = state_[7];

    for (usize i = 0; i < 64; ++i) {
        const u32 s1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
        const u32 ch = (e & f) ^ (~e & g);
        const u32 temp1 = h + s1 + ch + kRoundConstants[i] + w[i];
        const u32 s0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
        const u32 maj = (a & b) ^ (a & c) ^ (b & c);
        const u32 temp2 = s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256::Update(const void* data, usize bytes) {
    const auto* cursor = static_cast<const u8*>(data);
    totalBits_ += static_cast<u64>(bytes) * 8u;

    // Top up a partial block first, then run whole blocks straight out of the
    // caller's buffer, then keep the remainder.
    if (bufferLength_ > 0) {
        const usize wanted = 64 - bufferLength_;
        const usize taken = bytes < wanted ? bytes : wanted;
        std::memcpy(buffer_.data() + bufferLength_, cursor, taken);
        bufferLength_ += taken;
        cursor += taken;
        bytes -= taken;
        if (bufferLength_ == 64) {
            CompressBlock(buffer_.data());
            bufferLength_ = 0;
        }
    }

    while (bytes >= 64) {
        CompressBlock(cursor);
        cursor += 64;
        bytes -= 64;
    }

    if (bytes > 0) {
        std::memcpy(buffer_.data(), cursor, bytes);
        bufferLength_ = bytes;
    }
}

void Sha256::UpdateSized(const void* data, usize bytes) {
    UpdateU64(static_cast<u64>(bytes));
    if (bytes > 0) {
        Update(data, bytes);
    }
}

void Sha256::UpdateString(StringView text) {
    UpdateSized(text.data(), text.size());
}

void Sha256::UpdateU8(u8 value) {
    Update(&value, 1);
}

void Sha256::UpdateU32(u32 value) {
    const u8 bytes[4] = {static_cast<u8>(value & 0xFFu), static_cast<u8>((value >> 8) & 0xFFu),
                         static_cast<u8>((value >> 16) & 0xFFu),
                         static_cast<u8>((value >> 24) & 0xFFu)};
    Update(bytes, sizeof(bytes));
}

void Sha256::UpdateU64(u64 value) {
    u8 bytes[8];
    for (usize i = 0; i < 8; ++i) {
        bytes[i] = static_cast<u8>((value >> (i * 8)) & 0xFFu);
    }
    Update(bytes, sizeof(bytes));
}

void Sha256::UpdateF32(f32 value) {
    UpdateU32(std::bit_cast<u32>(value));
}

void Sha256::UpdateF64(f64 value) {
    UpdateU64(std::bit_cast<u64>(value));
}

Sha256::Digest Sha256::Finalize() {
    const u64 lengthBits = totalBits_;

    // The padding is written straight into the block buffer rather than through
    // Update, which counts what it is fed as message: 0x80, zeroes up to offset
    // 56, then the 64-bit big-endian bit count.
    buffer_[bufferLength_++] = 0x80u;
    if (bufferLength_ > 56) {
        std::memset(buffer_.data() + bufferLength_, 0, 64 - bufferLength_);
        CompressBlock(buffer_.data());
        bufferLength_ = 0;
    }
    std::memset(buffer_.data() + bufferLength_, 0, 56 - bufferLength_);
    for (usize i = 0; i < 8; ++i) {
        buffer_[56 + i] = static_cast<u8>((lengthBits >> (56 - i * 8)) & 0xFFu);
    }
    CompressBlock(buffer_.data());
    bufferLength_ = 0;

    Digest digest{};
    for (usize i = 0; i < 8; ++i) {
        digest[i * 4 + 0] = static_cast<u8>((state_[i] >> 24) & 0xFFu);
        digest[i * 4 + 1] = static_cast<u8>((state_[i] >> 16) & 0xFFu);
        digest[i * 4 + 2] = static_cast<u8>((state_[i] >> 8) & 0xFFu);
        digest[i * 4 + 3] = static_cast<u8>(state_[i] & 0xFFu);
    }
    return digest;
}

String Sha256::FinalizeHex() {
    const Digest digest = Finalize();
    constexpr char kHex[] = "0123456789abcdef";
    String out(kDigestBytes * 2, '0');
    for (usize i = 0; i < kDigestBytes; ++i) {
        out[i * 2 + 0] = kHex[(digest[i] >> 4) & 0x0Fu];
        out[i * 2 + 1] = kHex[digest[i] & 0x0Fu];
    }
    return out;
}

String Sha256Hex(const void* data, usize bytes) {
    Sha256 hasher;
    hasher.Update(data, bytes);
    return hasher.FinalizeHex();
}

}  // namespace quantiloom::core
