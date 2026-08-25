/**
 * @file test_sha256.cpp
 * @brief SHA-256 against the FIPS 180-4 vectors, plus the framing the keys rely on
 *
 * The digest itself is checked against published vectors rather than against
 * another implementation in this repo, because there is no other one -- these
 * are the only thing standing between a cache key and a silent collision.
 */

#include <gtest/gtest.h>

#include "core/Sha256.hpp"

#include <string>

using quantiloom::core::Sha256;
using quantiloom::core::Sha256Hex;

namespace {

std::string HexOf(const std::string& message) {
    Sha256 hasher;
    hasher.Update(message.data(), message.size());
    return hasher.FinalizeHex();
}

}  // namespace

// ---------------------------------------------------------------------------
// FIPS 180-4 / NIST CAVS known-answer vectors
// ---------------------------------------------------------------------------

TEST(Sha256, EmptyMessage) {
    EXPECT_EQ(HexOf(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(Sha256, Abc) {
    EXPECT_EQ(HexOf("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(Sha256, TwoBlockMessage) {
    // 448 bits: the case that needs a second block purely for the padding.
    EXPECT_EQ(HexOf("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(Sha256, MillionA) {
    Sha256 hasher;
    const std::string chunk(1000, 'a');
    for (int i = 0; i < 1000; ++i) {
        hasher.Update(chunk.data(), chunk.size());
    }
    EXPECT_EQ(hasher.FinalizeHex(),
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256, OneShotHelperMatchesStreaming) {
    const std::string message = "the quick brown fox";
    EXPECT_EQ(Sha256Hex(message.data(), message.size()), HexOf(message));
}

// ---------------------------------------------------------------------------
// Streaming: the block boundary must not be observable
// ---------------------------------------------------------------------------

TEST(Sha256, ByteAtATimeMatchesOneShot) {
    // Deliberately spans several block boundaries and ends mid-block.
    std::string message;
    for (int i = 0; i < 200; ++i) {
        message.push_back(static_cast<char>('a' + (i % 26)));
    }

    Sha256 streamed;
    for (const char c : message) {
        streamed.Update(&c, 1);
    }
    EXPECT_EQ(streamed.FinalizeHex(), HexOf(message));
}

TEST(Sha256, ExactBlockMultipleMatchesOneShot) {
    const std::string message(128, 'z');  // exactly two blocks

    Sha256 split;
    split.Update(message.data(), 64);
    split.Update(message.data() + 64, 64);
    EXPECT_EQ(split.FinalizeHex(), HexOf(message));
}

// ---------------------------------------------------------------------------
// Framing: what keeps a concatenated key unambiguous
// ---------------------------------------------------------------------------

TEST(Sha256, SizedUpdatesAreUnambiguous) {
    // The whole reason UpdateSized exists: raw Update cannot tell these apart.
    Sha256 a;
    a.UpdateString("ab");
    a.UpdateString("c");

    Sha256 b;
    b.UpdateString("a");
    b.UpdateString("bc");

    EXPECT_NE(a.FinalizeHex(), b.FinalizeHex());

    Sha256 rawA;
    rawA.Update("ab", 2);
    rawA.Update("c", 1);
    Sha256 rawB;
    rawB.Update("a", 1);
    rawB.Update("bc", 2);
    EXPECT_EQ(rawA.FinalizeHex(), rawB.FinalizeHex()) << "raw Update is unframed by design";
}

TEST(Sha256, EmptyAndAbsentSizedFieldsDiffer) {
    Sha256 withEmpty;
    withEmpty.UpdateString("");
    withEmpty.UpdateU32(7);

    Sha256 without;
    without.UpdateU32(7);

    EXPECT_NE(withEmpty.FinalizeHex(), without.FinalizeHex());
}

TEST(Sha256, TypedScalarsPinWidthAndOrder) {
    // A u32 1 and a u64 1 must not hash alike, or widening a key field would
    // silently keep every existing entry addressable under a new meaning.
    Sha256 asU32;
    asU32.UpdateU32(1u);
    Sha256 asU64;
    asU64.UpdateU64(1u);
    EXPECT_NE(asU32.FinalizeHex(), asU64.FinalizeHex());
}

TEST(Sha256, FloatBitPatternsDistinguishSignedZero) {
    Sha256 positive;
    positive.UpdateF32(0.0f);
    Sha256 negative;
    negative.UpdateF32(-0.0f);
    // Documented behaviour: costs at most a spurious cache miss.
    EXPECT_NE(positive.FinalizeHex(), negative.FinalizeHex());
}

TEST(Sha256, ScalarsAreHostOrderIndependent) {
    // Little-endian by construction, so the digest is a fixed function of the
    // value rather than of the machine.
    Sha256 typed;
    typed.UpdateU32(0x01020304u);

    const quantiloom::u8 expected[4] = {0x04, 0x03, 0x02, 0x01};
    Sha256 manual;
    manual.Update(expected, sizeof(expected));

    EXPECT_EQ(typed.FinalizeHex(), manual.FinalizeHex());
}
