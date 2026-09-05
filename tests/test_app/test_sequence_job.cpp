// ============================================================================
// Quantiloom - Unit Tests for src/app/SequenceJob.hpp
// ============================================================================
// Only the naming, which is the part with no device in it. What a frame's file
// is called decides whether a hundred renders land in a hundred files or all
// in one, and the padding is what keeps them sorting in the order they were
// rendered.
// ============================================================================

#include <gtest/gtest.h>

#include "SequenceJob.hpp"

using namespace quantiloom;
using namespace quantiloom::app;

TEST(SequenceJobTest, TheTickFillsInAndPads) {
    EXPECT_EQ(FormatFrameName("frame_{tick}.exr", 7, 0.35), "frame_7.exr");
    EXPECT_EQ(FormatFrameName("frame_{tick:05}.exr", 7, 0.35), "frame_00007.exr");
    EXPECT_EQ(FormatFrameName("frame_{tick:05}.exr", 123456, 0.0), "frame_123456.exr");
}

TEST(SequenceJobTest, TheTimeFillsInAndRounds) {
    EXPECT_EQ(FormatFrameName("t{time_s:.3f}.exr", 7, 0.35), "t0.350.exr");
    EXPECT_EQ(FormatFrameName("t{time_s:.0f}.exr", 7, 2.6), "t3.exr");
    EXPECT_EQ(FormatFrameName("t{time_s}.exr", 7, 0.35), "t0.35.exr");
}

TEST(SequenceJobTest, BothCanAppearAndSoCanNeither) {
    // 2.06 rather than 2.05: the latter is not exactly representable and lands
    // just below the halfway point, so printf rounds it down -- which is
    // correct and a poor thing to pin a naming test on.
    EXPECT_EQ(FormatFrameName("out/{tick:03}_{time_s:.1f}.exr", 42, 2.06),
              "out/042_2.1.exr");
    EXPECT_EQ(FormatFrameName("fixed.exr", 42, 2.0), "fixed.exr");
}

TEST(SequenceJobTest, AnUnknownPlaceholderIsLeftAlone) {
    // More likely a literal brace than a typo, and eating it would produce a
    // file under a name nobody asked for.
    EXPECT_EQ(FormatFrameName("{scene}_{tick}.exr", 3, 0.0), "{scene}_3.exr");
    EXPECT_EQ(FormatFrameName("unclosed_{tick", 3, 0.0), "unclosed_{tick");
    EXPECT_EQ(FormatFrameName("{}_{tick}.exr", 3, 0.0), "{}_3.exr");
}

TEST(SequenceJobTest, ANegativeTickKeepsItsSignOutsideThePadding) {
    EXPECT_EQ(FormatFrameName("frame_{tick:04}.exr", -7, 0.0), "frame_-0007.exr");
}
