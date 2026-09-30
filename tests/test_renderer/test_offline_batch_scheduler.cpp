#include <gtest/gtest.h>

#include "renderer/OfflineBatchScheduler.hpp"

#include <array>
#include <limits>

namespace quantiloom::rendercore {
namespace {

TEST(OfflineBatchSchedulerTest, StartsAtTwoAndNeverExceedsRemainingSamples) {
    OfflineBatchScheduler scheduler;

    EXPECT_EQ(scheduler.NextBatchSize(0), 0u);
    EXPECT_EQ(scheduler.NextBatchSize(1), 1u);
    EXPECT_EQ(scheduler.NextBatchSize(2), 2u);
    EXPECT_EQ(scheduler.NextBatchSize(99), 2u);
}

TEST(OfflineBatchSchedulerTest, KeepsFallbackWhenGpuTimingIsUnavailable) {
    OfflineBatchScheduler scheduler;
    const std::array invalidTimes{
        0.0f, -1.0f, std::numeric_limits<f32>::quiet_NaN(),
        std::numeric_limits<f32>::infinity()};

    scheduler.ObserveCompletedBatch(invalidTimes);

    EXPECT_EQ(scheduler.NextBatchSize(99), 2u);
}

TEST(OfflineBatchSchedulerTest, SlowSamplesImmediatelyReduceTheNextSubmit) {
    OfflineBatchScheduler scheduler;
    const std::array firstBatch{500.0f, 480.0f};

    scheduler.ObserveCompletedBatch(firstBatch);

    EXPECT_EQ(scheduler.NextBatchSize(99), 1u);
}

TEST(OfflineBatchSchedulerTest, OneFastSampleDoesNotEraseASlowHistory) {
    OfflineBatchScheduler scheduler;
    const std::array slowBatch{100.0f, 100.0f};
    const std::array fastBatch{1.0f};

    scheduler.ObserveCompletedBatch(slowBatch);
    scheduler.ObserveCompletedBatch(fastBatch);

    EXPECT_EQ(scheduler.NextBatchSize(99), 1u);
}

TEST(OfflineBatchSchedulerTest, FastSamplesGrowByAtMostTwoTimesPerBatch) {
    OfflineBatchScheduler scheduler;
    const std::array firstBatch{1.0f, 1.0f};
    const std::array secondBatch{1.0f, 1.0f, 1.0f, 1.0f};
    const std::array thirdBatch{1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f};

    scheduler.ObserveCompletedBatch(firstBatch);
    EXPECT_EQ(scheduler.NextBatchSize(99), 4u);
    EXPECT_EQ(scheduler.NextBatchSize(3), 3u);
    scheduler.ObserveCompletedBatch(secondBatch);
    EXPECT_EQ(scheduler.NextBatchSize(99), 8u);
    scheduler.ObserveCompletedBatch(thirdBatch);
    EXPECT_EQ(scheduler.NextBatchSize(99), 16u);
}

TEST(OfflineBatchSchedulerTest, BatchSizeHasAQueryPoolSafeUpperBound) {
    OfflineBatchScheduler scheduler;
    std::array<f32, OfflineBatchScheduler::kMaxSamples> fastBatch{};
    fastBatch.fill(0.01f);

    scheduler.ObserveCompletedBatch(fastBatch);

    EXPECT_EQ(scheduler.NextBatchSize(1000),
              OfflineBatchScheduler::kMaxSamples);
}

TEST(OfflineBatchSchedulerTest, ASlowOutlierWinsOverFastSamplesInItsBatch) {
    OfflineBatchScheduler scheduler;
    const std::array mixedBatch{1.0f, 1.0f, 60.0f, 1.0f};

    scheduler.ObserveCompletedBatch(mixedBatch);

    EXPECT_EQ(scheduler.NextBatchSize(99), 1u);
}

}  // namespace
}  // namespace quantiloom::rendercore
