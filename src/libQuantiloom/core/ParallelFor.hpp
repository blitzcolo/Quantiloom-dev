/**
 * @file ParallelFor.hpp
 * @brief Fixed-chunk parallel loops over an index range
 *
 * A stand-in for the C++17 parallel execution policies, which Apple libc++
 * only offers behind -fexperimental-library: split [0, count) into
 * contiguous chunks, run each chunk on its own std::thread, join. No
 * dependency beyond the C++ standard library.
 *
 * Determinism: the chunk count depends only on count and the hardware
 * concurrency, never on scheduling, and every chunk owns a disjoint slice
 * of the range -- ParallelReduce in particular combines per-chunk partials
 * in ascending chunk order, so its result is reproducible run to run.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

namespace quantiloom::par {
namespace detail {

// Chunk count for [0, count): hardware concurrency, capped so no chunk is
// empty. Fixed for a given count.
inline std::size_t ChunkCount(std::size_t count) {
    if (count == 0) {
        return 0;
    }
    const unsigned hw = std::thread::hardware_concurrency();
    const std::size_t threads = hw > 0 ? static_cast<std::size_t>(hw) : std::size_t{1};
    return std::min(count, threads);
}

}  // namespace detail

/// Call body(begin, end) over a partition of [0, count) into contiguous
/// chunks, each on its own std::thread; the last chunk runs on the calling
/// thread. Bodies must not throw: an exception inside a worker thread
/// terminates, matching std::terminate behaviour of the par policies.
template <typename Body>
void ParallelFor(std::size_t count, Body&& body) {
    const std::size_t chunks = detail::ChunkCount(count);
    if (chunks == 0) {
        return;
    }
    const std::size_t chunkSize = (count + chunks - 1) / chunks;
    if (chunks == 1) {
        body(0, count);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(chunks - 1);
    for (std::size_t c = 0; c + 1 < chunks; ++c) {
        const std::size_t begin = c * chunkSize;
        const std::size_t end = std::min(count, begin + chunkSize);
        threads.emplace_back([&body, begin, end] { body(begin, end); });
    }
    body((chunks - 1) * chunkSize, count);
    for (auto& thread : threads) {
        thread.join();
    }
}

/// Map-reduce over [0, count): chunk(begin, end) returns that chunk's
/// partial and the partials are folded with combine in ascending chunk
/// order. The fold order is fixed, so the result never depends on thread
/// scheduling.
template <typename T, typename Chunk, typename Combine>
T ParallelReduce(std::size_t count, T identity, Chunk&& chunk, Combine&& combine) {
    const std::size_t chunks = detail::ChunkCount(count);
    if (chunks == 0) {
        return identity;
    }
    const std::size_t chunkSize = (count + chunks - 1) / chunks;
    std::vector<T> partials(chunks, identity);
    const auto run = [&](std::size_t c) {
        const std::size_t begin = c * chunkSize;
        const std::size_t end = std::min(count, begin + chunkSize);
        partials[c] = chunk(begin, end);
    };
    if (chunks == 1) {
        run(0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(chunks - 1);
        for (std::size_t c = 0; c + 1 < chunks; ++c) {
            threads.emplace_back([&run, c] { run(c); });
        }
        run(chunks - 1);
        for (auto& thread : threads) {
            thread.join();
        }
    }
    T result = identity;
    for (std::size_t c = 0; c < chunks; ++c) {
        result = combine(std::move(result), std::move(partials[c]));
    }
    return result;
}

}  // namespace quantiloom::par
