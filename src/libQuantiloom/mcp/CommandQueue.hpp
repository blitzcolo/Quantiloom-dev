/**
 * @file CommandQueue.hpp
 * @brief Moves a tool call from the transport thread onto the host's thread
 *
 * The renderer has no synchronisation of its own. ExternalRenderContext's
 * setters write their fields and, several of them, reallocate GPU buffers and
 * rebind descriptors -- on the calling thread, with no fence and no lock. That
 * has been safe so far only because every caller has been the one thread that
 * also draws. An MCP server introduces a second thread, and this queue is what
 * keeps that from being a data race: the transport thread submits work and
 * waits, the host thread runs it at a point where no frame is in flight.
 *
 * The waiting is deliberate. MCP's request/response shape gives the agent one
 * answer per call, so the transport thread has to hold the connection until the
 * work is done anyway; making the wait explicit means a stalled host produces a
 * timeout with an explanation rather than a socket that never replies.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>

namespace quantiloom::mcp {

/**
 * @class CommandQueue
 * @brief Single-producer-friendly work queue with a bounded wait
 */
class CommandQueue {
public:
    CommandQueue() = default;

    CommandQueue(const CommandQueue&) = delete;
    CommandQueue& operator=(const CommandQueue&) = delete;

    /// Called from the transport thread when a command is enqueued, so a host
    /// that is idle knows to pump. Set once, before any Submit().
    void SetWakeCallback(std::function<void()> wake);

    /// Runs HostDispatch work instead of Drain(). Empty means "run it inline
    /// like everything else".
    void SetHostDispatch(std::function<void(std::function<void()>)> dispatch);

    /**
     * @brief Enqueue work and wait for the host to run it
     *
     * @param work        Runs on the host thread. Exceptions are caught by the
     *                    caller of Drain(), not here.
     * @param hostDispatch Route to the host's own scheduler rather than running
     *                    inline in Drain().
     * @param timeoutMs   How long to wait before giving up.
     * @return true if the work ran, false on timeout or after Shutdown().
     *
     * On timeout the work is *not* cancelled -- it may still run later. Anything
     * it writes must therefore outlive this call, which is why the result is
     * held in a shared block rather than on the caller's stack.
     */
    bool Submit(std::function<void()> work, bool hostDispatch, u32 timeoutMs);

    /**
     * @brief Run everything queued, in order
     *
     * Called from the host thread. Commands added while draining are left for
     * the next call, so a handler that somehow enqueues more work cannot spin
     * the frame callback forever.
     */
    void Drain();

    [[nodiscard]] u32 Pending() const;

    /// Wake every waiter with a failure and refuse further work.
    void Shutdown();

private:
    struct Slot {
        std::mutex mutex;
        std::condition_variable cv;
        /// Stop waiting. Set both when the work finished and when Shutdown()
        /// released the waiter without running it.
        bool settled = false;
        /// The handler actually executed. The two are not the same thing, and
        /// conflating them made a call interrupted by Shutdown() report success
        /// with an empty result.
        bool ran = false;
    };

    struct Entry {
        std::function<void()> work;
        std::shared_ptr<Slot> slot;
        bool hostDispatch = false;
    };

    static void Settle(const std::shared_ptr<Slot>& slot, bool ran);

    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    std::deque<Entry> m_queue;
    std::function<void()> m_wake;
    std::function<void(std::function<void()>)> m_hostDispatch;
    bool m_shutdown = false;
};

}  // namespace quantiloom::mcp
