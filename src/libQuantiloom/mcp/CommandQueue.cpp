#include "mcp/CommandQueue.hpp"

#include "core/Log.hpp"

#include <chrono>
#include <utility>

namespace quantiloom::mcp {

void CommandQueue::SetWakeCallback(std::function<void()> wake) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_wake = std::move(wake);
}

void CommandQueue::SetHostDispatch(std::function<void(std::function<void()>)> dispatch) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_hostDispatch = std::move(dispatch);
}

void CommandQueue::Settle(const std::shared_ptr<Slot>& slot, const bool ran) {
    {
        std::lock_guard<std::mutex> lock(slot->mutex);
        slot->settled = true;
        slot->ran = ran;
    }
    slot->cv.notify_all();
}

bool CommandQueue::Submit(std::function<void()> work, const bool hostDispatch, const u32 timeoutMs) {
    auto slot = std::make_shared<Slot>();

    // The wrapper owns a shared reference to the slot, so running late -- after
    // this call has already timed out and returned -- writes to a block that is
    // still alive rather than to a caller's stack that is not.
    Entry entry;
    entry.slot = slot;
    entry.hostDispatch = hostDispatch;
    entry.work = [work = std::move(work), slot]() mutable {
        try {
            work();
        } catch (const std::exception& e) {
            QL_LOG_ERROR("MCP: tool handler threw: {}", e.what());
        } catch (...) {
            QL_LOG_ERROR("MCP: tool handler threw a non-standard exception");
        }
        Settle(slot, true);
    };

    std::function<void()> wake;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_shutdown) {
            return false;
        }
        m_queue.push_back(std::move(entry));
        wake = m_wake;
    }

    // Outside the lock: a host that pumps synchronously from its wake callback
    // would otherwise deadlock against Drain().
    if (wake) {
        wake();
    }

    std::unique_lock<std::mutex> lock(slot->mutex);
    slot->cv.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                      [&slot] { return slot->settled; });
    return slot->ran;
}

void CommandQueue::Drain() {
    // Take the whole queue at once. Work enqueued while these run belongs to the
    // next Drain(), so a handler that submits more cannot hold the frame here.
    std::deque<Entry> batch;
    std::function<void(std::function<void()>)> hostDispatch;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_shutdown) {
            return;
        }
        batch.swap(m_queue);
        hostDispatch = m_hostDispatch;
    }

    for (auto& entry : batch) {
        if (entry.hostDispatch && hostDispatch) {
            hostDispatch(std::move(entry.work));
        } else {
            entry.work();
        }
    }
}

u32 CommandQueue::Pending() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return static_cast<u32>(m_queue.size());
}

void CommandQueue::Shutdown() {
    std::deque<Entry> pending;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_shutdown) {
            return;
        }
        m_shutdown = true;
        pending.swap(m_queue);
    }

    // Release the waiters without running their work: the host is going away,
    // and a handler that touches it now is the crash this avoids. They are told
    // it did not run, so the call reports a failure rather than an empty success.
    for (const auto& entry : pending) {
        Settle(entry.slot, false);
    }
}

}  // namespace quantiloom::mcp
