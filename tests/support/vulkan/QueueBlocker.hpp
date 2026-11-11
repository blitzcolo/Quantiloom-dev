/**
 * @file QueueBlocker.hpp
 * @brief Hold one submission on the device's queue until the host lets it run
 *
 * A test of a setter that must (or must not) wait for frames already submitted needs a
 * submitted frame that is deterministically still pending: the gated command buffer's
 * first command waits for a host signal, so everything recorded after it stays queued
 * until Release(). The interface is neutral; the implementation is Vulkan's (an event the
 * host sets, waited on from the queue). A Metal one would wait on an MTLSharedEvent. A
 * test on a backend without one skips (IsSupported).
 */

#pragma once

#include "core/Types.hpp"
#include "gpu/GpuTypes.hpp"
#include "gpu/Sync.hpp"
#include "renderer/GpuInterop.hpp"

#include <memory>

namespace quantiloom::gpu {
class Device;
}

namespace quantiloom::testing {

class QueueBlocker {
public:
    [[nodiscard]] static bool IsSupported(gpu::Backend backend);

    /// The gate's event, fence and command pool, in that order. Err with the code of
    /// the call that failed.
    [[nodiscard]] static Result<std::unique_ptr<QueueBlocker>, String> Create(gpu::Device& device);

    /// Waits for the queue when a submission is outstanding, then destroys the event,
    /// the fence and the pool. Release() must have been called, or the wait never ends.
    virtual ~QueueBlocker() = default;

    /// Begins the gated command buffer, whose first command waits for Release(). Record
    /// the held work into the returned recording, then Submit().
    [[nodiscard]] virtual Result<GpuCommandRecording, String> Block() = 0;

    /// Ends the command buffer and submits it, signalling the fence.
    [[nodiscard]] virtual Result<void, String> Submit() = 0;

    /// Opens the gate from the host. Callable from another thread.
    virtual void Release() = 0;

    /// Signaled once the gated submission has completed.
    [[nodiscard]] virtual Result<gpu::FenceStatus, String> Status() = 0;

    [[nodiscard]] virtual Result<void, String> Wait(u64 timeoutNs) = 0;
};

}  // namespace quantiloom::testing
