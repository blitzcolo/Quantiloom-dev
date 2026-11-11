// The Metal implementation of QueueBlocker: the gated command buffer waits on a shared
// event the host signals, and signals a second one when it completes.

#include "support/vulkan/QueueBlocker.hpp"

#include "core/ResultFail.hpp"
#include "gpu/metal/MetalBackend.hpp"
#include "renderer/interop/MetalInterop.hpp"

namespace quantiloom::testing {

namespace {

class MetalQueueBlocker final : public QueueBlocker {
public:
    MetalQueueBlocker(id<MTLDevice> device, id<MTLCommandQueue> queue)
        : queue_(queue), gate_([device newSharedEvent]), done_([device newSharedEvent]) {}

    ~MetalQueueBlocker() override {
        if (commandBuffer_ != nil && submitted_) [commandBuffer_ waitUntilCompleted];
    }

    Result<GpuCommandRecording, String> Block() override {
        commandBuffer_ = [queue_ commandBuffer];
        if (commandBuffer_ == nil) return Fail<GpuCommandRecording>("cannot create a Metal command buffer");
        [commandBuffer_ encodeWaitForEvent:gate_ value:1];
        return mtl::Commands((__bridge void*)commandBuffer_);
    }

    Result<void, String> Submit() override {
        [commandBuffer_ encodeSignalEvent:done_ value:1];
        [commandBuffer_ commit];
        submitted_ = true;
        return Result<void, String>::Ok();
    }

    void Release() override { gate_.signaledValue = 1; }

    Result<gpu::FenceStatus, String> Status() override {
        return [done_ signaledValue] >= 1 ? gpu::FenceStatus::Signaled : gpu::FenceStatus::NotReady;
    }

    Result<void, String> Wait(u64 timeoutNs) override {
        const u64 timeoutMs = timeoutNs == ~u64{0} ? ~u64{0} : (timeoutNs + 999'999) / 1'000'000;
        if (![done_ waitUntilSignaledValue:1 timeoutMS:timeoutMs]) return Fail<void>("timeout");
        return Result<void, String>::Ok();
    }

private:
    id<MTLCommandQueue> queue_;
    id<MTLSharedEvent> gate_;
    id<MTLSharedEvent> done_;
    id<MTLCommandBuffer> commandBuffer_ = nil;
    bool submitted_ = false;
};

}  // namespace

bool QueueBlocker::IsSupported(gpu::Backend backend) {
    return backend == gpu::Backend::Metal;
}

Result<std::unique_ptr<QueueBlocker>, String> QueueBlocker::Create(gpu::Device& device) {
    if (!IsSupported(device.Info().backend)) {
        return Fail<std::unique_ptr<QueueBlocker>>("no queue blocker for this backend");
    }
    auto& metal = static_cast<gpu::metal::MetalDevice&>(device);
    return std::unique_ptr<QueueBlocker>(std::make_unique<MetalQueueBlocker>(metal.NativeDevice(), metal.NativeQueue()));
}

}  // namespace quantiloom::testing
