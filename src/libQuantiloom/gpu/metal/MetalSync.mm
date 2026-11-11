#include "gpu/metal/MetalBackend.hpp"

#include "core/ResultFail.hpp"

namespace quantiloom::gpu::metal {

Result<FenceStatus, String> MetalFence::Status() {
    return [event_ signaledValue] >= target_ ? FenceStatus::Signaled : FenceStatus::NotReady;
}

Result<void, String> MetalFence::Wait(u64 timeoutNs) {
    const u64 timeoutMs = timeoutNs == ~u64{0} ? ~u64{0} : (timeoutNs + 999'999) / 1'000'000;
    if (![event_ waitUntilSignaledValue:target_ timeoutMS:timeoutMs]) {
        return Fail<void>("timeout");
    }
    return Result<void, String>::Ok();
}

Result<void, String> MetalFence::Reset() {
    // An event only counts up: an unsignaled fence is one whose target the event has not
    // reached, and the next submission that signals it signals the new target.
    target_ = std::max<u64>([event_ signaledValue], target_) + 1;
    return Result<void, String>::Ok();
}

Result<ResolveStatus, String> MetalTimestampPool::Resolve(u32 first, u32 count, TimestampResolve mode,
                                                          std::span<u64> out) {
    const size_t needed = mode == TimestampResolve::Wait ? count : 2ull * count;
    if (out.size() < needed) return Fail<ResolveStatus>("timestamp output too small");
    NSData* data = [samples_ resolveCounterRange:NSMakeRange(first, count)];
    const auto* results = data != nil ? static_cast<const MTLCounterResultTimestamp*>([data bytes]) : nullptr;
    const size_t available = data != nil ? [data length] / sizeof(MTLCounterResultTimestamp) : 0;
    bool ready = true;
    for (u32 i = 0; i < count; ++i) {
        const bool valid = results != nullptr && i < available && results[i].timestamp != MTLCounterErrorValue &&
                           results[i].timestamp != 0;
        const u64 value = valid ? results[i].timestamp : 0;
        ready = ready && valid;
        if (mode == TimestampResolve::Wait) {
            out[i] = value;
        } else {
            out[2 * i] = value;
            out[2 * i + 1] = valid ? 1 : 0;
        }
    }
    return ready ? ResolveStatus::Ready : ResolveStatus::NotReady;
}

MetalCommandBuffer::~MetalCommandBuffer() = default;

Result<void, String> MetalCommandBuffer::Reset() {
    list_.reset();
    return Result<void, String>::Ok();
}

Result<CommandList*, String> MetalCommandBuffer::Begin() {
    // A Metal command buffer is single-use: each Begin records into a fresh one.
    id<MTLCommandBuffer> native = [device_.NativeQueue() commandBuffer];
    if (native == nil) return Fail<CommandList*>("cannot create a Metal command buffer");
    list_ = std::make_unique<MetalCommandList>(device_, native);
    return static_cast<CommandList*>(list_.get());
}

Result<void, String> MetalCommandBuffer::End() {
    if (!list_) return Fail<void>("End without Begin");
    list_->Finish();
    return Result<void, String>::Ok();
}

id<MTLCommandBuffer> MetalCommandBuffer::Native() const {
    return list_ ? list_->Native() : nil;
}

}  // namespace quantiloom::gpu::metal
