/**
 * @file GpuThermalStepper.hpp
 * @brief GPU Crank-Nicolson thermal stepper (f32)
 *
 * Mirrors the CPU stepper's math line by line, in f32. Each element is one
 * thread; the inter-element coupling (radiative T^4 exchange) reads from a
 * ping-pong surface buffer, and the intra-element Thomas solve is thread-local.
 *
 * StepMany dispatches the whole batch in one ExecuteImmediate, which is 1-5 ms
 * at 15k elements — the point of putting the stepper on the GPU is that the
 * slider scrubs without per-step host round-trips.
 */

#pragma once

#include "renderer/VulkanContext.hpp"
#include "thermal/ThermalStepper.hpp"

#include <memory>

namespace quantiloom::rendercore {

class GpuThermalStepper final : public thermal::IThermalStepper {
public:
    static constexpr u32 kMaxNodes = 32;

    explicit GpuThermalStepper(VulkanContext& context);
    ~GpuThermalStepper() override;

    GpuThermalStepper(const GpuThermalStepper&) = delete;
    GpuThermalStepper& operator=(const GpuThermalStepper&) = delete;

    [[nodiscard]] bool IsValid() const;

    void Step(thermal::ThermalState& state, const Vector<thermal::ThermalElement>& elements,
              const Vector<thermal::ThermalMaterial>& materials,
              const thermal::ExchangeGeometry& exchange, const thermal::ThermalForcing& forcing,
              f64 dt_s, std::span<const f32> sunVisibility) override;

    void StepMany(thermal::ThermalState& state,
                  const Vector<thermal::ThermalElement>& elements,
                  const Vector<thermal::ThermalMaterial>& materials,
                  const thermal::ExchangeGeometry& exchange,
                  const thermal::SunVisibilityTable& sunTable,
                  std::span<const thermal::ThermalBatchStep> steps) override;

    [[nodiscard]] const char* Name() const override { return "GPU Crank-Nicolson f32"; }

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace quantiloom::rendercore
