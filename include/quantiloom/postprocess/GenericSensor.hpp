/**
 * @file GenericSensor.hpp
 * @brief Compatibility adapter from SensorParams and linear RGB input to the
 * versioned CPU CameraPipeline.
 *
 * RGB input carries no unique spectrum. This path uses a documented generic
 * one-nanometre flat surrogate at SensorParams::wavelength_nm, marks the
 * result as a fast-RGB approximation, and shares the physical detector,
 * Poisson/readout and ADC code with spectral camera captures.
 *
 * rawDN contains integer ADC codes. enhancedPreview is encoded sRGB derived
 * from those codes, not a radiance measurement.
 *
 * @see SensorModel for abstract base class
 * @see SensorParams for configuration parameters
 * @see SensorOutput for output data structure
 *
 * @author blitzcolo
 */

#pragma once

#include "SensorModel.hpp"
#include "core/Platform.hpp"
#include <memory>

namespace quantiloom {

/// Legacy API preserving SensorParams while using the one camera pipeline.
///
/// The chain's state -- RNG, FPN maps, seeding -- lives behind a pimpl. It used
/// to sit in this header, which put sizeof(GenericSensor) into the ABI: a
/// frontend calling make_unique<GenericSensor>() baked the layout in, so adding
/// a member silently broke a Quantiloom-Qt built against the previous SDK. That
/// has happened (see SdkGuard in Quantiloom-Qt). Now only Apply() is contract.
class QL_API GenericSensor final : public SensorModel {
public:
    GenericSensor();
    ~GenericSensor() override;

    /// Apply full sensor chain to HDR input
    auto Apply(const Image& hdr, const SensorParams& params)
        -> Result<SensorOutput, String> override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace quantiloom
