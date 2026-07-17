#pragma once

#include "atmos/AtmosModelPack.hpp"
#include "atmos/AtmosNNGpuTypes.hpp"
#include "atmos/AtmosphereNNConfig.hpp"
#include "core/Platform.hpp"
#include "core/Types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace quantiloom {

// ============================================================================
// AtmosphereBaker - CPU NN inference -> GPU spectral LUT
// ============================================================================
// One bake covers (render band, geometry, weather config, h1, sun geometry).
// The lambda target grid is exactly the renderer's spectral loop sample
// points, so the shader indexes by loop counter with no spectral
// interpolation on the GPU.
//
// Geometry selection:
//   h1 < 0.5 km          -> ground nets, axis a = ln(range_km) over
//                           [0.05, 20] km (log-uniform), zenith fixed at 90 deg
//   0.5 <= h1 <= 12 km   -> slant nets, axis a = cos(view_zenith) uniform
//                           over [cos 180, cos 110]; range from SlantRangeKm
//   h1 > 12 km           -> clamped to 12 km (warning)
//
// Radiance is converted from the network's native W cm^-2 sr^-1 / cm^-1 to
// W m^-2 sr^-1 nm^-1 per channel (factor 1e11 / lambda_nm^2), then
// box-averaged into each render lambda bin. Transmittance is averaged in
// tau space (band-effective transmittance). Rows whose min optical depth
// exceeds opaque_delta bake to tau = 0 (deployment gate).

struct AtmosBakeResult {
    AtmosNNHeaderGPU header;
    std::vector<float> data;  // Flat blob for binding 20
};

// The lambda grid a render band samples on the GPU. MUST mirror the loop
// constants in closesthit.rchit / miss.rmiss exactly -- the shader indexes
// the baked LUT by loop counter.
struct AtmosLambdaGrid {
    std::string band;              // vis/nir/swir/mwir/lwir; empty = no NN support
    std::vector<double> lambdasNm;
    double windowHalfWidthNm = 0.0;  // > 0 for discrete lambdas (RGB / SINGLE)
    std::string error;             // Set when a SINGLE lambda is out of coverage
};

QL_API AtmosLambdaGrid RenderBandLambdaGrid(SpectralMode mode, double wavelengthNm);

// Upper bound on the binding-20 blob so GPU buffers can be allocated once:
// 32 lambda x 256 a x (1 + 64 az) + ldown.
constexpr size_t kAtmosMaxDataFloats = 32u * 256u * 65u + 32u;

class QL_API AtmosphereBaker {
public:
    explicit AtmosphereBaker(AtmosModelPack& pack) : pack_(pack) {}

    // band: vis / nir / swir / mwir / lwir. lambdasNm: render sample centers
    // in nm, ascending for contiguous band loops. windowHalfWidthNm = 0 uses
    // contiguous midpoint bins (fused band integration); > 0 averages a
    // +/- window around each sample instead (RGB / SINGLE discrete lambdas,
    // any order). Throws std::runtime_error on missing networks.
    AtmosBakeResult Bake(const AtmosphereNNConfig& cfg, const std::string& band,
                         const std::vector<double>& lambdasNm,
                         double windowHalfWidthNm = 0.0);

    // Spherical no-refraction slant range, replicating atmoemu
    // manifest.py::slant_range_km (R = 6371 km, grazing clamp +0.05 deg).
    static double SlantRangeKm(double h1Km, double viewZenithDeg);

    // Hash over everything that changes baked data contents.
    static uint64_t BakeKey(const AtmosphereNNConfig& cfg, const std::string& band,
                            const std::vector<double>& lambdasNm);

private:
    AtmosModelPack& pack_;
};

}  // namespace quantiloom
