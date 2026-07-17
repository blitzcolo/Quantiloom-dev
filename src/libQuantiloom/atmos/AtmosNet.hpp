#pragma once

#include "atmos/ResMLP.hpp"
#include "core/Platform.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace quantiloom {

class SafetensorsFile;

// ============================================================================
// AtmosNet - one MODTRAN surrogate network pack (<band>_<geom>_<net>)
// ============================================================================
// Wraps a safetensors pack and reproduces the full Python inference contract
// (atmoemu/transforms.py + scripts/infer.py):
//   raw parameter rows -> InputSpec feature assembly (float64 math, fp32 X)
//   -> ResMLP fp32 forward
//   -> per-target inverse transform in float64:
//        z*std + mean -> exp(.) - log_eps on log channels
//        delta rows clamped to [0, delta_clamp]
//        radiance rows clamped >= 0 and capped at log_eps*1e4*2
//
// Raw parameter rows use the manifest feature layout (FeatureNames()):
// sampled columns in declaration order followed by the geometry block
// [h1_km, h2_km, cos_view_zenith, range_km]. Out-of-domain continuous
// features are clamped to the training range (one-time warning); discrete
// features snap to the nearest declared value.

struct AtmosBandGrid {
    std::string name;    // lwir / mwir / nir / swir / vis
    double v1_cm = 0.0;  // Grid start, wavenumber cm^-1
    double v2_cm = 0.0;  // Grid end
    double dv_cm = 0.0;  // Grid step
    int K = 0;           // Number of spectral channels
    bool thermal = false;

    // Channel i center in cm^-1 / nm (grid runs v1 -> v2 ascending in cm^-1).
    double WavenumberAt(int i) const { return v1_cm + dv_cm * i; }
    double WavelengthNmAt(int i) const { return 1e7 / WavenumberAt(i); }
    double MinWavelengthNm() const { return 1e7 / v2_cm; }
    double MaxWavelengthNm() const { return 1e7 / v1_cm; }
};

struct AtmosTargetRow {
    std::string block;   // tau / lpath / ldown
    std::string column;  // LOG_TOTAL / TOTAL_RAD / PTH_THRML / SOL_SCAT
    std::string kind;    // delta / radiance
};

class QL_API AtmosNet {
public:
    explicit AtmosNet(const std::filesystem::path& file);
    ~AtmosNet();
    AtmosNet(AtmosNet&&) noexcept;
    AtmosNet& operator=(AtmosNet&&) noexcept;

    const std::string& NetKind() const { return net_; }        // tau/lpath/ldown
    const std::string& PathType() const { return pathType_; }  // horizontal/...
    const AtmosBandGrid& Band() const { return band_; }
    const std::vector<AtmosTargetRow>& Targets() const { return targets_; }
    int T() const { return static_cast<int>(targets_.size()); }
    int K() const { return band_.K; }
    double OpaqueDelta() const { return opaqueDelta_; }
    double DeltaClamp() const { return deltaClamp_; }

    const std::vector<std::string>& FeatureNames() const { return featureNames_; }
    int NumFeatures() const { return static_cast<int>(featureNames_.size()); }
    // Index into a raw parameter row; -1 when the dataset lacks the feature.
    int FeatureIndex(const std::string& name) const;
    // Sampled-domain range from the training manifest; false if not sampled.
    bool FeatureRange(const std::string& name, double& lo, double& hi) const;
    // Declared values for a discrete feature; empty for continuous ones.
    std::vector<double> DiscreteValues(const std::string& name) const;
    // True if the network consumed any solar geometry feature.
    bool HasSunFeatures() const { return hasSunFeatures_; }

    // rows: [n, NumFeatures()] row-major float64 -> out: [n, T, K] float64
    // physical values (radiance in W cm^-2 sr^-1 / cm^-1).
    void Infer(const double* rows, size_t n, double* out) const;

    // Deployment gate: min(delta) > opaque_delta means the line of sight is
    // outside the tau training domain -> treat as fully opaque (tau = 0).
    bool IsOpaque(const double* deltaRow) const;

private:
    struct InputEntry;
    void AssembleFeatures(const double* rows, size_t n, float* X) const;

    std::string net_;
    std::string pathType_;
    AtmosBandGrid band_;
    std::vector<AtmosTargetRow> targets_;
    double opaqueDelta_ = 7.0;
    double deltaClamp_ = 20.0;
    std::vector<std::string> featureNames_;
    std::vector<InputEntry> entries_;
    int dIn_ = 0, dOut_ = 0;
    bool hasSunFeatures_ = false;
    ResMLP mlp_;
    // Per-target inverse-transform arrays, each [K]
    struct NormRow {
        std::vector<double> mean, std, logEps;
        std::vector<uint8_t> logMask;
        bool anyLog = false;
    };
    std::vector<NormRow> norm_;
    std::string sampledJson_;  // Raw sampled dist map for FeatureRange
    std::filesystem::path path_;
};

}  // namespace quantiloom
