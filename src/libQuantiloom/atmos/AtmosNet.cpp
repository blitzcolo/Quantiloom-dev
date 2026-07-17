#include "atmos/AtmosNet.hpp"

#include "atmos/Safetensors.hpp"
#include "core/Log.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace quantiloom {

using nlohmann::json;

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg2Rad = kPi / 180.0;
}  // namespace

struct AtmosNet::InputEntry {
    enum class Kind { Linear, Log, CosDeg, Onehot } kind = Kind::Linear;
    std::string name;
    int col = 0;
    double lo = 0.0, hi = 1.0;    // Continuous kinds
    std::vector<double> values;   // Onehot
    mutable bool warned = false;  // One-time out-of-domain warning
};

AtmosNet::AtmosNet(const std::filesystem::path& file) : path_(file) {
    SafetensorsFile st(file);
    const std::string& fmt = st.Metadata("format_version");
    if (fmt != "1")
        throw std::runtime_error("AtmosNet: " + file.string() +
                                 ": unsupported format_version '" + fmt + "'");
    net_ = st.Metadata("net");
    pathType_ = st.Metadata("path_type");
    opaqueDelta_ = std::stod(st.Metadata("opaque_delta"));
    deltaClamp_ = std::stod(st.Metadata("delta_clamp"));
    sampledJson_ = st.Metadata("sampled_json");

    json band, model, ispec, targets, featNames;
    try {
        band = json::parse(st.Metadata("band_json"));
        model = json::parse(st.Metadata("model_json"));
        ispec = json::parse(st.Metadata("input_spec_json"));
        targets = json::parse(st.Metadata("targets_json"));
        featNames = json::parse(st.Metadata("feature_names_json"));
    } catch (const json::exception& e) {
        throw std::runtime_error("AtmosNet: " + file.string() +
                                 ": malformed metadata JSON: " + e.what());
    }

    band_.name = band.at("name").get<std::string>();
    band_.v1_cm = band.at("v1_cm").get<double>();
    band_.v2_cm = band.at("v2_cm").get<double>();
    band_.dv_cm = band.at("dv_cm").get<double>();
    band_.K = band.at("K").get<int>();
    band_.thermal = band.at("thermal").get<bool>();

    for (auto& n : featNames) featureNames_.push_back(n.get<std::string>());

    dIn_ = model.at("d_in").get<int>();
    dOut_ = model.at("d_out").get<int>();

    for (auto& e : ispec.at("entries")) {
        InputEntry ie;
        const std::string kind = e.at("kind").get<std::string>();
        ie.name = e.at("name").get<std::string>();
        ie.col = e.at("col").get<int>();
        if (kind == "onehot") {
            ie.kind = InputEntry::Kind::Onehot;
            for (auto& v : e.at("values")) ie.values.push_back(v.get<double>());
        } else {
            if (kind == "linear") ie.kind = InputEntry::Kind::Linear;
            else if (kind == "log") ie.kind = InputEntry::Kind::Log;
            else if (kind == "cos_deg") ie.kind = InputEntry::Kind::CosDeg;
            else
                throw std::runtime_error("AtmosNet: " + file.string() +
                                         ": unknown input entry kind '" + kind + "'");
            ie.lo = e.at("lo").get<double>();
            ie.hi = e.at("hi").get<double>();
        }
        if (ie.col < 0 || ie.col >= static_cast<int>(featureNames_.size()))
            throw std::runtime_error("AtmosNet: " + file.string() +
                                     ": input entry '" + ie.name +
                                     "' col out of range");
        if (ie.name.find("sun") != std::string::npos) hasSunFeatures_ = true;
        entries_.push_back(std::move(ie));
    }

    // Consistency check: expanded feature count must equal d_in
    int expanded = 0;
    for (const auto& e : entries_)
        expanded += (e.kind == InputEntry::Kind::Onehot)
                        ? static_cast<int>(e.values.size())
                        : 1;
    if (expanded != dIn_)
        throw std::runtime_error("AtmosNet: " + file.string() +
                                 ": input_spec expands to " + std::to_string(expanded) +
                                 " features but model d_in=" + std::to_string(dIn_));

    // Targets and per-target normalization arrays
    const int K = targets.at("K").get<int>();
    if (K != band_.K)
        throw std::runtime_error("AtmosNet: " + file.string() +
                                 ": targets K != band K");
    int t = 0;
    for (auto& r : targets.at("rows")) {
        AtmosTargetRow row;
        row.block = r.at("block").get<std::string>();
        row.column = r.at("column").get<std::string>();
        row.kind = r.at("kind").get<std::string>();
        targets_.push_back(std::move(row));

        NormRow nr;
        const std::string p = "norm." + std::to_string(t) + ".";
        auto loadVec = [&](const char* suffix, std::vector<double>& out) {
            const TensorView& tv = st.Get(p + suffix);
            if (tv.shape.size() != 1 || tv.shape[0] != K)
                throw std::runtime_error("AtmosNet: " + file.string() + ": tensor '" +
                                         p + suffix + "' shape mismatch");
            out.assign(tv.F32Data(), tv.F32Data() + K);
        };
        loadVec("mean", nr.mean);
        loadVec("std", nr.std);
        loadVec("log_eps", nr.logEps);
        const TensorView& tm = st.Get(p + "log_mask");
        if (tm.shape.size() != 1 || tm.shape[0] != K)
            throw std::runtime_error("AtmosNet: " + file.string() + ": tensor '" +
                                     p + "log_mask' shape mismatch");
        nr.logMask.assign(tm.U8Data(), tm.U8Data() + K);
        for (uint8_t m : nr.logMask)
            if (m) { nr.anyLog = true; break; }
        norm_.push_back(std::move(nr));
        ++t;
    }
    if (static_cast<int>(targets_.size()) * K != dOut_)
        throw std::runtime_error("AtmosNet: " + file.string() +
                                 ": T*K != model d_out");

    mlp_.Load(st, dIn_, dOut_, model.at("width").get<int>(),
              model.at("blocks").get<int>(), model.at("pca_mode").get<std::string>(),
              model.at("n_pc").get<int>());
}

AtmosNet::~AtmosNet() = default;
AtmosNet::AtmosNet(AtmosNet&&) noexcept = default;
AtmosNet& AtmosNet::operator=(AtmosNet&&) noexcept = default;

int AtmosNet::FeatureIndex(const std::string& name) const {
    for (size_t i = 0; i < featureNames_.size(); ++i)
        if (featureNames_[i] == name) return static_cast<int>(i);
    return -1;
}

bool AtmosNet::FeatureRange(const std::string& name, double& lo, double& hi) const {
    json sampled = json::parse(sampledJson_);
    auto it = sampled.find(name);
    if (it == sampled.end()) return false;
    const json& dist = *it;
    if (dist.contains("values")) {
        lo = std::numeric_limits<double>::infinity();
        hi = -std::numeric_limits<double>::infinity();
        for (auto& v : dist["values"]) {
            const double x = v.get<double>();
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
        return true;
    }
    for (const char* k : {"uniform", "log_uniform"}) {
        if (dist.contains(k)) {
            lo = dist[k][0].get<double>();
            hi = dist[k][1].get<double>();
            return true;
        }
    }
    return false;
}

std::vector<double> AtmosNet::DiscreteValues(const std::string& name) const {
    json sampled = json::parse(sampledJson_);
    std::vector<double> out;
    auto it = sampled.find(name);
    if (it != sampled.end() && it->contains("values"))
        for (auto& v : (*it)["values"]) out.push_back(v.get<double>());
    return out;
}

void AtmosNet::AssembleFeatures(const double* rows, size_t n, float* X) const {
    const int P = NumFeatures();
    for (size_t r = 0; r < n; ++r) {
        const double* row = rows + r * P;
        float* x = X + r * dIn_;
        int j = 0;
        for (const InputEntry& e : entries_) {
            double v = row[e.col];
            if (e.kind == InputEntry::Kind::Onehot) {
                // Snap to the nearest declared value, then exact one-hot
                double best = e.values[0];
                for (double cand : e.values)
                    if (std::abs(cand - v) < std::abs(best - v)) best = cand;
                if (std::abs(best - v) > 1e-9 && !e.warned) {
                    e.warned = true;
                    QL_LOG_WARN("AtmosNet[{}_{}]: discrete feature '{}' value {} "
                                "snapped to nearest training value {}",
                                band_.name, net_, e.name, v, best);
                }
                for (double cand : e.values)
                    x[j++] = (std::abs(cand - best) < 1e-9) ? 1.0f : 0.0f;
                continue;
            }
            if (e.kind == InputEntry::Kind::Log)
                v = std::log(std::max(v, 1e-300));
            else if (e.kind == InputEntry::Kind::CosDeg)
                v = std::cos(v * kDeg2Rad);
            double u = 2.0 * (v - e.lo) / (e.hi - e.lo) - 1.0;
            if (u < -1.0 || u > 1.0) {
                if (!e.warned && (u < -1.0 - 1e-9 || u > 1.0 + 1e-9)) {
                    e.warned = true;
                    QL_LOG_WARN("AtmosNet[{}_{}]: feature '{}' outside training "
                                "domain, clamped (normalized {:.4f})",
                                band_.name, net_, e.name, u);
                }
                u = std::clamp(u, -1.0, 1.0);
            }
            x[j++] = static_cast<float>(u);
        }
    }
}

void AtmosNet::Infer(const double* rows, size_t n, double* out) const {
    if (n == 0) return;
    std::vector<float> X(n * dIn_);
    AssembleFeatures(rows, n, X.data());
    std::vector<float> Z(n * dOut_);
    mlp_.Forward(X.data(), n, Z.data());

    // Inverse transforms in float64, matching the Python reference exactly.
    const int K = band_.K;
    const int T = this->T();
    for (size_t r = 0; r < n; ++r) {
        for (int t = 0; t < T; ++t) {
            const NormRow& nr = norm_[t];
            const bool isDelta = targets_[t].kind == "delta";
            const float* z = Z.data() + r * dOut_ + static_cast<size_t>(t) * K;
            double* y = out + (r * T + t) * K;
            for (int k = 0; k < K; ++k) {
                double v = static_cast<double>(z[k]) * nr.std[k] + nr.mean[k];
                if (nr.anyLog && nr.logMask[k]) v = std::exp(v) - nr.logEps[k];
                if (isDelta) {
                    v = std::clamp(v, 0.0, deltaClamp_);
                } else {
                    v = std::max(v, 0.0);
                    // Physical ceiling: log_eps = 1e-4 * vmax_train, so
                    // 2 * vmax_train = log_eps * 1e4 * 2
                    if (nr.anyLog && nr.logMask[k])
                        v = std::min(v, nr.logEps[k] * 1e4 * 2.0);
                }
                y[k] = v;
            }
        }
    }
}

bool AtmosNet::IsOpaque(const double* deltaRow) const {
    double dmin = std::numeric_limits<double>::infinity();
    for (int k = 0; k < band_.K; ++k) dmin = std::min(dmin, deltaRow[k]);
    return dmin > opaqueDelta_;
}

}  // namespace quantiloom
