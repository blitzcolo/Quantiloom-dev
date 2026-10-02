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

int ReadDimension(const json& value, const char* name, bool allowZero = false) {
    const auto& number = value.at(name);
    if (!number.is_number_integer() ||
        (number.is_number_unsigned() && number.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int>::max())))
        throw std::runtime_error(std::string("AtmosNet: invalid integer dimension ") + name);
    const auto dimension = number.get<int64_t>();
    if (dimension < (allowZero ? 0 : 1) || dimension > std::numeric_limits<int>::max())
        throw std::runtime_error(std::string("AtmosNet: dimension out of range ") + name);
    return static_cast<int>(dimension);
}
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
    band_.K = ReadDimension(band, "K");
    band_.thermal = band.at("thermal").get<bool>();

    if (band_.K <= 0 || !std::isfinite(band_.v1_cm) || !std::isfinite(band_.v2_cm) ||
        !std::isfinite(band_.dv_cm) || band_.v1_cm <= 0 || band_.v2_cm < band_.v1_cm ||
        band_.dv_cm <= 0 || !std::isfinite(opaqueDelta_) || !std::isfinite(deltaClamp_) ||
        opaqueDelta_ < 0 || deltaClamp_ <= 0)
        throw std::runtime_error("AtmosNet: invalid spectral grid or deployment bounds");
    if (!featNames.is_array() || featNames.size() < 4 ||
        featNames.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("AtmosNet: missing geometry feature block");

    for (auto& n : featNames) featureNames_.push_back(n.get<std::string>());
    static constexpr const char* geometry[] = {"h1_km", "h2_km", "cos_view_zenith", "range_km"};
    for (size_t i = 0; i < 4; ++i)
        if (featureNames_[featureNames_.size() - 4 + i] != geometry[i])
            throw std::runtime_error("AtmosNet: geometry feature block is out of order");

    dIn_ = ReadDimension(model, "d_in");
    dOut_ = ReadDimension(model, "d_out");
    const int width = ReadDimension(model, "width");
    const int blocks = ReadDimension(model, "blocks", true);
    const int nPc = ReadDimension(model, "n_pc", true);
    if (dIn_ <= 0 || dOut_ <= 0 || width <= 0 || blocks < 0 || nPc < 0)
        throw std::runtime_error("AtmosNet: invalid model dimensions");
    // Each residual block owns six tensors; a tiny file cannot describe an
    // enormous block allocation merely by putting that count in metadata.
    if (static_cast<size_t>(blocks) > st.TensorNames().size() / 6)
        throw std::runtime_error("AtmosNet: block count exceeds available tensors");
    if (!ispec.at("entries").is_array() || ispec.at("entries").empty())
        throw std::runtime_error("AtmosNet: empty input entries");

    for (auto& e : ispec.at("entries")) {
        InputEntry ie;
        const std::string kind = e.at("kind").get<std::string>();
        ie.name = e.at("name").get<std::string>();
        ie.col = ReadDimension(e, "col", true);
        if (kind == "onehot") {
            ie.kind = InputEntry::Kind::Onehot;
            if (!e.at("values").is_array())
                throw std::runtime_error("AtmosNet: onehot values must be an array");
            for (auto& v : e.at("values")) ie.values.push_back(v.get<double>());
            if (ie.values.empty())
                throw std::runtime_error("AtmosNet: empty onehot values");
            for (double value : ie.values)
                if (!std::isfinite(value))
                    throw std::runtime_error("AtmosNet: nonfinite onehot value");
        } else {
            if (kind == "linear") ie.kind = InputEntry::Kind::Linear;
            else if (kind == "log") ie.kind = InputEntry::Kind::Log;
            else if (kind == "cos_deg") ie.kind = InputEntry::Kind::CosDeg;
            else
                throw std::runtime_error("AtmosNet: " + file.string() +
                                         ": unknown input entry kind '" + kind + "'");
            ie.lo = e.at("lo").get<double>();
            ie.hi = e.at("hi").get<double>();
            if (!std::isfinite(ie.lo) || !std::isfinite(ie.hi) || ie.hi <= ie.lo)
                throw std::runtime_error("AtmosNet: invalid input normalization range");
        }
        if (ie.col < 0 || ie.col >= static_cast<int>(featureNames_.size()))
            throw std::runtime_error("AtmosNet: " + file.string() +
                                     ": input entry '" + ie.name +
                                     "' col out of range");
        if (ie.name.find("sun") != std::string::npos) hasSunFeatures_ = true;
        entries_.push_back(std::move(ie));
    }

    // Consistency check: expanded feature count must equal d_in
    size_t expanded = 0;
    for (const auto& e : entries_) {
        const size_t count = e.kind == InputEntry::Kind::Onehot ? e.values.size() : 1;
        if (count > static_cast<size_t>(dIn_) - expanded)
            throw std::runtime_error("AtmosNet: expanded input count exceeds model d_in");
        expanded += count;
    }
    if (expanded != static_cast<size_t>(dIn_))
        throw std::runtime_error("AtmosNet: " + file.string() +
                                 ": input_spec expands to " + std::to_string(expanded) +
                                 " features but model d_in=" + std::to_string(dIn_));

    // Targets and per-target normalization arrays
    const int K = ReadDimension(targets, "K");
    if (K != band_.K)
        throw std::runtime_error("AtmosNet: " + file.string() +
                                 ": targets K != band K");
    const auto& targetRows = targets.at("rows");
    if (!targetRows.is_array() || targetRows.empty() ||
        targetRows.size() > static_cast<size_t>(dOut_) / static_cast<size_t>(K) ||
        targetRows.size() * static_cast<size_t>(K) != static_cast<size_t>(dOut_))
        throw std::runtime_error("AtmosNet: T*K != model d_out");
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
    if (targets_.size() * static_cast<size_t>(K) != static_cast<size_t>(dOut_))
        throw std::runtime_error("AtmosNet: " + file.string() +
                                 ": T*K != model d_out");

    mlp_.Load(st, dIn_, dOut_, width, blocks,
              model.at("pca_mode").get<std::string>(), nPc);
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
    const size_t maxElements = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (!rows || !out || n > maxElements / static_cast<size_t>(NumFeatures()) ||
        n > maxElements / static_cast<size_t>(dIn_) || n > maxElements / static_cast<size_t>(dOut_))
        throw std::runtime_error("AtmosNet: inference dimensions exceed addressable storage");
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
