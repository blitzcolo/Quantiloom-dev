#include "atmos/ResMLP.hpp"

#include "atmos/Safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace quantiloom {

namespace {

void LoadLinear(const SafetensorsFile& f, const std::string& prefix,
                int expectIn, int expectOut, std::vector<float>& w,
                std::vector<float>& b) {
    const TensorView& tw = f.Get(prefix + ".weight");
    const TensorView& tb = f.Get(prefix + ".bias");
    if (tw.shape.size() != 2 || tw.shape[0] != expectOut || tw.shape[1] != expectIn)
        throw std::runtime_error("ResMLP: " + f.Path().string() + ": tensor '" +
                                 prefix + ".weight' shape mismatch (expected [" +
                                 std::to_string(expectOut) + ", " +
                                 std::to_string(expectIn) + "])");
    if (tb.shape.size() != 1 || tb.shape[0] != expectOut)
        throw std::runtime_error("ResMLP: " + f.Path().string() + ": tensor '" +
                                 prefix + ".bias' shape mismatch");
    w.assign(tw.F32Data(), tw.F32Data() + tw.NumElements());
    b.assign(tb.F32Data(), tb.F32Data() + tb.NumElements());
}

void LoadVector(const SafetensorsFile& f, const std::string& name, int expect,
                std::vector<float>& out) {
    const TensorView& t = f.Get(name);
    if (t.shape.size() != 1 || t.shape[0] != expect)
        throw std::runtime_error("ResMLP: " + f.Path().string() + ": tensor '" +
                                 name + "' shape mismatch");
    out.assign(t.F32Data(), t.F32Data() + t.NumElements());
}

// y = W x + b, W row-major [out, in]
inline void Gemv(const std::vector<float>& w, const std::vector<float>& b,
                 int in, int out, const float* x, float* y) {
    for (int o = 0; o < out; ++o) {
        const float* row = w.data() + static_cast<size_t>(o) * in;
        float acc = 0.0f;
        for (int i = 0; i < in; ++i) acc += row[i] * x[i];
        y[o] = acc + b[o];
    }
}

}  // namespace

void ResMLP::Load(const SafetensorsFile& file, int dIn, int dOut, int width,
                  int blocks, const std::string& pcaMode, int nPc) {
    dIn_ = dIn;
    dOut_ = dOut;
    width_ = width;
    nPc_ = nPc;
    pcaMode_ = pcaMode;

    stem_.in = dIn;
    stem_.out = width;
    LoadLinear(file, "stem", dIn, width, stem_.w, stem_.b);

    blocks_.resize(blocks);
    for (int i = 0; i < blocks; ++i) {
        const std::string p = "blocks." + std::to_string(i) + ".";
        Block& blk = blocks_[i];
        blk.fc1.in = blk.fc1.out = width;
        blk.fc2.in = blk.fc2.out = width;
        LoadLinear(file, p + "fc1", width, width, blk.fc1.w, blk.fc1.b);
        LoadVector(file, p + "norm.weight", width, blk.normW);
        LoadVector(file, p + "norm.bias", width, blk.normB);
        LoadLinear(file, p + "fc2", width, width, blk.fc2.w, blk.fc2.b);
    }

    hasHead_ = true;
    if (pcaMode == "none") {
        head_.in = width;
        head_.out = dOut;
        LoadLinear(file, "head", width, dOut, head_.w, head_.b);
    } else if (pcaMode == "head") {
        if (nPc <= 0)
            throw std::runtime_error("ResMLP: " + file.Path().string() +
                                     ": pca_mode=head requires n_pc > 0");
        proj_.in = width;
        proj_.out = nPc;
        LoadLinear(file, "proj", width, nPc, proj_.w, proj_.b);
        head_.in = nPc;
        head_.out = dOut;
        LoadLinear(file, "head", nPc, dOut, head_.w, head_.b);
    } else if (pcaMode == "coeff") {
        if (nPc <= 0)
            throw std::runtime_error("ResMLP: " + file.Path().string() +
                                     ": pca_mode=coeff requires n_pc > 0");
        proj_.in = width;
        proj_.out = nPc;
        LoadLinear(file, "proj", width, nPc, proj_.w, proj_.b);
        hasHead_ = false;
        const TensorView& tb = file.Get("pca.basis");
        if (tb.shape.size() != 2 || tb.shape[0] != dOut || tb.shape[1] != nPc)
            throw std::runtime_error("ResMLP: " + file.Path().string() +
                                     ": 'pca.basis' shape mismatch");
        pcaBasis_.assign(tb.F32Data(), tb.F32Data() + tb.NumElements());
        LoadVector(file, "pca.mean", dOut, pcaMean_);
    } else {
        throw std::runtime_error("ResMLP: " + file.Path().string() +
                                 ": unknown pca_mode '" + pcaMode + "'");
    }
}

void ResMLP::ForwardRows(const float* X, size_t rowBegin, size_t rowEnd,
                         float* Z) const {
    constexpr float kLayerNormEps = 1e-5f;
    std::vector<float> h(width_), a(width_), t(width_);
    std::vector<float> c(std::max(nPc_, 1));

    for (size_t r = rowBegin; r < rowEnd; ++r) {
        const float* x = X + r * dIn_;
        float* z = Z + r * dOut_;

        Gemv(stem_.w, stem_.b, dIn_, width_, x, h.data());

        for (const Block& blk : blocks_) {
            Gemv(blk.fc1.w, blk.fc1.b, width_, width_, h.data(), a.data());
            // LayerNorm over the width axis; double accumulators keep the
            // mean/variance stable independent of summation order.
            double s1 = 0.0, s2 = 0.0;
            for (int i = 0; i < width_; ++i) {
                s1 += a[i];
                s2 += static_cast<double>(a[i]) * a[i];
            }
            const double mean = s1 / width_;
            const double var = std::max(s2 / width_ - mean * mean, 0.0);
            const float inv = 1.0f / std::sqrt(static_cast<float>(var) + kLayerNormEps);
            for (int i = 0; i < width_; ++i) {
                float v = (a[i] - static_cast<float>(mean)) * inv;
                v = v * blk.normW[i] + blk.normB[i];
                a[i] = v / (1.0f + std::exp(-v));  // SiLU
            }
            Gemv(blk.fc2.w, blk.fc2.b, width_, width_, a.data(), t.data());
            for (int i = 0; i < width_; ++i) h[i] += t[i];
        }

        if (pcaMode_ == "none") {
            Gemv(head_.w, head_.b, width_, dOut_, h.data(), z);
        } else {
            Gemv(proj_.w, proj_.b, width_, nPc_, h.data(), c.data());
            if (hasHead_) {
                Gemv(head_.w, head_.b, nPc_, dOut_, c.data(), z);
            } else {
                // coeff mode: decode with the fixed PCA basis
                for (int o = 0; o < dOut_; ++o) {
                    const float* row = pcaBasis_.data() + static_cast<size_t>(o) * nPc_;
                    float acc = 0.0f;
                    for (int i = 0; i < nPc_; ++i) acc += row[i] * c[i];
                    z[o] = acc + pcaMean_[o];
                }
            }
        }
    }
}

void ResMLP::Forward(const float* X, size_t n, float* Z) const {
    if (n == 0) return;
    const size_t hw = std::max<size_t>(std::thread::hardware_concurrency(), 1);
    const size_t numThreads = std::min(n, hw);
    if (numThreads <= 1) {
        ForwardRows(X, 0, n, Z);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(numThreads);
    const size_t chunk = (n + numThreads - 1) / numThreads;
    for (size_t t = 0; t < numThreads; ++t) {
        const size_t lo = t * chunk;
        const size_t hi = std::min(lo + chunk, n);
        if (lo >= hi) break;
        pool.emplace_back([this, X, lo, hi, Z] { ForwardRows(X, lo, hi, Z); });
    }
    for (auto& th : pool) th.join();
}

}  // namespace quantiloom
