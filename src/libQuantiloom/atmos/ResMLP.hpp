#pragma once

#include "core/Platform.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace quantiloom {

class SafetensorsFile;

// ============================================================================
// ResMLP - hand-written fp32 forward pass for the MODTRAN surrogate networks
// ============================================================================
// Architecture (mirrors atmoemu/model.py):
//   stem(Linear d_in->width)
//   N x residual block: x + fc2(SiLU(LayerNorm(fc1(x))))     eps = 1e-5
//   [proj(Linear width->n_pc)] [head(Linear ...->d_out)]
//   [coeff mode only: z @ pca.basis^T + pca.mean]
//
// Weights are loaded from a safetensors file using the torch state-dict
// names. Batch rows are processed in parallel with std::thread.

class ResMLP {
public:
    // Loads and shape-checks all weights. Throws std::runtime_error naming
    // the file and tensor on any mismatch.
    void Load(const SafetensorsFile& file, int dIn, int dOut, int width,
              int blocks, const std::string& pcaMode, int nPc);

    // X: [n, dIn] row-major fp32 -> Z: [n, dOut] row-major fp32.
    void Forward(const float* X, size_t n, float* Z) const;

    int InputDim() const { return dIn_; }
    int OutputDim() const { return dOut_; }

private:
    struct Linear {
        int in = 0, out = 0;
        std::vector<float> w;  // [out, in] row-major (torch layout)
        std::vector<float> b;  // [out]
    };
    struct Block {
        Linear fc1;
        std::vector<float> normW, normB;  // LayerNorm affine [width]
        Linear fc2;
    };

    void ForwardRows(const float* X, size_t rowBegin, size_t rowEnd, float* Z) const;

    int dIn_ = 0, dOut_ = 0, width_ = 0, nPc_ = 0;
    std::string pcaMode_ = "none";
    Linear stem_;
    std::vector<Block> blocks_;
    Linear proj_;                       // Used when pcaMode != none
    Linear head_;                       // Absent only in coeff mode
    bool hasHead_ = true;
    std::vector<float> pcaBasis_;       // [dOut, nPc], coeff mode only
    std::vector<float> pcaMean_;        // [dOut], coeff mode only
};

}  // namespace quantiloom
