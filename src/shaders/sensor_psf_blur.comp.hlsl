/**
 * @file sensor_psf_blur.comp.hlsl
 * @brief GPU Sensor Simulation - Pass 3/4: PSF Blur (Separable Gaussian)
 *
 * Applies Point Spread Function (PSF) blur using separable Gaussian convolution.
 * Two passes required:
 *   - SENSOR_PSF_HORIZONTAL: Horizontal blur
 *   - SENSOR_PSF_VERTICAL: Vertical blur
 *
 * Sigma arrives as a push constant. The host derives it in PSFSigmaPixels()
 * (include/quantiloom/postprocess/SensorModel.hpp) -- either an explicit
 * override, or the width matched to the diffraction-limited Airy core:
 *   σ_psf = 0.437 × λ × f# / pixel_pitch
 * A kernelRadius of 0 means "no blur": the single centre tap has unit weight,
 * so the pass copies. Sigma is still non-zero there, because the weight below
 * is 0/0 at x = 0 when sigma = 0.
 *
 * @author blitzcolo
 */

// Define which pass to compile
// SENSOR_PSF_HORIZONTAL = 0
// SENSOR_PSF_VERTICAL = 1

// ============================================================================
// Push Constants
// ============================================================================

struct SensorPSFParams {
    float sigma;                  // Gaussian sigma (pixels)
    uint kernelRadius;            // Kernel radius (pixels)
    uint imageWidth;              // Image width
    uint imageHeight;             // Image height

    uint passIndex;               // 0 = horizontal, 1 = vertical
    uint padding[3];
};

[[vk::push_constant]] SensorPSFParams params;

// ============================================================================
// Bindings
// ============================================================================

// Input image (storage image for compute shader)
[[vk::binding(0, 0)]] RWTexture2D<float4> inputImage;

// Output image
[[vk::binding(1, 0)]] RWTexture2D<float4> outputImage;

// ============================================================================
// Helper Functions
// ============================================================================

// Gaussian kernel weight
float gaussian(float x, float sigma) {
    return exp(-0.5 * (x * x) / (sigma * sigma));
}

// ============================================================================
// Main Compute Shader
// ============================================================================

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 coord = dispatchThreadID.xy;

    // Bounds check
    if (coord.x >= params.imageWidth || coord.y >= params.imageHeight) {
        return;
    }

    float3 sum = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;

#ifdef SENSOR_PSF_HORIZONTAL
    // Horizontal convolution
    for (int dx = -int(params.kernelRadius); dx <= int(params.kernelRadius); ++dx) {
        int x = int(coord.x) + dx;
        if (x >= 0 && x < int(params.imageWidth)) {
            float weight = gaussian(float(dx), params.sigma);
            float3 sample = inputImage[uint2(x, coord.y)].rgb;
            sum += sample * weight;
            weightSum += weight;
        }
    }
#endif

#ifdef SENSOR_PSF_VERTICAL
    // Vertical convolution
    for (int dy = -int(params.kernelRadius); dy <= int(params.kernelRadius); ++dy) {
        int y = int(coord.y) + dy;
        if (y >= 0 && y < int(params.imageHeight)) {
            float weight = gaussian(float(dy), params.sigma);
            float3 sample = inputImage[uint2(coord.x, y)].rgb;
            sum += sample * weight;
            weightSum += weight;
        }
    }
#endif

    // Normalize
    float3 result = sum / max(weightSum, 1e-10);

    // Store result
    outputImage[coord] = float4(result, 1.0);
}
