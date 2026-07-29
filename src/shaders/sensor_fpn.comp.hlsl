/**
 * @file sensor_fpn.comp.hlsl
 * @brief GPU Sensor Simulation - FPN (Fixed Pattern Noise) Pass
 *
 * Applies PRNU (vertical stripes) and DSNU (horizontal stripes) to electron image.
 * FPN maps are pre-generated on CPU and uploaded as storage images.
 *
 * PRNU: multiplicative gain non-uniformity  signal *= (1 + prnu)
 * DSNU: additive dark signal non-uniformity  signal += dsnu
 * NUC:  reduces FPN residual by nucEfficiency factor
 *
 * @author blitzcolo
 */

// ============================================================================
// Push Constants
// ============================================================================

struct SensorFPNParams {
    uint enableFPN;        // FPN enable flag
    uint enableNUC;        // NUC enable flag
    float nucEfficiency;   // NUC efficiency [0, 1]
    uint imageWidth;
    uint imageHeight;
    uint padding[3];
};

[[vk::push_constant]] SensorFPNParams params;

// ============================================================================
// Bindings
// ============================================================================

// Electron image (in/out)
[[vk::binding(0, 0)]] RWTexture2D<float4> inputImage;
// Output image (same as input for in-place, but kept for layout compatibility)
[[vk::binding(1, 0)]] RWTexture2D<float4> outputImage;
// PRNU map (2D, single channel stored in .r)
[[vk::binding(2, 0)]] RWTexture2D<float4> prnuMap;
// DSNU map (2D, single channel stored in .r)
[[vk::binding(3, 0)]] RWTexture2D<float4> dsnuMap;

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

    if (!params.enableFPN) {
        // FPN disabled, just copy input to output
        outputImage[coord] = inputImage[coord];
        return;
    }

    // Read electron values
    float3 electrons = inputImage[coord].rgb;

    // Read FPN maps
    float prnu = prnuMap[coord].r;  // Multiplicative gain error
    float dsnu = dsnuMap[coord].r;  // Additive dark signal error

    // Apply NUC correction (reduces FPN residual)
    if (params.enableNUC) {
        prnu *= (1.0 - params.nucEfficiency);  // Residual PRNU after NUC
        dsnu *= (1.0 - params.nucEfficiency);  // Residual DSNU after NUC
    }

    // Apply PRNU: multiplicative gain non-uniformity
    // signal' = signal * (1 + prnu)
    electrons *= (1.0 + prnu);

    // Apply DSNU: additive dark signal non-uniformity
    // signal'' = signal' + dsnu
    electrons += dsnu;

    // Write result
    outputImage[coord] = float4(electrons, 1.0);
}
