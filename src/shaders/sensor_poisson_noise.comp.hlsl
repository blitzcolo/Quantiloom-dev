/**
 * @file sensor_poisson_noise.comp.hlsl
 * @brief GPU Sensor Simulation - Pass 2: Add Poisson + Read Noise
 *
 * Adds noise to photo-electron signal:
 * - Poisson noise (shot noise): σ = sqrt(N)
 * - Read noise (Gaussian): σ = read_noise_e_rms
 *
 * Uses PCG random number generator for GPU-friendly noise generation.
 *
 * @author wtflmao
 */

// ============================================================================
// Push Constants
// ============================================================================

struct SensorNoiseParams {
    uint frameIndex;              // For noise seed variation
    uint enablePoissonNoise;      // Flag: add shot noise
    float readNoise_e_rms;        // Read noise (e⁻ RMS)
    uint enableReadNoise;         // Flag: add read noise

    float wellCapacity_e;         // For clamping
    uint imageWidth;              // Image width
    uint imageHeight;             // Image height
    uint padding;
};

[[vk::push_constant]] SensorNoiseParams params;

// ============================================================================
// Bindings
// ============================================================================

// In/Out: Photo-electrons (e⁻)
[[vk::binding(0, 0)]] RWTexture2D<float4> electronsImage;

// ============================================================================
// PCG Random Number Generator (GPU-friendly)
// ============================================================================

// PCG hash function
uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate uniform random [0, 1]
float random(inout uint seed) {
    seed = pcg_hash(seed);
    return float(seed) / 4294967296.0;
}

// Box-Muller transform: uniform → Gaussian N(0,1)
float randomGaussian(inout uint seed) {
    float u1 = random(seed);
    float u2 = random(seed);
    // Avoid log(0)
    u1 = max(u1, 1e-10);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
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

    // Load current electron count
    float4 data = electronsImage[coord];
    float3 numElectrons = data.rgb;

    // Initialize RNG seed (unique per pixel and frame)
    uint seed = coord.x + coord.y * 65536u + params.frameIndex * 16777216u;

    // Add Poisson noise (shot noise): σ = sqrt(N)
    // Approximated using Gaussian for large N (N > 20)
    if (params.enablePoissonNoise != 0) {
        float3 shotNoise = float3(
            randomGaussian(seed) * sqrt(max(numElectrons.r, 0.0)),
            randomGaussian(seed) * sqrt(max(numElectrons.g, 0.0)),
            randomGaussian(seed) * sqrt(max(numElectrons.b, 0.0))
        );
        numElectrons += shotNoise;
    }

    // Add read noise (Gaussian)
    if (params.enableReadNoise != 0) {
        float3 readNoise = float3(
            randomGaussian(seed) * params.readNoise_e_rms,
            randomGaussian(seed) * params.readNoise_e_rms,
            randomGaussian(seed) * params.readNoise_e_rms
        );
        numElectrons += readNoise;
    }

    // Clamp to [0, well_capacity]
    numElectrons = clamp(numElectrons,
                         float3(0.0, 0.0, 0.0),
                         float3(params.wellCapacity_e, params.wellCapacity_e, params.wellCapacity_e));

    // Store result
    electronsImage[coord] = float4(numElectrons, 1.0);
}
