// ============================================================================
// Quantiloom - Spectral Reconstruction Compute Shader
// ============================================================================
// GPU-accelerated spectral interpolation for adaptive hyperspectral rendering
//
// Reconstructs full spectral cube from adaptively-sampled sparse data using
// Catmull-Rom cubic spline interpolation. Each pixel's spectrum is processed
// independently in parallel.
//
// Interpolation algorithm: Catmull-Rom spline (tension = 0.5)
// - Smooth, C1 continuous
// - Local: requires 4 points (p0, p1, p2, p3)
// - Interpolates between p1 and p2
//
// Memory layout:
// - Input: SparseCube stored BSQ (band-sequential) in storage buffer
// - Output: FullCube stored BSQ in storage buffer
// - Wavelengths: storage buffers for source and target grids
//
// This was GLSL (spectral_reconstruct.comp, compiled by glslc) until it was
// the last shader outside HLSL; it is a line-for-line translation, so the
// arithmetic and its order are unchanged. Host side:
// hs_core/GpuSpectralReconstructor.cpp, which binds five storage buffers at
// bindings 0-4 and a 32-byte push constant block.
//
// Reference: "Catmull-Rom Splines" (Catmull & Rom, 1974)
// ============================================================================

#include "ql_bind.hlsli"

// ============================================================================
// Bindings
// ============================================================================

// Push constants for reconstruction parameters
struct ReconstructPush {
    uint width;             // Image width
    uint height;            // Image height
    uint srcBands;          // Number of source (sparse) bands
    uint dstBands;          // Number of destination (full) bands
    uint interpolationMode; // 0=Linear, 1=CatmullRom, 2=Akima
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
QL_PUSH_CONSTANTS(ReconstructPush, params);

// Source sparse cube data (BSQ layout: band0_all_pixels, band1_all_pixels, ...)
StructuredBuffer<float>   sparseCube     : register(t0, space0);

// Destination full cube data (BSQ layout)
RWStructuredBuffer<float> fullCube       : register(u1, space0);

// Source wavelengths (sparse grid)
StructuredBuffer<float>   srcWavelengths : register(t2, space0);

// Target wavelengths (full grid). Bound by the host; the mapping table below
// already carries each target wavelength, so the shader does not read it.
StructuredBuffer<float>   dstWavelengths : register(t3, space0);

// Reconstruction mapping (pre-computed on CPU)
// One entry per target band, packed as uint4 + float2 (6 x u32):
// x: targetBand
// y: sourceBand (exact index, or the interval's left/lower index)
// z: sourceNextBand (the interval's upper index)
// w: wasRendered (1 = direct copy, 0 = interpolate)
// float[0]: targetWavelength
// float[1]: interpWeight (t between the two source bands)
// The search depends only on the two wavelength grids, so the host builds
// the table once and the shader reads it instead of scanning the source
// grid for every pixel and band.
StructuredBuffer<uint>    mappingData    : register(t4, space0);

// ============================================================================
// Constants
// ============================================================================

#define INTERP_MODE_LINEAR      0
#define INTERP_MODE_CATMULL_ROM 1
#define INTERP_MODE_AKIMA       2

// ============================================================================
// Helper Functions
// ============================================================================

// Get pixel index in BSQ layout
uint GetPixelIndex(uint x, uint y, uint band) {
    uint pixelOffset = y * params.width + x;
    return band * params.width * params.height + pixelOffset;
}

// Read source value at (x, y, band)
float ReadSparse(uint x, uint y, uint band) {
    return sparseCube[GetPixelIndex(x, y, band)];
}

// Write destination value at (x, y, band)
void WriteFull(uint x, uint y, uint band, float value) {
    fullCube[GetPixelIndex(x, y, band)] = value;
}

// ============================================================================
// Reconstruction Mapping Readers
// ============================================================================

uint4 ReadMappingIndices(uint dstBand) {
    const uint base = dstBand * 6u;
    return uint4(mappingData[base], mappingData[base + 1u],
                 mappingData[base + 2u], mappingData[base + 3u]);
}

float2 ReadMappingWavelengthWeight(uint dstBand) {
    const uint base = dstBand * 6u;
    return float2(asfloat(mappingData[base + 4u]),
                  asfloat(mappingData[base + 5u]));
}

// ============================================================================
// Linear Interpolation
// ============================================================================

float LinearInterp(float v0, float v1, float t) {
    return v0 + t * (v1 - v0);
}

// ============================================================================
// Catmull-Rom Cubic Spline Interpolation
// ============================================================================
// Interpolates between p1 and p2 using points p0, p1, p2, p3
// Parameter t in [0, 1]

float CatmullRomInterp(float p0, float p1, float p2, float p3, float t) {
    float t2 = t * t;
    float t3 = t2 * t;

    // Catmull-Rom coefficients (tension = 0.5)
    float a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    float a1 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    float a2 = -0.5 * p0 + 0.5 * p2;
    float a3 = p1;

    return a0 * t3 + a1 * t2 + a2 * t + a3;
}

// ============================================================================
// Akima Spline Interpolation (Simplified GPU Version)
// ============================================================================
// Akima splines are more robust to outliers but require more computation
// This is a simplified version suitable for GPU execution
// idx is the interval's left source band from the mapping table.

float AkimaInterp(uint x, uint y, uint idx, float targetWl) {
    uint numSrc = params.srcBands;

    if (idx >= numSrc - 1) {
        return ReadSparse(x, y, numSrc - 1);
    }

    // Compute slopes for 5 points centered around idx
    float m[5];
    for (int j = -2; j <= 2; ++j) {
        int i0 = clamp(int(idx) + j, 0, int(numSrc) - 2);
        int i1 = i0 + 1;
        m[j + 2] = (ReadSparse(x, y, i1) - ReadSparse(x, y, i0)) /
                   (srcWavelengths[i1] - srcWavelengths[i0]);
    }

    // Akima weights
    float w1 = abs(m[3] - m[2]);
    float w2 = abs(m[1] - m[0]);

    float d0, d1;
    if (w1 + w2 < 1e-10) {
        d0 = 0.5 * (m[1] + m[2]);
    } else {
        d0 = (w1 * m[1] + w2 * m[2]) / (w1 + w2);
    }

    w1 = abs(m[4] - m[3]);
    w2 = abs(m[2] - m[1]);
    if (w1 + w2 < 1e-10) {
        d1 = 0.5 * (m[2] + m[3]);
    } else {
        d1 = (w1 * m[2] + w2 * m[3]) / (w1 + w2);
    }

    // Hermite interpolation
    float h = srcWavelengths[idx + 1] - srcWavelengths[idx];
    float t = (targetWl - srcWavelengths[idx]) / h;
    float t2 = t * t;
    float t3 = t2 * t;

    float v0 = ReadSparse(x, y, idx);
    float v1 = ReadSparse(x, y, idx + 1);

    float h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    float h10 = t3 - 2.0 * t2 + t;
    float h01 = -2.0 * t3 + 3.0 * t2;
    float h11 = t3 - t2;

    return h00 * v0 + h10 * h * d0 + h01 * v1 + h11 * h * d1;
}

// ============================================================================
// Reconstruct Single Pixel Spectrum
// ============================================================================

void ReconstructPixel(uint x, uint y) {
    // Process each target band
    for (uint dstBand = 0; dstBand < params.dstBands; ++dstBand) {
        const uint4 mapIdx = ReadMappingIndices(dstBand);
        const float2 wlWeight = ReadMappingWavelengthWeight(dstBand);
        const uint sourceBand = mapIdx.y;
        const uint sourceNextBand = mapIdx.z;
        const float interpWeight = wlWeight.y;
        float value;

        if (mapIdx.w != 0u) {
            // This wavelength was directly rendered - no interpolation needed
            value = ReadSparse(x, y, sourceBand);
        } else {
            // Interpolation needed
            switch (params.interpolationMode) {
                case INTERP_MODE_LINEAR: {
                    value = LinearInterp(
                        ReadSparse(x, y, sourceBand),
                        ReadSparse(x, y, sourceNextBand),
                        interpWeight
                    );
                    break;
                }

                case INTERP_MODE_CATMULL_ROM: {
                    const uint numSrc = params.srcBands;
                    const uint idx1 = sourceBand;
                    const uint idx2 = sourceNextBand;
                    const uint idx0 = (idx1 > 0u) ? idx1 - 1u : 0u;
                    const uint idx3 = (idx2 < numSrc - 1u) ? idx2 + 1u : numSrc - 1u;
                    value = CatmullRomInterp(
                        ReadSparse(x, y, idx0),
                        ReadSparse(x, y, idx1),
                        ReadSparse(x, y, idx2),
                        ReadSparse(x, y, idx3),
                        interpWeight
                    );
                    break;
                }

                case INTERP_MODE_AKIMA: {
                    value = AkimaInterp(x, y, sourceBand, wlWeight.x);
                    break;
                }

                default:
                    value = 0.0;
                    break;
            }
        }

        // Ensure non-negative radiance
        value = max(value, 0.0);

        WriteFull(x, y, dstBand, value);
    }
}

// ============================================================================
// Main Compute Shader
// ============================================================================
// Each thread processes one pixel's entire spectrum reconstruction
// Using 16x16 tiles for good occupancy on most GPUs

[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    uint2 coord = dispatchId.xy;

    // Bounds check
    if (coord.x >= params.width || coord.y >= params.height) {
        return;
    }

    ReconstructPixel(coord.x, coord.y);
}
