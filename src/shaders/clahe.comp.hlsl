/**
 * @file clahe.comp.hlsl
 * @brief CLAHE (Contrast Limited Adaptive Histogram Equalization) compute shaders
 *
 * GPU implementation of CLAHE for real-time display enhancement.
 * Three passes:
 *   Pass 1 (CLAHE_PASS_HISTOGRAM): Build per-tile histograms
 *   Pass 2 (CLAHE_PASS_CDF): Clip, redistribute, compute CDFs
 *   Pass 3 (CLAHE_PASS_APPLY): Apply interpolated mapping
 *
 * Designed for HDR images - handles floating point input with auto-normalization.
 *
 * @author blitzcolo
 */

// Define which pass to compile (set via specialization constant or #define)
// CLAHE_PASS_HISTOGRAM = 0
// CLAHE_PASS_CDF = 1
// CLAHE_PASS_APPLY = 2

// ============================================================================
// Common Definitions
// ============================================================================

#define HISTOGRAM_BINS 256
#define MAX_TILES_X 64
#define MAX_TILES_Y 64

// Push constants for all passes
struct CLAHEPushConstants {
    uint imageWidth;
    uint imageHeight;
    uint tileCountX;
    uint tileCountY;
    float clipLimit;           // Typically 2.0-4.0
    uint luminanceOnly;        // 1 = process luminance only, 0 = all channels
    float inputMin;            // Pre-computed min value for normalization
    float inputMax;            // Pre-computed max value for normalization
    uint passIndex;            // Which pass is executing (0, 1, or 2)
    uint padding[3];
};

[[vk::push_constant]] CLAHEPushConstants params;

// Input image (R32G32B32A32_SFLOAT) - raw HDR output from ray tracer
[[vk::binding(0, 0)]] RWTexture2D<float4> inputImage;

// Output image (R32G32B32A32_SFLOAT) - CLAHE processed display image
[[vk::binding(1, 0)]] RWTexture2D<float4> outputImage;

// Histogram buffer: tileCountX * tileCountY * HISTOGRAM_BINS * uint
// Layout: histogram[tileY][tileX][bin]
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> histogramBuffer;

// CDF buffer: same layout as histogram, stores cumulative distribution
[[vk::binding(3, 0)]] RWStructuredBuffer<float> cdfBuffer;

// Tile min/max buffer for HDR normalization: tileCountX * tileCountY * 2
[[vk::binding(4, 0)]] RWStructuredBuffer<float> tileMinMaxBuffer;

// ============================================================================
// Helper Functions
// ============================================================================

// BT.709 RGB to luminance
float RGBToLuminance(float3 rgb) {
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

// Get histogram buffer index
uint GetHistogramIndex(uint tileX, uint tileY, uint bin) {
    return (tileY * params.tileCountX + tileX) * HISTOGRAM_BINS + bin;
}

// Get tile min/max buffer index
uint GetMinMaxIndex(uint tileX, uint tileY) {
    return (tileY * params.tileCountX + tileX) * 2;
}

// Normalize value to [0, 1] range
float NormalizeValue(float v, float minVal, float maxVal) {
    if (maxVal <= minVal) return 0.5f;
    return saturate((v - minVal) / (maxVal - minVal));
}

// Map normalized value to histogram bin
uint ValueToBin(float normalizedValue) {
    uint bin = (uint)(normalizedValue * (HISTOGRAM_BINS - 1) + 0.5f);
    return min(bin, HISTOGRAM_BINS - 1);
}

// Get tile index from pixel coordinates
uint2 GetTileIndex(uint2 pixelCoord) {
    uint tileWidth = (params.imageWidth + params.tileCountX - 1) / params.tileCountX;
    uint tileHeight = (params.imageHeight + params.tileCountY - 1) / params.tileCountY;
    return uint2(pixelCoord.x / tileWidth, pixelCoord.y / tileHeight);
}

// Get tile bounds
void GetTileBounds(uint tileX, uint tileY, out uint2 minCoord, out uint2 maxCoord) {
    uint tileWidth = (params.imageWidth + params.tileCountX - 1) / params.tileCountX;
    uint tileHeight = (params.imageHeight + params.tileCountY - 1) / params.tileCountY;

    minCoord = uint2(tileX * tileWidth, tileY * tileHeight);
    maxCoord = uint2(min((tileX + 1) * tileWidth, params.imageWidth),
                     min((tileY + 1) * tileHeight, params.imageHeight));
}

// ============================================================================
// Pass 1: Build Per-Tile Histograms
// ============================================================================
// Each workgroup processes one tile
// Dispatch: (tileCountX, tileCountY, 1)

#ifdef CLAHE_PASS_HISTOGRAM

groupshared uint localHistogram[HISTOGRAM_BINS];

[numthreads(16, 16, 1)]
void main(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID, uint localIndex : SV_GroupIndex) {
    uint tileX = groupId.x;
    uint tileY = groupId.y;

    // Initialize shared memory histogram to zero
    if (localIndex < HISTOGRAM_BINS) {
        localHistogram[localIndex] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Get tile bounds
    uint2 minCoord, maxCoord;
    GetTileBounds(tileX, tileY, minCoord, maxCoord);
    uint tileWidth = maxCoord.x - minCoord.x;
    uint tileHeight = maxCoord.y - minCoord.y;

    // Use global min/max for normalization (pre-computed on CPU)
    float globalMin = params.inputMin;
    float globalMax = params.inputMax;

    // Each thread processes multiple pixels and builds histogram
    for (uint dy = localId.y; dy < tileHeight; dy += 16) {
        for (uint dx = localId.x; dx < tileWidth; dx += 16) {
            uint2 pixelCoord = minCoord + uint2(dx, dy);
            if (pixelCoord.x >= params.imageWidth || pixelCoord.y >= params.imageHeight)
                continue;

            float4 pixel = inputImage[pixelCoord];

            // Skip invalid pixels
            if (isinf(pixel.r) || isnan(pixel.r) ||
                isinf(pixel.g) || isnan(pixel.g) ||
                isinf(pixel.b) || isnan(pixel.b))
                continue;

            // Compute luminance for histogram
            float value = RGBToLuminance(pixel.rgb);

            // Normalize and bin
            float normalized = NormalizeValue(value, globalMin, globalMax);
            uint bin = ValueToBin(normalized);

            InterlockedAdd(localHistogram[bin], 1);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Write histogram to global memory
    if (localIndex < HISTOGRAM_BINS) {
        uint idx = GetHistogramIndex(tileX, tileY, localIndex);
        histogramBuffer[idx] = localHistogram[localIndex];
    }
}

#endif // CLAHE_PASS_HISTOGRAM

// ============================================================================
// Pass 2: Clip, Redistribute, and Compute CDF
// ============================================================================
// Each workgroup processes one tile's histogram
// Dispatch: (tileCountX, tileCountY, 1)

#ifdef CLAHE_PASS_CDF

groupshared uint histogram[HISTOGRAM_BINS];
groupshared float cdf[HISTOGRAM_BINS];

[numthreads(256, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint localIndex : SV_GroupIndex) {
    uint tileX = groupId.x;
    uint tileY = groupId.y;

    // Load histogram to shared memory
    if (localIndex < HISTOGRAM_BINS) {
        uint idx = GetHistogramIndex(tileX, tileY, localIndex);
        histogram[localIndex] = histogramBuffer[idx];
    }
    GroupMemoryBarrierWithGroupSync();

    // Calculate total pixel count in this tile
    uint totalPixels = 0;
    if (localIndex == 0) {
        for (uint i = 0; i < HISTOGRAM_BINS; ++i) {
            totalPixels += histogram[i];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Broadcast totalPixels (stored in histogram[0] temporarily)
    if (localIndex == 0) {
        cdf[0] = (float)totalPixels;
    }
    GroupMemoryBarrierWithGroupSync();
    totalPixels = (uint)cdf[0];

    // Calculate clip limit (absolute count)
    float avgBinCount = (float)totalPixels / (float)HISTOGRAM_BINS;
    uint clipThreshold = (uint)(params.clipLimit * avgBinCount);
    if (clipThreshold == 0) clipThreshold = 1;

    // Clip and count excess
    uint excess = 0;
    if (localIndex < HISTOGRAM_BINS) {
        if (histogram[localIndex] > clipThreshold) {
            excess = histogram[localIndex] - clipThreshold;
            histogram[localIndex] = clipThreshold;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Sum excess (simple reduction)
    if (localIndex == 0) {
        uint totalExcess = 0;
        for (uint i = 0; i < HISTOGRAM_BINS; ++i) {
            if (histogramBuffer[GetHistogramIndex(tileX, tileY, i)] > clipThreshold) {
                totalExcess += histogramBuffer[GetHistogramIndex(tileX, tileY, i)] - clipThreshold;
            }
        }
        cdf[1] = (float)totalExcess;  // Store for redistribution
    }
    GroupMemoryBarrierWithGroupSync();

    // Redistribute excess equally
    uint totalExcess = (uint)cdf[1];
    uint redistributePerBin = totalExcess / HISTOGRAM_BINS;
    uint remainder = totalExcess % HISTOGRAM_BINS;

    if (localIndex < HISTOGRAM_BINS) {
        histogram[localIndex] += redistributePerBin;
        if (localIndex < remainder) {
            histogram[localIndex] += 1;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Compute CDF (prefix sum)
    if (localIndex == 0) {
        float cumulative = 0.0f;
        float scale = (totalPixels > 0) ? 1.0f / (float)totalPixels : 1.0f;
        for (uint i = 0; i < HISTOGRAM_BINS; ++i) {
            cumulative += (float)histogram[i];
            cdf[i] = cumulative * scale;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Write CDF to global memory
    if (localIndex < HISTOGRAM_BINS) {
        uint idx = GetHistogramIndex(tileX, tileY, localIndex);
        cdfBuffer[idx] = cdf[localIndex];
    }
}

#endif // CLAHE_PASS_CDF

// ============================================================================
// Pass 3: Apply Interpolated Mapping
// ============================================================================
// Each thread processes one pixel
// Dispatch: ((width+15)/16, (height+15)/16, 1)

#ifdef CLAHE_PASS_APPLY

// Sample CDF from a tile
float SampleCDF(uint tileX, uint tileY, uint bin) {
    tileX = clamp(tileX, 0, params.tileCountX - 1);
    tileY = clamp(tileY, 0, params.tileCountY - 1);
    bin = clamp(bin, 0, HISTOGRAM_BINS - 1);
    uint idx = GetHistogramIndex(tileX, tileY, bin);
    return cdfBuffer[idx];
}

// Bilinear interpolation of CDF values from 4 surrounding tiles
float InterpolateCDF(float2 pixelCenter, uint bin) {
    float tileWidth = (float)params.imageWidth / (float)params.tileCountX;
    float tileHeight = (float)params.imageHeight / (float)params.tileCountY;

    // Find the four surrounding tile centers
    float tileFracX = (pixelCenter.x / tileWidth) - 0.5f;
    float tileFracY = (pixelCenter.y / tileHeight) - 0.5f;

    int tileX0 = (int)floor(tileFracX);
    int tileY0 = (int)floor(tileFracY);
    int tileX1 = tileX0 + 1;
    int tileY1 = tileY0 + 1;

    float fx = tileFracX - (float)tileX0;
    float fy = tileFracY - (float)tileY0;

    // Clamp tile indices
    tileX0 = clamp(tileX0, 0, (int)params.tileCountX - 1);
    tileY0 = clamp(tileY0, 0, (int)params.tileCountY - 1);
    tileX1 = clamp(tileX1, 0, (int)params.tileCountX - 1);
    tileY1 = clamp(tileY1, 0, (int)params.tileCountY - 1);

    // Sample CDF from 4 tiles
    float cdf00 = SampleCDF(tileX0, tileY0, bin);
    float cdf10 = SampleCDF(tileX1, tileY0, bin);
    float cdf01 = SampleCDF(tileX0, tileY1, bin);
    float cdf11 = SampleCDF(tileX1, tileY1, bin);

    // Bilinear interpolation
    float cdf0 = lerp(cdf00, cdf10, fx);
    float cdf1 = lerp(cdf01, cdf11, fx);
    return lerp(cdf0, cdf1, fy);
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    uint2 pixelCoord = dispatchId.xy;

    if (pixelCoord.x >= params.imageWidth || pixelCoord.y >= params.imageHeight)
        return;

    float4 pixel = inputImage[pixelCoord];
    float4 result = pixel;

    // Skip invalid pixels
    if (isinf(pixel.r) || isnan(pixel.r) ||
        isinf(pixel.g) || isnan(pixel.g) ||
        isinf(pixel.b) || isnan(pixel.b)) {
        outputImage[pixelCoord] = pixel;
        return;
    }

    float globalMin = params.inputMin;
    float globalMax = params.inputMax;
    float2 pixelCenter = float2(pixelCoord) + 0.5f;

    if (params.luminanceOnly) {
        // Process luminance only, preserve colors
        float luminance = RGBToLuminance(pixel.rgb);
        float normalizedLum = NormalizeValue(luminance, globalMin, globalMax);
        uint bin = ValueToBin(normalizedLum);

        // Get interpolated CDF value (already in [0, 1] range)
        float mappedLum = InterpolateCDF(pixelCenter, bin);

        // Scale RGB by luminance ratio
        // Output is normalized to [0, 1] for display (not HDR physical values)
        if (normalizedLum > 1e-6f) {
            float scale = mappedLum / normalizedLum;
            result.rgb = saturate(float3(
                NormalizeValue(pixel.r, globalMin, globalMax) * scale,
                NormalizeValue(pixel.g, globalMin, globalMax) * scale,
                NormalizeValue(pixel.b, globalMin, globalMax) * scale
            ));
        } else {
            result.rgb = float3(mappedLum, mappedLum, mappedLum);
        }
    } else {
        // Process each channel independently
        float3 normalizedRGB = float3(
            NormalizeValue(pixel.r, globalMin, globalMax),
            NormalizeValue(pixel.g, globalMin, globalMax),
            NormalizeValue(pixel.b, globalMin, globalMax)
        );

        uint3 bins = uint3(
            ValueToBin(normalizedRGB.r),
            ValueToBin(normalizedRGB.g),
            ValueToBin(normalizedRGB.b)
        );

        // Get interpolated CDF values for each channel
        // Output is directly in [0, 1] range for display
        result.r = InterpolateCDF(pixelCenter, bins.r);
        result.g = InterpolateCDF(pixelCenter, bins.g);
        result.b = InterpolateCDF(pixelCenter, bins.b);
    }

    result.a = pixel.a;
    outputImage[pixelCoord] = result;
}

#endif // CLAHE_PASS_APPLY
