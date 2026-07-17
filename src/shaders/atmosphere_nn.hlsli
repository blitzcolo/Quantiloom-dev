// ============================================================================
// Quantiloom - Neural Network Atmosphere LUT Sampling
// ============================================================================
// Replaces the analytic Rayleigh/Mie atmosphere with LUTs baked on the CPU
// from MODTRAN surrogate networks (see src/libQuantiloom/atmos/).
//
// Composition contract (MODTRAN deployment model):
//   L_pixel(lambda) = tau_view(lambda) * L_surface(lambda) + L_path(lambda)
//
// Data layout (binding 17 header + binding 20 flat float blob):
//   tau:   [numLambda, countA]           tauOffset + iLambda*countA + ia
//   lpath: [numLambda, countA, countAz]  lpathOffset + (iLambda*countA+ia)*countAz + iaz
//   ldown: [numLambda]                   ldownOffset + iLambda (thermal bands)
//   sky:   reserved (hasSky/skyOffset), future <band>_sky networks
//
// Axis a: pathMode 0 (ground camera)  a = ln(range_km)
//         pathMode 1 (slant/airborne) a = cos(view_zenith)   (world Y-up)
// Axis az: relative azimuth |view - sun| horizontal angle in degrees [0,180].
//
// The lambda axis is exactly the renderer's spectral loop sample points, so
// shaders index it by loop counter -- no spectral interpolation on the GPU.
// ============================================================================

#ifndef QUANTILOOM_ATMOSPHERE_NN_HLSLI
#define QUANTILOOM_ATMOSPHERE_NN_HLSLI

// Mirror of quantiloom::AtmosNNHeaderGPU (80 bytes)
struct AtmosNNHeader {
    uint  enabled;
    uint  pathMode;      // 0 = ground (range axis), 1 = slant (cos zenith axis)
    uint  numLambda;
    uint  countA;
    uint  countAz;
    uint  hasLdown;
    uint  hasSky;
    uint  thermalBand;
    float aStart;
    float aStep;
    float azStart;       // degrees
    float azStep;        // degrees
    uint  tauOffset;
    uint  lpathOffset;
    uint  ldownOffset;
    uint  skyOffset;
    float3 sunDirWorld;  // unit vector toward the sun
    float worldUnitsToMeters;
};

// ----------------------------------------------------------------------------
// Axis coordinates
// ----------------------------------------------------------------------------

// Continuous a coordinate for a ray: ground mode uses the hit distance
// (converted to km), slant mode uses the view direction's cos(zenith).
// rayDir points from camera toward the scene; hitDist is in world units.
float AtmosCoordA(AtmosNNHeader h, float3 rayDir, float hitDist)
{
    if (h.pathMode == 0) {
        float rangeKm = max(hitDist * h.worldUnitsToMeters * 1e-3, 1e-4);
        return log(rangeKm);
    }
    // Slant LUT is parameterized by view zenith of the DOWNWARD ray
    // (110..180 deg). World is Y-up: cos(view_zenith) = rayDir.y for a ray
    // leaving the camera (zenith measured from local up at the camera).
    return rayDir.y;
}

// Relative azimuth between the viewing direction and the sun's horizontal
// projection, in degrees [0, 180].
float AtmosRelAz(AtmosNNHeader h, float3 rayDir)
{
    float2 v = float2(rayDir.x, rayDir.z);
    float2 s = float2(h.sunDirWorld.x, h.sunDirWorld.z);
    float lv = length(v);
    float ls = length(s);
    if (lv < 1e-6 || ls < 1e-6)
        return 0.0;
    float c = clamp(dot(v / lv, s / ls), -1.0, 1.0);
    return degrees(acos(c));
}

// ----------------------------------------------------------------------------
// LUT sampling (linear interpolation along a / bilinear over a,az)
// ----------------------------------------------------------------------------

// Fractional index on the a axis, clamped to the table (out-of-domain rays --
// e.g. ground camera looking up at airborne geometry -- clamp to the edge).
float AtmosIndexA(AtmosNNHeader h, float a, out uint i0, out uint i1)
{
    float fi = (a - h.aStart) / max(h.aStep, 1e-12);
    fi = clamp(fi, 0.0, (float)(h.countA - 1));
    i0 = (uint)floor(fi);
    i1 = min(i0 + 1, h.countA - 1);
    return fi - (float)i0;
}

float SampleAtmosTau(AtmosNNHeader h, StructuredBuffer<float> data,
                     uint iLambda, float a)
{
    uint i0, i1;
    float f = AtmosIndexA(h, a, i0, i1);
    uint base = h.tauOffset + iLambda * h.countA;
    return lerp(data[base + i0], data[base + i1], f);
}

float SampleAtmosLpath(AtmosNNHeader h, StructuredBuffer<float> data,
                       uint iLambda, float a, float azDeg)
{
    uint i0, i1;
    float f = AtmosIndexA(h, a, i0, i1);
    uint base = h.lpathOffset + iLambda * h.countA * h.countAz;
    if (h.countAz <= 1) {
        return lerp(data[base + i0 * h.countAz],
                    data[base + i1 * h.countAz], f);
    }
    float fj = (azDeg - h.azStart) / max(h.azStep, 1e-12);
    fj = clamp(fj, 0.0, (float)(h.countAz - 1));
    uint j0 = (uint)floor(fj);
    uint j1 = min(j0 + 1, h.countAz - 1);
    float g = fj - (float)j0;
    float v0 = lerp(data[base + i0 * h.countAz + j0],
                    data[base + i0 * h.countAz + j1], g);
    float v1 = lerp(data[base + i1 * h.countAz + j0],
                    data[base + i1 * h.countAz + j1], g);
    return lerp(v0, v1, f);
}

// Thermal downwelling radiance (W m^-2 sr^-1 nm^-1); one spectrum per bake.
float SampleAtmosLdown(AtmosNNHeader h, StructuredBuffer<float> data,
                       uint iLambda)
{
    return data[h.ldownOffset + iLambda];
}

#endif // QUANTILOOM_ATMOSPHERE_NN_HLSLI
