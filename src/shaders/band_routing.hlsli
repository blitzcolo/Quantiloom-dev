// ============================================================================
// Quantiloom - Single-wavelength branch routing
// ============================================================================
// SPECTRAL_MODE_SINGLE and SPECTRAL_MODE_CAMERA_MEASUREMENT both carry exactly
// one wavelength per ray -- SINGLE reads it from the camera push constants, a
// camera-measurement ray carries it in the payload's heroLambda -- and route
// it through whichever band branch of closesthit.rchit / miss.rmiss would own
// a fused sweep over that wavelength. The cut points are the fused bands' own
// edges: at or below SPECTRAL_VIS_LAMBDA_MAX the ray takes the single-visible
// branch, at or above SPECTRAL_MWIR_LAMBDA_MIN the thermal branch, and a NIR
// wavelength between them lands in the SWIR branch, where the reflective-IR
// integrator lives (the NIR branch is fused-only and never sees these rays).
//
// Include AFTER common.hlsli: these helpers read SPEC_SPECTRAL_MODE and
// pushConsts, both declared there.
// ============================================================================
#ifndef QUANTILOOM_BAND_ROUTING_HLSLI
#define QUANTILOOM_BAND_ROUTING_HLSLI

/// The two modes that carry one wavelength per ray.
bool IsSingleWavelengthMode() {
    return SPEC_SPECTRAL_MODE == SPECTRAL_MODE_SINGLE ||
           SPEC_SPECTRAL_MODE == SPECTRAL_MODE_CAMERA_MEASUREMENT;
}

/// The wavelength the current ray carries: the push constant under SINGLE,
/// heroLambda under CAMERA_MEASUREMENT. A fused-mode caller gets heroLambda
/// back, which is the bounce-ray sample the band loops want anyway.
float SingleModeWavelength(float payloadHeroLambda) {
    return SPEC_SPECTRAL_MODE == SPECTRAL_MODE_SINGLE
               ? pushConsts.camera.wavelength_nm
               : payloadHeroLambda;
}

/// Where one carried wavelength routes in the band chain.
bool RoutesToVisibleSingle(float lambda) {
    return lambda <= SPECTRAL_VIS_LAMBDA_MAX;
}
bool RoutesToSwirBranch(float lambda) {
    return lambda > SPECTRAL_VIS_LAMBDA_MAX && lambda < SPECTRAL_MWIR_LAMBDA_MIN;
}
bool RoutesToThermalBranch(float lambda) {
    return lambda >= SPECTRAL_MWIR_LAMBDA_MIN;
}
#endif // QUANTILOOM_BAND_ROUTING_HLSLI
