// ============================================================================
// Quantiloom - Thermal Step Compute Shader
// ============================================================================
// One step of the Crank-Nicolson surface energy balance, on the GPU.
// Mirrors CpuCrankNicolsonStepper line-by-line, in f32.
//
// Each thread owns one element. The inter-element coupling (radiative T^4
// exchange) reads from a ping-pong surface buffer seeded before the dispatch;
// the intra-element solve is a Thomas tridiagonal, thread-local.
//
// Non-participating elements (conductivity zero) are never written to the
// state buffer, preserving the bit-exact contract the CPU stepper keeps.
// They DO write their temperature into the surface buffer so the next step's
// radiative read is correct.
// ============================================================================

#define QL_THERMAL_MAX_NODES 32

struct ThermalElementGpu {
    float3 centre;
    float  area;
    float3 normal;
    uint   materialId;
};

struct ThermalMaterialGpu {
    float conductivity;      // W/(mK), 0 = non-participating
    float density;           // kg/m^3
    float specificHeat;      // J/(kgK)
    float thickness;         // m
    float convection;        // W/(m^2 K)
    float shortwaveAbsorptivity;
    float longwaveEmissivity;
    uint  interiorBcFixed;   // 0 = adiabatic, 1 = fixed temperature
    float interiorTemperature;
    float wetnessFactor;     // 0 = dry, 1 = open water
};

[[vk::binding(0, 0)]] StructuredBuffer<ThermalElementGpu>   elements;
[[vk::binding(1, 0)]] StructuredBuffer<ThermalMaterialGpu>  materials;
[[vk::binding(2, 0)]] StructuredBuffer<uint>                csrRowStart;
[[vk::binding(3, 0)]] StructuredBuffer<uint>                csrColumn;
[[vk::binding(4, 0)]] StructuredBuffer<float>               csrValue;
[[vk::binding(5, 0)]] StructuredBuffer<float>               skyFraction;
[[vk::binding(6, 0)]] StructuredBuffer<float>               sunVisTable;
[[vk::binding(7, 0)]] RWStructuredBuffer<float>             state;
[[vk::binding(8, 0)]] RWStructuredBuffer<float>             surface;
// Baked short wave: what reached the element off something else. The reflected
// table has the same K-column layout as sunVisTable and is read on the same
// indices; the diffuse gain does not depend on the sun, so it is one column.
[[vk::binding(9, 0)]] StructuredBuffer<float>               reflectedGainTable;
[[vk::binding(10, 0)]] StructuredBuffer<float>              diffuseGain;
// dT/dv per node: the tangent of the trajectory with respect to this element's
// own sun visibility, advanced by the same operator as the temperature. Same
// element-major layout as `state`. A one-float placeholder is bound when the
// host is not carrying it, and pc.carryTangent is what stops this shader
// touching it -- see ThermalState::sunSensitivity_K for what it is for.
[[vk::binding(11, 0)]] RWStructuredBuffer<float>            sensitivity;

struct StepPushConstants {
    float3 sunDirection;
    float  dt_s;
    float  airTemperature_K;
    float  sunIrradiance;
    float  skyTemperature_K;
    float  sunBlend;
    uint   elementCount;
    uint   nodeCount;
    uint   parity;
    uint   sunSampleA;
    uint   sunSampleB;
    float  diffuseIrradiance;
    float  relativeHumidity;   // percent
    float  convection_W_m2K;   // 0 means "the material's own"
    uint   hasReflectedGain;   // 0 when the table is a placeholder
    uint   carryTangent;       // 0 when binding 11 is a placeholder
};
[[vk::push_constant]] StepPushConstants pc;

static const float kStefanBoltzmann = 5.670374419e-8;

// Air at the surface, for the latent term. Mirrors the CPU constants.
static const float kAirPressure_Pa = 101325.0;
static const float kAirSpecificHeat_J_kgK = 1005.0;
static const float kLatentHeatVaporisation_J_kg = 2.45e6;

// Saturation specific humidity, kg/kg, by Magnus-Tetens. The clamp mirrors the
// CPU stepper's: the state clamp lets a diverging element reach 5000 K, where
// Magnus returns 1e10 Pa, drives the mixing-ratio denominator negative and
// hands back a NaN that then spreads down every view factor.
float SaturationHumidity(float temperature_K) {
    const float tC = clamp(temperature_K - 273.15, -80.0, 80.0);
    const float e = 610.94 * exp(17.625 * tC / (tC + 243.04));
    return 0.622 * e / (kAirPressure_Pa - 0.378 * e);
}

// The same, and its slope in temperature -- the reason the latent term is
// linearised into the matrix rather than left explicit.
void SaturationHumidityAndSlope(float temperature_K, out float q, out float dq_dT) {
    const float tC = clamp(temperature_K - 273.15, -80.0, 80.0);
    const float denominator = tC + 243.04;
    const float e = 610.94 * exp(17.625 * tC / denominator);
    const float de_dT = e * (17.625 * 243.04) / (denominator * denominator);

    const float mixed = kAirPressure_Pa - 0.378 * e;
    q = 0.622 * e / mixed;
    dq_dT = 0.622 * kAirPressure_Pa * de_dT / (mixed * mixed);
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint e = tid.x;
    if (e >= pc.elementCount) return;

    const ThermalElementGpu element = elements[e];
    const uint matId = element.materialId;
    const ThermalMaterialGpu mat = materials[matId];
    const uint nodes = pc.nodeCount;

    // Read this element's current surface temperature from the ping-pong
    // buffer (seeded before the first dispatch of the batch).
    const uint readSlot = pc.parity * pc.elementCount;
    const uint writeSlot = (1 - pc.parity) * pc.elementCount;
    const float mySurface = surface[readSlot + e];

    // Non-participating: write surface for neighbour reads, skip state.
    if (mat.conductivity <= 0.0) {
        surface[writeSlot + e] = mySurface;
        return;
    }

    const float k = mat.conductivity;
    const float rhoC = mat.density * mat.specificHeat;
    const float dx = mat.thickness / float(nodes - 1);
    if (rhoC <= 0.0 || dx <= 0.0) {
        surface[writeSlot + e] = mySurface;
        return;
    }

    const float r = (k * pc.dt_s) / (rhoC * dx * dx);
    const uint base = e * nodes;

    // Read current temperatures into thread-local array
    float T[QL_THERMAL_MAX_NODES];
    for (uint i = 0; i < nodes; ++i) {
        T[i] = state[base + i];
    }

    // ----------------------------------------------------------------
    // Exposed face
    // ----------------------------------------------------------------
    const float emissivity = mat.longwaveEmissivity;
    float surfaceFlux = 0.0;

    // Sun: interpolate visibility from the sun table
    float sunVis = 0.0;
    if (pc.sunSampleA == pc.sunSampleB) {
        sunVis = sunVisTable[pc.sunSampleA * pc.elementCount + e];
    } else {
        const float vA = sunVisTable[pc.sunSampleA * pc.elementCount + e];
        const float vB = sunVisTable[pc.sunSampleB * pc.elementCount + e];
        sunVis = vA + pc.sunBlend * (vB - vA);
    }

    // d(flux)/dv alongside the flux itself: the only term in the balance that
    // moves when this element's own sun visibility does. The reflected gain
    // below is driven by what other elements see and the diffuse by the sky,
    // so neither appears in the tangent.
    float directPerVisibility = 0.0;
    if (pc.sunIrradiance > 0.0) {
        const float cosTheta = dot(element.normal, normalize(pc.sunDirection));
        if (cosTheta > 0.0) {
            directPerVisibility = mat.shortwaveAbsorptivity * pc.sunIrradiance * cosTheta;
            surfaceFlux += directPerVisibility * sunVis;
        }
    }

    // Sun off a neighbour, read on the same indices as the visibility it was
    // baked from. A gather over the whole hemisphere, so no cos(theta) here --
    // that went into the surfaces that did the reflecting.
    if (pc.sunIrradiance > 0.0 && pc.hasReflectedGain != 0) {
        float reflected = reflectedGainTable[pc.sunSampleA * pc.elementCount + e];
        if (pc.sunSampleA != pc.sunSampleB) {
            const float rB = reflectedGainTable[pc.sunSampleB * pc.elementCount + e];
            reflected = reflected + pc.sunBlend * (rB - reflected);
        }
        surfaceFlux += mat.shortwaveAbsorptivity * pc.sunIrradiance * reflected;
    }

    // Sky, diffuse. Binding 10 is the baked gain, or the bare sky fraction
    // when nothing was baked -- the host binds one or the other, so there is
    // no fallback branch here.
    if (pc.diffuseIrradiance > 0.0) {
        surfaceFlux += mat.shortwaveAbsorptivity * pc.diffuseIrradiance * diffuseGain[e];
    }

    // Long-wave exchange
    float incoming = 0.0;
    const uint rowBegin = csrRowStart[e];
    const uint rowEnd = csrRowStart[e + 1];
    for (uint n = rowBegin; n < rowEnd; ++n) {
        const uint j = csrColumn[n];
        const float Tj = surface[readSlot + j];
        incoming += csrValue[n] * Tj * Tj * Tj * Tj;
    }
    const float Tsky = pc.skyTemperature_K;
    incoming += skyFraction[e] * Tsky * Tsky * Tsky * Tsky;

    const float Ti = mySurface;
    surfaceFlux += emissivity * kStefanBoltzmann * (incoming - Ti * Ti * Ti * Ti);

    // The forcing's coefficient wins when it has one, matching
    // CpuCrankNicolsonStepper. A constant cannot describe a day: the
    // wind and the stability of the air over the surface both reverse
    // between afternoon and midnight.
    const float h = pc.convection_W_m2K > 0.0 ? pc.convection_W_m2K
                                             : mat.convection;
    const float halfCell = rhoC * dx / (2.0 * pc.dt_s);

    // Evaporation, linearised about the previous surface temperature and split
    // half-and-half like the convection.
    float latentAdmittance = 0.0;
    float latentFlux = 0.0;
    if (mat.wetnessFactor > 0.0) {
        float qSurface;
        float dq_dT;
        SaturationHumidityAndSlope(Ti, qSurface, dq_dT);
        const float qAir = SaturationHumidity(pc.airTemperature_K);

        const float humidity = clamp(pc.relativeHumidity, 0.0, 100.0) / 100.0;
        const float coefficient = mat.wetnessFactor *
                                  (h / kAirSpecificHeat_J_kgK) *
                                  kLatentHeatVaporisation_J_kg;
        latentFlux = coefficient * (qSurface - humidity * qAir);
        latentAdmittance = coefficient * dq_dT;
    }

    // Build tridiagonal system in thread-local arrays
    float lower[QL_THERMAL_MAX_NODES];
    float diag[QL_THERMAL_MAX_NODES];
    float upper[QL_THERMAL_MAX_NODES];
    float rhs[QL_THERMAL_MAX_NODES];
    // The tangent's right-hand side. One more array rather than one more of
    // everything: it shares the matrix exactly, which is the whole reason
    // carrying dT/dv costs an elimination pass and not a second solve. The
    // current sigma is read straight from the buffer -- it is only written at
    // the end, so nothing it needs has been clobbered.
    float rhsTangent[QL_THERMAL_MAX_NODES];
    const bool carryTangent = pc.carryTangent != 0;

    // Row 0: exposed face
    lower[0] = 0.0;
    diag[0] = halfCell + 0.5 * (k / dx + h + latentAdmittance);
    upper[0] = -0.5 * (k / dx);
    rhs[0] = halfCell * T[0] - 0.5 * (k / dx) * (T[0] - T[1]) +
             0.5 * h * (2.0 * pc.airTemperature_K - T[0]) + surfaceFlux -
             latentFlux + 0.5 * latentAdmittance * T[0];

    // The same row differentiated in v. Constants of v drop (the air
    // temperature, the latent flux); the explicit long-wave loss leaves its
    // own slope; the short wave leaves the only source. The neighbours' share
    // of `incoming` is deliberately not differentiated -- see the CPU stepper
    // for why that keeps this a per-element quantity.
    if (carryTangent) {
        const float radiativeSlope = 4.0 * emissivity * kStefanBoltzmann * Ti * Ti * Ti;
        const float s0 = sensitivity[base + 0];
        const float s1 = sensitivity[base + 1];
        rhsTangent[0] = halfCell * s0 - 0.5 * (k / dx) * (s0 - s1) -
                        0.5 * h * s0 - radiativeSlope * s0 -
                        0.5 * latentAdmittance * s0 + directPerVisibility;
    }

    // Interior nodes
    for (uint i = 1; i + 1 < nodes; ++i) {
        lower[i] = -0.5 * r;
        diag[i] = 1.0 + r;
        upper[i] = -0.5 * r;
        rhs[i] = T[i] + 0.5 * r * (T[i - 1] - 2.0 * T[i] + T[i + 1]);
        if (carryTangent) {
            rhsTangent[i] = sensitivity[base + i] +
                            0.5 * r * (sensitivity[base + i - 1] -
                                       2.0 * sensitivity[base + i] +
                                       sensitivity[base + i + 1]);
        }
    }

    // Back face
    const uint last = nodes - 1;
    if (mat.interiorBcFixed != 0) {
        lower[last] = 0.0;
        diag[last] = 1.0;
        upper[last] = 0.0;
        rhs[last] = mat.interiorTemperature;
        // A room held at its own temperature does not care about the sun.
        if (carryTangent) rhsTangent[last] = 0.0;
    } else {
        lower[last] = -0.5 * (k / dx);
        diag[last] = halfCell + 0.5 * (k / dx);
        upper[last] = 0.0;
        rhs[last] = halfCell * T[last] - 0.5 * (k / dx) * (T[last] - T[last - 1]);
        if (carryTangent) {
            const float sLast = sensitivity[base + last];
            rhsTangent[last] = halfCell * sLast -
                               0.5 * (k / dx) * (sLast - sensitivity[base + last - 1]);
        }
    }

    // Thomas elimination (forward)
    for (uint i = 1; i < nodes; ++i) {
        const float factor = lower[i] / diag[i - 1];
        diag[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
        if (carryTangent) rhsTangent[i] -= factor * rhsTangent[i - 1];
    }
    // Back substitution
    rhs[last] /= diag[last];
    if (carryTangent) rhsTangent[last] /= diag[last];
    for (uint i = last; i-- > 0;) {
        rhs[i] = (rhs[i] - upper[i] * rhs[i + 1]) / diag[i];
        if (carryTangent) {
            rhsTangent[i] = (rhsTangent[i] - upper[i] * rhsTangent[i + 1]) / diag[i];
        }
    }

    // Write solved temperatures, clamped
    for (uint i = 0; i < nodes; ++i) {
        state[base + i] = clamp(rhs[i], 1.0, 5000.0);
    }
    // And the tangent, on its own clamp: a sensitivity is a derivative, not a
    // temperature, so [1, 5000] would be meaningless for it.
    if (carryTangent) {
        for (uint i = 0; i < nodes; ++i) {
            sensitivity[base + i] = clamp(rhsTangent[i], -1000.0, 1000.0);
        }
    }

    // Write new surface to ping-pong buffer for next step's radiative read
    surface[writeSlot + e] = clamp(rhs[0], 1.0, 5000.0);
}
