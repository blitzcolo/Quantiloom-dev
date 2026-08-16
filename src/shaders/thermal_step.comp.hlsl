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
    float pad;
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
    uint   pad0;
    uint   pad1;
    uint   pad2;
};
[[vk::push_constant]] StepPushConstants pc;

static const float kStefanBoltzmann = 5.670374419e-8;

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

    if (pc.sunIrradiance > 0.0) {
        const float cosTheta = dot(element.normal, normalize(pc.sunDirection));
        if (cosTheta > 0.0) {
            surfaceFlux += mat.shortwaveAbsorptivity * pc.sunIrradiance *
                           cosTheta * sunVis;
        }
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

    const float h = mat.convection;
    const float halfCell = rhoC * dx / (2.0 * pc.dt_s);

    // Build tridiagonal system in thread-local arrays
    float lower[QL_THERMAL_MAX_NODES];
    float diag[QL_THERMAL_MAX_NODES];
    float upper[QL_THERMAL_MAX_NODES];
    float rhs[QL_THERMAL_MAX_NODES];

    // Row 0: exposed face
    lower[0] = 0.0;
    diag[0] = halfCell + 0.5 * (k / dx + h);
    upper[0] = -0.5 * (k / dx);
    rhs[0] = halfCell * T[0] - 0.5 * (k / dx) * (T[0] - T[1]) +
             0.5 * h * (2.0 * pc.airTemperature_K - T[0]) + surfaceFlux;

    // Interior nodes
    for (uint i = 1; i + 1 < nodes; ++i) {
        lower[i] = -0.5 * r;
        diag[i] = 1.0 + r;
        upper[i] = -0.5 * r;
        rhs[i] = T[i] + 0.5 * r * (T[i - 1] - 2.0 * T[i] + T[i + 1]);
    }

    // Back face
    const uint last = nodes - 1;
    if (mat.interiorBcFixed != 0) {
        lower[last] = 0.0;
        diag[last] = 1.0;
        upper[last] = 0.0;
        rhs[last] = mat.interiorTemperature;
    } else {
        lower[last] = -0.5 * (k / dx);
        diag[last] = halfCell + 0.5 * (k / dx);
        upper[last] = 0.0;
        rhs[last] = halfCell * T[last] - 0.5 * (k / dx) * (T[last] - T[last - 1]);
    }

    // Thomas elimination (forward)
    for (uint i = 1; i < nodes; ++i) {
        const float factor = lower[i] / diag[i - 1];
        diag[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }
    // Back substitution
    rhs[last] /= diag[last];
    for (uint i = last; i-- > 0;) {
        rhs[i] = (rhs[i] - upper[i] * rhs[i + 1]) / diag[i];
    }

    // Write solved temperatures, clamped
    for (uint i = 0; i < nodes; ++i) {
        state[base + i] = clamp(rhs[i], 1.0, 5000.0);
    }

    // Write new surface to ping-pong buffer for next step's radiative read
    surface[writeSlot + e] = clamp(rhs[0], 1.0, 5000.0);
}
