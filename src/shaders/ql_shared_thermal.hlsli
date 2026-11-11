// ============================================================================
// Quantiloom - structs shared by the thermal kernels and their hosts
// ============================================================================
// thermal_exchange.rayq.hlsl (who sees whom) and thermal_step.comp.hlsl (one
// Crank-Nicolson step) read these; renderer/ThermalExchangePrecompute.cpp and
// renderer/GpuThermalStepper.cpp fill them. Each is defined here once, for both
// languages. ql_shared_prelude.hlsli has the layout rules and how a shader asks
// for one struct (QL_SHARED_WANT_<Name>).
//
// No include guard: a shader includes this once per struct it asks for, and
// every section guards itself (QL_SHARED_HAS_<Name>).
// ============================================================================

#include "ql_shared_prelude.hlsli"

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_ThermalElementGpu)) && !defined(QL_SHARED_HAS_ThermalElementGpu)
#define QL_SHARED_HAS_ThermalElementGpu
QL_SHARED_BEGIN
// One per element, in the order BuildThermalMesh produced them. Both kernels
// read the same array.
struct ThermalElementGpu {
    // `centre`, not `centroid`: the latter is an HLSL interpolation modifier
    // and a member by that name does not parse.
    float3 centre;
    float  area;
    float3 normal;
    uint   materialId;
};
QL_SHARED_LAYOUT(ThermalElementGpu, 32)
QL_SHARED_OFFSET(ThermalElementGpu, centre, 0)
QL_SHARED_OFFSET(ThermalElementGpu, area, 12)
QL_SHARED_OFFSET(ThermalElementGpu, normal, 16)
QL_SHARED_OFFSET(ThermalElementGpu, materialId, 28)
QL_SHARED_END
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_ThermalMaterialGpu)) && !defined(QL_SHARED_HAS_ThermalMaterialGpu)
#define QL_SHARED_HAS_ThermalMaterialGpu
QL_SHARED_BEGIN
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
QL_SHARED_LAYOUT(ThermalMaterialGpu, 40)
QL_SHARED_OFFSET(ThermalMaterialGpu, conductivity, 0)
QL_SHARED_OFFSET(ThermalMaterialGpu, density, 4)
QL_SHARED_OFFSET(ThermalMaterialGpu, specificHeat, 8)
QL_SHARED_OFFSET(ThermalMaterialGpu, thickness, 12)
QL_SHARED_OFFSET(ThermalMaterialGpu, convection, 16)
QL_SHARED_OFFSET(ThermalMaterialGpu, shortwaveAbsorptivity, 20)
QL_SHARED_OFFSET(ThermalMaterialGpu, longwaveEmissivity, 24)
QL_SHARED_OFFSET(ThermalMaterialGpu, interiorBcFixed, 28)
QL_SHARED_OFFSET(ThermalMaterialGpu, interiorTemperature, 32)
QL_SHARED_OFFSET(ThermalMaterialGpu, wetnessFactor, 36)
QL_SHARED_END
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_StepPushConstants)) && !defined(QL_SHARED_HAS_StepPushConstants)
#define QL_SHARED_HAS_StepPushConstants
QL_SHARED_BEGIN
// thermal_step.comp.hlsl's push constants.
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
    // The forcing's convective coefficient, or 0 to mean the material's own.
    // Mirrors ThermalForcing::convection_W_m2K so that a trajectory stepped on
    // the GPU matches one stepped on the CPU; test_thermal_step_gpu compares
    // them over 24 h and would otherwise start disagreeing the moment a forcing
    // file carried a ninth column.
    float  convection_W_m2K;   // 0 means "the material's own"
    uint   hasReflectedGain;   // 0 when the table is a placeholder
    uint   carryTangent;       // 0 when binding 11 is a placeholder
};
QL_SHARED_LAYOUT(StepPushConstants, 72)
QL_SHARED_OFFSET(StepPushConstants, sunDirection, 0)
QL_SHARED_OFFSET(StepPushConstants, dt_s, 12)
QL_SHARED_OFFSET(StepPushConstants, airTemperature_K, 16)
QL_SHARED_OFFSET(StepPushConstants, sunIrradiance, 20)
QL_SHARED_OFFSET(StepPushConstants, skyTemperature_K, 24)
QL_SHARED_OFFSET(StepPushConstants, sunBlend, 28)
QL_SHARED_OFFSET(StepPushConstants, elementCount, 32)
QL_SHARED_OFFSET(StepPushConstants, nodeCount, 36)
QL_SHARED_OFFSET(StepPushConstants, parity, 40)
QL_SHARED_OFFSET(StepPushConstants, sunSampleA, 44)
QL_SHARED_OFFSET(StepPushConstants, sunSampleB, 48)
QL_SHARED_OFFSET(StepPushConstants, diffuseIrradiance, 52)
QL_SHARED_OFFSET(StepPushConstants, relativeHumidity, 56)
QL_SHARED_OFFSET(StepPushConstants, convection_W_m2K, 60)
QL_SHARED_OFFSET(StepPushConstants, hasReflectedGain, 64)
QL_SHARED_OFFSET(StepPushConstants, carryTangent, 68)
QL_SHARED_END
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_ExchangePushConstants)) && !defined(QL_SHARED_HAS_ExchangePushConstants)
#define QL_SHARED_HAS_ExchangePushConstants
QL_SHARED_BEGIN
// thermal_exchange.rayq.hlsl's push constants.
struct ExchangePushConstants {
    float3 sunDirection;   // from surface toward the sun, normalised
    uint   elementCount;
    uint   rayCount;       // hemisphere rays per element; 0 = sun-only pass
    uint   sunRayCount;    // rays toward the sun, for a soft shadow edge
    float  sunAngularRadius;  // radians; 0.00465 is the real sun
    float  rayOffset;      // floor of how far off the surface a ray starts
    uint   sunOutputOffset;   // element offset into sunVisibility for this direction
};
QL_SHARED_LAYOUT(ExchangePushConstants, 36)
QL_SHARED_OFFSET(ExchangePushConstants, sunDirection, 0)
QL_SHARED_OFFSET(ExchangePushConstants, elementCount, 12)
QL_SHARED_OFFSET(ExchangePushConstants, rayCount, 16)
QL_SHARED_OFFSET(ExchangePushConstants, sunRayCount, 20)
QL_SHARED_OFFSET(ExchangePushConstants, sunAngularRadius, 24)
QL_SHARED_OFFSET(ExchangePushConstants, rayOffset, 28)
QL_SHARED_OFFSET(ExchangePushConstants, sunOutputOffset, 32)
QL_SHARED_END
#endif
