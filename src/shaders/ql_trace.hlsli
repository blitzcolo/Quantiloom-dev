#ifndef QUANTILOOM_QL_TRACE_HLSLI
#define QUANTILOOM_QL_TRACE_HLSLI

#include "shade_types.hlsli"

// The acceleration structure every ray in this pipeline is traced against.
// Kept here, and not in a shared resource header, so that a stage which never
// traces (the sky evaluator) does not acquire a binding it must then satisfy.
RaytracingAccelerationStructure scene : register(t1, space0);

#endif // QUANTILOOM_QL_TRACE_HLSLI
