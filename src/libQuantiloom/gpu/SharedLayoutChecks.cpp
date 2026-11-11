/**
 * @file SharedLayoutChecks.cpp
 * @brief The C++ half of the public structs' layout locks
 *
 * Six structs the shaders read are public API, so their C++ definitions stay in
 * include/quantiloom/ -- the only thing the SDK installs -- while the HLSL keeps
 * its own in src/shaders/ql_shared_public.hlsli. That header's C++ branch asserts
 * the size of each public struct and the offset of every one of its members,
 * beside the HLSL definition they must agree with. This file is the one place it
 * is compiled, so no public header includes it. It defines nothing.
 */

// By a path relative to this file, as renderer/RtBindings.hpp includes its manifest.
#include "../../shaders/ql_shared_public.hlsli"
