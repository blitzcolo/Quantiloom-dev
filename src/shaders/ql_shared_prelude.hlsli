// ============================================================================
// Quantiloom - prelude of the structs the host and the shaders share
// ============================================================================
// A struct that a shader reads out of a buffer or a push-constant block, and
// that the host fills, is written once, in a ql_shared_*.hlsli header which DXC
// (HLSL 2021, its default: the shader build passes no -HV) and MSVC (C++20)
// both compile:
//
//   - HLSL sees the struct exactly as it was written before it moved here.
//     Every macro below expands to nothing on that side, so a shader's
//     preprocessed tokens -- and with them its .spv and its DXIL -- are the ones
//     the hand-written struct gave.
//   - C++ sees it in namespace quantiloom::shader, with float3 = glm::vec3 and
//     the other aliases below, and static_asserts its size and the offset of
//     every member. A host file includes the header by a path relative to itself
//     (as renderer/RtBindings.hpp includes ql_rt_bindings.hlsli) and names the
//     struct through an alias, keeping its own name where that differed:
//         using DisplayRangePush = shader::RangePush;
//
// Names are the HLSL ones. DXC writes OpName/OpMemberName into the SPIR-V, so
// renaming a struct or a member changes the .spv even when no code changes.
//
// Layout rules, under which SPIR-V (std430 with relaxed vectors), DXIL
// (structured buffers packed tight; push constants in a cbuffer off SPIR-V, see
// QL_PUSH_CONSTANTS in ql_bind.hlsli) and C++ (glm::vec3 is 12 bytes, 4-aligned)
// put every member at the same offset:
//
//   - a float3 starts on a 16-byte boundary and is followed by a 4-byte scalar;
//   - no bool (4 bytes in an HLSL buffer, 1 in C++) and no matrix;
//   - no array in a push-constant struct (a cbuffer pads each element to 16
//     bytes); a nested struct in one starts on 16 bytes.
//
// QL_SHARED_LAYOUT and QL_SHARED_OFFSET hold the C++ side to the offsets those
// rules give, so a member that moves fails the C++ build.
//
// Asking for a struct. In HLSL a shared header emits only the structs its
// includer asked for, each at most once per shader, by a WANT macro defined
// just before the include, where the struct is needed:
//
//     #define QL_SHARED_WANT_ThermalElementGpu
//     #include "ql_shared_thermal.hlsli"
//
// so a shader's token stream keeps the order it had when the struct text stood
// there, and gains no struct it does not use. C++ always gets every struct of
// the header. Each struct sits in a section of its own:
//
//     #if (defined(__cplusplus) || defined(QL_SHARED_WANT_<Name>)) && !defined(QL_SHARED_HAS_<Name>)
//     #define QL_SHARED_HAS_<Name>
//     QL_SHARED_BEGIN
//     struct <Name> { ... };
//     QL_SHARED_LAYOUT(<Name>, <size>)
//     QL_SHARED_OFFSET(<Name>, <member>, <offset>)    one per member
//     QL_SHARED_END
//     #endif
//
// A host default member initialiser rides in QL_DEFAULT(...): the braced
// initialiser in C++ and nothing in HLSL, so `uint channelCount QL_DEFAULT(1);`
// is `u32 channelCount{1};` on the host and `uint channelCount;` in a shader.
// The assertion macros carry their own semicolon and are written without one,
// so HLSL never sees an empty declaration.
//
// DXC never defines __cplusplus for HLSL; MSVC defines it, and this build passes
// /Zc:__cplusplus so it carries the real value.
// ============================================================================

#ifndef QUANTILOOM_QL_SHARED_PRELUDE_HLSLI
#define QUANTILOOM_QL_SHARED_PRELUDE_HLSLI

#if defined(__cplusplus)
    #include "core/Types.hpp"
    #include <glm/glm.hpp>
    #include <cstddef>
    #include <type_traits>

    namespace quantiloom::shader {
    using float2 = glm::vec2;
    using float3 = glm::vec3;
    using float4 = glm::vec4;
    using uint = ::quantiloom::u32;
    using int2 = glm::ivec2;
    using uint2 = glm::uvec2;
    using uint4 = glm::uvec4;
    }  // namespace quantiloom::shader

    #define QL_SHARED_BEGIN namespace quantiloom::shader {
    #define QL_SHARED_END   }
    #define QL_SHARED_LAYOUT(T, n) \
        static_assert(sizeof(T) == (n) && std::is_trivially_copyable_v<T>, #T " size");
    #define QL_SHARED_OFFSET(T, f, n) static_assert(offsetof(T, f) == (n), #T "::" #f " offset");
    #define QL_DEFAULT(...) {__VA_ARGS__}
#else
    #define QL_SHARED_BEGIN
    #define QL_SHARED_END
    #define QL_SHARED_LAYOUT(T, n)
    #define QL_SHARED_OFFSET(T, f, n)
    #define QL_DEFAULT(...)
#endif

#endif // QUANTILOOM_QL_SHARED_PRELUDE_HLSLI
