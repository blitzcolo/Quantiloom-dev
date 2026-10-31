// ============================================================================
// Quantiloom - binding constructs with no portable HLSL spelling
// ============================================================================
// Resources are declared in plain HLSL -- `register(<class><binding>, space<set>)`
// -- and that one annotation serves every target:
//
//   - DXC's SPIR-V backend, given no [[vk::binding]], takes the register number
//     as the Vulkan binding and the space as the descriptor set. A Vulkan binding
//     number is unique within its set whatever the descriptor type, so t3, u5 and
//     s7 can never land on one binding. The .spv are byte-identical to the
//     [[vk::binding]] spelling this replaced, except that a stage which also
//     declares an unbounded array (below) lists its DescriptorSet/Binding
//     decorations in a different order -- same ids, same numbers.
//   - DXIL reads the register as written, and Metal Shader Converter lays out its
//     argument buffer from the DXIL that results. So the Vulkan layout, the DXIL
//     root signature and the Metal layout all number a resource the same way.
//
// Three things still differ by target, and only these go through a macro:
//
//   QL_PUSH_CONSTANTS(Type, name)
//       SPIR-V: a [[vk::push_constant]] block. Elsewhere: a ConstantBuffer in
//       space 1, register b0 -- one per pipeline, as Vulkan allows one push
//       constant block. Members read the same either way: `name.field`.
//
//   QL_SPEC_CONSTANT(id, Type, name, value)
//       SPIR-V: a specialisation constant with that default. Elsewhere there is
//       no specialisation, so it is a compile-time constant: a host that needs
//       another value compiles a variant with -D on the value's macro.
//
//   QL_VK_BINDING(slot, set)
//       Only for an unbounded array. DXIL gives one every register from its base
//       upward, so it cannot share a space with numbered bindings and is declared
//       in space 1 at register 0; this macro keeps its Vulkan slot. Nothing else
//       uses it -- a resource with a plain register needs no second annotation.
//
// DXC defines __spirv__ when it is given -spirv.
// ============================================================================

#ifndef QUANTILOOM_QL_BIND_HLSLI
#define QUANTILOOM_QL_BIND_HLSLI

#if defined(__spirv__)
    #define QL_VK_BINDING(slot, set)                [[vk::binding(slot, set)]]
    #define QL_PUSH_CONSTANTS(Type, name)           [[vk::push_constant]] Type name
    #define QL_SPEC_CONSTANT(id, Type, name, value) [[vk::constant_id(id)]] const Type name = value
#else
    #define QL_VK_BINDING(slot, set)
    #define QL_PUSH_CONSTANTS(Type, name)           ConstantBuffer<Type> name : register(b0, space1)
    #define QL_SPEC_CONSTANT(id, Type, name, value) static const Type name = value

    // The root signature: the binding contract a DXIL consumer reads, and the
    // one Metal Shader Converter lays its top-level argument buffer out from.
    // Tables mirror the declarations above -- every numbered binding of Vulkan
    // set 0 by register class, the two unbounded arrays in space 1, the push
    // constants at b0 space 1 -- so MSC numbers a resource the way Vulkan does.
    // MSC requires one for ray tracing stages and for unbounded arrays, and
    // checks that it covers every resource a shader declares: a binding number
    // past these ranges fails the build's portability step until the range here
    // grows with it.
    //
    // Ray tracing libraries carry it as a subobject. An inline-ray-query kernel
    // takes it through DXC's -rootsig-define QL_ROOT_SIGNATURE, which the
    // portability step passes for every .rayq.hlsl. A plain compute kernel has
    // neither and gets MSC's linear layout.
    #define QL_ROOT_SIGNATURE "DescriptorTable(SRV(t0, numDescriptors=38), UAV(u0, numDescriptors=37)), DescriptorTable(Sampler(s0, numDescriptors=22)), DescriptorTable(SRV(t0, space=1, numDescriptors=unbounded)), DescriptorTable(Sampler(s0, space=1, numDescriptors=unbounded)), CBV(b0, space=1)"

    #if __SHADER_TARGET_STAGE == __SHADER_STAGE_LIBRARY
    GlobalRootSignature qlGlobalRootSignature = { QL_ROOT_SIGNATURE };
    #endif
#endif

#endif // QUANTILOOM_QL_BIND_HLSLI
