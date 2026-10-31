/**
 * @file Kernel.hpp
 * @brief Kernel layouts and compute kernels of the compute RHI (internal)
 */

#pragma once

#include "gpu/Binding.hpp"

#include <span>

namespace quantiloom::gpu {

struct KernelLayoutDesc {
    std::span<const BindingLayout* const> spaces;  // index == space; contiguous from 0
    u32 pushConstantBytes = 0;                      // 0 = none
};

class KernelLayout {
public:
    virtual ~KernelLayout() = default;
    [[nodiscard]] virtual u32 PushConstantBytes() const = 0;
};

/// QL_SPEC_CONSTANT (ql_bind.hlsli). On Metal a value that differs from the default
/// selects a -D variant by name.
struct SpecializationConstant {
    u32 id = 0;
    u32 value = 0;
};

struct ShaderCode;  // ShaderLibrary.hpp

struct ComputeKernelDesc {
    const ShaderCode* code = nullptr;
    const KernelLayout* layout = nullptr;
    std::span<const SpecializationConstant> specialization = {};
};

class ComputeKernel {
public:
    virtual ~ComputeKernel() = default;
    [[nodiscard]] virtual const KernelLayout& Layout() const = 0;
    [[nodiscard]] virtual StringView Name() const = 0;
};

}  // namespace quantiloom::gpu
