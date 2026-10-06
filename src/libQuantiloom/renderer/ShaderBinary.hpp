/**
 * @file ShaderBinary.hpp
 * @brief Locating and loading a .spv that ships beside the executable
 *
 * Internal pipelines (camera, display range, thermal stepper/exchange, the
 * pick and CLAHE compute helpers) all find their shaders by the same search:
 * the bare name, beside the executable, then shaders/ under each of those,
 * then the repo-layout fallbacks -- installed SDKs put the .spv next to the
 * binary, a build tree leaves it in src/shaders. That list lives here once;
 * this file is the reason four copies of it do not have to be kept in step.
 */

#pragma once

#include "core/Types.hpp"
#include <filesystem>

namespace quantiloom::rendercore {

/// Directory the running executable lives in, per platform.
std::filesystem::path ShaderExecutableDir();

/// Every candidate path for @p name, in search order. LoadSpirv() returns the
/// first that holds a plausible shader; a caller that must find several
/// sibling shaders in ONE directory (the CLAHE triple) iterates this list
/// itself and picks the first directory that has them all.
Vector<std::filesystem::path> SpirvSearchPaths(StringView name);

/// First readable candidate for @p name as SPIR-V words, or empty. Files
/// whose size is zero or not a whole number of words are skipped -- handing
/// one to vkCreateShaderModule is a validation error, not a shader.
Vector<u32> LoadSpirv(StringView name);

} // namespace quantiloom::rendercore
