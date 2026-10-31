/**
 * @file ShaderBinary.hpp
 * @brief Locating and loading a .spv from the SDK's resolved resource root
 *
 * Internal pipelines (camera, display range, thermal stepper/exchange, the
 * pick and CLAHE compute helpers) all find their shaders through the one
 * resource locator in core/ResourceLocator.hpp: an installed SDK reads
 * `<root>/shaders/spirv`, a build tree falls back to the working-directory
 * development candidates. There is exactly one accepted root, so siblings
 * that must come from the same build (the CLAHE triple) cannot be mixed.
 */

#pragma once

#include "core/Types.hpp"
#include <filesystem>

namespace quantiloom::rendercore {

/// Every candidate path for @p name, in search order. LoadSpirv() returns the
/// first that holds a plausible shader; a caller that must find several
/// sibling shaders in ONE directory (the CLAHE triple) iterates this list
/// itself and picks the first directory that has them all.
Vector<std::filesystem::path> SpirvSearchPaths(StringView name);

/// Read one SPIR-V file, or empty when it is missing, empty, not a whole
/// number of words, or only partially readable -- handing any of those to
/// vkCreateShaderModule is a validation error, not a shader.
Vector<u32> ReadSpirvFile(const std::filesystem::path& path);

/// First readable candidate for @p name as SPIR-V words, or empty.
Vector<u32> LoadSpirv(StringView name);

} // namespace quantiloom::rendercore
