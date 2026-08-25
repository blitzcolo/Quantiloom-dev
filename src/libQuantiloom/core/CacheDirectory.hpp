/**
 * @file CacheDirectory.hpp
 * @brief Where this machine keeps Quantiloom's caches
 *
 * Internal. Lived as a file-static in ExternalRenderContext.cpp for the Vulkan
 * pipeline cache; hoisted here when the thermal solve cache needed the same
 * answer, because two implementations of "the cache directory" is how a user
 * ends up with two of them.
 */

#pragma once

#include "core/Types.hpp"

namespace quantiloom::core {

/**
 * @brief The platform's cache directory for Quantiloom, created if absent
 *
 *   Windows: %LOCALAPPDATA%/Quantiloom/cache
 *   macOS:   ~/Library/Caches/Quantiloom
 *   Linux:   $XDG_CACHE_HOME/Quantiloom, else ~/.cache/Quantiloom
 *
 * Falls back to the system temp directory when the home location cannot be
 * determined, and to "." when even that cannot be created -- a cache is an
 * optimisation, so failing to place one is a warning rather than an error.
 */
[[nodiscard]] String GetDefaultCacheDirectory();

}  // namespace quantiloom::core
