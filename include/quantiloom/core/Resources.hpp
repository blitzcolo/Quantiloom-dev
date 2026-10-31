/**
 * @file Resources.hpp
 * @brief Where the SDK looks for the files it needs at runtime
 *
 * The installed SDK keeps its shaders and assets under
 * `<prefix>/share/quantiloom`, not beside the binaries, so a host that
 * relocates the library relocates its resources with it. The lookup order is
 *
 *   1. SetResourceRoot() -- exclusive: the only root that is tried
 *   2. QUANTILOOM_RESOURCE_ROOT -- exclusive likewise
 *   3. <directory of the loaded library>/../share/quantiloom
 *   4. <directory of the loaded library>/share/quantiloom
 *   5. the working-directory development fallback (src/shaders, shaders,
 *      ../shaders, plus assets/), used only when 1-4 found nothing
 *
 * Candidates 1-4 are accepted only when their `shaders/spirv/SHADERSET`
 * stamp equals the digest of the shader set this library was built with, so
 * a stale copy refuses to run instead of quietly misrendering.
 */

#pragma once

#include "core/Platform.hpp"
#include "core/Types.hpp"

namespace quantiloom {

/// Pin the resource root. Exclusive: when set, no other root is tried, so a
/// stale copy elsewhere cannot be picked up silently. An empty string clears
/// the override. Call before creating a render context; a later call takes
/// effect for the lookups made after it.
QL_API void SetResourceRoot(const String& root);

/// The accepted resource root, or an empty string when the working-directory
/// development fallback is what resolved. Hosts use it to find sibling data
/// such as `assets/atmos_models`.
QL_API String GetResourceRoot();

}  // namespace quantiloom
