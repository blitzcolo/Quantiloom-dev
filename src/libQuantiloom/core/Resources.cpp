#include "core/Resources.hpp"

#include "core/ResourceLocator.hpp"

namespace quantiloom {

void SetResourceRoot(const String& root) {
    core::SetResourceRootOverride(root.empty() ? std::filesystem::path()
                                               : std::filesystem::path(root));
}

String GetResourceRoot() {
    return core::GetResolvedResources().root.string();
}

}  // namespace quantiloom
