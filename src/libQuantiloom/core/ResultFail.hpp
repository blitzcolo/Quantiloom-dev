/**
 * @file ResultFail.hpp
 * @brief Fail<T>(message) -- the one-line Err every module grew its own copy of
 *
 * Result<void, E> is a specialization whose Err is a static factory, not a
 * nested type, so `typename Result<T, E>::Err` does not compile for void --
 * which is why the per-file copies of this helper drifted apart in signature
 * (const char*, const String&, String) and in whether they could fail a void
 * call at all. One definition, one signature.
 */

#pragma once

#include "core/Types.hpp"
#include <type_traits>

namespace quantiloom {

template<class T>
Result<T, String> Fail(const String& message) {
    if constexpr (std::is_void_v<T>) {
        return Result<T, String>::Err(message);
    } else {
        return typename Result<T, String>::Err(message);
    }
}

} // namespace quantiloom
