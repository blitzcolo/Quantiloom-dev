#pragma once
#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <sddl.h>
#    include <string>
#    include <vector>
#    include <stdexcept>

namespace quantiloom::dataset::detail {
// Atomically apply owner-only access and the host's mandatory integrity level.
// Same-user code at the same integrity level can already control this process.
class PrivateDirectorySecurity {
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), nullptr, FALSE};

public:
    PrivateDirectorySecurity() {
        HANDLE token = nullptr;
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token))
            throw std::runtime_error("cannot read process security token");
        struct TokenGuard {
            HANDLE value;
            ~TokenGuard() {
                CloseHandle(value);
            }
        } tokenGuard{token};
        const auto tokenInfo = [&](TOKEN_INFORMATION_CLASS kind) {
            DWORD length = 0;
            GetTokenInformation(token, kind, nullptr, 0, &length);
            std::vector<unsigned char> buffer(length);
            if (!length || !GetTokenInformation(token, kind, buffer.data(), length, &length))
                throw std::runtime_error("cannot read process security identity");
            return buffer;
        };
        const auto user = tokenInfo(TokenUser), integrity = tokenInfo(TokenIntegrityLevel);
        const auto sidText = [](PSID sid) {
            LPWSTR text = nullptr;
            if (!ConvertSidToStringSidW(sid, &text))
                throw std::runtime_error("cannot format security SID");
            struct Guard {
                LPWSTR value;
                ~Guard() {
                    LocalFree(value);
                }
            } guard{text};
            return std::wstring(text);
        };
        const auto owner = sidText(reinterpret_cast<const TOKEN_USER*>(user.data())->User.Sid);
        const auto label =
            sidText(reinterpret_cast<const TOKEN_MANDATORY_LABEL*>(integrity.data())->Label.Sid);
        const auto sddl =
            L"O:" + owner + L"D:P(A;OICI;FA;;;" + owner + L")S:(ML;OICI;NW;;;" + label + L")";
        if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(), SDDL_REVISION_1,
                                                                  &descriptor, nullptr))
            throw std::runtime_error("cannot build private directory security descriptor");
        attributes.lpSecurityDescriptor = descriptor;
    }
    PrivateDirectorySecurity(const PrivateDirectorySecurity&) = delete;
    PrivateDirectorySecurity& operator=(const PrivateDirectorySecurity&) = delete;
    ~PrivateDirectorySecurity() {
        if (descriptor)
            LocalFree(descriptor);
    }
    SECURITY_ATTRIBUTES& Attributes() {
        return attributes;
    }
};
} // namespace quantiloom::dataset::detail
#endif
