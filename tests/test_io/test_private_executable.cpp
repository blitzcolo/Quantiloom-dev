#ifdef _WIN32
#include <gtest/gtest.h>
#include "../../src/tools/QLTrans/PrivateExecutable.hpp"
#include <filesystem>
#include <fstream>
#include <aclapi.h>

TEST(PrivateExecutableTest, UsesUniqueProtectedDirectoriesAndPreventsReplacement) {
    const char bytes[] = "embedded test image";
    std::filesystem::path firstPath;
    {
        qltrans::detail::PrivateExecutable first(bytes, sizeof(bytes));
        qltrans::detail::PrivateExecutable second(bytes, sizeof(bytes));
        firstPath = first.Path();
        EXPECT_NE(first.Path().parent_path(), second.Path().parent_path());
        PSECURITY_DESCRIPTOR security = nullptr;
        PACL acl = nullptr;
        ASSERT_EQ(GetNamedSecurityInfoW(const_cast<wchar_t*>(first.Path().parent_path().c_str()),
            SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, &acl, nullptr, &security), ERROR_SUCCESS);
        struct SecurityGuard { PSECURITY_DESCRIPTOR value; ~SecurityGuard() { LocalFree(value); } } guard{security};
        SECURITY_DESCRIPTOR_CONTROL control{};
        DWORD revision = 0;
        ASSERT_TRUE(GetSecurityDescriptorControl(security, &control, &revision));
        EXPECT_NE(control & SE_DACL_PROTECTED, 0);
        ASSERT_NE(acl, nullptr);
        EXPECT_EQ(acl->AceCount, 1u);
        EXPECT_FALSE(DeleteFileW(first.Path().c_str()));
        EXPECT_FALSE(MoveFileExW(first.Path().parent_path().c_str(),
            (first.Path().parent_path().wstring() + L"_replaced").c_str(), 0));
        std::ifstream input(first.Path(), std::ios::binary);
        const std::string contents{std::istreambuf_iterator<char>(input), {}};
        EXPECT_EQ(contents, std::string(bytes, sizeof(bytes)));
        qltrans::detail::Handle writer;
        writer.value = CreateFileW(first.Path().c_str(), GENERIC_WRITE, FILE_SHARE_READ,
            nullptr, OPEN_EXISTING, 0, nullptr);
        EXPECT_EQ(writer.value, INVALID_HANDLE_VALUE);
    }
    EXPECT_FALSE(std::filesystem::exists(firstPath));
    EXPECT_FALSE(std::filesystem::exists(firstPath.parent_path()));
}

TEST(PrivateExecutableTest, ExtractedExecutableLaunchesWithReadLeaseHeld) {
    std::vector<wchar_t> systemDirectory(32768);
    const auto length = GetSystemDirectoryW(systemDirectory.data(), static_cast<UINT>(systemDirectory.size()));
    ASSERT_GT(length, 0u);
    ASSERT_LT(length, systemDirectory.size());
    std::ifstream source(std::filesystem::path(systemDirectory.data()) / L"cmd.exe", std::ios::binary);
    ASSERT_TRUE(source);
    const std::string bytes{std::istreambuf_iterator<char>(source), {}};
    ASSERT_FALSE(bytes.empty());
    qltrans::detail::PrivateExecutable file(bytes.data(), static_cast<DWORD>(bytes.size()));
    std::wstring command = L"\"" + file.Path().wstring() + L"\" /c exit 0";
    STARTUPINFOW startup{}; startup.cb = sizeof(startup);
    PROCESS_INFORMATION launched{};
    ASSERT_TRUE(CreateProcessW(file.Path().c_str(), command.data(), nullptr, nullptr,
        FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &launched));
    qltrans::detail::Handle process, thread;
    process.value = launched.hProcess;
    thread.value = launched.hThread;
    ASSERT_EQ(WaitForSingleObject(process.value, 30000), WAIT_OBJECT_0);
    DWORD exitCode = 1;
    ASSERT_TRUE(GetExitCodeProcess(process.value, &exitCode));
    EXPECT_EQ(exitCode, 0u);
}

TEST(PrivateExecutableTest, LaunchFailureCleansUpExtractedBytes) {
    std::filesystem::path extracted;
    {
        const char invalid[] = "not a PE executable";
        qltrans::detail::PrivateExecutable file(invalid, sizeof(invalid));
        extracted = file.Path();
        STARTUPINFOW startup{}; startup.cb = sizeof(startup);
        PROCESS_INFORMATION process{};
        EXPECT_FALSE(CreateProcessW(extracted.c_str(), nullptr, nullptr, nullptr,
            FALSE, 0, nullptr, nullptr, &startup, &process));
    }
    EXPECT_FALSE(std::filesystem::exists(extracted.parent_path()));
}
#endif
