/**
 * @file Platform.hpp
 * @brief DLL export/import macros for libSpectraForge
 */

#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #if defined(SF_BUILD_SHARED)
        #define SF_API __declspec(dllexport)
    #elif defined(SF_USE_SHARED)
        #define SF_API __declspec(dllimport)
    #else
        #define SF_API
    #endif
#else
    #if defined(SF_BUILD_SHARED)
        #define SF_API __attribute__((visibility("default")))
    #else
        #define SF_API
    #endif
#endif
