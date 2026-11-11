// ============================================================================
// VMA Implementation File
// ============================================================================
// VMA (Vulkan Memory Allocator) is a header-only library, but requires
// exactly ONE .cpp file to define VMA_IMPLEMENTATION to generate the
// function implementations.
//
// This file serves that purpose. Do NOT define VMA_IMPLEMENTATION anywhere else.
// ============================================================================

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4100) // unreferenced parameter
#pragma warning(disable: 4189) // local variable initialized but not referenced
#pragma warning(disable: 4324) // structure was padded
#endif

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif
