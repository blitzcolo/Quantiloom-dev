// ============================================================================
// VMA Implementation File
// ============================================================================
// VMA (Vulkan Memory Allocator) is a header-only library, but requires
// exactly ONE .cpp file to define VMA_IMPLEMENTATION to generate the
// function implementations.
//
// This file serves that purpose. Do NOT define VMA_IMPLEMENTATION anywhere else.
// ============================================================================

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#include <vk_mem_alloc.h>
