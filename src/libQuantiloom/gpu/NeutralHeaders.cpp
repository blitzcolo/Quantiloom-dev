// Compile-time check that no gpu/ header drags in a backend header. Built without the
// precompiled header (quantiloom_core's PCH includes <vulkan/vulkan.h>), see CMakeLists.txt.
// Each migration step adds the migrated module's header below.

#include "gpu/Binding.hpp"
#include "gpu/CommandList.hpp"
#include "gpu/Device.hpp"
#include "gpu/GpuTypes.hpp"
#include "gpu/Kernel.hpp"
#include "gpu/Resources.hpp"
#include "gpu/ShaderLibrary.hpp"
#include "hs_core/GpuSpectralReconstructor.hpp"
#include "renderer/GpuDisplayRange.hpp"
#include "renderer/GpuThermalStepper.hpp"

#if defined(VK_VERSION_1_0)
#error "a gpu/ or migrated module header pulled in vulkan.h"
#endif
