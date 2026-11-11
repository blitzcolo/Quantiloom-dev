// Compile-time check that no gpu/ header drags in a backend header. Built without the
// precompiled header (quantiloom_core's PCH includes <vulkan/vulkan.h>), see CMakeLists.txt.
// Each migration step adds the migrated module's header below.

#include "gpu/Binding.hpp"
#include "gpu/CommandList.hpp"
#include "gpu/Device.hpp"
#include "gpu/DeviceFactory.hpp"
#include "gpu/GpuTypes.hpp"
#include "gpu/Kernel.hpp"
#include "gpu/Resources.hpp"
#include "gpu/RtScene.hpp"
#include "gpu/ShaderLibrary.hpp"
#include "gpu/Sync.hpp"
#include "gpu/Trace.hpp"
#include "hs_core/BatchRenderer.hpp"
#include "hs_core/GpuSpectralReconstructor.hpp"
#include "renderer/AccelerationStructure.hpp"
#include "renderer/AsyncPixelReadback.hpp"
#include "renderer/DeferredBufferUploads.hpp"
#include "renderer/GpuAlloc.hpp"
#include "renderer/GpuCameraPipeline.hpp"
#include "renderer/GpuClahe.hpp"
#include "renderer/GpuDisplayRange.hpp"
#include "renderer/GpuThermalStepper.hpp"
#include "renderer/PerformanceLogger.hpp"
#include "renderer/RayTracingPipeline.hpp"
#include "renderer/RenderCore.hpp"
#include "renderer/RtBindings.hpp"
#include "renderer/TextureManager.hpp"
#include "renderer/TextureTransitions.hpp"
#include "renderer/ThermalExchangePrecompute.hpp"

#if defined(VK_VERSION_1_0)
#error "a gpu/ or migrated module header pulled in vulkan.h"
#endif
