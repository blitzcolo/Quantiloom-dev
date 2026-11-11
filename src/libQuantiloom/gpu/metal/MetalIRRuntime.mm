// The one translation unit that defines Metal Shader Converter's runtime companion
// functions (IRDescriptorTableSet*, IRRaytracingSetAccelerationStructure, ...); every
// other file includes the header for the declarations only.

#import <Metal/Metal.h>

#define IR_PRIVATE_IMPLEMENTATION
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>
