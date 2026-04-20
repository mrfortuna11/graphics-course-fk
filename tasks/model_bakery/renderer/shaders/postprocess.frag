#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "luminance_common.glsl"

layout(set = 0, binding = 0) uniform sampler2D hdrImage;
layout(set = 0, binding = 1) readonly buffer LuminanceStats
{
  uint  minLogLumBits;
  uint  maxLogLumBits;
  uint  histogram[HISTOGRAM_BINS];
  float smoothedExposure;
} stats;

layout(push_constant) uniform PC
{
  uint tonemapMode;  // 0 = Reinhard, 1 = ACES, 2 = None (linear clamp)
} pc;

layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 out_color;

vec3 reinhard(vec3 x) { return x / (1.0 + x); }

// Narkowicz 2015 ACES fit 
vec3 aces(vec3 x)
{
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main()
{
  vec3 hdr = texture(hdrImage, vTexCoord).rgb;
  float exposure = stats.smoothedExposure;
  if (exposure <= 0.0) exposure = 1.0; // safety for the very first frame

  vec3 exposed = hdr * exposure;

  vec3 ldr;
  if (pc.tonemapMode == 0u)      ldr = reinhard(exposed);
  else if (pc.tonemapMode == 1u) ldr = aces(exposed);
  else                           ldr = clamp(exposed, 0.0, 1.0);

  out_color = vec4(ldr, 1.0);
}
