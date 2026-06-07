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
layout(set = 0, binding = 2) uniform sampler2D fogImage; // rgb=inscatter, a=transmittance

layout(push_constant) uniform PC
{
  uint tonemapMode;  // 0 = Reinhard, 1 = ACES, 2 = None (linear clamp)
  uint fogEnabled;   // 1 = fog before tonemapping
} pc;

layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 out_color;

vec3 reinhard(vec3 x) { return x / (1.0 + x); }

// Stephen Hill ACES: sRGB -> AP1 -> sRGB color space transforms
vec3 aces(vec3 x)
{
  // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
  const mat3 m1 = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777);
  // ODT_SAT => XYZ => D60_2_D65 => sRGB
  const mat3 m2 = mat3(
     1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602);
  vec3 v = m1 * x;
  vec3 a = v * (v + 0.0245786) - 0.000090537;
  vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return clamp(m2 * (a / b), 0.0, 1.0);
}

void main()
{
  vec3 hdr = texture(hdrImage, vTexCoord).rgb;

  if (pc.fogEnabled == 1u)
  {
    vec4 fog = texture(fogImage, vTexCoord);
    hdr = hdr * fog.a + fog.rgb;
  }

  float exposure = stats.smoothedExposure;
  if (exposure <= 0.0) exposure = 1.0; // safety for the very first frame

  vec3 exposed = hdr * exposure;

  vec3 ldr;
  if (pc.tonemapMode == 0u)      
    ldr = reinhard(exposed);
  else if (pc.tonemapMode == 1u) 
    ldr = aces(exposed);
  else                           
    ldr = clamp(exposed, 0.0, 1.0);

  out_color = vec4(ldr, 1.0);
}
