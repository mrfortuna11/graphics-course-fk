// Shared definitions for the adaptive-exposure compute pipeline

#ifndef LUMINANCE_COMMON_GLSL
#define LUMINANCE_COMMON_GLSL

const uint  HISTOGRAM_BINS   = 128u;
const float LOG_LUM_OFFSET   = 20.0;  // shift applied to log2(lum) before atomic packing
const float LOG_LUM_RANGE    = 40.0;  // valid [-20, +20] → [0, +40] after shift

// Rec.709 relative luminance 
float compute_luminance(vec3 col)
{
  return dot(col, vec3(0.2126, 0.7152, 0.0722));
}

#endif
