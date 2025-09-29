#ifndef UNIFORM_PARAMS_H_INCLUDED
#define UNIFORM_PARAMS_H_INCLUDED

#include "cpp_glsl_compat.h"

struct UniformParams
{
  shader_vec2  iResolution;
  shader_vec2  iMouse;
  shader_float iTime;
};


#endif // UNIFORM_PARAMS_H_INCLUDED