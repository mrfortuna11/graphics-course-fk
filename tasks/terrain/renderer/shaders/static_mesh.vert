#version 450
#extension GL_GOOGLE_include_directive : require

#include "quant.h"


layout(location = 0) in vec4 vPosNorm;
layout(location = 1) in vec4 vTexCoordAndTang;

layout(push_constant) uniform params_t
{
  mat4 mProjView;
}
params;

layout(std430, binding = 0) restrict readonly buffer matr
{
  mat4 mModels[];
};

layout(location = 0) out VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 wTangent;
  vec2 texCoord;
}
vOut;

out gl_PerVertex
{
  vec4 gl_Position;
};

void main(void)
{
  const vec4 wNorm = vec4(dequantize3fnorm(floatBitsToUint(vPosNorm.w)), 0.0f);
  const vec4 wTang = vec4(dequantize3fnorm(floatBitsToUint(vTexCoordAndTang.z)), 0.0f);
  const mat4 mModel = mModels[gl_InstanceIndex];

  vOut.wPos = (mModel * vec4(vPosNorm.xyz, 1.0f)).xyz;
  vOut.wNorm = normalize(mat3(transpose(inverse(mModel))) * wNorm.xyz);
  vOut.wTangent = normalize(mat3(transpose(inverse(mModel))) * wTang.xyz);
  vOut.texCoord = vTexCoordAndTang.xy;

  gl_Position = params.mProjView * vec4(vOut.wPos, 1.0);
}
