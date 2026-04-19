#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "unpack_attributes.glsl"


layout(location = 0) in vec4 vPosNorm;
layout(location = 1) in vec4 vTexCoordAndTang;

layout(push_constant) uniform params_t
{
  mat4 mProjView;
  vec4 baseColorFactor;
  vec4 materialParams;
  vec4 cameraPos;
  uint isBaked;
  uint debugMode;
} params;

layout(set = 0, binding = 0) buffer InstanceMatrices
{
  mat4 model[];
};


layout (location = 0 ) out VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 wTangent;
  vec2 texCoord;
  float tangentSign;
} vOut;

out gl_PerVertex { vec4 gl_Position; };

void main(void)
{
  const vec4 wNorm = vec4(decode_normal(floatBitsToInt(vPosNorm.w)),     0.0f);
  const vec4 wTang = vec4(decode_normal(floatBitsToInt(vTexCoordAndTang.z)), 0.0f);

  mat4 modelTm = model[gl_InstanceIndex];
  vOut.wPos   = (modelTm * vec4(vPosNorm.xyz, 1.0f)).xyz;
  vOut.wNorm  = normalize(mat3(transpose(inverse(modelTm))) * wNorm.xyz);
  vOut.wTangent = normalize(mat3(transpose(inverse(modelTm))) * wTang.xyz);
  vOut.texCoord = vTexCoordAndTang.xy;
  vOut.tangentSign = vTexCoordAndTang.w;

  gl_Position   = params.mProjView * vec4(vOut.wPos, 1.0);
}
