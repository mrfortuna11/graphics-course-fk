#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec2 inGridPos;

layout(binding = 0) uniform sampler2DArray heightmapArray;

layout(binding = 2) readonly buffer LevelDataBlock {
  vec4 data[]; // xy=levelOrigin (world XZ), z=gridStep, w=unused
} levels;

#include "clipmap_vertex_common.glsl"

layout(push_constant) uniform PC
{
  mat4 lightViewProj; 
  vec4 eyeAndScale;   // .w (heightScale)
  vec4 morphParams;   // .x (morphWidth)
};

layout(location = 0) out vec3 wPos;
layout(location = 1) flat out int instanceIdx;

void main()
{
  vec2  _hmUVSelf, _parentHmUV;
  float _morphAlpha, _gridStep, _parentStep;
  int   _parentIdx;
  bool  _isOutermost;

  wPos = computeClipmapWorldPos(
    inGridPos, gl_InstanceIndex, eyeAndScale.w, morphParams.x,
    _hmUVSelf, _parentHmUV, _morphAlpha, _parentIdx,
    _gridStep, _parentStep, _isOutermost);

  instanceIdx = gl_InstanceIndex;
  gl_Position = lightViewProj * vec4(wPos, 1.0);

  const int NUM_LEVELS = 10;
  float depthBias = float(NUM_LEVELS - 1 - gl_InstanceIndex) * 5e-5;
  gl_Position.z += depthBias * gl_Position.w;
}
