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
  mat4  mProjView;
  vec4  sunDir;
  vec4  sunColor;
  vec4  eyeAndScale; // xyz=camera world pos, w=heightScale (negative = debug)
  vec4  morphParams; // morphWidth, showMorphAlpha, liveLevelsCount, detailTilePeriod
  vec4  splatParams; // used in fragment shader
};

layout(location = 0) out vec3 wPos;
layout(location = 1) out vec2 hmUV;
layout(location = 2) flat out int instanceIdx;
layout(location = 3) out float morphAlpha;
layout(location = 4) out vec2 parentHmUV; // same world point in parent's heightmap
layout(location = 5) out vec3 wNormal;    // heightmap-derived, blended with parent

void main()
{
  int   parentIdx;
  float gridStep, parentStep;
  bool  isOutermost;

  wPos = computeClipmapWorldPos(
    inGridPos,
    gl_InstanceIndex,
    eyeAndScale.w,
    morphParams.x,
    hmUV,
    parentHmUV,
    morphAlpha,
    parentIdx,
    gridStep,
    parentStep,
    isOutermost);

  float hs = abs(eyeAndScale.w);
  const float TEX_STEP = 1.0 / HEIGHTMAP_SIZE;
  float layer = float(gl_InstanceIndex);

  float hL_s = texture(heightmapArray, vec3(hmUV + vec2(-TEX_STEP, 0.0), layer)).r;
  float hR_s = texture(heightmapArray, vec3(hmUV + vec2( TEX_STEP, 0.0), layer)).r;
  float hD_s = texture(heightmapArray, vec3(hmUV + vec2(0.0, -TEX_STEP), layer)).r;
  float hU_s = texture(heightmapArray, vec3(hmUV + vec2(0.0,  TEX_STEP), layer)).r;
  vec3 nSelf = normalize(vec3(
    (hL_s - hR_s) * hs,
    2.0 * gridStep,
    (hD_s - hU_s) * hs));

  vec3 nParent = nSelf;
  if (!isOutermost)
  {
    float hL_p = texture(heightmapArray, vec3(parentHmUV + vec2(-TEX_STEP, 0.0), float(parentIdx))).r;
    float hR_p = texture(heightmapArray, vec3(parentHmUV + vec2( TEX_STEP, 0.0), float(parentIdx))).r;
    float hD_p = texture(heightmapArray, vec3(parentHmUV + vec2(0.0, -TEX_STEP), float(parentIdx))).r;
    float hU_p = texture(heightmapArray, vec3(parentHmUV + vec2(0.0,  TEX_STEP), float(parentIdx))).r;
    nParent = normalize(vec3(
      (hL_p - hR_p) * hs,
      2.0 * parentStep,
      (hD_p - hU_p) * hs));
  }

  wNormal = normalize(mix(nSelf, nParent, morphAlpha));

  instanceIdx = gl_InstanceIndex;
  gl_Position = mProjView * vec4(wPos, 1.0);
  const int NUM_LEVELS = 10;
  float depthBias = float(NUM_LEVELS - 1 - gl_InstanceIndex) * 5e-5;
  gl_Position.z += depthBias * gl_Position.w;
}
