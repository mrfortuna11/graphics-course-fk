#ifndef CLIPMAP_VERTEX_COMMON_GLSL
#define CLIPMAP_VERTEX_COMMON_GLSL


const float HEIGHTMAP_SIZE = 256.0;
const float GRID_N = 255.0;
const float GRID_HALF = (GRID_N - 1.0) * 0.5;

vec3 computeClipmapWorldPos(
  vec2  inGridPos,
  int   instanceIdx,
  float heightScale,
  float morphWidth,
  out vec2  hmUVSelf,
  out vec2  parentHmUV,
  out float morphAlpha,
  out int   parentIdx,
  out float gridStep,
  out float parentStep,
  out bool  isOutermost)
{
  vec4  ld = levels.data[instanceIdx];
  vec2  origin = ld.xy;
  gridStep = ld.z;
  float layer  = float(instanceIdx);

  float w = max(morphWidth, 1.0);
  vec2  distFromCenter = abs(inGridPos - vec2(GRID_HALF));
  float chebyshev = max(distFromCenter.x, distFromCenter.y);
  float morphStart = GRID_HALF - w;
  morphAlpha = clamp((chebyshev - morphStart) / w, 0.0, 1.0);

  isOutermost = (instanceIdx == 0);
  if (isOutermost) morphAlpha = 0.0;

  vec2 gridParent  = floor(inGridPos * 0.5) * 2.0;
  vec2 gridMorphed = mix(inGridPos, gridParent, morphAlpha);

  vec2 worldXZ = origin + gridMorphed * gridStep;
  hmUVSelf = (gridMorphed + 0.5) / HEIGHTMAP_SIZE;

  float hSelf = texture(heightmapArray, vec3(hmUVSelf, layer)).r;
  float hParent = hSelf;
  parentHmUV = hmUVSelf;
  parentIdx = instanceIdx;
  parentStep = gridStep;
  if (!isOutermost)
  {
    parentIdx = instanceIdx - 1;
    vec4 parentLD = levels.data[parentIdx];
    vec2 parentO = parentLD.xy;
    parentStep = parentLD.z;
    vec2 parentGrid = (worldXZ - parentO) / parentStep;
    parentHmUV = (parentGrid + 0.5) / HEIGHTMAP_SIZE;
    hParent = texture(heightmapArray, vec3(parentHmUV, float(parentIdx))).r;
  }

  float h = mix(hSelf, hParent, morphAlpha) * abs(heightScale);
  return vec3(worldXZ.x, h, worldXZ.y);
}

#endif
