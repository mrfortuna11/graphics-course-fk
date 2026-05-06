#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inGridPos;

layout(binding = 0) uniform sampler2DArray heightmapArray;

layout(binding = 2) readonly buffer LevelDataBlock {
  vec4 data[]; // xy=levelOrigin (world XZ), z=gridStep, w=unused
} levels;

layout(push_constant) uniform PC
{
  mat4  mProjView;
  vec4  sunDir;
  vec4  sunColor;
  vec4  eyeAndScale; // xyz=camera world pos, w=heightScale (negative = debug)
  vec4  morphParams; // x=morphWidth (texels), y=showMorphAlpha
};

const float HEIGHTMAP_SIZE = 256.0;
const float GRID_N = 255.0; 
const float GRID_HALF = (GRID_N - 1.0) * 0.5; // = 127.0, vertex center index

layout(location = 0) out vec3 wPos;
layout(location = 1) out vec2 hmUV;
layout(location = 2) flat out int instanceIdx;
layout(location = 3) out float morphAlpha;

void main()
{
  vec4  ld     = levels.data[gl_InstanceIndex];
  vec2  origin = ld.xy;
  float step   = ld.z;
  float layer = float(gl_InstanceIndex);

  // LOD morph
  float morphWidth = max(morphParams.x, 1.0);
  vec2  distFromCenter = abs(inGridPos - vec2(GRID_HALF));
  float chebyshev = max(distFromCenter.x, distFromCenter.y);
  float morphStart = GRID_HALF - morphWidth;
  float alpha = clamp((chebyshev - morphStart) / morphWidth, 0.0, 1.0);

  bool isOutermost = (gl_InstanceIndex == 0);
  if (isOutermost)
    alpha = 0.0;

  vec2 gridSelf = inGridPos;
  vec2 gridParent = floor(inGridPos * 0.5) * 2.0;
  vec2 gridMorphed = mix(gridSelf, gridParent, alpha);

  vec2 worldXZ = origin + gridMorphed * step;

  // Sample own height at the morphed grid position
  vec2 hmUVSelf = (gridMorphed + 0.5) / HEIGHTMAP_SIZE;
  float hSelf = texture(heightmapArray, vec3(hmUVSelf, layer)).r;

  // Sample parent (coarser) level's height at the same world point
  float hParent = hSelf;
  if (!isOutermost)
  {
    int   parentIdx = gl_InstanceIndex - 1;
    vec4  parentLD  = levels.data[parentIdx];
    vec2  parentOrigin = parentLD.xy;
    float parentStep = parentLD.z;
    vec2  parentGrid = (worldXZ - parentOrigin) / parentStep;
    vec2  parentUV = (parentGrid + 0.5) / HEIGHTMAP_SIZE;
    hParent = texture(heightmapArray, vec3(parentUV, float(parentIdx))).r;
  }

  float h = mix(hSelf, hParent, alpha) * abs(eyeAndScale.w);

  hmUV = hmUVSelf;
  wPos = vec3(worldXZ.x, h, worldXZ.y);
  instanceIdx = gl_InstanceIndex;
  morphAlpha = alpha;
  gl_Position = mProjView * vec4(wPos, 1.0);
}
