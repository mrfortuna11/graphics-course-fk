#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inGridPos;

layout(binding = 0) uniform sampler2D heightmap;

layout(push_constant) uniform PC
{
  mat4  mProjView;
  vec4  sunDir;                    // xyz = towards sun, w = gridStep
  vec4  sunColor;                  // rgb = color, a = intensity
  vec4  levelOriginAndHeightScale; // xy = levelOrigin (world XZ), z = heightScale
  vec4  fpOriginAndEye;            // xy = footprint origin (grid units), zw = eye.xz
};

// World bounds of the heightmap (fixed for MVP).
const vec2 HM_WORLD_SIZE = vec2(1024.0);
const vec2 HM_WORLD_MIN  = vec2(-512.0);

layout(location = 0) out vec3 wPos;
layout(location = 1) out vec2 hmUV;

void main()
{
  vec2 levelOrigin = levelOriginAndHeightScale.xy;
  float gridStep   = sunDir.w;
  float heightScale = levelOriginAndHeightScale.z;
  vec2 fpOrigin    = fpOriginAndEye.xy;

  vec2 gridCoord = fpOrigin + inGridPos;
  vec2 worldXZ   = levelOrigin + gridCoord * gridStep;

  hmUV = (worldXZ - HM_WORLD_MIN) / HM_WORLD_SIZE;
  float h = texture(heightmap, clamp(hmUV, 0.0, 1.0)).r * heightScale;

  wPos = vec3(worldXZ.x, h, worldXZ.y);
  gl_Position = mProjView * vec4(wPos, 1.0);
}
