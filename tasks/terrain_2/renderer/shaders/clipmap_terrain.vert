#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inGridPos;

layout(binding = 0) uniform sampler2D heightmap;

layout(binding = 2) readonly buffer LevelDataBlock {
  vec4 data[]; // xy=levelOrigin (world XZ), z=gridStep, w=unused
} levels;

layout(push_constant) uniform PC
{
  mat4  mProjView;
  vec4  sunDir;
  vec4  sunColor;
  vec4  eyeAndScale; // xyz=camera world pos, w=heightScale (negative = debug)
};

const vec2 HM_WORLD_SIZE = vec2(1024.0);
const vec2 HM_WORLD_MIN  = vec2(-512.0);

layout(location = 0) out vec3 wPos;
layout(location = 1) out vec2 hmUV;
layout(location = 2) flat out int instanceIdx;

void main()
{
  vec4  ld     = levels.data[gl_InstanceIndex];
  vec2  origin = ld.xy;
  float step   = ld.z;

  vec2 worldXZ = origin + inGridPos * step;

  hmUV = (worldXZ - HM_WORLD_MIN) / HM_WORLD_SIZE;
  float h = texture(heightmap, hmUV).r * abs(eyeAndScale.w);

  wPos        = vec3(worldXZ.x, h, worldXZ.y);
  instanceIdx = gl_InstanceIndex;
  gl_Position = mProjView * vec4(wPos, 1.0);
}
