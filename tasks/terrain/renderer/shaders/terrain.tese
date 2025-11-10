#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(quads, fractional_even_spacing, cw) in;

layout(binding = 0) uniform sampler2D perlinNoise;
layout(push_constant) uniform params_t
{
  mat4 mProjView;
  vec3 eye;
};

const float zScale = 200.0;

layout(location = 0) in vec3 WorldPos_ES_in[];
layout(location = 1) in vec2 TexCoord_ES_in[];
layout(location = 2) in float GridSize[];

layout(location = 0) out VS_OUT
{
  vec3 wPos;
  vec2 texCoord;
};

out gl_PerVertex
{
  vec4 gl_Position;
};

vec3 calcNorm(vec2 texCoord)
{
  const float pixSize = GridSize[0];
  float left = texture(perlinNoise, texCoord - vec2(-1, 0) * pixSize).x * zScale;
  float right = texture(perlinNoise, texCoord + vec2(-1, 0) * pixSize).x * zScale;
  float up = texture(perlinNoise, texCoord - vec2(0, -1) * pixSize).x * zScale;
  float down = texture(perlinNoise, texCoord + vec2(0, -1) * pixSize).x * zScale;

  return normalize(vec3(right - left, 2.0 * pixSize, up - down));
}

void main()
{
  wPos.y = WorldPos_ES_in[0].y;
  wPos.xz = mix(WorldPos_ES_in[0].xz, WorldPos_ES_in[1].xz, gl_TessCoord.xy);
  texCoord = mix(TexCoord_ES_in[0], TexCoord_ES_in[1], gl_TessCoord.xy);
  wPos.y += texture(perlinNoise, texCoord).x * zScale;
  gl_Position = mProjView * vec4(wPos, 1.0);
}
