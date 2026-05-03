#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 out_fragColor;

layout(binding = 1) uniform sampler2D normalMap;

layout(push_constant) uniform PC
{
  mat4  mProjView;
  vec4  sunDir;                    // xyz = towards sun, w = gridStep
  vec4  sunColor;                  // rgb = color, a = intensity
  vec4  levelOriginAndHeightScale;
  vec4  fpOriginAndEye;
};

layout(location = 0) in vec3 wPos;
layout(location = 1) in vec2 hmUV;

void main()
{
  vec3 wNorm = normalize(texture(normalMap, hmUV).xyz);

  const vec3 surfaceColor = vec3(0.45, 0.42, 0.35);
  const vec3 L = normalize(sunDir.xyz);

  const float NdotL      = max(dot(wNorm, L), 0.0);
  const vec3  lightColor = sunColor.rgb * sunColor.a;
  vec3 diffuse = surfaceColor / 3.14159265 * lightColor * NdotL;

  const vec3 SKY_ZENITH  = vec3(0.18, 0.32, 0.70);
  const vec3 SKY_HORIZON = vec3(0.78, 0.86, 0.95);
  const vec3 SKY_GROUND  = vec3(0.18, 0.16, 0.14);
  vec3 skyAmbient = wNorm.y >= 0.0
    ? mix(SKY_HORIZON, SKY_ZENITH, smoothstep(0.0, 0.55, wNorm.y))
    : mix(SKY_HORIZON, SKY_GROUND, smoothstep(0.0, 0.30, -wNorm.y));
  vec3 ambient = skyAmbient * surfaceColor * 0.5;

  out_fragColor = vec4(diffuse + ambient, 1.0);
}
