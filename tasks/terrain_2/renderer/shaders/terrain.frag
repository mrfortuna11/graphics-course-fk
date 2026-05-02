#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 out_fragColor;

layout(binding = 1) uniform sampler2D normalMap;

layout(push_constant) uniform params_t
{
  mat4 mProjView;
  vec4 eye;       // xyz = camera world pos
  vec4 sunDir;    // xyz = normalized sun direction (towards sun)
  vec4 sunColor;  // rgb = color, a = intensity
};

layout(location = 0) in VS_OUT
{
  vec3 wPos;
  vec2 texCoord;
} surf;

void main()
{
  // ground albedo: tinted by slope/height for some variation
  const vec3 surfaceColor = vec3(0.45, 0.42, 0.35);

  vec3 wNorm = texture(normalMap, surf.texCoord).xyz;
  wNorm = normalize(wNorm);

  const vec3 L = normalize(sunDir.xyz);
  const vec3 V = normalize(eye.xyz - surf.wPos);

  const float NdotL = max(dot(wNorm, L), 0.0);
  const vec3 lightColor = sunColor.rgb * sunColor.a;

  // diffuse (Lambert) in linear HDR — exposure/ACES handle the rest
  vec3 diffuse = surfaceColor / 3.14159265 * lightColor * NdotL;

  // sky ambient (same constants as static_mesh.frag for consistency)
  const vec3 SKY_ZENITH  = vec3(0.18, 0.32, 0.70);
  const vec3 SKY_HORIZON = vec3(0.78, 0.86, 0.95);
  const vec3 SKY_GROUND  = vec3(0.18, 0.16, 0.14);
  vec3 skyAmbient = wNorm.y >= 0.0
    ? mix(SKY_HORIZON, SKY_ZENITH, smoothstep(0.0, 0.55, wNorm.y))
    : mix(SKY_HORIZON, SKY_GROUND, smoothstep(0.0, 0.30, -wNorm.y));
  vec3 ambient = skyAmbient * surfaceColor * 0.5;

  out_fragColor.rgb = diffuse + ambient;
  out_fragColor.a = 1.0;
}
