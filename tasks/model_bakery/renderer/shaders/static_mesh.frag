#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require


layout(location = 0) out vec4 out_fragColor;

layout(set = 1, binding = 0) uniform sampler2D albedoTexture;

layout(location = 0) in VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 wTangent;
  vec2 texCoord;
} surf;

void main()
{
  const vec3 wLightPos = vec3(10, 10, 10);
  const vec3 surfaceColor = texture(albedoTexture, surf.texCoord).rgb;

  const vec3 lightColor = vec3(1.0f, 1.0f, 1.0f);

  const vec3 lightDir   = normalize(wLightPos - surf.wPos);
  const vec3 diffuse = max(dot(surf.wNorm, lightDir), 0.0f) * lightColor;
  const float ambient = 0.05;
  out_fragColor.rgb = (diffuse + ambient) * surfaceColor;
  out_fragColor.a = 1.0f;
}
