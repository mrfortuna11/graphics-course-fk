#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require


layout(location = 0) out vec4 out_fragColor;

layout(set = 1, binding = 0) uniform sampler2D baseColorTexture;
layout(set = 1, binding = 1) uniform sampler2D metallicRoughnessTexture;
layout(set = 1, binding = 2) uniform sampler2D normalTexture;
layout(set = 1, binding = 3) uniform sampler2D occlusionTexture;

layout(push_constant) uniform params_t
{
  mat4 mProjView;
  vec4 baseColorFactor;
  vec4 materialParams;
  uint isBaked;
  uint debugMode;
} params;

layout(location = 0) in VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 wTangent;
  vec2 texCoord;
} surf;

void main()
{
  const vec4 baseColorSample = texture(baseColorTexture, surf.texCoord);
  const vec4 mrSample        = texture(metallicRoughnessTexture, surf.texCoord);
  const vec3 normalSample    = texture(normalTexture, surf.texCoord).rgb;
  const float occlusionSample = texture(occlusionTexture, surf.texCoord).r;

  const vec3 albedo    = baseColorSample.rgb * params.baseColorFactor.rgb;
  const float metallic  = mrSample.b * params.materialParams.x;
  const float roughness = mrSample.g * params.materialParams.y;
  const float ao        = mix(1.0, occlusionSample, params.materialParams.w);

  // Debug channels — raw previews of each material texture for validation.
  if (params.debugMode == 1u) { out_fragColor = vec4(albedo, 1.0); return; }
  if (params.debugMode == 2u) { out_fragColor = vec4(normalSample, 1.0); return; }
  if (params.debugMode == 3u) { out_fragColor = vec4(metallic, roughness, 0.0, 1.0); return; }
  if (params.debugMode == 4u) { out_fragColor = vec4(vec3(ao), 1.0); return; }

  // Lambertian shading with a single directional light and ambient term
  const vec3 wLightPos = vec3(20, 20, 20);
  const vec3 lightColor = vec3(1.0);
  const vec3 lightDir   = normalize(wLightPos - surf.wPos);
  const float ndl = max(dot(surf.wNorm, lightDir), 0.0);
  const float ambient = 0.05;
  out_fragColor.rgb = (ndl * lightColor + ambient) * albedo * ao;
  out_fragColor.a = 1.0;
}
