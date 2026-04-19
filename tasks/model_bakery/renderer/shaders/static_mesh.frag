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
  vec4 cameraPos;
  uint isBaked;
  uint debugMode;
} params;

layout(location = 0) in VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 wTangent;
  vec2 texCoord;
  float tangentSign;
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
  const float normalScale = params.materialParams.z;
  const float ao        = mix(1.0, occlusionSample, params.materialParams.w);

  // tangent.w == 0 marks meshes without TANGENT data
  const vec3 N = normalize(surf.wNorm);
  vec3 shadingNormal = N;
  if (abs(surf.tangentSign) > 0.5)
  {
    const vec3 T = normalize(surf.wTangent - N * dot(N, surf.wTangent));
    const vec3 B = cross(N, T) * surf.tangentSign;
    vec3 nTs = normalSample * 2.0 - 1.0;
    nTs.xy *= normalScale;
    shadingNormal = normalize(mat3(T, B, N) * nTs);
  }

  // Debug channels — raw previews of each material texture for validation
  if (params.debugMode == 1u) { out_fragColor = vec4(albedo, 1.0); return; }
  if (params.debugMode == 2u) { out_fragColor = vec4(shadingNormal * 0.5 + 0.5, 1.0); return; }
  if (params.debugMode == 3u) { out_fragColor = vec4(metallic, roughness, 0.0, 1.0); return; }
  if (params.debugMode == 4u) { out_fragColor = vec4(vec3(ao), 1.0); return; }

  // GGX metallic-roughness BRDF 
  const vec3 wLightPos = vec3(20, 20, 20);
  const vec3 lightColor = vec3(3.0); // slightly punchy directional for now
  const vec3 sN = shadingNormal;
  const vec3 V = normalize(params.cameraPos.xyz - surf.wPos);
  const vec3 L = normalize(wLightPos - surf.wPos);
  const vec3 H = normalize(V + L);

  const float NdotL = max(dot(sN, L), 0.0);
  const float NdotV = max(dot(sN, V), 0.0);
  const float NdotH = max(dot(sN, H), 0.0);
  const float VdotH = max(dot(V, H), 0.0);

  // α = roughness²
  const float alpha = roughness * roughness;
  const float alpha2 = alpha * alpha;

  // F0 is the reflectance at normal incidence
  const vec3 F0 = mix(vec3(0.04), albedo, metallic);

  // Schlick Fresnel
  const vec3 F = F0 + (vec3(1.0) - F0) * pow(1.0 - VdotH, 5.0);

  // GGX/Trowbridge-Reitz normal distribution
  const float denomD = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
  const float D = alpha2 / (3.14159265 * denomD * denomD + 1e-7);

  // Smith height-correlated visibility 
  const float gV = NdotL * sqrt(NdotV * NdotV * (1.0 - alpha2) + alpha2);
  const float gL = NdotV * sqrt(NdotL * NdotL * (1.0 - alpha2) + alpha2);
  const float Vvis = 0.5 / (gV + gL + 1e-7);

  const vec3 specular = D * Vvis * F;
  // Diffuse uses remaining energy; metals have no diffuse lobe
  const vec3 diffuse = (vec3(1.0) - F) * (1.0 - metallic) * albedo / 3.14159265;

  const vec3 Lo = (diffuse + specular) * lightColor * NdotL;

  // Temporary constant ambient until IBL lands
  const vec3 ambient = vec3(0.03) * albedo * ao;

  out_fragColor.rgb = ambient + Lo;
  out_fragColor.a = 1.0;
}
