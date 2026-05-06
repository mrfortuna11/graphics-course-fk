#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) out vec4 out_fragColor;

struct RelemMat
{
  uint baseColorIdx;
  uint metalRoughIdx;
  uint normalIdx;
  uint occlusionIdx;
  vec4 baseColorFactor;
  vec4 materialParams;  // x=metallic, y=roughness, z=normalScale, w=occlusionStrength
  uint emissiveIdx;
  uint _pad0;
  uint _pad1;
  uint _pad2;
  vec4 emissiveFactor;  // rgb=color
};

layout(set = 1, binding = 0) readonly buffer RelemMaterialBuffer { RelemMat data[]; } relemMats;

// Bindless texture array — all scene textures in one descriptor set
layout(set = 2, binding = 0) uniform sampler2D textures[];

layout(push_constant) uniform params_t
{
  mat4 mProjView;
  vec4 cameraPos;
  vec4 sunDir;
  vec4 sunColor;
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
  flat uint relemIdx;
} surf;

void main()
{
  RelemMat mat = relemMats.data[surf.relemIdx];

  const vec4 baseColorSample  = texture(textures[nonuniformEXT(mat.baseColorIdx)],  surf.texCoord);
  const vec4 mrSample         = texture(textures[nonuniformEXT(mat.metalRoughIdx)], surf.texCoord);
  const vec3 normalSample     = texture(textures[nonuniformEXT(mat.normalIdx)],      surf.texCoord).rgb;
  const float occlusionSample = texture(textures[nonuniformEXT(mat.occlusionIdx)],   surf.texCoord).r;
  const vec3 emissiveSample   = texture(textures[nonuniformEXT(mat.emissiveIdx)],    surf.texCoord).rgb;

  const vec3 albedo    = baseColorSample.rgb * mat.baseColorFactor.rgb;
  const float metallic  = mrSample.b * mat.materialParams.x;
  const float roughness = mrSample.g * mat.materialParams.y;
  const float normalScale = mat.materialParams.z;
  const float ao        = mix(1.0, occlusionSample, mat.materialParams.w);

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

  if (params.debugMode == 1u) { out_fragColor = vec4(albedo, 1.0); return; }
  if (params.debugMode == 2u) { out_fragColor = vec4(shadingNormal * 0.5 + 0.5, 1.0); return; }
  if (params.debugMode == 3u) { out_fragColor = vec4(metallic, roughness, 0.0, 1.0); return; }
  if (params.debugMode == 4u) { out_fragColor = vec4(vec3(ao), 1.0); return; }

  const vec3 lightColor = params.sunColor.rgb * params.sunColor.a;
  const vec3 sN = shadingNormal;
  const vec3 V = normalize(params.cameraPos.xyz - surf.wPos);
  const vec3 L = normalize(params.sunDir.xyz);
  const vec3 H = normalize(V + L);

  const float NdotL = max(dot(sN, L), 0.0);
  const float NdotV = max(dot(sN, V), 0.0);
  const float NdotH = max(dot(sN, H), 0.0);
  const float VdotH = max(dot(V, H), 0.0);

  const float alpha  = roughness * roughness;
  const float alpha2 = alpha * alpha;

  const vec3 F0 = mix(vec3(0.04), albedo, metallic);
  const vec3 F = F0 + (vec3(1.0) - F0) * pow(1.0 - VdotH, 5.0);

  const float denomD = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
  const float D = alpha2 / (3.14159265 * denomD * denomD + 1e-7);

  const float gV = NdotL * sqrt(NdotV * NdotV * (1.0 - alpha2) + alpha2);
  const float gL = NdotV * sqrt(NdotL * NdotL * (1.0 - alpha2) + alpha2);
  const float Vvis = 0.5 / (gV + gL + 1e-7);

  const vec3 specular = D * Vvis * F;
  const vec3 diffuse  = (vec3(1.0) - F) * (1.0 - metallic) * albedo / 3.14159265;
  const vec3 Lo = (diffuse + specular) * lightColor * NdotL;

  const vec3 SKY_ZENITH  = vec3(0.18, 0.32, 0.70);
  const vec3 SKY_HORIZON = vec3(0.78, 0.86, 0.95);
  const vec3 SKY_GROUND  = vec3(0.18, 0.16, 0.14);

  vec3 skyAmbient = sN.y >= 0.0
    ? mix(SKY_HORIZON, SKY_ZENITH, smoothstep(0.0, 0.55, sN.y))
    : mix(SKY_HORIZON, SKY_GROUND, smoothstep(0.0, 0.30, -sN.y));
  vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
  vec3 ambient = skyAmbient * albedo * kD * ao * 0.5;

  vec3 R = reflect(-V, sN);
  vec3 skyRefl = R.y >= 0.0
    ? mix(SKY_HORIZON, SKY_ZENITH, smoothstep(0.0, 0.55, R.y))
    : mix(SKY_HORIZON, SKY_GROUND, smoothstep(0.0, 0.30, -R.y));
  vec3 Fa = F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - NdotV, 5.0);
  vec3 ambientSpec = skyRefl * Fa * ao * (1.0 - roughness * 0.7);

  const vec3 emissive = emissiveSample * mat.emissiveFactor.rgb;

  out_fragColor.rgb = ambient + ambientSpec + Lo + emissive;
  out_fragColor.a = 1.0;
}
