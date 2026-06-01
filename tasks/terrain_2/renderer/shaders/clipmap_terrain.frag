#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 out_fragColor;

layout(binding = 2) readonly buffer LevelDataBlock {
  vec4 data[]; // xy=levelOrigin, z=gridStep
} levels;

layout(binding = 3) uniform sampler2DArray albedoArray;
layout(binding = 4) uniform sampler2DArray detailColorArray;
layout(binding = 5) uniform sampler2DArray detailHeightArray;
layout(binding = 6) uniform sampler2D shadowMap;

layout(push_constant) uniform PC
{
  mat4  mProjView;
  vec4  sunDir;       // xyz=normalized sun direction, w=detailNormalStrength
  vec4  sunColor;     // rgb=color, a=intensity
  vec4  eyeAndScale;  // xyz=camera world pos, w=heightScale (negative = debug colors)
  vec4  morphParams;  // x=morphWidth (texels), y=showMorphAlpha (0/1), z=liveLevelsCount, w=detailTilePeriod
  vec4  splatParams;  // x=heightLow, y=heightHigh, z=blendSharp, w=slopeThreshold
  mat4  lightViewProj; 
  vec4  shadowParams;  
};

layout(location = 0) in vec3 wPos;
layout(location = 1) in vec2 hmUV;
layout(location = 2) flat in int instanceIdx;
layout(location = 3) in float morphAlpha;
layout(location = 4) in vec2 parentHmUV;
layout(location = 5) in vec3 wNormal;

const int NUM_LEVELS = 10;
const float GRID_SIZE = 255.0;

vec3 levelColor(int idx)
{
  const vec3 colors[10] = vec3[10](
    vec3(0.80, 0.10, 0.10), vec3(0.85, 0.45, 0.10),
    vec3(0.85, 0.80, 0.10), vec3(0.40, 0.75, 0.15),
    vec3(0.10, 0.70, 0.55), vec3(0.10, 0.55, 0.85),
    vec3(0.30, 0.15, 0.85), vec3(0.65, 0.10, 0.85),
    vec3(0.85, 0.10, 0.55), vec3(0.90, 0.90, 0.90)
  );
  return colors[clamp(idx, 0, 9)];
}

void main()
{
  if (instanceIdx < NUM_LEVELS - 1)
  {
    vec4  inner    = levels.data[instanceIdx + 1];
    float overlap  = inner.z * 2.0;
    vec2  innerMin = inner.xy + vec2(overlap);
    vec2  innerMax = inner.xy + (GRID_SIZE - 1.0) * inner.z - vec2(overlap);
    if (wPos.x >= innerMin.x && wPos.x < innerMax.x &&
        wPos.z >= innerMin.y && wPos.z < innerMax.y)
      discard;
  }

  bool debugLevels      = eyeAndScale.w < 0.0;
  int  liveLevels       = int(morphParams.z + 0.5);
  bool useMaterialCache = (instanceIdx < NUM_LEVELS - liveLevels);
  float tilePeriod      = max(morphParams.w, 1.0);
  float normalStrength  = sunDir.w;

  // Macro surface normal from vertex shader (heightmap-derived, morph-blended)
  vec3 macroN = normalize(wNormal);
  if (macroN.y < 0.0) macroN = -macroN;
  float slope = 1.0 - clamp(macroN.y, 0.0, 1.0);

  // Smooth height/slope weights (ground=0, grass=1, rock=2, snow=3)
  float heightLow  = splatParams.x;
  float heightHigh = splatParams.y;
  float blendSharp = max(splatParams.z, 0.5);
  float slopeThr   = splatParams.w;

  float wGround = 1.0 - smoothstep(heightLow - blendSharp, heightLow + blendSharp, wPos.y);
  float wSnow   = smoothstep(heightHigh - blendSharp, heightHigh + blendSharp, wPos.y);
  float wGrass  = clamp(1.0 - wGround - wSnow, 0.0, 1.0);
  float wRock   = smoothstep(slopeThr - 0.15, slopeThr + 0.05, slope);
  wGrass  *= (1.0 - wRock);
  wGround *= (1.0 - wRock);
  wSnow   *= (1.0 - wRock);

  vec2 tileUV = wPos.xz / tilePeriod;
  vec3 wNorm = macroN; 
  vec3 splatColor;

  if (useMaterialCache)
  {
    vec3 albedoSelf = texture(albedoArray, vec3(hmUV, float(instanceIdx))).rgb;
    vec3 albedoParent = albedoSelf;
    if (instanceIdx > 0)
      albedoParent = texture(albedoArray, vec3(parentHmUV, float(instanceIdx - 1))).rgb;
    splatColor = mix(albedoSelf, albedoParent, morphAlpha);
  }
  else
  {

    float dh0 = texture(detailHeightArray, vec3(tileUV, 0.0)).r;
    float dh1 = texture(detailHeightArray, vec3(tileUV, 1.0)).r;
    float dh2 = texture(detailHeightArray, vec3(tileUV, 2.0)).r;
    float dh3 = texture(detailHeightArray, vec3(tileUV, 3.0)).r;

    const float HB_BIAS = 0.2;
    float r0 = wGround + dh0;
    float r1 = wGrass  + dh1;
    float r2 = wRock   + dh2;
    float r3 = wSnow   + dh3;
    float rMax = max(max(r0, r1), max(r2, r3));
    float hbBias = rMax - HB_BIAS;
    r0 = max(r0 - hbBias, 0.0);
    r1 = max(r1 - hbBias, 0.0);
    r2 = max(r2 - hbBias, 0.0);
    r3 = max(r3 - hbBias, 0.0);
    float rTot = r0 + r1 + r2 + r3;
    if (rTot < 1e-6) rTot = 1.0;
    r0 /= rTot; r1 /= rTot; r2 /= rTot; r3 /= rTot;

    // Sample and blend detail colors
    vec3 c0 = texture(detailColorArray, vec3(tileUV, 0.0)).rgb;
    vec3 c1 = texture(detailColorArray, vec3(tileUV, 1.0)).rgb;
    vec3 c2 = texture(detailColorArray, vec3(tileUV, 2.0)).rgb;
    vec3 c3 = texture(detailColorArray, vec3(tileUV, 3.0)).rgb;
    vec3 liveColor = c0*r0 + c1*r1 + c2*r2 + c3*r3;

    bool parentIsBaked = (instanceIdx > 0)
                      && ((instanceIdx - 1) < NUM_LEVELS - liveLevels);
    if (parentIsBaked)
    {
      vec3 parentBaked = texture(albedoArray, vec3(parentHmUV, float(instanceIdx - 1))).rgb;
      splatColor = mix(liveColor, parentBaked, morphAlpha);
    }
    else
    {
      splatColor = liveColor;
    }

    const float DET_STEP = 1.0 / 1024.0;
    float hL = 0.0, hR = 0.0, hD = 0.0, hU = 0.0;
    float sw[4] = float[4](wGround, wGrass, wRock, wSnow);
    for (int i = 0; i < 4; ++i)
    {
      float w = sw[i];
      if (w < 1e-4) continue;
      hL += w * texture(detailHeightArray, vec3(tileUV + vec2(-DET_STEP, 0.0), float(i))).r;
      hR += w * texture(detailHeightArray, vec3(tileUV + vec2( DET_STEP, 0.0), float(i))).r;
      hD += w * texture(detailHeightArray, vec3(tileUV + vec2(0.0, -DET_STEP), float(i))).r;
      hU += w * texture(detailHeightArray, vec3(tileUV + vec2(0.0,  DET_STEP), float(i))).r;
    }
    vec3 perturbation = vec3((hL - hR), 0.0, (hD - hU)) * normalStrength;
    wNorm = normalize(macroN + perturbation);
  }

  // Lighting
  const vec3 L = normalize(sunDir.xyz);
  float NdotL  = max(dot(wNorm, L), 0.0);

  vec3 surfaceColor = debugLevels ? levelColor(instanceIdx) : splatColor;

  vec3 diffuse = surfaceColor / 3.14159265
    * sunColor.rgb * sunColor.a * NdotL;

  float shadow = 1.0;
  if (shadowParams.x > 0.5)
  {
    vec4 lp  = lightViewProj * vec4(wPos, 1.0);
    vec3 ndc = lp.xyz / lp.w;
    vec2 sUV = ndc.xy * 0.5 + 0.5;

    if (all(greaterThanEqual(sUV, vec2(0.0))) &&
        all(lessThanEqual(sUV, vec2(1.0))))
    {
      float NdotL_macro = max(dot(macroN, L), 0.0);
      float bias = max(0.002 * (1.0 - NdotL_macro), 0.0005);

      float sampleDepth = texture(shadowMap, sUV).r;
      shadow = (ndc.z - bias > sampleDepth) ? 0.0 : 1.0;
    }
  }

  const vec3 SKY_ZENITH  = vec3(0.18, 0.32, 0.70);
  const vec3 SKY_HORIZON = vec3(0.78, 0.86, 0.95);
  const vec3 SKY_GROUND  = vec3(0.18, 0.16, 0.14);
  vec3 skyAmbient = wNorm.y >= 0.0
    ? mix(SKY_HORIZON, SKY_ZENITH,  smoothstep(0.0, 0.55,  wNorm.y))
    : mix(SKY_HORIZON, SKY_GROUND,  smoothstep(0.0, 0.30, -wNorm.y));

  vec3 finalColor = shadow * diffuse + skyAmbient * surfaceColor * 0.5;

  if (morphParams.y > 0.5)
    finalColor = mix(finalColor, vec3(0.05, 1.0, 0.05), morphAlpha);

  out_fragColor = vec4(finalColor, 1.0);
}
