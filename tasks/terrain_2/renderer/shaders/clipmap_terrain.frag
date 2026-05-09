#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 out_fragColor;

layout(binding = 2) readonly buffer LevelDataBlock {
  vec4 data[]; // xy=levelOrigin, z=gridStep
} levels;

layout(binding = 3) uniform sampler2DArray albedoArray;

layout(push_constant) uniform PC
{
  mat4  mProjView;
  vec4  sunDir;
  vec4  sunColor;
  vec4  eyeAndScale; // xyz=camera world pos, w=heightScale (negative = debug colors)
  vec4  morphParams; // x=morphWidth (texels), y=showMorphAlpha (0/1), z=useMaterialClipmap (0/1)
  vec4  splatParams; // x=heightLow, y=heightHigh, z=blendSharp, w=slopeThreshold
};

layout(location = 0) in vec3 wPos;
layout(location = 1) in vec2 hmUV;
layout(location = 2) flat in int instanceIdx;
layout(location = 3) in float morphAlpha;
layout(location = 4) in vec2 parentHmUV;
layout(location = 5) in vec3 wNormal; // from vert: heightmap-derived, parent-blended

const int NUM_LEVELS = 10;
const float GRID_SIZE = 255.0; // n

const vec3 COL_SAND  = vec3(0.78, 0.70, 0.50);
const vec3 COL_GRASS = vec3(0.30, 0.45, 0.18);
const vec3 COL_ROCK  = vec3(0.42, 0.38, 0.34);
const vec3 COL_SNOW  = vec3(0.92, 0.94, 0.97);

vec3 levelColor(int idx)
{
  const vec3 colors[10] = vec3[10](
    vec3(0.80, 0.10, 0.10),
    vec3(0.85, 0.45, 0.10),
    vec3(0.85, 0.80, 0.10),
    vec3(0.40, 0.75, 0.15),
    vec3(0.10, 0.70, 0.55),
    vec3(0.10, 0.55, 0.85),
    vec3(0.30, 0.15, 0.85),
    vec3(0.65, 0.10, 0.85),
    vec3(0.85, 0.10, 0.55),
    vec3(0.90, 0.90, 0.90)
  );
  return colors[clamp(idx, 0, 9)];
}

// Cheap hash-based 0..1 noise; one value per integer cell
float hash21(vec2 p)
{
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Smooth value-noise: bilinear interp between 4 hashes
float valueNoise(vec2 p)
{
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash21(i + vec2(0.0, 0.0));
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// blend 4 base colors by height + slope
// p = world-space XZ for color variation
// h = world-space Y (height)
// slope = 1 - n.y, in [0..1] (0=flat, 1=vertical)
vec3 splatAlbedo(vec3 p, float h, float slope)
{
  float heightLow  = splatParams.x;
  float heightHigh = splatParams.y;
  float blendSharp = max(splatParams.z, 0.5);
  float slopeThr   = splatParams.w;

  float n0 = valueNoise(p.xz * 0.05);          // ~20m features
  float n1 = valueNoise(p.xz * 0.5  + 17.3);   // ~2m grain
  float variation = mix(0.85, 1.15, n0 * 0.6 + n1 * 0.4);

  // Height-driven weights (smooth bands)
  float wSand  = 1.0 - smoothstep(heightLow  - blendSharp, heightLow  + blendSharp, h);
  float wSnow  = smoothstep(heightHigh - blendSharp, heightHigh + blendSharp, h);
  float wGrass = clamp(1.0 - wSand - wSnow, 0.0, 1.0);

  float wRock = smoothstep(slopeThr - 0.15, slopeThr + 0.05, slope);
  wGrass *= (1.0 - wRock);
  wSand  *= (1.0 - wRock);
  wSnow  *= (1.0 - wRock);

  vec3 albedo = COL_SAND  * wSand
              + COL_GRASS * wGrass
              + COL_SNOW  * wSnow
              + COL_ROCK  * wRock;

  return albedo * variation;
}

void main()
{
  // Discard fragments covered by the next inner (finer) level
  // Instance 0 = outermost, NUM_LEVELS-1 = innermost
  if (instanceIdx < NUM_LEVELS - 1)
  {
    vec4  inner    = levels.data[instanceIdx + 1];
    vec2  innerMin = inner.xy;
    vec2  innerMax = inner.xy + (GRID_SIZE - 1.0) * inner.z;
    if (wPos.x >= innerMin.x && wPos.x <= innerMax.x &&
        wPos.z >= innerMin.y && wPos.z <= innerMax.y)
      discard;
  }

  vec3 wNorm = normalize(wNormal);
  if (wNorm.y < 0.0)
    wNorm = -wNorm;

  bool debugLevels = eyeAndScale.w < 0.0;
  const vec3 L = normalize(sunDir.xyz);
  float NdotL  = max(dot(wNorm, L), 0.0);

  bool useMaterialCache = morphParams.z > 0.5;
  float slope = 1.0 - clamp(wNorm.y, 0.0, 1.0);
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
    splatColor = splatAlbedo(wPos, wPos.y, slope);
  }

  vec3 surfaceColor = debugLevels ? levelColor(instanceIdx) : splatColor;

  vec3 diffuse = surfaceColor / 3.14159265
    * sunColor.rgb * sunColor.a * NdotL;

  const vec3 SKY_ZENITH  = vec3(0.18, 0.32, 0.70);
  const vec3 SKY_HORIZON = vec3(0.78, 0.86, 0.95);
  const vec3 SKY_GROUND  = vec3(0.18, 0.16, 0.14);
  vec3 skyAmbient = wNorm.y >= 0.0
    ? mix(SKY_HORIZON, SKY_ZENITH,  smoothstep(0.0, 0.55,  wNorm.y))
    : mix(SKY_HORIZON, SKY_GROUND,  smoothstep(0.0, 0.30, -wNorm.y));

  vec3 finalColor = diffuse + skyAmbient * surfaceColor * 0.5;

  // Debug: visualize morph alpha as a green tint at the ring edges
  if (morphParams.y > 0.5)
  {
    finalColor = mix(finalColor, vec3(0.05, 1.0, 0.05), morphAlpha);
  }

  out_fragColor = vec4(finalColor, 1.0);
}
