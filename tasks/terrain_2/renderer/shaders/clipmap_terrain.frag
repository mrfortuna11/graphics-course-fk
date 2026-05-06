#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 out_fragColor;

layout(binding = 2) readonly buffer LevelDataBlock {
  vec4 data[]; // xy=levelOrigin, z=gridStep
} levels;

layout(push_constant) uniform PC
{
  mat4  mProjView;
  vec4  sunDir;
  vec4  sunColor;
  vec4  eyeAndScale; // xyz=camera world pos, w=heightScale (negative = debug colors)
  vec4  morphParams; // x=morphWidth (texels), y=showMorphAlpha (0/1)
};

layout(location = 0) in vec3 wPos;
layout(location = 1) in vec2 hmUV;
layout(location = 2) flat in int instanceIdx;
layout(location = 3) in float morphAlpha;

const int NUM_LEVELS = 10;
const float GRID_SIZE = 255.0; // n

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

 
  vec3 dPdx = dFdx(wPos);
  vec3 dPdy = dFdy(wPos);
  vec3 wNorm = normalize(cross(dPdx, dPdy));
  if (wNorm.y < 0.0)
    wNorm = -wNorm;

  bool debugLevels = eyeAndScale.w < 0.0;
  const vec3 L = normalize(sunDir.xyz);
  float NdotL  = max(dot(wNorm, L), 0.0);

  vec3 surfaceColor = debugLevels
    ? levelColor(instanceIdx)
    : vec3(0.45, 0.42, 0.35);

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
