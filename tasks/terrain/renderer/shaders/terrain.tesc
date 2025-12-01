#version 450

layout(vertices = 2) out;

layout(push_constant) uniform params_t
{
  mat4 mProjView;
  vec3 eye;
};


const vec3 startPos = vec3(-512, -100, -512);
const uint squareSize = 32;

layout(location = 0) in uint InstanceIndex[];

layout(location = 0) out vec3 WorldPos_ES_in[];
layout(location = 1) out vec2 TexCoord_ES_in[];
layout(location = 2) out float GridSize[];

const float maxDist = 600.0; // probably too big
const float minDist = 1.0;

const float maxTess = 20.0;
const float minTess = 4.0;

vec3 calcPos(in uint cornerNum)
{
  const uint gridSize = 1024 / squareSize;
  uvec2 sqrCoord = uvec2(InstanceIndex[0] % gridSize, (1023u - InstanceIndex[0]) / gridSize);
  uvec2 offset = uvec2(cornerNum / 2u, 1u - (cornerNum % 2u));
  return vec3((sqrCoord + offset) * squareSize, 0).xzy + startPos;
}

vec2 calcTexcoord(in uint cornerNum)
{
  const uint gridSize = 1024 / squareSize;
  uvec2 sqrCoord = uvec2(InstanceIndex[0] % gridSize, InstanceIndex[0] / gridSize);
  uvec2 offset = uvec2(cornerNum / 2u, cornerNum % 2u);
  return vec2((sqrCoord + offset) * squareSize) / 1024;
}

float GetTessLevel(in float dist)
{
  dist = clamp((dist - minDist) / (maxDist - minDist), 0.0, 1.0);
  dist = sqrt(dist);
  return mix(maxTess, minTess, dist);
}

bool within(float a, float x, float b)
{
  return a <= x && x <= b;
}

bool Seen(vec3 point)
{
  vec4 proj = mProjView * vec4(point, 1);
  return within(0.0, proj.z, proj.w);
}

bool Cull(vec3 minP, vec3 maxP)
{
  for (uint mask = 0; mask < 8u; ++mask)
  {
    vec3 corner;
    for (uint i = 0; i < 3; ++i)
    {
      corner[i] = ((mask & (1u << i)) > 0) ? maxP[i] : minP[i];
    }
    if (Seen(corner))
    {
      return false;
    }
  }
  vec3 center = (maxP + minP) / 2.0;
  if (Seen(center))
  {
    return false;
  }
  return true;
}

void main()
{
  const vec3 corner00 = calcPos(0);
  const vec3 corner01 = calcPos(1);
  const vec3 corner10 = calcPos(2);
  const vec3 corner11 = calcPos(3);

  vec3 minP = corner01;
  minP.y += -200.0;
  vec3 maxP = corner10;
  maxP.y += 200.0;

  if (Cull(minP, maxP))
  {
    for (uint i = 0; i < 4; ++i)
      gl_TessLevelOuter[i] = -1.0;
    gl_TessLevelInner[0] = -1.0;
    gl_TessLevelInner[1] = -1.0;
    return;
  }

  // we only consider the horizontal distance, because the real height of a chunk is variable
  const float dist00 = distance(corner00.xz, eye.xz);
  const float dist01 = distance(corner01.xz, eye.xz);
  const float dist10 = distance(corner10.xz, eye.xz);
  const float dist11 = distance(corner11.xz, eye.xz);

  const float ol0 = min(dist00, dist01);
  const float ol1 = min(dist00, dist10);
  const float ol2 = min(dist10, dist11);
  const float ol3 = min(dist01, dist11);

  const float centerDist = min(ol0, ol3);

  gl_TessLevelOuter[0] = GetTessLevel(ol0);
  gl_TessLevelOuter[1] = GetTessLevel(ol1);
  gl_TessLevelOuter[2] = GetTessLevel(ol2);
  gl_TessLevelOuter[3] = GetTessLevel(ol3);
  float centerTess = GetTessLevel(centerDist);
  gl_TessLevelInner[0] = centerTess;
  gl_TessLevelInner[1] = centerTess;

  WorldPos_ES_in[gl_InvocationID] = gl_InvocationID == 0 ? corner00 : corner11;
  TexCoord_ES_in[gl_InvocationID] = gl_InvocationID == 0 ? calcTexcoord(0) : calcTexcoord(3);
  GridSize[gl_InvocationID] = squareSize / (centerTess * 1024.0);
}
