#version 450

layout(binding = 2) readonly buffer LevelDataBlock {
  vec4 data[]; // xy=levelOrigin, z=gridStep
} levels;

layout(location = 0) in vec3 wPos;
layout(location = 1) flat in int instanceIdx;

const int   NUM_LEVELS = 10;
const float GRID_SIZE  = 255.0;

void main()
{
  if (instanceIdx < NUM_LEVELS - 1)
  {
    vec4  inner = levels.data[instanceIdx + 1];
    float overlap = inner.z * 2.0;
    vec2  innerMin = inner.xy + vec2(overlap);
    vec2  innerMax = inner.xy + (GRID_SIZE - 1.0) * inner.z - vec2(overlap);
    if (wPos.x >= innerMin.x && wPos.x < innerMax.x &&
        wPos.z >= innerMin.y && wPos.z < innerMax.y)
      discard;
  }
}
