#version 450

layout(push_constant) uniform PC
{
  mat4 invProjView;
  vec4 cameraPos;
  vec4 sunDir;       // xyz = direction to the sun in world space
  vec4 sunColor;     // rgb = sun colour, a = intensity multiplier
} pc;

layout(location = 0) out vec3 vWorldDir;

void main()
{
  vec2 xy = gl_VertexIndex == 0 ? vec2(-1.0, -1.0)
          : gl_VertexIndex == 1 ? vec2( 3.0, -1.0)
                                : vec2(-1.0,  3.0);

  // Far plane in clip space; perspective divide gives a world-space point on the
  // far frustum face, from which we derive a direction by subtracting the camera
  vec4 worldFar = pc.invProjView * vec4(xy, 1.0, 1.0);
  vec3 worldPos = worldFar.xyz / worldFar.w;
  vWorldDir = worldPos - pc.cameraPos.xyz;

  gl_Position = vec4(xy, 1.0, 1.0);
}
