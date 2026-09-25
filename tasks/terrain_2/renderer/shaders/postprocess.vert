#version 450
#extension GL_ARB_separate_shader_objects : enable

// Fullscreen triangle covering [-1,1]^2 with UVs in [0,1]^2
layout(location = 0) out vec2 vTexCoord;

void main()
{
  vec2 xy = gl_VertexIndex == 0 ? vec2(-1, -1)
          : gl_VertexIndex == 1 ? vec2( 3, -1)
                                : vec2(-1,  3);
  gl_Position = vec4(xy, 0, 1);
  vTexCoord = xy * 0.5 + 0.5;
}
