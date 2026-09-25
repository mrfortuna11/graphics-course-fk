#version 450

// Procedural sky: zenith→horizon→ground gradient + a sun disk and soft halo.
// HDR output: sun is several × 1.0 so adaptive exposure has something to react to.

layout(push_constant) uniform PC
{
  mat4 invProjView;
  vec4 cameraPos;
  vec4 sunDir;
  vec4 sunColor;
} pc;

layout(location = 0) in vec3 vWorldDir;
layout(location = 0) out vec4 out_color;

void main()
{
  vec3 dir = normalize(vWorldDir);
  vec3 sunDir = normalize(pc.sunDir.xyz);

  vec3 zenith  = vec3(0.18, 0.32, 0.70);
  vec3 horizon = vec3(0.78, 0.86, 0.95);
  vec3 ground  = vec3(0.18, 0.16, 0.14);

  float upT = clamp(dir.y, -1.0, 1.0);
  vec3 sky = upT >= 0.0
    ? mix(horizon, zenith, smoothstep(0.0, 0.55, upT))
    : mix(horizon, ground, smoothstep(0.0, 0.30, -upT));

  float sunDot = dot(dir, sunDir);
  float disk = smoothstep(0.9985, 0.9999, sunDot);
  float halo = pow(max(sunDot, 0.0), 96.0) * 0.4;
  vec3 sunContrib = pc.sunColor.rgb * pc.sunColor.a * (disk * 25.0 + halo);

  float horizonGlow = pow(max(1.0 - abs(dir.y), 0.0), 6.0)
                    * pow(max(sunDot * 0.5 + 0.5, 0.0), 3.0);
  vec3 glow = vec3(1.0, 0.65, 0.35) * horizonGlow * 0.35;

  out_color = vec4(sky + sunContrib + glow, 1.0);
}
