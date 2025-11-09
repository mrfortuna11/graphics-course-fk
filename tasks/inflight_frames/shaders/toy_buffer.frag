#version 430
#extension GL_GOOGLE_include_directive : require

#include "UniformParams.h"

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 fragColor;

layout(binding = 7) uniform Params
{
  UniformParams params;
};


#define camera_z 1.0

const float I0 = 11.0;
const vec3 sigma_r = vec3(0.33, 0.78, 1.89);
const float S_r = 0.17;
const float S_m = 0.05;
const float S_s = 0.07;
const float mie_w = 0.25;
const float sun_w = 0.0002;


vec3 sky_full(vec3 vd, vec3 sd) {

    float phase_r = 0.06 * (1.0 + dot(vd, sd) * dot(vd, sd));
    float phase_m = mie_w / (1.0 + mie_w + dot(vd, sd));
    phase_m = 0.3 * phase_m * phase_m;
    float phase_s = sun_w / (1.0 + sun_w + dot(vd, sd));
    phase_s = 4.0 * phase_s * phase_s;
    
    vec3 sigma_sum = S_r * sigma_r + S_m;
    vec3 phase_sum = S_r * sigma_r * phase_r + S_m * phase_m + S_s * phase_s;
    
    return I0 * sd.y / (sd.y + vd.y) * (phase_sum / sigma_sum) * 
           (exp(sigma_sum / sd.y) - exp(-sigma_sum / vd.y));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord.xy / params.iResolution.xy;
    
    uv.y = 1.0 - uv.y;
    
    vec3 view_dir = normalize(vec3(uv*2.0-1.0, camera_z));
    
    vec3 sun_dir = -normalize(vec3(-4.8, 0.3, 0.2));
    
    vec3 col = sky_full(view_dir, sun_dir);
    col = pow(col, vec3(1.0/2.2));
    
    fragColor = vec4(col, 1.0);
}

void main()
{
  mainImage(fragColor, texCoord * params.iResolution);
}