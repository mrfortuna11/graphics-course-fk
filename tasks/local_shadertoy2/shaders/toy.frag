#version 430
#extension GL_GOOGLE_include_directive : require

#include "UniformParams.h"

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 fragColor;

layout(binding = 0, set = 1) uniform sampler2D   iChannel0;
layout(binding = 1, set = 1) uniform sampler2D   iChannel1;

layout(binding = 7) uniform Params
{
  UniformParams params;
};

#define INF 1e5
#define POINTS 8
#define PI      3.141592
#define TWO_PI  6.2831852

const float kEps = 0.01;

const float PalmTrunkLength = 1.5;   
const float PalmTrunkRadius = 0.02;  
const float PalmTrunkSkew = 1.8;     
const float TrunkPower = 2.0;        
const vec3  TrunkColor = vec3(0.48, 0.29, 0.070) * 0.7;
const float TrunkPowerFactor = 0.03; 



float noise(vec3 pos, float speed)
{
    float time = params.iTime * 2.0 * speed;
    return (
        sin(pos.x * 1.7 + pos.z * 0.3 + time) * 0.3 +
        cos(pos.x * 1.3 + pos.z * 0.5 + time * 0.7) * 0.2 +
        cos(pos.x * 0.4 + pos.z * 1.5 - time * 0.3) * 0.4 +
        cos(-pos.x * 0.4 + pos.z * 1.5 + time * 1.2) * 0.3
    ) / 2.0 + 0.5;
}

vec2 rotate(vec2 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}

vec3 rotateX(vec3 p, float a) {
	float sa = sin(a);
	float ca = cos(a);
	return vec3(p.x, ca * p.y + sa * p.z, ca * p.z - sa * p.y);
}

vec3 rotateY(vec3 p, float a) {
	float sa = sin(a);
	float ca = cos(a);
	return vec3(ca * p.x + sa * p.z, p.y, ca * p.z - sa * p.x);
}

vec3 rotateZ(vec3 p, float a) {
	float sa = sin(a);
	float ca = cos(a);
	return vec3(ca * p.x + sa * p.y, ca * p.y - sa * p.x, p.z);
}


vec3 rgba(int red, int green, int blue, int alpha)
{
    return vec3(float(red) / 256.0, float(green) / 256.0, float(blue) / 256.0);
}

float luma(vec3 col) {
	return dot(col, vec3(0.2126, 0.7152, 0.0722));
}


vec3 texcube(sampler2D sa, vec3 p, vec3 n) {
	vec3 x = texture(sa, p.yz).xyz;
	vec3 y = texture(sa, p.zx).xyz;
	vec3 z = texture(sa, p.xy).xyz;

	return x * abs(n.x) + y * abs(n.y) + z * abs(n.z);
}

vec3 bumpMap(sampler2D tex, vec3 pos, vec3 nor, float amount) {
	float e = 0.001;

	float ref = luma(texcube(tex, pos, nor));

	vec3 gra = -vec3(luma(texcube(tex, vec3(pos.x + e, pos.y, pos.z), nor)) - ref,
					 luma(texcube(tex, vec3(pos.x, pos.y + e, pos.z), nor)) - ref,
					 luma(texcube(tex, vec3(pos.x, pos.y, pos.z + e), nor)) - ref) / e;

	vec3 tgrad = gra - nor * dot(nor, gra);
	return normalize(nor - amount * tgrad);
}



float island(vec3 pos)
{

    float sphere = length(pos + vec3(0.0,0.4,0.0)) - 0.5;

    return max(sphere, -pos.y);
}

float sdTrunk(vec3 pos)
{

    vec3 p = pos - vec3(0.0, 0.1, 0.0); 


    float f = clamp(p.y / PalmTrunkLength, 0.0, 1.0);
    p.xy = rotate(p.xy, -p.y * p.y * PalmTrunkSkew);


    float d = length(p.xz) - (PalmTrunkRadius - pow(f, TrunkPower) * TrunkPowerFactor * 10.0);


    d = max(d, -p.y); 
    d = max(d, p.y - PalmTrunkLength); 

    return d;
}


float water(vec3 pos)
{
    float tinyRipples = noise(pos * 50.0, 1.0) * 0.01;
    return pos.y - tinyRipples;
}



float sdCutHollowSphere(vec3 p, float r, float h, float t, vec3 n)
{

    n = normalize(n);


    float axis = dot(p, n);


    float radial = length(p - n * axis);

    float w = sqrt(r*r - h*h);
    vec2 q = vec2(radial, axis);

    return ((h*q.x < w*q.y) ? length(q - vec2(w,h)) : abs(length(q) - r)) - t;
}


float globalSdf(int object, vec3 pos)
{
    if (object == 1)
        return water(pos);
    else {
        float islandSDF = island(pos);
        float trunkSDF = sdTrunk(pos);


        float leafSDF = INF;
        for (int i = 0; i < POINTS; i++)
        {
            float angle = float(i) * TWO_PI / float(POINTS);
            vec3 leafPos = pos - vec3(-0.064, PalmTrunkLength-1.083, 0.0);
            float lSDF = sdCutHollowSphere(leafPos, 0.1, 0.0, 0.01, vec3(1,-1,0));
            leafSDF = min(leafSDF, lSDF);
        }

        return min(min(islandSDF, trunkSDF), leafSDF);
    }
}

float trace(int object, vec3 start, vec3 ray, out vec3 position, out vec3 normal)
{
    int iterations = 999;
    
    vec3 pos = start;

    if (object == 1 && ray.y < 0.0)
    {
        pos -= ray * (pos.y / ray.y);
    }
    
    float sdf = 0.0;
    for (int iter = 0; iter < iterations; ++iter)
    {
        sdf = globalSdf(object, pos);
        pos += normalize(ray) * sdf;

        if (object == 0 && abs(pos.z) > 1.0)
            break;

        if (object == 1 && ray.y >= 0.0)
            break;
    }

    position = pos;
    normal.x = globalSdf(object, pos + vec3(kEps, 0.0, 0.0)) - sdf;
    normal.y = globalSdf(object, pos + vec3(0.0, kEps, 0.0)) - sdf;
    normal.z = globalSdf(object, pos + vec3(0.0, 0.0, kEps)) - sdf;
    normal = normalize(normal);

    return sdf;
}



vec3 deepBlue(vec3 ray)
{
    vec3 zenith = rgba(47, 88, 110, 1);
    vec3 horizon = rgba(47, 88, 110, 1);
    float interpolation = pow(normalize(ray).y + 1.0, 1.0);

    vec3 color = zenith * interpolation + horizon * (1.0 - interpolation);

    return color;
}

vec3 sky(vec3 ray)
{
    vec3 zenith = rgba(116, 92, 97, 1);
    vec3 horizon = rgba(179, 124, 100, 1);
    float interpolation = smoothstep(0.0, 1.0, normalize(ray).y / 2.0 + 0.5);

    vec3 color = zenith * interpolation + horizon * (1.0 - interpolation);

    return color;
}

vec3 islandColor(vec3 normal) {
    return rgba(237, 201, 175, 1); 
}

vec2 trunkUV(vec3 pos) {

    float u = atan(pos.z, pos.x) / (2.0*PI) + 0.5;

    float v = clamp(pos.y / PalmTrunkLength, 0.0, 2.0);
    v = 1.0 - v;
    
    return vec2(u, v);
}

vec3 trunkColor(vec3 pos, vec3 normal) {
    vec2 uv = trunkUV(pos);


    vec3 texColor = texture(iChannel1, uv).rgb;


    float n = noise(pos * 5.0, 1.0);
    texColor = mix(texColor, texColor * 0.7, n*0.3);


    normal = bumpMap(iChannel1, pos*2.0, normal, 0.2);

    return texColor;
}


vec3 kSun = normalize(vec3(-4.8, 0.3, 0.2));

float specular(vec3 normal, vec3 view)
{

    vec3 bisector = normalize(-view - kSun);
    return pow(max(0.0, dot(bisector, normal)), 170.0);
}

vec3 shade(vec3 normal, vec3 view)
{
    vec3 albedo = islandColor(normal);
    vec3 ao = albedo * (sky(normal) + vec3(1.0)) / 2.0;
    vec3 direct = albedo * dot(normal, -kSun);
    vec3 spec = vec3(1.0) * specular(normal, view);
    return direct * 0.3 + ao * 0.9  + spec * 0.2;
}

vec3 leafColor(vec3 pos)
{

    float n = noise(pos * 5.0, 1.0);
    return mix(vec3(0.1,0.6,0.2), vec3(0.05,0.4,0.1), n);
}


vec3 shadeIsland(vec3 pos, vec3 normal)
{

    bool isTrunk = sdTrunk(pos) < 0.001;


    float leafSDF = INF;
    bool isLeaf = leafSDF < 1.01;
    

    vec3 albedo = isLeaf ? leafColor(pos)
                     : (isTrunk ? trunkColor(pos, normal)
                                : islandColor(normal));


    vec3 sunDir = normalize(vec3(-4.8, 0.3, 0.2));
    vec3 sunColor = rgba(189, 188, 14, 1);

    float diffuse = max(0.0, dot(normal, sunDir));
    float ambient = 0.3;
    float light = diffuse * 0.8 + ambient;

    return albedo * (sunColor * light);
}

vec3 colorCorrect(vec3 color)
{
    vec3 result = color;
    result.x = smoothstep(0.0, 1.0, color.x);
    result.y = smoothstep(0.0, 1.0, color.y);
    result.z = smoothstep(0.0, 1.0, color.z);
    return result;
}

vec3 shadeWater(vec3 pos, vec3 normal, vec3 view)
{
    vec3 refracted = refract(view, normal, 1.0 / 1.33);
    vec3 reflected = reflect(view, normal);

    vec3 reflectedSky = sky(reflected);
    
    vec3 islandPosition;
    vec3 islandNormal;
    float dist = trace(0, pos, refracted, islandPosition, islandNormal);


    vec3 result = mix(deepBlue(refracted), rgba(115, 224, 238, 1), pos.y / 0.6);
    result = mix(result, reflectedSky, pow(1.0 - abs(view.y), 3.0) * 0.75);
    if (dist < kEps)
    {
        result = shade(islandNormal, refracted);
        vec3 mixedWaterColor = deepBlue(islandNormal);
        mixedWaterColor = mix(mixedWaterColor, rgba(55, 99, 180, 1), 0.4);
        result = mix(result, mixedWaterColor, 0.6);
    }

    if (island(pos) < noise(pos * 100.0, 3.0) * 0.01 + 0.005)
    {
        result = rgba(216, 240, 243, 1);
    }

   

    result += specular(normal, view) * vec3(1.0) * 0.04;

    return result;
}

void setCamera(vec2 pix, out vec3 ro, out vec3 rd) {
	const float fov = 0.64;

	vec2 coord = (-params.iResolution.xy + 2.0 * pix) / params.iResolution.y;
	coord *= fov;

	vec2 mo = params.iMouse.xy / params.iResolution.xy - vec2(0.5);
	float t = params.iTime * 0.2 + mo.x * 4.0;

	vec3 target = vec3(0.0, 0.19, 0.0);
	ro = vec3(0.0, 0.5, -1.0);

	float s = -0.1 + clamp(mo.y, 0.0, 1.0);
	ro = rotateX(ro, s);
	ro = rotateY(ro, t);

	vec3 dir = normalize(target - ro);

	vec3 up = vec3(0.0, 1.0, 0.0);
	up = rotateX(up, s);
	up = rotateY(up, t);

	up = normalize(up);

	vec3 right = normalize(cross(dir, up));

	rd = normalize(dir + coord.x * right + coord.y * up);
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec3 ro, rd;
	setCamera(fragCoord, ro, rd);


    vec3 islandPosition;
    vec3 islandNormal;
    float islandDelta = trace(0, ro, rd, islandPosition, islandNormal);

    vec3 waterPosition;
    vec3 waterNormal;
    float waterDelta = trace(1, ro, rd, waterPosition, waterNormal);

    vec3 skyTex = texture(iChannel0, vec2(
    0.4 + 0.4*rd.x,
    0.55 + 0.55*rd.y
    )).rgb;
    vec3 col = skyTex;

    if ((distance(islandPosition, ro) < distance(waterPosition, ro) || waterDelta > kEps) &&
        islandDelta < kEps)
    {
        col = shadeIsland(islandPosition, islandNormal);
    }
    else if (waterDelta < kEps)
    {
        col = shadeWater(waterPosition, waterNormal, normalize(waterPosition - ro));
    }
    col = colorCorrect(col);

    fragColor = vec4(col,1.0);
}

void main()
{
  mainImage(fragColor, texCoord * params.iResolution);
}
