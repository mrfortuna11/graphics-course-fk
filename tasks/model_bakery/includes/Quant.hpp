#ifndef QUANT_H_INCLUDED
#define QUANT_H_INCLUDED

#ifdef __cplusplus

#include <cstdint>
#include <glm/glm.hpp>
#include <algorithm>
#include <bit>


inline uint32_t quantize4fnorm(const glm::vec4& in)
{
  int8_t coords[4] = {
    static_cast<int8_t>(std::round(in.x * 127.f)),
    static_cast<int8_t>(std::round(in.y * 127.f)),
    static_cast<int8_t>(std::round(in.z * 127.f)),
    static_cast<int8_t>(std::round(in.w * 127.f))};
  return std::bit_cast<uint32_t>(coords);
}

inline glm::vec3 dequantize3fnorm(uint32_t q)
{
  const uint32_t encX = (q & 0x000000FFu);
  const uint32_t encY = ((q & 0x0000FF00u) >> 8);
  const uint32_t encZ = ((q & 0x00FF0000u) >> 16);

  auto decode = [](uint32_t v) -> float {
    int32_t s = (v <= 127) ? static_cast<int32_t>(v) : static_cast<int32_t>(v) - 256;
    return std::max(static_cast<float>(s) / 127.0f, -1.0f);
  };

  return glm::vec3(decode(encX), decode(encY), decode(encZ));
}

#else

vec3 dequantize3fnorm(uint q)
{
  const uint encX = (q & 0x000000FFu);
  const uint encY = ((q & 0x0000FF00u) >> 8);
  const uint encZ = ((q & 0x00FF0000u) >> 16);

  const uint usX = uint(encX & 0x000000FFu);
  const uint usY = uint(encY & 0x000000FFu);
  const uint usZ = uint(encZ & 0x000000FFu);

  const int sX = (usX <= 127) ? int(usX) : (int(usX) - 256);
  const int sY = (usY <= 127) ? int(usY) : (int(usY) - 256);
  const int sZ = (usZ <= 127) ? int(usZ) : (int(usZ) - 256);

  const float x = max(float(sX) / 127.0f, -1.0f);
  const float y = max(float(sY) / 127.0f, -1.0f);
  const float z = max(float(sZ) / 127.0f, -1.0f);

  return vec3(x, y, z);
}
#endif

#endif