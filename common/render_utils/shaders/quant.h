#ifndef QUANT_H_INCLUDED
#define QUANT_H_INCLUDED

#ifdef __cplusplus

#include <glm/glm.hpp>
#include <algorithm>
#include <bit>

inline uint32_t quantize4fnorm(glm::vec4 normal)
{

  const int8_t x = static_cast<int8_t>(round(normal.x * 127.0));
  const int8_t y = static_cast<int8_t>(round(normal.y * 127.0));
  const int8_t z = static_cast<int8_t>(round(normal.z * 127.0));
  const int8_t w = static_cast<int8_t>(round(normal.w * 127.0));

  int8_t arr[4] = {x, y, z, w};

  auto res = std::bit_cast<uint32_t>(arr);

  return res;
}

inline std::uint32_t quantize4fnorm(glm::vec3 normal)
{
  return quantize4fnorm(glm::vec4{normal, 0});
}

glm::vec3 dequantize3fnorm(uint32_t q)
{
  const uint32_t aEncX = (q & 0x000000FFu);
  const uint32_t aEncY = ((q & 0x0000FF00u) >> 8);
  const uint32_t aEncZ = ((q & 0x00FF0000u) >> 16);

  const uint32_t usX = uint32_t(aEncX & 0x000000FFu);
  const uint32_t usY = uint32_t(aEncY & 0x000000FFu);
  const uint32_t usZ = uint32_t(aEncZ & 0x000000FFu);

  const int32_t sX = (usX <= 127) ? usX : (int32_t(usX) - 256);
  const int32_t sY = (usY <= 127) ? usY : (int32_t(usY) - 256);
  const int32_t sZ = (usZ <= 127) ? usZ : (int32_t(usZ) - 256);

  const float x = std::max(float(sX) / 127.0f, -1.0f);
  const float y = std::max(float(sY) / 127.0f, -1.0f);
  const float z = std::max(float(sZ) / 127.0f, -1.0f);

  return glm::vec3{x, y, z};
}

#else

vec3 dequantize3fnorm(uint q)
{
  const uint aEncX = (q & 0x000000FFu);
  const uint aEncY = ((q & 0x0000FF00u) >> 8);
  const uint aEncZ = ((q & 0x00FF0000u) >> 16);

  const uint usX = uint(aEncX & 0x000000FFu);
  const uint usY = uint(aEncY & 0x000000FFu);
  const uint usZ = uint(aEncZ & 0x000000FFu);

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