#include "ClipmapMesh.hpp"

#include <etna/GlobalContext.hpp>
#include <span>
#include <stdexcept>

ClipmapMesh::ClipmapMesh(uint32_t n, etna::BlockingTransferHelper& transferHelper): n_(n)
{
  if (n < 3 || (n % 2) == 0)
    throw std::invalid_argument("ClipmapMesh: n must be odd and >= 3");
  buildGeometry(transferHelper);
}

void ClipmapMesh::buildGeometry(etna::BlockingTransferHelper& transferHelper)
{
  // n×n vertex grid; (n-1)×(n-1) quads; 2*(n-1)^2 triangles
  std::vector<glm::vec2> verts;
  verts.reserve(n_ * n_);
  for (uint32_t r = 0; r < n_; ++r)
    for (uint32_t c = 0; c < n_; ++c)
      verts.push_back(glm::vec2(static_cast<float>(c), static_cast<float>(r)));

  std::vector<uint32_t> indices;
  indices.reserve(2u * (n_ - 1) * (n_ - 1) * 3u);
  for (uint32_t r = 0; r < n_ - 1; ++r)
  {
    for (uint32_t c = 0; c < n_ - 1; ++c)
    {
      uint32_t tl = r * n_ + c;
      uint32_t bl = (r + 1) * n_ + c;
      uint32_t br = (r + 1) * n_ + (c + 1);
      uint32_t tr = r * n_ + (c + 1);
      // CCW winding
      indices.push_back(tl);
      indices.push_back(bl);
      indices.push_back(br);
      indices.push_back(tl);
      indices.push_back(br);
      indices.push_back(tr);
    }
  }

  indexCount_ = static_cast<uint32_t>(indices.size());

  auto& ctx = etna::get_context();

  vbo = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = verts.size() * sizeof(glm::vec2),
      .bufferUsage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "clipmap_vbo",
    });
  ibo = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = indices.size() * sizeof(uint32_t),
      .bufferUsage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "clipmap_ibo",
    });

  {
    auto oneShot = ctx.createOneShotCmdMgr();
    transferHelper.uploadBuffer<glm::vec2>(*oneShot, vbo, 0, std::span<const glm::vec2>(verts));
  }
  {
    auto oneShot = ctx.createOneShotCmdMgr();
    transferHelper.uploadBuffer<uint32_t>(*oneShot, ibo, 0, std::span<const uint32_t>(indices));
  }
}

std::vector<ClipmapMesh::Footprint> ClipmapMesh::buildLevelFootprints() const
{
  return {Footprint{
    .firstIndex = 0,
    .indexCount = indexCount_,
    .vertexOffset = 0,
    .localOriginGrid = glm::vec2(0.f),
  }};
}
