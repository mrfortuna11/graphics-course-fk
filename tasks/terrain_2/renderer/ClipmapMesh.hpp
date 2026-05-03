#pragma once

#include <etna/Buffer.hpp>
#include <etna/BlockingTransferHelper.hpp>
#include <glm/glm.hpp>
#include <vector>

class ClipmapMesh
{
public:
  struct Footprint
  {
    uint32_t  firstIndex;
    uint32_t  indexCount;
    int32_t   vertexOffset; 
    glm::vec2 localOriginGrid; 
  };

  
  explicit ClipmapMesh(uint32_t n, etna::BlockingTransferHelper& transferHelper);

 
  std::vector<Footprint> buildLevelFootprints() const;

  vk::Buffer vertexBuffer() const { return vbo.get(); }
  vk::Buffer indexBuffer()  const { return ibo.get(); }

  uint32_t n() const { return n_; }

private:
  void buildGeometry(etna::BlockingTransferHelper& transferHelper);

  uint32_t n_;
  etna::Buffer vbo;
  etna::Buffer ibo;
  uint32_t indexCount_ = 0;
};
