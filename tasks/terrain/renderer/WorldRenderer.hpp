#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/Buffer.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/GpuSharedResource.hpp>
#include <glm/glm.hpp>

#include "scene/SceneManager.hpp"
#include "wsi/Keyboard.hpp"

#include "FramePacket.hpp"


class WorldRenderer
{
public:
  WorldRenderer(const etna::GpuWorkCount& workCount);

  void loadScene(std::filesystem::path path);

  void loadShaders();
  void allocateResources(glm::uvec2 swapchain_resolution);
  void setupPipelines(vk::Format swapchain_format);

  void debugInput(const Keyboard& kb);
  void update(const FramePacket& packet);
  void drawGui();
  void renderWorld(
    vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view);

private:
  void renderScene(
    vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm, vk::PipelineLayout pipeline_layout);
  void createTerrainMap(vk::CommandBuffer cmd_buf);
  void renderTerrain(vk::CommandBuffer cmd_buf);

  bool shouldCull(glm::mat4 mModel, BoundingBox box);

private:
  std::unique_ptr<SceneManager> sceneMgr;

  etna::Image mainViewDepth;
  etna::Image perlinTex;
  etna::Image normalMap;
  etna::GpuSharedResource<etna::Buffer> modelMatrices;

  glm::mat4x4 worldViewProj;
  glm::vec3 eye;
  glm::mat4x4 lightMatrix;
  float nearPlane;
  float farPlane;

  etna::GraphicsPipeline staticMeshPipeline{};
  etna::ComputePipeline perlinPipeline{};
  etna::ComputePipeline normalPipeline{};
  etna::GraphicsPipeline terrainPipeline{};

  struct TerrainPushConst
  {
    glm::mat4 proj;
    glm::vec3 eye;
  };

  etna::Sampler perlinSampler;

  glm::uvec2 resolution;
};
