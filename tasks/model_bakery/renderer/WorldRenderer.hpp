#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/Buffer.hpp>
#include <etna/BlockingTransferHelper.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <cstdint>
#include <filesystem>
#include <vector>
#include <glm/glm.hpp>

#include "scene/SceneManager.hpp"
#include "wsi/Keyboard.hpp"

#include "FramePacket.hpp"


class WorldRenderer
{
public:
  enum class SceneType
  {
    SimpleMeshes,
    LowPolyDarkTown,
    Avocado,
  };

  WorldRenderer();

  void loadScene(std::filesystem::path path);
  void loadSceneByType(SceneType type);

  void loadShaders();
  void allocateResources(glm::uvec2 swapchain_resolution);
  void setupPipelines(vk::Format swapchain_format);

  void debugInput(const Keyboard& kb);
  void update(const FramePacket& packet);
  void drawGui();
  void renderWorld(
    vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view);

private:
  // Performs frustum culling and prepares instance data for rendering
  struct CulledRenderElements
  {
    struct Element
    {
      std::size_t relemIdx;
      std::vector<glm::mat4x4> visibleMatrices;
    };
    std::vector<Element> elements;
  };

  void renderScene(
    vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm, vk::PipelineLayout pipeline_layout);

  void performFrustumCulling(const glm::mat4x4& projView);

  void prepareInstanceMatrices();

  void uploadSceneTextures();

  std::filesystem::path modifyPathForBaking(std::filesystem::path path) const;


private:
  std::unique_ptr<SceneManager> sceneMgr;

  etna::Image mainViewDepth;
  std::unique_ptr<etna::BlockingTransferHelper> transferHelper;

  // Buffers for instanced rendering and culling
  etna::Buffer instanceMatricesBuffer;
  CulledRenderElements culledElements;

  struct PushConstants
  {
    glm::mat4x4 projView;
    uint32_t isBaked = 0;
  } pushConst2M;

  glm::mat4x4 worldViewProj;

  etna::GraphicsPipeline staticMeshPipeline{};
  etna::Sampler albedoSampler{};
  std::vector<etna::Image> albedoTextures;

  bool logEnabled = true;
  std::uint32_t logFrameCounter = 0;

  bool bakedEnabled = true;
  SceneType selectedScene = SceneType::LowPolyDarkTown;
  float imguiScale = 1.5f;

  glm::uvec2 resolution;
};
