#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/Buffer.hpp>
#include <etna/BlockingTransferHelper.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/ComputePipeline.hpp>
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

  void performFrustumCulling(const glm::mat4x4& proj_view);

  void prepareInstanceMatrices();

  void uploadSceneTextures();

  std::filesystem::path modifyPathForBaking(std::filesystem::path path) const;


private:
  std::unique_ptr<SceneManager> sceneMgr;
  
  etna::Image mainViewDepth;
  etna::Image hdrTarget;
  etna::Sampler hdrSampler;
  std::unique_ptr<etna::BlockingTransferHelper> transferHelper;

  // Buffers for instanced rendering and culling
  etna::Buffer instanceMatricesBuffer;
  CulledRenderElements culledElements;

  struct PushConstants
  {
    glm::mat4x4 proj_view;
    // xyz=world-space camera position (needed for Fresnel / view vector), w unused
    glm::vec4 cameraPos{0.f, 0.f, 0.f, 0.f};
    uint32_t isBaked{0};
    uint32_t debugMode{0};
  } pushConst;

  // GPU-side material table: one entry per render element (all relems in the scene).
  // Filled once at scene load; contains texture indices + factor overrides.
  struct RelemMat
  {
    uint32_t baseColorIdx;
    uint32_t metalRoughIdx;
    uint32_t normalIdx;
    uint32_t occlusionIdx;
    glm::vec4 baseColorFactor;
    glm::vec4 materialParams;
  };
  etna::Buffer relemMaterialsBuffer;
  etna::Buffer indirectBuffer;    // VkDrawIndexedIndirectCommand[]
  etna::Buffer drawMappingBuffer; // uint32_t[] — draw_id → relemIdx

  // Persistent bindless descriptor set — one binding with all scene textures.
  // Created once per scene load; never recreated per frame.
  etna::PersistentDescriptorSet bindlessTextureSet;

  glm::mat4x4 worldViewProj;
  glm::vec3 cameraWorldPos{0.f};

  float previousTime = 0.f;
  float deltaTime = 0.f;

  // Adaptive-exposure tuning (exposed via ImGui later).
  float adaptationSpeed = 2.5f;
  float keyValue = 0.18f;
  float minExposure = 0.01f;
  float maxExposure = 100.f;
  int tonemapMode = 1; // 0 Reinhard, 1 ACES, 2 None

  // Sun direction shared between sky and main BRDF for visual consistency.
  glm::vec3 sunDirection{20.f, 20.f, 20.f};
  glm::vec3 sunColor{1.0f, 0.95f, 0.85f};
  float sunIntensity = 3.0f;

  etna::GraphicsPipeline staticMeshPipeline{};
  etna::GraphicsPipeline postprocessPipeline{};
  etna::Buffer luminanceStatsBuffer;
  etna::ComputePipeline clearStatsPipeline{};
  etna::ComputePipeline minmaxPipeline{};
  etna::ComputePipeline histogramPipeline{};
  etna::ComputePipeline reducePipeline{};
  etna::GraphicsPipeline skyboxPipeline{};
  etna::Sampler albedoSampler{};
  // All GPU images referenced by scene materials, indexed by TextureId
  // Includes built-in fallback textures at the start of the array
  std::vector<etna::Image> sceneTextures;

  bool logEnabled = true;
  std::uint32_t logFrameCounter = 0;

  bool bakedEnabled = false;
  SceneType selectedScene = SceneType::LowPolyDarkTown;
  float imguiScale = 1.5f;

  // 0=Shaded, 1=BaseColor, 2=Normal, 3=MetalRough, 4=Occlusion
  int debugMode = 0;

  glm::uvec2 resolution;
};
