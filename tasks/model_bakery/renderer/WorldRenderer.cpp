#include "WorldRenderer.hpp"

#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Profiling.hpp>
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <glm/ext.hpp>

#include <cstddef>
#include <string>

WorldRenderer::WorldRenderer()
  : sceneMgr{std::make_unique<SceneManager>()}
{
}

void WorldRenderer::allocateResources(glm::uvec2 swapchain_resolution)
{
  resolution = swapchain_resolution;

  auto& ctx = etna::get_context();

  mainViewDepth = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "main_view_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
  });

  transferHelper = std::make_unique<etna::BlockingTransferHelper>(
    etna::BlockingTransferHelper::CreateInfo{.stagingSize = 4u * 1024u * 1024u});

  albedoSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "albedo_sampler",
  });

  const vk::DeviceSize maxInstancesBytes = 10000u * sizeof(glm::mat4x4);
  instanceMatricesBuffer = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = maxInstancesBytes,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "instanceMatrices",
  });
}

void WorldRenderer::loadScene(std::filesystem::path path)
{
  path = modifyPathForBaking(path);
  sceneMgr->selectScene(path, bakedEnabled);
  uploadSceneTextures();
}

void WorldRenderer::uploadSceneTextures()
{
  albedoTextures.clear();

  auto cpuTextures = sceneMgr->getAlbedoTextures();
  albedoTextures.reserve(cpuTextures.size());

  auto& ctx = etna::get_context();
  auto oneShot = ctx.createOneShotCmdMgr();

  for (std::size_t i = 0; i < cpuTextures.size(); ++i)
  {
    const auto& texture = cpuTextures[i];
    auto gpuImage = ctx.createImage(etna::Image::CreateInfo{
      .extent = vk::Extent3D{texture.width, texture.height, 1},
      .name = std::string("albedo_texture_") + std::to_string(i),
      .format = vk::Format::eR8G8B8A8Srgb,
      .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    });

    transferHelper->uploadImage(
      *oneShot,
      gpuImage,
      0,
      0,
      std::span<const std::byte>{
        reinterpret_cast<const std::byte*>(texture.rgba8.data()), texture.rgba8.size()});

    albedoTextures.push_back(std::move(gpuImage));
  }

  const auto relems = sceneMgr->getRenderElements();
  std::size_t texturedRelems = 0;
  std::size_t untexturedRelems = 0;
  for (const auto& relem : relems)
  {
    if (relem.albedoTextureIdx != 0u)
      ++texturedRelems;
    else
      ++untexturedRelems;
  }

  spdlog::info(
    "Scene materials: {} textured relems, {} untextured relems, {} uploaded albedo textures",
    texturedRelems,
    untexturedRelems,
    albedoTextures.size());
}

void WorldRenderer::loadShaders()
{
  etna::create_program(
    "static_mesh_material",
    {MODEL_BAKERY_RENDERER_SHADERS_ROOT "static_mesh.frag.spv",
     MODEL_BAKERY_RENDERER_SHADERS_ROOT "static_mesh.vert.spv"});
  etna::create_program("static_mesh", {MODEL_BAKERY_RENDERER_SHADERS_ROOT "static_mesh.vert.spv"});
}

void WorldRenderer::setupPipelines(vk::Format swapchain_format)
{
  etna::VertexShaderInputDescription sceneVertexInputDesc{
    .bindings = {etna::VertexShaderInputDescription::Binding{
      .byteStreamDescription = sceneMgr->getVertexFormatDescription(),
    }},
  };

  auto& pipelineManager = etna::get_context().getPipelineManager();

  staticMeshPipeline = {};
  staticMeshPipeline = pipelineManager.createGraphicsPipeline(
    "static_mesh_material",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = sceneVertexInputDesc,
      .rasterizationConfig =
        vk::PipelineRasterizationStateCreateInfo{
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eBack,
          .frontFace = vk::FrontFace::eCounterClockwise,
          .lineWidth = 1.f,
        },
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {swapchain_format},
          .depthAttachmentFormat = vk::Format::eD32Sfloat,
        },
    });
}

void WorldRenderer::debugInput(const Keyboard&) {}

void WorldRenderer::update(const FramePacket& packet)
{
  ZoneScoped;

  // calc camera matrix
  {
    const float aspect = float(resolution.x) / float(resolution.y);
    worldViewProj = packet.mainCam.projTm(aspect) * packet.mainCam.viewTm();
  }
}

void WorldRenderer::performFrustumCulling(const glm::mat4x4& projView)
{
  auto meshes = sceneMgr->getMeshes();
  auto instanceMeshes = sceneMgr->getInstanceMeshes();
  auto instanceMatrices = sceneMgr->getInstanceMatrices();
  auto relemAABBs = sceneMgr->getRenderElementAABBs();

  CulledRenderElements result;

  for (std::size_t meshIdx = 0; meshIdx < meshes.size(); ++meshIdx)
  {
    for (std::size_t j = 0; j < meshes[meshIdx].relemCount; ++j)
    {
      const auto relemIdx = meshes[meshIdx].firstRelem + j;
      const auto& aabb = relemAABBs[relemIdx];

      std::vector<glm::mat4x4> visibleMatrices;
      visibleMatrices.reserve(16);

      for (std::size_t instIdx = 0; instIdx < instanceMeshes.size(); ++instIdx)
      {
        if (instanceMeshes[instIdx] != static_cast<uint32_t>(meshIdx))
          continue;

        const glm::mat4x4 model = instanceMatrices[instIdx];
        const glm::mat4x4 mvp = projView * model;

        bool visible = false;
        for (int cx = 0; cx < 2 && !visible; ++cx)
          for (int cy = 0; cy < 2 && !visible; ++cy)
            for (int cz = 0; cz < 2 && !visible; ++cz)
            {
              glm::vec3 corner{
                cx ? aabb.max.x : aabb.min.x,
                cy ? aabb.max.y : aabb.min.y,
                cz ? aabb.max.z : aabb.min.z,
              };
              glm::vec4 clip = mvp * glm::vec4(corner, 1.0f);
              if (clip.w == 0.0f)
                continue;
              glm::vec3 ndc = glm::vec3(clip) / clip.w;
              if (
                ndc.x >= -1.0f && ndc.x <= 1.0f && ndc.y >= -1.0f && ndc.y <= 1.0f &&
                ndc.z >= -1.0f && ndc.z <= 1.0f)
                visible = true;
            }

        if (visible)
          visibleMatrices.push_back(model);
      }

      if (!visibleMatrices.empty())
        result.elements.push_back({relemIdx, std::move(visibleMatrices)});
    }
  }
  culledElements = std::move(result);
}

void WorldRenderer::prepareInstanceMatrices()
{
  std::vector<glm::mat4x4> allVisibleMatrices;
  for (const auto& element : culledElements.elements)
  {
    allVisibleMatrices.insert(
      allVisibleMatrices.end(), element.visibleMatrices.begin(), element.visibleMatrices.end());
  }

  if (!allVisibleMatrices.empty())
  {
    auto& ctx = etna::get_context();
    auto oneShot = ctx.createOneShotCmdMgr();
    transferHelper->uploadBuffer<glm::mat4x4>(
      *oneShot, instanceMatricesBuffer, 0, std::span<const glm::mat4x4>(allVisibleMatrices));
  }
}

void WorldRenderer::renderScene(
  vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm, vk::PipelineLayout pipeline_layout)
{
  if (!sceneMgr->getVertexBuffer())
    return;

  cmd_buf.bindVertexBuffers(0, {sceneMgr->getVertexBuffer()}, {0});
  cmd_buf.bindIndexBuffer(sceneMgr->getIndexBuffer(), 0, vk::IndexType::eUint32);

  pushConst2M.projView = glob_tm;
  pushConst2M.isBaked = bakedEnabled ? 1u : 0u;

  auto relems = sceneMgr->getRenderElements();
  auto programInfo = etna::get_shader_program("static_mesh_material");

  auto instanceSet = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmd_buf,
    {etna::Binding{0, instanceMatricesBuffer.genBinding()}});
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, {instanceSet.getVkSet()}, {});

  uint32_t instanceOffset = 0;
  for (const auto& element : culledElements.elements)
  {
    const auto& relem = relems[element.relemIdx];
    const uint32_t instanceCount = static_cast<uint32_t>(element.visibleMatrices.size());

    cmd_buf.pushConstants<PushConstants>(
      pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, {pushConst2M});

    const auto textureIdx = relem.albedoTextureIdx < albedoTextures.size() ? relem.albedoTextureIdx
                                                                           : 0u;

    auto materialSet = etna::create_descriptor_set(
      programInfo.getDescriptorLayoutId(1),
      cmd_buf,
      {etna::Binding{0,
                     albedoTextures[textureIdx].genBinding(
                       albedoSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)}});

    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics, pipeline_layout, 1, {materialSet.getVkSet()}, {});

    cmd_buf.drawIndexed(
      relem.indexCount, instanceCount, relem.indexOffset, relem.vertexOffset, instanceOffset);

    instanceOffset += instanceCount;
  }
}

void WorldRenderer::renderWorld(
  vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view)
{
  ETNA_PROFILE_GPU(cmd_buf, renderWorld);

  performFrustumCulling(worldViewProj);

  if (logEnabled)
  {
    const std::size_t totalInstances = sceneMgr->getInstanceMatrices().size();
    std::size_t visibleInstances = 0;
    for (const auto& element : culledElements.elements)
      visibleInstances += element.visibleMatrices.size();

    if ((logFrameCounter++ % 60u) == 0u)
      spdlog::info(
        "Frustum culling: visible instances = {}, total instances = {}",
        visibleInstances,
        totalInstances);
  }

  prepareInstanceMatrices();

  // draw final scene to screen
  {
    ETNA_PROFILE_GPU(cmd_buf, renderForward);

    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},
      {{.image = target_image, .view = target_image_view}},
      {.image = mainViewDepth.get(), .view = mainViewDepth.getView({})});

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, staticMeshPipeline.getVkPipeline());

    renderScene(cmd_buf, worldViewProj, staticMeshPipeline.getVkPipelineLayout());
  }
}

std::filesystem::path WorldRenderer::modifyPathForBaking(std::filesystem::path path) const
{
  if (!bakedEnabled)
    return path;

  std::string pathStr = path.string();
  const std::string bakedSuffix = "_baked";

  // Find the extension
  size_t extensionPos = pathStr.rfind('.');
  if (extensionPos != std::string::npos)
  {
    pathStr.insert(extensionPos, bakedSuffix);
  }

  return pathStr;
}

void WorldRenderer::loadSceneByType(SceneType type)
{
  std::filesystem::path path;
  switch (type)
  {
  case SceneType::SimpleMeshes:
    path = GRAPHICS_COURSE_RESOURCES_ROOT "/scenes/SimpleMeshes/glTF/SimpleMeshes.gltf";
    break;
  case SceneType::LowPolyDarkTown:
    path = GRAPHICS_COURSE_RESOURCES_ROOT "/scenes/low_poly_dark_town/scene.gltf";
    break;
  case SceneType::Avocado:
    path = GRAPHICS_COURSE_RESOURCES_ROOT "/scenes/Avocado/Avocado.gltf";
    break;
  }

  path = modifyPathForBaking(path);
  sceneMgr->selectScene(path, bakedEnabled);
  uploadSceneTextures();
}

void WorldRenderer::drawGui()
{
  ImGui::Begin("Simple render settings");

  if (ImGui::InputFloat("ImGui scale", &imguiScale, 0.1f, 0.5f, "%.1f"))
  {
    if (imguiScale < 0.5f)
      imguiScale = 0.5f;
    else if (imguiScale > 3.0f)
      imguiScale = 3.0f;
    ImGui::GetIO().FontGlobalScale = imguiScale;
  }

  ImGui::Checkbox("Enable logging", &logEnabled);

  bool bakedChanged = ImGui::Checkbox("Use baked scene", &bakedEnabled);

  const char* scenes[] = {"SimpleMeshes", "Low Poly Dark Town", "Avocado"};
  int currentSceneIdx = static_cast<int>(selectedScene);

  bool sceneChanged = ImGui::Combo("Scene", &currentSceneIdx, scenes, IM_ARRAYSIZE(scenes));

  if (sceneChanged || bakedChanged)
  {
    selectedScene = static_cast<SceneType>(currentSceneIdx);
    loadSceneByType(selectedScene);
  }

  ImGui::Text(
    "Application average %.1f ms/frame (%.1f FPS)",
    1000.0f / ImGui::GetIO().Framerate,
    ImGui::GetIO().Framerate);

  ImGui::End();
}
