#include "WorldRenderer.hpp"

#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Profiling.hpp>
#include <glm/ext.hpp>
#include <iostream>


WorldRenderer::WorldRenderer()
  : sceneMgr{std::make_unique<SceneManager>()},
  maxDrawnInstances(4096)
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

  mModels.emplace(
    ctx.getMainWorkCount(),
    [&ctx, maxDrawnInstances = maxDrawnInstances](std::size_t i) {
      return ctx.createBuffer(etna::Buffer::CreateInfo{
        .size = maxDrawnInstances * sizeof(glm::mat4x4),
        .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
        .name = fmt::format("instanceMatrix{}", i)});
    });

  mModelsCount.resize(maxDrawnInstances, 0);
}

void WorldRenderer::loadScene(std::filesystem::path path)
{
  sceneMgr->selectBakerScene(path);
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

void WorldRenderer::renderScene(
  vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm, vk::PipelineLayout pipeline_layout)
{
  if (!sceneMgr->getVertexBuffer())
    return;

  cmd_buf.bindVertexBuffers(0, {sceneMgr->getVertexBuffer()}, {0});
  cmd_buf.bindIndexBuffer(sceneMgr->getIndexBuffer(), 0, vk::IndexType::eUint32);

  auto& mModelsBuffer = mModels->get();
  mModelsBuffer.map();
  glm::mat4x4* mModelsVec = std::bit_cast<glm::mat4x4*>(mModelsBuffer.data());
  std::size_t mModelsId = 0;

  auto instanceMeshes = sceneMgr->getInstanceMeshes();
  auto instanceMatrices = sceneMgr->getInstanceMatrices();

  auto meshes = sceneMgr->getMeshes();
  auto relems = sceneMgr->getRenderElements();
  auto bounds = sceneMgr->getBounds();

  for (std::size_t instIdx = 0; instIdx < instanceMeshes.size(); ++instIdx) {
    auto& mModel = instanceMatrices[instIdx];
    const auto meshIdx = instanceMeshes[instIdx];
    for (std::size_t j = 0; j < meshes[meshIdx].relemCount; ++j) {
      const auto relemIdx = meshes[meshIdx].firstRelem + j;
      if (!frustumCulled(bounds[meshIdx], glob_tm * mModel)) {
        mModelsVec[mModelsId++] = mModel;
        ++mModelsCount[relemIdx];
      }
    }
  }

  mModelsBuffer.unmap();

  auto descriptor_set = etna::create_descriptor_set(
    etna::get_shader_program("static_mesh_material").getDescriptorLayoutId(0),
    cmd_buf,
    {etna::Binding{0, mModelsBuffer.genBinding()}});
  auto vkSet = descriptor_set.getVkSet();
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1, &vkSet, 0, nullptr);

  pushConst2M.projView = glob_tm;
  cmd_buf.pushConstants<PushConstants>(
      pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, {pushConst2M});

  uint32_t offset = 0;
  for (std::size_t relemIdx = 0; relemIdx < mModelsCount.size(); ++relemIdx) {
    if (mModelsCount[relemIdx] != 0) {
      cmd_buf.drawIndexed(relems[relemIdx].indexCount, mModelsCount[relemIdx],
                          relems[relemIdx].indexOffset, relems[relemIdx].vertexOffset, offset);
      offset += mModelsCount[relemIdx];
      mModelsCount[relemIdx] = 0;
    }
  }
}

void WorldRenderer::renderWorld(
  vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view)
{
  ETNA_PROFILE_GPU(cmd_buf, renderWorld);

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

bool WorldRenderer::frustumCulled(const std::pair<glm::vec3, glm::vec3> bound, const glm::mat4x4 globalTransform) {
  float leeway = bound.second.x - bound.first.x +\
                 bound.second.y - bound.first.y +\
                 bound.second.z - bound.first.z;
  for (std::size_t i = 0; i < 8; ++i) {
    float x = bound.first.x * (i & 1) + bound.second.x * ((i & 1) ^ 1);
    float y = bound.first.y * (i & 2) + bound.second.y * ((i & 2) ^ 1);
    float z = bound.first.z * (i & 4) + bound.second.z * ((i & 4) ^ 1);
    glm::vec4 pos = globalTransform * glm::vec4(x, y, z, 1.0f);
    bool onScreen = pos.x >= -pos.w - leeway && pos.x <= pos.w + leeway &&
                    pos.y >= -pos.w - leeway && pos.y <= pos.w + leeway &&
                    pos.z >= -pos.w - leeway && pos.z <= pos.w + leeway;
    if (onScreen) {
      return false;
    }
  }
  glm::vec3 center = (bound.first + bound.second) / 2.0f; // helps with huge objects
  glm::vec4 pos = globalTransform * glm::vec4(center.x, center.y, center.z, 1.0f);
  bool onScreen = pos.x >= -pos.w - leeway && pos.x <= pos.w + leeway &&
                  pos.y >= -pos.w - leeway && pos.y <= pos.w + leeway &&
                  pos.z >= -pos.w - leeway && pos.z <= pos.w + leeway;
  return !onScreen;
}