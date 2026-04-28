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

  hdrTarget = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "hdr_target",
    .format = vk::Format::eB10G11R11UfloatPack32,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  hdrSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eClampToEdge,
    .name = "hdr_sampler",
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

  const vk::DeviceSize maxRelemMatsBytes = 20000u * sizeof(RelemMat);
  relemMaterialsBuffer = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = maxRelemMatsBytes,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "relemMaterials",
  });

  constexpr vk::DeviceSize LUMINANCE_STATS_BYTES =
    sizeof(std::uint32_t) * 2u + sizeof(std::uint32_t) * 128u + sizeof(float);
  luminanceStatsBuffer = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = LUMINANCE_STATS_BYTES,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "luminance_stats",
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
  sceneTextures.clear();

  auto cpuTextures = sceneMgr->getTextures();
  sceneTextures.reserve(cpuTextures.size());

  auto& ctx = etna::get_context();
  auto oneShot = ctx.createOneShotCmdMgr();

  for (std::size_t i = 0; i < cpuTextures.size(); ++i)
  {
    const auto& texture = cpuTextures[i];

    const vk::Format format =
      texture.isSrgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;

    auto gpuImage = ctx.createImage(etna::Image::CreateInfo{
      .extent = vk::Extent3D{texture.width, texture.height, 1},
      .name = std::string("scene_texture_") + std::to_string(i),
      .format = format,
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

    sceneTextures.push_back(std::move(gpuImage));
  }

  const auto materials = sceneMgr->getMaterials();
  std::size_t withBaseColor = 0;
  std::size_t withMetalRough = 0;
  std::size_t withNormal = 0;
  std::size_t withOcclusion = 0;
  for (const auto& m : materials)
  {
    if (m.baseColorTex != TextureId::DefaultBaseColor)
      ++withBaseColor;
    if (m.metallicRoughnessTex != TextureId::DefaultMetallicRoughness)
      ++withMetalRough;
    if (m.normalTex != TextureId::DefaultNormal)
      ++withNormal;
    if (m.occlusionTex != TextureId::DefaultOcclusion)
      ++withOcclusion;
  }

  spdlog::info(
    "Scene: {} materials, {} textures. Non-default: baseColor={}, metalRough={}, normal={}, "
    "occlusion={}",
    materials.size(),
    sceneTextures.size(),
    withBaseColor,
    withMetalRough,
    withNormal,
    withOcclusion);

  {
    auto relems = sceneMgr->getRenderElements();
    std::vector<RelemMat> cpuMats;
    cpuMats.reserve(relems.size());

    auto resolveIdx = [&](TextureId id, TextureId fallback) -> uint32_t {
      const auto idx = static_cast<uint32_t>(id);
      if (idx < sceneTextures.size())
        return idx;
      return static_cast<uint32_t>(fallback);
    };

    for (const auto& relem : relems)
    {
      const auto matIdx = static_cast<uint32_t>(relem.materialId);
      const auto& m = materials[matIdx < materials.size() ? matIdx : 0u];

      cpuMats.push_back(RelemMat{
        .baseColorIdx = resolveIdx(m.baseColorTex, TextureId::DefaultBaseColor),
        .metalRoughIdx = resolveIdx(m.metallicRoughnessTex, TextureId::DefaultMetallicRoughness),
        .normalIdx = resolveIdx(m.normalTex, TextureId::DefaultNormal),
        .occlusionIdx = resolveIdx(m.occlusionTex, TextureId::DefaultOcclusion),
        .baseColorFactor = m.baseColorFactor,
        .materialParams = glm::vec4(
          m.metallicFactor, m.roughnessFactor, m.normalScale, m.occlusionStrength),
      });
    }

    if (!cpuMats.empty())
    {
      auto oneShotMat = ctx.createOneShotCmdMgr();
      transferHelper->uploadBuffer<RelemMat>(
        *oneShotMat,
        relemMaterialsBuffer,
        0,
        std::span<const RelemMat>(cpuMats));
    }
  }

  {
    auto programInfo = etna::get_shader_program("static_mesh_material");
    auto layoutId    = programInfo.getDescriptorLayoutId(2);

    std::vector<etna::Binding> texBindings;
    texBindings.reserve(sceneTextures.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(sceneTextures.size()); ++i)
    {
      texBindings.emplace_back(
        0u,
        sceneTextures[i].genBinding(albedoSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal),
        i);
    }

    bindlessTextureSet =
      etna::get_context().getPersistentDescriptorPool().allocateSet(
        layoutId, std::move(texBindings), /*allow_unbound_slots=*/true);
  }
}

void WorldRenderer::loadShaders()
{
  etna::create_program(
    "static_mesh_material",
    {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "static_mesh.frag.spv",
     BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "static_mesh.vert.spv"});
  etna::create_program("static_mesh", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "static_mesh.vert.spv"});

  etna::create_program(
    "postprocess",
    {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "postprocess.vert.spv",
     BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "postprocess.frag.spv"});

  etna::create_program(
    "clear_stats", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "clear_stats.comp.spv"});
  etna::create_program(
    "minmax", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "minmax.comp.spv"});
  etna::create_program(
    "histogram", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "histogram.comp.spv"});
  etna::create_program(
    "reduce", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "reduce.comp.spv"});

  etna::create_program(
    "skybox",
    {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "skybox.vert.spv",
     BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "skybox.frag.spv"});
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
          .colorAttachmentFormats = {vk::Format::eB10G11R11UfloatPack32},
          .depthAttachmentFormat = vk::Format::eD32Sfloat,
        },
    });

  postprocessPipeline = pipelineManager.createGraphicsPipeline(
    "postprocess",
    etna::GraphicsPipeline::CreateInfo{
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {swapchain_format},
        },
    });

  clearStatsPipeline = pipelineManager.createComputePipeline("clear_stats", {});
  minmaxPipeline = pipelineManager.createComputePipeline("minmax", {});
  histogramPipeline = pipelineManager.createComputePipeline("histogram", {});
  reducePipeline = pipelineManager.createComputePipeline("reduce", {});

  // Skybox: depth test enabled, depth write OFF. Vertex shader puts the
  // fullscreen tri at z = 1, so it only passes where nothing was drawn
  skyboxPipeline = pipelineManager.createGraphicsPipeline(
    "skybox",
    etna::GraphicsPipeline::CreateInfo{
      .rasterizationConfig =
        vk::PipelineRasterizationStateCreateInfo{
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eNone,
          .frontFace = vk::FrontFace::eCounterClockwise,
          .lineWidth = 1.f,
        },
      .depthConfig =
        vk::PipelineDepthStencilStateCreateInfo{
          .depthTestEnable = vk::True,
          .depthWriteEnable = vk::False,
          .depthCompareOp = vk::CompareOp::eLessOrEqual,
          .maxDepthBounds = 1.f,
        },
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {vk::Format::eB10G11R11UfloatPack32},
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
    cameraWorldPos = packet.mainCam.position;
  }

  // Track frame delta for temporal exposure smoothing.
  if (previousTime <= 0.f)
    deltaTime = 1.f / 60.f;
  else
    deltaTime = std::max(packet.currentTime - previousTime, 0.f);
  previousTime = packet.currentTime;
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

  pushConst.projView = glob_tm;
  pushConst.cameraPos = glm::vec4(cameraWorldPos, 0.0f);
  pushConst.isBaked = bakedEnabled ? 1u : 0u;
  pushConst.debugMode = static_cast<std::uint32_t>(debugMode);

  auto programInfo = etna::get_shader_program("static_mesh_material");

  // Set 0: instance matrices (per-frame).
  auto instanceSet = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmd_buf,
    {etna::Binding{0, instanceMatricesBuffer.genBinding()}});
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, {instanceSet.getVkSet()}, {});

  // Set 1: RelemMaterials SSBO — bound once for the whole scene.
  auto materialsSet = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(1),
    cmd_buf,
    {etna::Binding{0, relemMaterialsBuffer.genBinding()}});
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics, pipeline_layout, 1, {materialsSet.getVkSet()}, {});

  // Set 2: bindless texture array — persistent, bound once.
  vk::DescriptorSet bindlessVkSet = bindlessTextureSet.getVkSet();
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics, pipeline_layout, 2, {bindlessVkSet}, {});

  auto relems = sceneMgr->getRenderElements();

  uint32_t instanceOffset = 0;
  for (const auto& element : culledElements.elements)
  {
    const auto& relem = relems[element.relemIdx];
    const uint32_t instanceCount = static_cast<uint32_t>(element.visibleMatrices.size());

    // Only relemIdx changes per draw — no per-relem descriptor-set updates.
    pushConst.relemIdx = static_cast<uint32_t>(element.relemIdx);
    cmd_buf.pushConstants<PushConstants>(
      pipeline_layout,
      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
      0,
      {pushConst});

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

  // Render the world into the HDR offscreen target
  {
    ETNA_PROFILE_GPU(cmd_buf, renderForward);

    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},
      {{.image = hdrTarget.get(), .view = hdrTarget.getView({})}},
      {.image = mainViewDepth.get(), .view = mainViewDepth.getView({})});

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, staticMeshPipeline.getVkPipeline());

    renderScene(cmd_buf, worldViewProj, staticMeshPipeline.getVkPipelineLayout());

    {
      ETNA_PROFILE_GPU(cmd_buf, skybox);

      struct SkyPush
      {
        glm::mat4 invProjView;
        glm::vec4 cameraPos;
        glm::vec4 sunDir;
        glm::vec4 sunColor;
      } skyPush{
        glm::inverse(worldViewProj),
        glm::vec4(cameraWorldPos, 0.f),
        glm::vec4(glm::normalize(sunDirection), 0.f),
        glm::vec4(sunColor, sunIntensity),
      };

      const auto skyLayout = skyboxPipeline.getVkPipelineLayout();
      cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, skyboxPipeline.getVkPipeline());
      cmd_buf.pushConstants<SkyPush>(
        skyLayout,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        0,
        {skyPush});
      cmd_buf.draw(3, 1, 0, 0);
    }
  }

  // Adaptive-exposure compute: transition HDR once for sampling in compute + frag
  etna::set_state(
    cmd_buf,
    hdrTarget.get(),
    vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmd_buf);

  // Clear per-frame stats (histogram zeros + min/max seeds)
  {
    ETNA_PROFILE_GPU(cmd_buf, clearStats);

    auto info = etna::get_shader_program("clear_stats");
    auto statsBind = luminanceStatsBuffer.genBinding();
    auto descSet = etna::create_descriptor_set(
      info.getDescriptorLayoutId(0), cmd_buf, {etna::Binding{0, statsBind}});
    vk::DescriptorSet vkSet = descSet.getVkSet();

    cmd_buf.bindPipeline(
      vk::PipelineBindPoint::eCompute, clearStatsPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      clearStatsPipeline.getVkPipelineLayout(),
      0, 1, &vkSet, 0, nullptr);
    cmd_buf.dispatch(1, 1, 1);
  }

  // Barrier: clear writes → minmax reads/writes
  etna::set_state(
    cmd_buf,
    luminanceStatsBuffer.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
  etna::flush_barriers(cmd_buf);

  // Min/max log-luminance via shared-memory reduction
  {
    ETNA_PROFILE_GPU(cmd_buf, minmax);

    auto info = etna::get_shader_program("minmax");
    auto statsBind = luminanceStatsBuffer.genBinding();
    auto hdrBind = hdrTarget.genBinding(
      hdrSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal);
    auto descSet = etna::create_descriptor_set(
      info.getDescriptorLayoutId(0),
      cmd_buf,
      {etna::Binding{0, statsBind}, etna::Binding{1, hdrBind}});
    vk::DescriptorSet vkSet = descSet.getVkSet();

    cmd_buf.bindPipeline(
      vk::PipelineBindPoint::eCompute, minmaxPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      minmaxPipeline.getVkPipelineLayout(),
      0, 1, &vkSet, 0, nullptr);

    const uint32_t groupsX = (resolution.x + 15u) / 16u;
    const uint32_t groupsY = (resolution.y + 15u) / 16u;
    cmd_buf.dispatch(groupsX, groupsY, 1);
  }

  // Barrier: minmax writes → histogram reads/writes
  etna::set_state(
    cmd_buf,
    luminanceStatsBuffer.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
  etna::flush_barriers(cmd_buf);

  // Histogram: bin every HDR pixel into 128 log-luminance buckets
  {
    ETNA_PROFILE_GPU(cmd_buf, histogram);

    auto info = etna::get_shader_program("histogram");
    auto statsBind = luminanceStatsBuffer.genBinding();
    auto hdrBind = hdrTarget.genBinding(
      hdrSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal);
    auto descSet = etna::create_descriptor_set(
      info.getDescriptorLayoutId(0),
      cmd_buf,
      {etna::Binding{0, statsBind}, etna::Binding{1, hdrBind}});
    vk::DescriptorSet vkSet = descSet.getVkSet();

    cmd_buf.bindPipeline(
      vk::PipelineBindPoint::eCompute, histogramPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      histogramPipeline.getVkPipelineLayout(),
      0, 1, &vkSet, 0, nullptr);

    const uint32_t groupsX = (resolution.x + 15u) / 16u;
    const uint32_t groupsY = (resolution.y + 15u) / 16u;
    cmd_buf.dispatch(groupsX, groupsY, 1);
  }

  // Barrier: histogram writes → reduce reads/writes
  etna::set_state(
    cmd_buf,
    luminanceStatsBuffer.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
  etna::flush_barriers(cmd_buf);

  // Reduce: weighted-average log-luminance → exposure → temporal smoothing
  {
    ETNA_PROFILE_GPU(cmd_buf, reduce);

    struct ReducePush
    {
      float deltaTime;
      float adaptationSpeed;
      float keyValue;
      float minExposure;
      float maxExposure;
    } reducePush{deltaTime, adaptationSpeed, keyValue, minExposure, maxExposure};

    auto info = etna::get_shader_program("reduce");
    auto statsBind = luminanceStatsBuffer.genBinding();
    auto descSet = etna::create_descriptor_set(
      info.getDescriptorLayoutId(0), cmd_buf, {etna::Binding{0, statsBind}});
    vk::DescriptorSet vkSet = descSet.getVkSet();

    cmd_buf.bindPipeline(
      vk::PipelineBindPoint::eCompute, reducePipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      reducePipeline.getVkPipelineLayout(),
      0, 1, &vkSet, 0, nullptr);
    cmd_buf.pushConstants<ReducePush>(
      reducePipeline.getVkPipelineLayout(),
      vk::ShaderStageFlagBits::eCompute,
      0,
      {reducePush});
    cmd_buf.dispatch(1, 1, 1);
  }

  // Barrier: reduce writes smoothedExposure → frag reads it
  etna::set_state(
    cmd_buf,
    luminanceStatsBuffer.get(),
    vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderStorageRead);
  etna::flush_barriers(cmd_buf);

  // Post-process pass: sample HDR, apply exposure + tonemap, write to LDR swapchain
  {
    ETNA_PROFILE_GPU(cmd_buf, postProcess);

    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},
      {{.image = target_image, .view = target_image_view}},
      {});

    auto programInfo = etna::get_shader_program("postprocess");
    auto hdrBinding =
      hdrTarget.genBinding(hdrSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal);
    auto statsBinding = luminanceStatsBuffer.genBinding();
    auto descSet = etna::create_descriptor_set(
      programInfo.getDescriptorLayoutId(0),
      cmd_buf,
      {etna::Binding{0, hdrBinding}, etna::Binding{1, statsBinding}});

    vk::DescriptorSet vkSet = descSet.getVkSet();
    const auto layout = postprocessPipeline.getVkPipelineLayout();
    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, postprocessPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics, layout, 0, 1, &vkSet, 0, nullptr);

    const std::uint32_t tonemapModeU = static_cast<std::uint32_t>(tonemapMode);
    cmd_buf.pushConstants<std::uint32_t>(
      layout, vk::ShaderStageFlagBits::eFragment, 0, {tonemapModeU});

    cmd_buf.draw(3, 1, 0, 0);
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

  const char* debugModes[] = {
    "Shaded", "BaseColor", "Normal (raw)", "MetalRough", "Occlusion"};
  ImGui::Combo("Debug view", &debugMode, debugModes, IM_ARRAYSIZE(debugModes));

  ImGui::Separator();
  ImGui::Text("Adaptive exposure");
  const char* tonemaps[] = {"Reinhard", "ACES", "None (clamp)"};
  ImGui::Combo("Tonemap", &tonemapMode, tonemaps, IM_ARRAYSIZE(tonemaps));
  ImGui::SliderFloat("Adaptation speed", &adaptationSpeed, 0.1f, 10.f, "%.2f");
  ImGui::SliderFloat("Key value", &keyValue, 0.01f, 1.0f, "%.3f");
  ImGui::SliderFloat("Min exposure", &minExposure, 0.001f, 1.f, "%.3f");
  ImGui::SliderFloat("Max exposure", &maxExposure, 1.f, 1000.f, "%.1f");

  ImGui::Text(
    "Application average %.1f ms/frame (%.1f FPS)",
    1000.0f / ImGui::GetIO().Framerate,
    ImGui::GetIO().Framerate);

  ImGui::End();
}
