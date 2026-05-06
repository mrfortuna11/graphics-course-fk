#include "WorldRenderer.hpp"
#include "vulkan/vulkan.hpp"

#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Profiling.hpp>
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <glm/ext.hpp>

#include <array>
#include <cstddef>
#include <string>

WorldRenderer::WorldRenderer()
  : sceneMgr{std::make_unique<SceneManager>()}
  , perlinSampler{etna::Sampler::CreateInfo{
      .filter = vk::Filter::eLinear,
      .addressMode = vk::SamplerAddressMode::eClampToEdge,
      .name = "perlin_sampler"}}
  , clipmapSampler{etna::Sampler::CreateInfo{
      .filter = vk::Filter::eLinear,
      .addressMode = vk::SamplerAddressMode::eRepeat,
      .name = "clipmap_sampler"}}
{
}

void WorldRenderer::allocateResources(glm::uvec2 swapchain_resolution)
{
  resolution = swapchain_resolution;

  auto& ctx = etna::get_context();

  mainViewDepth = ctx.createImage(
    etna::Image::CreateInfo{
      .extent = vk::Extent3D{resolution.x, resolution.y, 1},
      .name = "main_view_depth",
      .format = vk::Format::eD32Sfloat,
      .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
    });

  hdrTarget = ctx.createImage(
    etna::Image::CreateInfo{
      .extent = vk::Extent3D{resolution.x, resolution.y, 1},
      .name = "hdr_target",
      .format = vk::Format::eB10G11R11UfloatPack32,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
    });

  hdrSampler = etna::Sampler(
    etna::Sampler::CreateInfo{
      .filter = vk::Filter::eLinear,
      .addressMode = vk::SamplerAddressMode::eClampToEdge,
      .name = "hdr_sampler",
    });

  transferHelper = std::make_unique<etna::BlockingTransferHelper>(
    etna::BlockingTransferHelper::CreateInfo{.stagingSize = 4u * 1024u * 1024u});

  albedoSampler = etna::Sampler(
    etna::Sampler::CreateInfo{
      .addressMode = vk::SamplerAddressMode::eRepeat,
      .name = "albedo_sampler",
    });

  const vk::DeviceSize maxInstancesBytes = 10000u * sizeof(glm::mat4x4);
  instanceMatricesBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = maxInstancesBytes,
      .bufferUsage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "instanceMatrices",
    });

  const vk::DeviceSize maxRelemMatsBytes = 20000u * sizeof(RelemMat);
  relemMaterialsBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = maxRelemMatsBytes,
      .bufferUsage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "relemMaterials",
    });

  sceneAllInstanceMatricesBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = 10000u * sizeof(glm::mat4),
      .bufferUsage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "sceneAllInstanceMatrices",
    });
  sceneInstanceMeshIdBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = 10000u * sizeof(uint32_t),
      .bufferUsage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "sceneInstanceMeshId",
    });
  sceneRelemAabbBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = 20000u * 2u * sizeof(glm::vec4),
      .bufferUsage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "sceneRelemAabb",
    });
  sceneMeshRelemRangeBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = 5000u * sizeof(glm::uvec2),
      .bufferUsage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "sceneMeshRelemRange",
    });
  sceneRelemDrawTemplateBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = 20000u * sizeof(glm::uvec4),
      .bufferUsage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "sceneRelemDrawTemplate",
    });

  // Per-frame culling intermediates
  relemVisibleCountsBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = 20000u * sizeof(uint32_t),
      .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "relemVisibleCounts",
    });
  cullReadbackBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = 20000u * sizeof(uint32_t),
      .bufferUsage = vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_AUTO,
      .allocationCreate = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
      .name = "cullReadback",
    });
  cullReadbackBuffer.map();
  relemInstanceOffsetsBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = 20000u * sizeof(uint32_t),
      .bufferUsage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "relemInstanceOffsets",
    });
  relemWriteCursorsBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = 20000u * sizeof(uint32_t),
      .bufferUsage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "relemWriteCursors",
    });

  // indirectBuffer: VkDrawIndexedIndirectCommand (5×uint32)
  // drawMappingBuffer: one uint32 per visible relem
  const vk::DeviceSize maxDrawsBytes = 10000u * sizeof(vk::DrawIndexedIndirectCommand);
  indirectBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = maxDrawsBytes,
      .bufferUsage = vk::BufferUsageFlagBits::eIndirectBuffer |
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "indirectDraws",
    });

  const vk::DeviceSize maxDrawMappingBytes = 10000u * sizeof(uint32_t);
  drawMappingBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = maxDrawMappingBytes,
      .bufferUsage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "drawMapping",
    });

  constexpr vk::DeviceSize LUMINANCE_STATS_BYTES =
    sizeof(std::uint32_t) * 2u + sizeof(std::uint32_t) * 128u + sizeof(float);
  luminanceStatsBuffer = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = LUMINANCE_STATS_BYTES,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "luminance_stats",
  });

  perlinTex = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{4096, 4096, 1},
    .name = "perlin_noise",
    .format = vk::Format::eR32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
  });

  normalMap = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{4096, 4096, 1},
    .name = "terrain_normal_map",
    .format = vk::Format::eR8G8B8A8Snorm,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
  });

  clipmapMesh = std::make_unique<ClipmapMesh>(255, *transferHelper);
  clipmapFootprint = clipmapMesh->buildLevelFootprints().front();

  clipmapHeightmapArray = ctx.createImage(etna::Image::CreateInfo{
    .extent    = vk::Extent3D{256, 256, 1},
    .name      = "clipmap_heightmap_array",
    .format    = vk::Format::eR32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
    .layers    = static_cast<std::size_t>(CLIPMAP_LEVELS),
  });

  clipmapLevelsBuffer = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size        = CLIPMAP_LEVELS * sizeof(glm::vec4),
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
    .name        = "clipmap_levels",
  });
  clipmapLevelsBuffer.map();
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

    auto gpuImage = ctx.createImage(
      etna::Image::CreateInfo{
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

      cpuMats.push_back(
        RelemMat{
          .baseColorIdx = resolveIdx(m.baseColorTex, TextureId::DefaultBaseColor),
          .metalRoughIdx = resolveIdx(m.metallicRoughnessTex, TextureId::DefaultMetallicRoughness),
          .normalIdx = resolveIdx(m.normalTex, TextureId::DefaultNormal),
          .occlusionIdx = resolveIdx(m.occlusionTex, TextureId::DefaultOcclusion),
          .baseColorFactor = m.baseColorFactor,
          .materialParams =
            glm::vec4(m.metallicFactor, m.roughnessFactor, m.normalScale, m.occlusionStrength),
          .emissiveIdx = resolveIdx(m.emissiveTex, TextureId::DefaultEmissive),
          ._pad0 = 0,
          ._pad1 = 0,
          ._pad2 = 0,
          .emissiveFactor = glm::vec4(m.emissiveFactor, 0.0f),
        });
    }

    if (!cpuMats.empty())
    {
      auto oneShotMat = ctx.createOneShotCmdMgr();
      transferHelper->uploadBuffer<RelemMat>(
        *oneShotMat, relemMaterialsBuffer, 0, std::span<const RelemMat>(cpuMats));
    }
  }

  // Upload persistent scene geometry buffers for GPU culling
  {
    auto instanceMatrices = sceneMgr->getInstanceMatrices();
    auto instanceMeshes = sceneMgr->getInstanceMeshes();
    auto relems = sceneMgr->getRenderElements();
    auto meshes = sceneMgr->getMeshes();
    auto aabbs = sceneMgr->getRenderElementAABBs();

    sceneInstanceCount = static_cast<uint32_t>(instanceMatrices.size());
    sceneRelemCount = static_cast<uint32_t>(relems.size());

    if (sceneInstanceCount > 0)
    {
      transferHelper->uploadBuffer<glm::mat4>(
        *oneShot,
        sceneAllInstanceMatricesBuffer,
        0,
        std::span<const glm::mat4>(instanceMatrices.data(), instanceMatrices.size()));
    }

    if (sceneInstanceCount > 0)
    {
      transferHelper->uploadBuffer<uint32_t>(
        *oneShot,
        sceneInstanceMeshIdBuffer,
        0,
        std::span<const uint32_t>(instanceMeshes.data(), instanceMeshes.size()));
    }

    if (sceneRelemCount > 0)
    {
      // AABB in two vec4: mn.xyz+pad, mx.xyz+pad
      std::vector<glm::vec4> gpuAabbs;
      gpuAabbs.reserve(aabbs.size() * 2);
      for (const auto& a : aabbs)
      {
        gpuAabbs.push_back(glm::vec4(a.min, 0.f));
        gpuAabbs.push_back(glm::vec4(a.max, 0.f));
      }
      transferHelper->uploadBuffer<glm::vec4>(
        *oneShot, sceneRelemAabbBuffer, 0, std::span<const glm::vec4>(gpuAabbs));
    }

    if (!meshes.empty())
    {
      std::vector<glm::uvec2> gpuRanges;
      gpuRanges.reserve(meshes.size());
      for (const auto& m : meshes)
        gpuRanges.push_back(glm::uvec2(m.firstRelem, m.relemCount));
      transferHelper->uploadBuffer<glm::uvec2>(
        *oneShot, sceneMeshRelemRangeBuffer, 0, std::span<const glm::uvec2>(gpuRanges));
    }

    if (sceneRelemCount > 0)
    {
      std::vector<glm::uvec4> gpuTemplates;
      gpuTemplates.reserve(relems.size());
      for (const auto& r : relems)
      {
        gpuTemplates.push_back(
          glm::uvec4(r.indexCount, r.indexOffset, static_cast<uint32_t>(r.vertexOffset), 0u));
      }
      transferHelper->uploadBuffer<glm::uvec4>(
        *oneShot, sceneRelemDrawTemplateBuffer, 0, std::span<const glm::uvec4>(gpuTemplates));
    }

    spdlog::info(
      "GPU culling buffers uploaded: {} instances, {} relems, {} meshes",
      sceneInstanceCount,
      sceneRelemCount,
      meshes.size());
  }

  {
    auto programInfo = etna::get_shader_program("static_mesh_material");
    auto layoutId = programInfo.getDescriptorLayoutId(2);

    std::vector<etna::Binding> texBindings;
    texBindings.reserve(sceneTextures.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(sceneTextures.size()); ++i)
    {
      texBindings.emplace_back(
        0u,
        sceneTextures[i].genBinding(albedoSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal),
        i);
    }

    bindlessTextureSet = etna::create_persistent_descriptor_set(
      layoutId, std::move(texBindings), /*allow_unbound_slots=*/true);
  }
}

void WorldRenderer::loadShaders()
{
  etna::create_program(
    "static_mesh_material",
    {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "static_mesh.frag.spv",
     BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "static_mesh.vert.spv"});
  etna::create_program(
    "static_mesh", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "static_mesh.vert.spv"});

  etna::create_program(
    "postprocess",
    {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "postprocess.vert.spv",
     BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "postprocess.frag.spv"});

  etna::create_program(
    "clear_stats", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "clear_stats.comp.spv"});
  etna::create_program("minmax", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "minmax.comp.spv"});
  etna::create_program("histogram", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "histogram.comp.spv"});
  etna::create_program("reduce", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "reduce.comp.spv"});

  etna::create_program(
    "skybox",
    {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "skybox.vert.spv",
     BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "skybox.frag.spv"});

  etna::create_program(
    "cull_count", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "cull_count.comp.spv"});
  etna::create_program(
    "prefix_sum", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "prefix_sum.comp.spv"});
  etna::create_program(
    "cull_write", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "cull_write.comp.spv"});

  etna::create_program("perlin", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "perlin.comp.spv"});
  etna::create_program("normal", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "normal.comp.spv"});
  etna::create_program(
    "clipmap_terrain",
    {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "clipmap_terrain.vert.spv",
     BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "clipmap_terrain.frag.spv"});
  etna::create_program(
    "clipmap_fill", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "clipmap_fill.comp.spv"});
  etna::create_program(
    "terrain_render",
    {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "quad.vert.spv",
     BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "terrain.tesc.spv",
     BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "terrain.tese.spv",
     BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "terrain.frag.spv"});
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
  cullCountPipeline  = pipelineManager.createComputePipeline("cull_count", {});
  prefixSumPipeline  = pipelineManager.createComputePipeline("prefix_sum", {});
  cullWritePipeline  = pipelineManager.createComputePipeline("cull_write", {});
  perlinPipeline     = pipelineManager.createComputePipeline("perlin", {});
  normalPipeline     = pipelineManager.createComputePipeline("normal", {});
  clipmapFillPipeline = pipelineManager.createComputePipeline("clipmap_fill", {});

  clipmapTerrainPipeline = pipelineManager.createGraphicsPipeline(
    "clipmap_terrain",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput =
        etna::VertexShaderInputDescription{
          .bindings = {etna::VertexShaderInputDescription::Binding{
            .byteStreamDescription =
              etna::VertexByteStreamFormatDescription{
                .stride     = sizeof(glm::vec2),
                .attributes = {etna::VertexByteStreamFormatDescription::Attribute{
                  .format = vk::Format::eR32G32Sfloat,
                  .offset = 0,
                }},
              },
          }},
        },
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

  terrainPipeline = pipelineManager.createGraphicsPipeline(
    "terrain_render",
    etna::GraphicsPipeline::CreateInfo{
      .inputAssemblyConfig = {.topology = vk::PrimitiveTopology::ePatchList},
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

  if (previousTime <= 0.f)
    deltaTime = 1.f / 60.f;
  else
    deltaTime = std::max(packet.currentTime - previousTime, 0.f);
  previousTime = packet.currentTime;
}

void WorldRenderer::performFrustumCulling(const glm::mat4x4& proj_view)
{
  auto meshes = sceneMgr->getMeshes();
  auto instanceMeshes = sceneMgr->getInstanceMeshes();
  auto instanceMatrices = sceneMgr->getInstanceMatrices();
  auto relemAABBs = sceneMgr->getRenderElementAABBs();

  CulledRenderElements result;

  auto row = [&](int i) {
    return glm::vec4(proj_view[0][i], proj_view[1][i], proj_view[2][i], proj_view[3][i]);
  };
  const glm::vec4 r0 = row(0);
  const glm::vec4 r1 = row(1);
  const glm::vec4 r2 = row(2);
  const glm::vec4 r3 = row(3);
  const std::array<glm::vec4, 6> planes = {
    r3 + r0, // left
    r3 - r0, // right
    r3 + r1, // bottom
    r3 - r1, // top
    r2,      // near (Vulkan: z >= 0)
    r3 - r2, // far
  };

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

        glm::vec3 worldCorners[8];
        for (int c = 0; c < 8; ++c)
        {
          glm::vec3 local{
            ((c & 1) != 0) ? aabb.max.x : aabb.min.x,
            ((c & 2) != 0) ? aabb.max.y : aabb.min.y,
            ((c & 4) != 0) ? aabb.max.z : aabb.min.z,
          };
          worldCorners[c] = glm::vec3(model * glm::vec4(local, 1.0f));
        }

        // cull only if some plane has ALL 8 corners on the outside
        bool culled = false;
        for (int p = 0; p < 6; ++p)
        {
          const glm::vec3 n{planes[p].x, planes[p].y, planes[p].z};
          const float d = planes[p].w;
          bool allOutside = true;
          for (int c = 0; c < 8; ++c)
          {
            if (glm::dot(n, worldCorners[c]) + d >= 0.0f)
            {
              allOutside = false;
              break;
            }
          }
          if (allOutside)
          {
            culled = true;
            break;
          }
        }

        if (!culled)
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
  std::vector<vk::DrawIndexedIndirectCommand> indirectCmds;
  std::vector<uint32_t> drawMapping;

  auto relems = sceneMgr->getRenderElements();
  uint32_t instanceOffset = 0;

  for (const auto& element : culledElements.elements)
  {
    const auto& relem = relems[element.relemIdx];
    const uint32_t instanceCount = static_cast<uint32_t>(element.visibleMatrices.size());

    allVisibleMatrices.insert(
      allVisibleMatrices.end(), element.visibleMatrices.begin(), element.visibleMatrices.end());

    indirectCmds.push_back(
      vk::DrawIndexedIndirectCommand{
        .indexCount = relem.indexCount,
        .instanceCount = instanceCount,
        .firstIndex = relem.indexOffset,
        .vertexOffset = relem.vertexOffset,
        .firstInstance = instanceOffset,
      });

    drawMapping.push_back(static_cast<uint32_t>(element.relemIdx));

    instanceOffset += instanceCount;
  }

  if (!allVisibleMatrices.empty())
  {
    auto& ctx = etna::get_context();
    auto oneShot = ctx.createOneShotCmdMgr();
    transferHelper->uploadBuffer<glm::mat4x4>(
      *oneShot, instanceMatricesBuffer, 0, std::span<const glm::mat4x4>(allVisibleMatrices));
  }

  if (!indirectCmds.empty())
  {
    auto& ctx = etna::get_context();
    {
      auto oneShot = ctx.createOneShotCmdMgr();
      transferHelper->uploadBuffer<vk::DrawIndexedIndirectCommand>(
        *oneShot, indirectBuffer, 0, std::span<const vk::DrawIndexedIndirectCommand>(indirectCmds));
    }
    {
      auto oneShot = ctx.createOneShotCmdMgr();
      transferHelper->uploadBuffer<uint32_t>(
        *oneShot, drawMappingBuffer, 0, std::span<const uint32_t>(drawMapping));
    }
  }
}

void WorldRenderer::renderScene(
  vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm, vk::PipelineLayout pipeline_layout)
{
  if (!sceneMgr->getVertexBuffer() || sceneRelemCount == 0)
    return;

  cmd_buf.bindVertexBuffers(0, {sceneMgr->getVertexBuffer()}, {0});
  cmd_buf.bindIndexBuffer(sceneMgr->getIndexBuffer(), 0, vk::IndexType::eUint32);

  pushConst.proj_view = glob_tm;
  pushConst.cameraPos = glm::vec4(cameraWorldPos, 0.0f);
  pushConst.sunDir = glm::vec4(glm::normalize(sunDirection), 0.0f);
  pushConst.sunColor = glm::vec4(sunColor, sunIntensity);
  pushConst.isBaked = bakedEnabled ? 1u : 0u;
  pushConst.debugMode = static_cast<std::uint32_t>(debugMode);

  cmd_buf.pushConstants<PushConstants>(
    pipeline_layout,
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
    0,
    {pushConst});

  auto programInfo = etna::get_shader_program("static_mesh_material");

  // Set 0: instance matrices + draw->relem mapping
  auto instanceSet = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmd_buf,
    {
      etna::Binding{0, instanceMatricesBuffer.genBinding()},
      etna::Binding{1, drawMappingBuffer.genBinding()},
    });
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, {instanceSet.getVkSet()}, {});

  // Set 1: RelemMaterials SSBO — persistent, bound once
  auto materialsSet = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(1),
    cmd_buf,
    {etna::Binding{0, relemMaterialsBuffer.genBinding()}});
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics, pipeline_layout, 1, {materialsSet.getVkSet()}, {});

  // Set 2: bindless texture array — persistent, bound once
  bindlessTextureSet.processBarriers(cmd_buf);
  vk::DescriptorSet bindlessVkSet = bindlessTextureSet.getVkSet();
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics, pipeline_layout, 2, {bindlessVkSet}, {});

  // One indirect draw call
  cmd_buf.drawIndexedIndirect(
    indirectBuffer.get(), 0, sceneRelemCount, sizeof(vk::DrawIndexedIndirectCommand));
}

void WorldRenderer::renderWorld(
  vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view)
{
  ETNA_PROFILE_GPU(cmd_buf, renderWorld);

  /*if (logEnabled && (logFrameCounter++ % 60u) == 0u)
    spdlog::info(
      "GPU culling: {} instances, {} relems",
      sceneInstanceCount,
      sceneRelemCount);
  */
  if (sceneInstanceCount > 0 && sceneRelemCount > 0)
  {
    ETNA_PROFILE_GPU(cmd_buf, gpuCulling);

    struct CullPC
    {
      glm::mat4 projView;
      uint32_t instanceCount;
      uint32_t relemCount;
    };
    const CullPC cullPc{worldViewProj, sceneInstanceCount, sceneRelemCount};

    cmd_buf.fillBuffer(
      relemVisibleCountsBuffer.get(),
      0,
      static_cast<vk::DeviceSize>(sceneRelemCount) * sizeof(uint32_t),
      0u);
    {
      vk::BufferMemoryBarrier2 memorybarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask =
          vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
        .buffer = relemVisibleCountsBuffer.get(),
        .offset = 0,
        .size = vk::WholeSize,
      };
      cmd_buf.pipelineBarrier2(
        vk::DependencyInfo{.bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &memorybarrier});
    }

    {
      auto info = etna::get_shader_program("cull_count");
      auto ds = etna::create_descriptor_set(
        info.getDescriptorLayoutId(0),
        cmd_buf,
        {
          etna::Binding{0, sceneAllInstanceMatricesBuffer.genBinding()},
          etna::Binding{1, sceneInstanceMeshIdBuffer.genBinding()},
          etna::Binding{2, sceneMeshRelemRangeBuffer.genBinding()},
          etna::Binding{3, sceneRelemAabbBuffer.genBinding()},
          etna::Binding{4, relemVisibleCountsBuffer.genBinding()},
        });
      cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, cullCountPipeline.getVkPipeline());
      cmd_buf.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        cullCountPipeline.getVkPipelineLayout(),
        0,
        {ds.getVkSet()},
        {});
      cmd_buf.pushConstants<CullPC>(
        cullCountPipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {cullPc});
      cmd_buf.dispatch((sceneInstanceCount + 63u) / 64u, 1, 1);
    }

    etna::set_state(
      cmd_buf,
      relemVisibleCountsBuffer.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageRead);
    etna::flush_barriers(cmd_buf);

    {
      struct PrefixPC
      {
        uint32_t relemCount;
      };
      auto info = etna::get_shader_program("prefix_sum");
      auto ds = etna::create_descriptor_set(
        info.getDescriptorLayoutId(0),
        cmd_buf,
        {
          etna::Binding{0, relemVisibleCountsBuffer.genBinding()},
          etna::Binding{1, relemInstanceOffsetsBuffer.genBinding()},
          etna::Binding{2, relemWriteCursorsBuffer.genBinding()},
          etna::Binding{3, indirectBuffer.genBinding()},
          etna::Binding{4, sceneRelemDrawTemplateBuffer.genBinding()},
          etna::Binding{5, drawMappingBuffer.genBinding()},
        });
      cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, prefixSumPipeline.getVkPipeline());
      cmd_buf.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        prefixSumPipeline.getVkPipelineLayout(),
        0,
        {ds.getVkSet()},
        {});
      cmd_buf.pushConstants<PrefixPC>(
        prefixSumPipeline.getVkPipelineLayout(),
        vk::ShaderStageFlagBits::eCompute,
        0,
        {PrefixPC{sceneRelemCount}});
      cmd_buf.dispatch(1, 1, 1);
    }

    {
      vk::BufferMemoryBarrier2 barriers[] = {
        {
          .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
          .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
          .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
          .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
          .buffer = relemInstanceOffsetsBuffer.get(),
          .offset = 0,
          .size = vk::WholeSize,
        },
        {
          .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
          .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
          .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
          .dstAccessMask =
            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
          .buffer = relemWriteCursorsBuffer.get(),
          .offset = 0,
          .size = vk::WholeSize,
        },
        {
          .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
          .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
          .dstStageMask = vk::PipelineStageFlagBits2::eDrawIndirect,
          .dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead,
          .buffer = indirectBuffer.get(),
          .offset = 0,
          .size = vk::WholeSize,
        },
      };
      cmd_buf.pipelineBarrier2(
        vk::DependencyInfo{
          .bufferMemoryBarrierCount = 3,
          .pBufferMemoryBarriers = barriers,
        });
    }

    {
      auto info = etna::get_shader_program("cull_write");
      auto ds = etna::create_descriptor_set(
        info.getDescriptorLayoutId(0),
        cmd_buf,
        {
          etna::Binding{0, sceneAllInstanceMatricesBuffer.genBinding()},
          etna::Binding{1, sceneInstanceMeshIdBuffer.genBinding()},
          etna::Binding{2, sceneMeshRelemRangeBuffer.genBinding()},
          etna::Binding{3, sceneRelemAabbBuffer.genBinding()},
          etna::Binding{4, relemInstanceOffsetsBuffer.genBinding()},
          etna::Binding{5, relemWriteCursorsBuffer.genBinding()},
          etna::Binding{6, instanceMatricesBuffer.genBinding()},
        });
      cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, cullWritePipeline.getVkPipeline());
      cmd_buf.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        cullWritePipeline.getVkPipelineLayout(),
        0,
        {ds.getVkSet()},
        {});
      cmd_buf.pushConstants<CullPC>(
        cullWritePipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {cullPc});
      cmd_buf.dispatch((sceneInstanceCount + 63u) / 64u, 1, 1);
    }

    {
      vk::BufferMemoryBarrier2 memorybarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eVertexShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
        .buffer = instanceMatricesBuffer.get(),
        .offset = 0,
        .size = vk::WholeSize,
      };
      cmd_buf.pipelineBarrier2(
        vk::DependencyInfo{.bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &memorybarrier});
    }

    if (sceneRelemCount > 0)
    {
      vk::BufferMemoryBarrier2 memorybarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .buffer = relemVisibleCountsBuffer.get(),
        .offset = 0,
        .size = vk::WholeSize,
      };
      cmd_buf.pipelineBarrier2(
        vk::DependencyInfo{.bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &memorybarrier});

      vk::BufferCopy region{0, 0, sceneRelemCount * sizeof(uint32_t)};
      cmd_buf.copyBuffer(relemVisibleCountsBuffer.get(), cullReadbackBuffer.get(), 1, &region);
    }
  }

  // Fill per-level heightmap textures before entering the render pass
  if (selectedScene == SceneType::Terrain && useClipmapTerrain)
    updateClipmapHeightmaps(cmd_buf);

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

    if (selectedScene == SceneType::Terrain)
    {
      if (useClipmapTerrain)
        renderClipmapTerrain(cmd_buf);
      else
        renderTerrain(cmd_buf);
    }

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

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, clearStatsPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      clearStatsPipeline.getVkPipelineLayout(),
      0,
      1,
      &vkSet,
      0,
      nullptr);
    cmd_buf.dispatch(1, 1, 1);
  }

  // Barrier: clear writes -> minmax reads/writes
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
    auto hdrBind = hdrTarget.genBinding(hdrSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal);
    auto descSet = etna::create_descriptor_set(
      info.getDescriptorLayoutId(0),
      cmd_buf,
      {etna::Binding{0, statsBind}, etna::Binding{1, hdrBind}});
    vk::DescriptorSet vkSet = descSet.getVkSet();

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, minmaxPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      minmaxPipeline.getVkPipelineLayout(),
      0,
      1,
      &vkSet,
      0,
      nullptr);

    const uint32_t groupsX = (resolution.x + 15u) / 16u;
    const uint32_t groupsY = (resolution.y + 15u) / 16u;
    cmd_buf.dispatch(groupsX, groupsY, 1);
  }

  // Barrier: minmax writes -> histogram reads/writes
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
    auto hdrBind = hdrTarget.genBinding(hdrSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal);
    auto descSet = etna::create_descriptor_set(
      info.getDescriptorLayoutId(0),
      cmd_buf,
      {etna::Binding{0, statsBind}, etna::Binding{1, hdrBind}});
    vk::DescriptorSet vkSet = descSet.getVkSet();

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, histogramPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      histogramPipeline.getVkPipelineLayout(),
      0,
      1,
      &vkSet,
      0,
      nullptr);

    const uint32_t groupsX = (resolution.x + 15u) / 16u;
    const uint32_t groupsY = (resolution.y + 15u) / 16u;
    cmd_buf.dispatch(groupsX, groupsY, 1);
  }

  // Barrier: histogram writes -> reduce reads/writes
  etna::set_state(
    cmd_buf,
    luminanceStatsBuffer.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
  etna::flush_barriers(cmd_buf);

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

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, reducePipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      reducePipeline.getVkPipelineLayout(),
      0,
      1,
      &vkSet,
      0,
      nullptr);
    cmd_buf.pushConstants<ReducePush>(
      reducePipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {reducePush});
    cmd_buf.dispatch(1, 1, 1);
  }

  // Barrier: reduce writes smoothedExposure -> frag reads it
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
    cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, 1, &vkSet, 0, nullptr);

    const std::uint32_t tonemapModeU = static_cast<std::uint32_t>(tonemapMode);
    cmd_buf.pushConstants<std::uint32_t>(
      layout, vk::ShaderStageFlagBits::eFragment, 0, {tonemapModeU});

    cmd_buf.draw(3, 1, 0, 0);
  }
}

void WorldRenderer::initTerrainIfNeeded()
{
  if (terrainInitialized)
    return;

  auto& ctx = etna::get_context();
  auto cmdManager = ctx.createOneShotCmdMgr();
  auto cmdBuf = cmdManager->start();
  ETNA_CHECK_VK_RESULT(cmdBuf.begin(vk::CommandBufferBeginInfo{}));
  createTerrainMap(cmdBuf);
  ETNA_CHECK_VK_RESULT(cmdBuf.end());
  cmdManager->submitAndWait(cmdBuf);

  terrainInitialized = true;
  spdlog::info("Terrain heightmap and normal map generated (4096x4096).");
}

void WorldRenderer::createTerrainMap(vk::CommandBuffer cmd_buf)
{
  // Pass 1: Perlin noise -> heightmap (R32F)
  {
    auto info = etna::get_shader_program("perlin");
    auto binding = perlinTex.genBinding(perlinSampler.get(), vk::ImageLayout::eGeneral, {});
    auto set = etna::create_descriptor_set(
      info.getDescriptorLayoutId(0), cmd_buf, {etna::Binding{0, binding}});
    vk::DescriptorSet vkSet = set.getVkSet();

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, perlinPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      perlinPipeline.getVkPipelineLayout(),
      0, 1, &vkSet, 0, nullptr);
    etna::flush_barriers(cmd_buf);
    cmd_buf.dispatch(4096 / 32, 4096 / 32, 1);
  }

  // Pass 2: heightmap -> normalMap (RGBA8_SNORM)
  {
    auto info = etna::get_shader_program("normal");
    auto bind0 = perlinTex.genBinding(perlinSampler.get(), vk::ImageLayout::eGeneral, {});
    auto bind1 = normalMap.genBinding(perlinSampler.get(), vk::ImageLayout::eGeneral, {});

    etna::set_state(
      cmd_buf,
      perlinTex.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageRead,
      vk::ImageLayout::eGeneral,
      vk::ImageAspectFlagBits::eColor);

    auto set = etna::create_descriptor_set(
      info.getDescriptorLayoutId(0),
      cmd_buf,
      {etna::Binding{0, bind0}, etna::Binding{1, bind1}});
    vk::DescriptorSet vkSet = set.getVkSet();

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, normalPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      normalPipeline.getVkPipelineLayout(),
      0, 1, &vkSet, 0, nullptr);
    etna::flush_barriers(cmd_buf);
    cmd_buf.dispatch(4096 / 32, 4096 / 32, 1);
  }

  // Transition both to read-only optimal for sampling in rendering passes
  etna::set_state(
    cmd_buf,
    perlinTex.get(),
    vk::PipelineStageFlagBits2::eTessellationEvaluationShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmd_buf,
    normalMap.get(),
    vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmd_buf);
}

void WorldRenderer::renderTerrain(vk::CommandBuffer cmd_buf)
{
  auto info = etna::get_shader_program("terrain_render");
  auto bind0 = perlinTex.genBinding(perlinSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal);
  auto bind1 = normalMap.genBinding(perlinSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal);

  auto descSet = etna::create_descriptor_set(
    info.getDescriptorLayoutId(0),
    cmd_buf,
    {etna::Binding{0, bind0}, etna::Binding{1, bind1}});
  auto vkSet = descSet.getVkSet();
  auto layout = terrainPipeline.getVkPipelineLayout();

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, terrainPipeline.getVkPipeline());
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics, layout, 0, 1, &vkSet, 0, nullptr);

  TerrainPushConst pc{
    worldViewProj,
    glm::vec4(cameraWorldPos, 0.f),
    glm::vec4(glm::normalize(sunDirection), 0.f),
    glm::vec4(sunColor, sunIntensity),
  };
  cmd_buf.pushConstants<TerrainPushConst>(
    layout,
    vk::ShaderStageFlagBits::eTessellationControl
      | vk::ShaderStageFlagBits::eTessellationEvaluation
      | vk::ShaderStageFlagBits::eFragment,
    0,
    {pc});

  // 4 vertices per patch; instance count = number of chunks (32x32 grid)
  const uint32_t chunkCount = (4096u * 4096u) / (128u * 128u); // = 1024
  cmd_buf.draw(4, chunkCount, 0, 0);
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
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());

  std::filesystem::path path;
  switch (type)
  {
  case SceneType::SimpleMeshes:
    path = GRAPHICS_COURSE_RESOURCES_ROOT "/scenes/SimpleMeshes/glTF/SimpleMeshes.gltf";
    break;
  case SceneType::LowPolyDarkTown:
    path = GRAPHICS_COURSE_RESOURCES_ROOT "/scenes/low_poly_dark_town/scene.gltf";
    break;
  case SceneType::LovelyTown:
    path = GRAPHICS_COURSE_RESOURCES_ROOT "/scenes/lovely_town/scene.gltf";
    break;
  case SceneType::Avocado:
    path = GRAPHICS_COURSE_RESOURCES_ROOT "/scenes/Avocado/Avocado.gltf";
    break;
  case SceneType::Terrain:
    sceneMgr = std::make_unique<SceneManager>();
    sceneTextures.clear();
    sceneInstanceCount = 0;
    sceneRelemCount    = 0;
    initTerrainIfNeeded();
    return;
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

  const char* scenes[] = {"SimpleMeshes", "Low Poly Dark Town", "Lovely Town", "Avocado", "Terrain"};
  int currentSceneIdx = static_cast<int>(selectedScene);

  bool sceneChanged = ImGui::Combo("Scene", &currentSceneIdx, scenes, IM_ARRAYSIZE(scenes));

  if (sceneChanged || bakedChanged)
  {
    selectedScene = static_cast<SceneType>(currentSceneIdx);
    loadSceneByType(selectedScene);
  }

  if (selectedScene == SceneType::Terrain)
  {
    ImGui::Separator();
    ImGui::Text("Terrain mode");
    ImGui::Checkbox("Use Clipmap (no tessellation)", &useClipmapTerrain);
    if (useClipmapTerrain)
    {
      ImGui::Checkbox("Debug: show LOD levels", &debugClipmapLevels);
      ImGui::Checkbox("Debug: show morph alpha", &showMorphAlpha);
      ImGui::SliderFloat("Morph width (texels)", &clipmapMorphWidth, 1.f, 64.f, "%.1f");
    }
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

  if (sceneRelemCount > 0)
  {
    const auto* counts = reinterpret_cast<const uint32_t*>(cullReadbackBuffer.data());
    uint32_t visible = 0;
    for (uint32_t i = 0; i < sceneRelemCount; ++i)
      visible += counts[i];
    lastFrameVisibleInstances = visible;
    ImGui::Text("Culling: %u visible / %u total instances", visible, sceneInstanceCount);
  }
  ImGui::End();
}

float WorldRenderer::findZFar() const
{
  if (selectedScene == SceneType::Terrain)
    return 100000.f;
  return 1000.f;
}

float WorldRenderer::findZNear() const
{
  if (selectedScene == SceneType::Terrain)
    return 5.f;
  return 0.01f;
}

void WorldRenderer::updateClipmapHeightmaps(vk::CommandBuffer cmd_buf)
{
  ETNA_PROFILE_GPU(cmd_buf, clipmapFill);

  const float halfGrid = static_cast<float>(clipmapMesh->n() - 1) * 0.5f;
  const glm::vec2 camXZ{cameraWorldPos.x, cameraWorldPos.z};

  etna::set_state(
    cmd_buf,
    clipmapHeightmapArray.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderStorageWrite,
    vk::ImageLayout::eGeneral,
    vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmd_buf);

  struct FillPC { glm::vec2 levelOrigin; float gridStep; int levelIdx; };

  auto fillInfo = etna::get_shader_program("clipmap_fill");
  auto fillBind = clipmapHeightmapArray.genBinding(
    clipmapSampler.get(),
    vk::ImageLayout::eGeneral,
    etna::Image::ViewParams{.type = vk::ImageViewType::e2DArray});
  auto fillSet = etna::create_descriptor_set(
    fillInfo.getDescriptorLayoutId(0), cmd_buf, {etna::Binding{0, fillBind}});
  vk::DescriptorSet fillVkSet = fillSet.getVkSet();

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, clipmapFillPipeline.getVkPipeline());
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eCompute,
    clipmapFillPipeline.getVkPipelineLayout(),
    0, 1, &fillVkSet, 0, nullptr);

  for (int level = CLIPMAP_LEVELS - 1; level >= 0; --level)
  {
    const float step   = clipmapBaseStep * static_cast<float>(1 << level);
    const float snap   = 2.f * step;
    const glm::vec2 center = glm::floor(camXZ / snap) * snap;
    const glm::vec2 origin = center - halfGrid * step;

    const int instanceIdx = CLIPMAP_LEVELS - 1 - level;
    const FillPC pc{origin, step, instanceIdx};
    cmd_buf.pushConstants<FillPC>(
      clipmapFillPipeline.getVkPipelineLayout(),
      vk::ShaderStageFlagBits::eCompute, 0, {pc});

    // 256x256 texels, 16x16 threadgroup = 16x16 groups
    cmd_buf.dispatch(16, 16, 1);
  }

  // Transition to shader-read for the vertex shader
  etna::set_state(
    cmd_buf,
    clipmapHeightmapArray.get(),
    vk::PipelineStageFlagBits2::eVertexShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmd_buf);
}

void WorldRenderer::renderClipmapTerrain(vk::CommandBuffer cmd_buf)
{
  ETNA_PROFILE_GPU(cmd_buf, clipmapTerrain);

  const float heightScale = 200.0f;
  const float halfGrid    = static_cast<float>(clipmapMesh->n() - 1) * 0.5f;
  const glm::vec2 camXZ{cameraWorldPos.x, cameraWorldPos.z};

  // Write per-level data to mapped SSBO: instance 0 = outermost, last = innermost
  auto* levelData = reinterpret_cast<glm::vec4*>(clipmapLevelsBuffer.data());
  for (int level = CLIPMAP_LEVELS - 1; level >= 0; --level)
  {
    const float step   = clipmapBaseStep * static_cast<float>(1 << level);
    const float snap   = 2.f * step;
    const glm::vec2 center = glm::floor(camXZ / snap) * snap;
    const glm::vec2 origin = center - halfGrid * step;
    levelData[CLIPMAP_LEVELS - 1 - level] = glm::vec4(origin, step, 0.f);
  }

  auto info  = etna::get_shader_program("clipmap_terrain");
  auto bind0 = clipmapHeightmapArray.genBinding(
    perlinSampler.get(),
    vk::ImageLayout::eShaderReadOnlyOptimal,
    etna::Image::ViewParams{.type = vk::ImageViewType::e2DArray});
  auto bind2 = clipmapLevelsBuffer.genBinding();

  auto descSet = etna::create_descriptor_set(
    info.getDescriptorLayoutId(0),
    cmd_buf,
    {etna::Binding{0, bind0}, etna::Binding{2, bind2}});
  auto layout = clipmapTerrainPipeline.getVkPipelineLayout();

  const float scaleSigned = debugClipmapLevels ? -heightScale : heightScale;
  const ClipmapPushConst pc{
    worldViewProj,
    glm::vec4(glm::normalize(sunDirection), 0.f),
    glm::vec4(sunColor, sunIntensity),
    glm::vec4(cameraWorldPos, scaleSigned),
    glm::vec4(clipmapMorphWidth, showMorphAlpha ? 1.0f : 0.0f, 0.f, 0.f),
  };

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, clipmapTerrainPipeline.getVkPipeline());
  vk::DescriptorSet vkSet = descSet.getVkSet();
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics, layout, 0, 1, &vkSet, 0, nullptr);
  cmd_buf.bindVertexBuffers(0, {clipmapMesh->vertexBuffer()}, {vk::DeviceSize{0}});
  cmd_buf.bindIndexBuffer(clipmapMesh->indexBuffer(), 0, vk::IndexType::eUint32);
  cmd_buf.pushConstants<ClipmapPushConst>(
    layout,
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
    0, {pc});

  cmd_buf.drawIndexed(
    clipmapFootprint.indexCount, CLIPMAP_LEVELS,
    clipmapFootprint.firstIndex, clipmapFootprint.vertexOffset, 0);
}
