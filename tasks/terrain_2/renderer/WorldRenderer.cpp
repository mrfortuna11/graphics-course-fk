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

#include <stb_image.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <sstream>
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
  loadSettings("terrain_settings.txt");

  auto& ctx = etna::get_context();
  auto device = ctx.getDevice();
  const float maxAniso =
    std::min(4.0f, ctx.getPhysicalDevice().getProperties().limits.maxSamplerAnisotropy);
  vk::SamplerCreateInfo si{
    .magFilter = vk::Filter::eLinear,
    .minFilter = vk::Filter::eLinear,
    .mipmapMode = vk::SamplerMipmapMode::eLinear,
    .addressModeU = vk::SamplerAddressMode::eRepeat,
    .addressModeV = vk::SamplerAddressMode::eRepeat,
    .addressModeW = vk::SamplerAddressMode::eRepeat,
    .anisotropyEnable = vk::True,
    .maxAnisotropy = maxAniso,
    .compareEnable = vk::False,
    .minLod = 0.0f,
    .maxLod = VK_LOD_CLAMP_NONE,
  };
  detailSampler = device.createSamplerUnique(si).value;
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
      .imageUsage =
        vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
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

  // Volumetric fog
  fogResolution = (resolution + glm::uvec2(1u)) / 2u;
  fogTarget = ctx.createImage(
    etna::Image::CreateInfo{
      .extent = vk::Extent3D{fogResolution.x, fogResolution.y, 1},
      .name = "fog_target",
      .format = vk::Format::eR16G16B16A16Sfloat,
      .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
    });

  // Linear clamp
  fogSampler = etna::Sampler(
    etna::Sampler::CreateInfo{
      .filter = vk::Filter::eLinear,
      .addressMode = vk::SamplerAddressMode::eClampToEdge,
      .name = "fog_sampler",
    });

  // Nearest clamp
  depthPointSampler = etna::Sampler(
    etna::Sampler::CreateInfo{
      .filter = vk::Filter::eNearest,
      .addressMode = vk::SamplerAddressMode::eClampToEdge,
      .mipmapMode = vk::SamplerMipmapMode::eNearest,
      .name = "depth_point_sampler",
    });

  shadowMap = ctx.createImage(
    etna::Image::CreateInfo{
      .extent = vk::Extent3D{SHADOWMAP_DIM, SHADOWMAP_DIM, 1},
      .name = "shadow_map",
      .format = vk::Format::eD32Sfloat,
      .imageUsage =
        vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
    });

  {
    vk::SamplerCreateInfo si{
      .magFilter = vk::Filter::eLinear,
      .minFilter = vk::Filter::eLinear,
      .mipmapMode = vk::SamplerMipmapMode::eLinear,
      .addressModeU = vk::SamplerAddressMode::eClampToBorder,
      .addressModeV = vk::SamplerAddressMode::eClampToBorder,
      .addressModeW = vk::SamplerAddressMode::eClampToBorder,
      .compareEnable = vk::True,
      .compareOp = vk::CompareOp::eLessOrEqual,
      .minLod = 0.f,
      .maxLod = VK_LOD_CLAMP_NONE,
      .borderColor = vk::BorderColor::eFloatOpaqueWhite,
    };
    shadowSampler = ctx.getDevice().createSamplerUnique(si).value;
  }

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
  luminanceStatsBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = LUMINANCE_STATS_BYTES,
      .bufferUsage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "luminance_stats",
    });

  perlinTex = ctx.createImage(
    etna::Image::CreateInfo{
      .extent = vk::Extent3D{4096, 4096, 1},
      .name = "perlin_noise",
      .format = vk::Format::eR32Sfloat,
      .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
    });

  normalMap = ctx.createImage(
    etna::Image::CreateInfo{
      .extent = vk::Extent3D{4096, 4096, 1},
      .name = "terrain_normal_map",
      .format = vk::Format::eR8G8B8A8Snorm,
      .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
    });

  clipmapMesh = std::make_unique<ClipmapMesh>(255, *transferHelper);
  clipmapFootprint = clipmapMesh->buildLevelFootprints().front();

  clipmapHeightmapArray = ctx.createImage(
    etna::Image::CreateInfo{
      .extent = vk::Extent3D{256, 256, 1},
      .name = "clipmap_heightmap_array",
      .format = vk::Format::eR32Sfloat,
      .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
      .layers = static_cast<std::size_t>(CLIPMAP_LEVELS),
    });

  clipmapAlbedoArray = ctx.createImage(
    etna::Image::CreateInfo{
      .extent = vk::Extent3D{256, 256, 1},
      .name = "clipmap_albedo_array",
      .format = vk::Format::eR8G8B8A8Unorm,
      .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
      .layers = static_cast<std::size_t>(CLIPMAP_LEVELS),
      .mipLevels = static_cast<std::size_t>(CLIPMAP_ALBEDO_MIPS),
    });

  detailTex = ctx.createImage(
    etna::Image::CreateInfo{
      .extent = vk::Extent3D{DETAIL_TEX_SIZE, DETAIL_TEX_SIZE, 1},
      .name = "clipmap_detail_tile",
      .format = vk::Format::eR32Sfloat,
      .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
      .mipLevels = static_cast<std::size_t>(DETAIL_TEX_MIPS),
    });

  clipmapLevelsBuffer = ctx.createBuffer(
    etna::Buffer::CreateInfo{
      .size = CLIPMAP_LEVELS * sizeof(glm::vec4),
      .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
      .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
      .name = "clipmap_levels",
    });
  clipmapLevelsBuffer.map();

  static constexpr std::size_t DETAIL_TEX_MIPS = 11;
  detailColorArray = ctx.createImage(
    etna::Image::CreateInfo{
      .extent = vk::Extent3D{1024, 1024, 1},
      .name = "detail_color_array",
      .format = vk::Format::eR8G8B8A8Srgb,
      .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eTransferSrc,
      .layers = static_cast<std::size_t>(DETAIL_LAYERS),
      .mipLevels = DETAIL_TEX_MIPS,
    });

  detailHeightArray = ctx.createImage(
    etna::Image::CreateInfo{
      .extent = vk::Extent3D{1024, 1024, 1},
      .name = "detail_height_array",
      .format = vk::Format::eR8Unorm,
      .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eTransferSrc,
      .layers = static_cast<std::size_t>(DETAIL_LAYERS),
      .mipLevels = DETAIL_TEX_MIPS,
    });

  loadDetailTextures();
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

void WorldRenderer::loadDetailTextures()
{
  static const char* colorPaths[DETAIL_LAYERS] = {
    GRAPHICS_COURSE_RESOURCES_ROOT "/textures/detail/ground079_color.png",
    GRAPHICS_COURSE_RESOURCES_ROOT "/textures/detail/grass_color.png",
    GRAPHICS_COURSE_RESOURCES_ROOT "/textures/detail/rock_color.png",
    GRAPHICS_COURSE_RESOURCES_ROOT "/textures/detail/snow_color.png",
  };
  static const char* heightPaths[DETAIL_LAYERS] = {
    GRAPHICS_COURSE_RESOURCES_ROOT "/textures/detail/ground079_height.png",
    GRAPHICS_COURSE_RESOURCES_ROOT "/textures/detail/grass_height.png",
    GRAPHICS_COURSE_RESOURCES_ROOT "/textures/detail/rock_height.png",
    GRAPHICS_COURSE_RESOURCES_ROOT "/textures/detail/snow_height.png",
  };

  auto& ctx = etna::get_context();
  auto oneShot = ctx.createOneShotCmdMgr();

  for (int i = 0; i < DETAIL_LAYERS; ++i)
  {
    int w = 0, h = 0, c = 0;
    stbi_uc* data = stbi_load(colorPaths[i], &w, &h, &c, 4);
    if (!data)
      spdlog::error("loadDetailTextures: failed to load '{}'", colorPaths[i]);
    else
    {
      transferHelper->uploadImage(
        *oneShot,
        detailColorArray,
        0,
        static_cast<std::uint32_t>(i),
        std::span<const std::byte>{
          reinterpret_cast<const std::byte*>(data), static_cast<std::size_t>(w * h * 4)});
      stbi_image_free(data);
    }
  }

  for (int i = 0; i < DETAIL_LAYERS; ++i)
  {
    int w = 0, h = 0, c = 0;
    stbi_uc* data = stbi_load(heightPaths[i], &w, &h, &c, 1);
    if (!data)
      spdlog::error("loadDetailTextures: failed to load '{}'", heightPaths[i]);
    else
    {
      transferHelper->uploadImage(
        *oneShot,
        detailHeightArray,
        0,
        static_cast<std::uint32_t>(i),
        std::span<const std::byte>{
          reinterpret_cast<const std::byte*>(data), static_cast<std::size_t>(w * h * 1)});
      stbi_image_free(data);
    }
  }
  const auto generateMips = [&](etna::Image& img) {
    constexpr uint32_t MIPS = 11; // log2(1024)+1
    constexpr uint32_t LAYERS = DETAIL_LAYERS;

    auto cmdBuf = oneShot->start();
    ETNA_CHECK_VK_RESULT(cmdBuf.begin(vk::CommandBufferBeginInfo{}));

    auto barrierMip = [&](
                        uint32_t baseMip,
                        uint32_t levelCount,
                        vk::ImageLayout oldL,
                        vk::ImageLayout newL,
                        vk::PipelineStageFlags2 srcS,
                        vk::AccessFlags2 srcA,
                        vk::PipelineStageFlags2 dstS,
                        vk::AccessFlags2 dstA) {
      vk::ImageMemoryBarrier2 b{
        .srcStageMask = srcS,
        .srcAccessMask = srcA,
        .dstStageMask = dstS,
        .dstAccessMask = dstA,
        .oldLayout = oldL,
        .newLayout = newL,
        .image = img.get(),
        .subresourceRange =
          vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = baseMip,
            .levelCount = levelCount,
            .baseArrayLayer = 0,
            .layerCount = LAYERS,
          },
      };
      cmdBuf.pipelineBarrier2(
        vk::DependencyInfo{
          .imageMemoryBarrierCount = 1,
          .pImageMemoryBarriers = &b,
        });
    };

    barrierMip(
      0,
      1,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageLayout::eTransferSrcOptimal,
      vk::PipelineStageFlagBits2::eAllCommands,
      vk::AccessFlagBits2::eShaderRead,
      vk::PipelineStageFlagBits2::eBlit,
      vk::AccessFlagBits2::eTransferRead);

    int srcW = 1024, srcH = 1024;
    for (uint32_t m = 1; m < MIPS; ++m)
    {
      const int dstW = std::max(srcW / 2, 1);
      const int dstH = std::max(srcH / 2, 1);

      barrierMip(
        m,
        1,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal,
        vk::PipelineStageFlagBits2::eNone,
        vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eBlit,
        vk::AccessFlagBits2::eTransferWrite);

      vk::ImageBlit blit{
        .srcSubresource =
          vk::ImageSubresourceLayers{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = m - 1,
            .baseArrayLayer = 0,
            .layerCount = LAYERS,
          },
        .srcOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{srcW, srcH, 1}},
        .dstSubresource =
          vk::ImageSubresourceLayers{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = m,
            .baseArrayLayer = 0,
            .layerCount = LAYERS,
          },
        .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{dstW, dstH, 1}},
      };
      cmdBuf.blitImage(
        img.get(),
        vk::ImageLayout::eTransferSrcOptimal,
        img.get(),
        vk::ImageLayout::eTransferDstOptimal,
        1,
        &blit,
        vk::Filter::eLinear);

      // Mip m: TransferDst -> TransferSrc (ready as source for next iteration)
      barrierMip(
        m,
        1,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::PipelineStageFlagBits2::eBlit,
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eBlit,
        vk::AccessFlagBits2::eTransferRead);

      srcW = dstW;
      srcH = dstH;
    }

    barrierMip(
      0,
      MIPS,
      vk::ImageLayout::eTransferSrcOptimal,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::PipelineStageFlagBits2::eBlit,
      vk::AccessFlagBits2::eTransferRead,
      vk::PipelineStageFlagBits2::eAllCommands,
      vk::AccessFlagBits2::eShaderRead);

    ETNA_CHECK_VK_RESULT(cmdBuf.end());
    oneShot->submitAndWait(std::move(cmdBuf));
  };

  generateMips(detailColorArray);
  generateMips(detailHeightArray);
}

void WorldRenderer::saveSettings(const std::filesystem::path& path) const
{
  std::ofstream f(path);
  if (!f)
  {
    spdlog::warn("saveSettings: cannot open '{}'", path.string());
    return;
  }

  f << "# terrain_2 settings\n";

#define SAVE_SCALAR(field) f << #field << " " << (field) << "\n"
#define SAVE_BOOL(field) f << #field << " " << ((field) ? 1 : 0) << "\n"
#define SAVE_VEC3(field)                                                                           \
  f << #field << " " << (field).x << " " << (field).y << " " << (field).z << "\n"

  // Terrain shape
  SAVE_SCALAR(terrainHeightScale);
  SAVE_SCALAR(terrainHillsWeight);
  SAVE_SCALAR(terrainRidgesWeight);
  SAVE_SCALAR(terrainDetailAmplitude);
  SAVE_SCALAR(terrainBaseFreq);
  SAVE_SCALAR(terrainOctaves);
  SAVE_SCALAR(terrainPersistence);
  SAVE_SCALAR(terrainLacunarity);
  SAVE_SCALAR(terrainBiasPower);
  SAVE_SCALAR(terrainOceanCut);
  SAVE_SCALAR(terrainMountainMaskFreq);
  SAVE_SCALAR(terrainMountainMaskOffset);
  SAVE_SCALAR(terrainMountainMaskWidth);
  SAVE_SCALAR(terrainWarpAmp);
  SAVE_SCALAR(terrainWarpFreq);
  SAVE_SCALAR(terrainRidgedSharpness);

  // Splatting
  SAVE_SCALAR(terrainHeightLow);
  SAVE_SCALAR(terrainHeightHigh);
  SAVE_SCALAR(terrainBlendSharpness);
  SAVE_SCALAR(terrainSlopeThreshold);

  // Detail textures
  SAVE_SCALAR(detailTilePeriod);
  SAVE_SCALAR(detailNormalStrength);

  // Clipmap
  SAVE_SCALAR(clipmapMorphWidth);
  SAVE_SCALAR(liveLevelsCount);
  SAVE_BOOL(debugClipmapLevels);
  SAVE_BOOL(showMorphAlpha);
  SAVE_BOOL(useClipmapTerrain);

  // Sun / sky
  SAVE_VEC3(sunDirection);
  SAVE_VEC3(sunColor);
  SAVE_SCALAR(sunIntensity);

  // Shadows
  SAVE_BOOL(enableShadows);
  SAVE_SCALAR(shadowOrthoHalfSize);

  // Volumetric fog
  SAVE_BOOL(fogEnabled);
  SAVE_SCALAR(fogDensity);
  SAVE_SCALAR(fogHeightFalloff);
  SAVE_SCALAR(fogGroundLevel);
  SAVE_SCALAR(fogScatterCoef);
  SAVE_SCALAR(fogExtinctionCoef);
  SAVE_SCALAR(fogPhaseG);
  SAVE_SCALAR(fogSteps);
  SAVE_SCALAR(fogMaxDistance);
  SAVE_SCALAR(fogNoiseScale);
  SAVE_SCALAR(fogNoiseStrength);
  f << "fogWindDir " << fogWindDir.x << " " << fogWindDir.y << "\n";
  SAVE_SCALAR(fogWindSpeed);

  // Tonemap / exposure
  SAVE_SCALAR(tonemapMode);
  SAVE_SCALAR(adaptationSpeed);
  SAVE_SCALAR(keyValue);
  SAVE_SCALAR(minExposure);
  SAVE_SCALAR(maxExposure);

  // Debug
  SAVE_SCALAR(debugMode);

#undef SAVE_SCALAR
#undef SAVE_BOOL
#undef SAVE_VEC3

  spdlog::info("saveSettings: wrote '{}'", path.string());
}

void WorldRenderer::loadSettings(const std::filesystem::path& path)
{
  std::ifstream f(path);
  if (!f)
  {
    spdlog::info("loadSettings: '{}' not found, using defaults", path.string());
    return;
  }

  std::string line;
  while (std::getline(f, line))
  {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream iss(line);
    std::string key;
    if (!(iss >> key))
      continue;

#define LOAD_SCALAR(field) else if (key == #field) iss >> field
#define LOAD_BOOL(field)                                                                           \
  else if (key == #field) do                                                                       \
  {                                                                                                \
    int _v = 0;                                                                                    \
    iss >> _v;                                                                                     \
    field = (_v != 0);                                                                             \
  }                                                                                                \
  while (0)
#define LOAD_VEC3(field) else if (key == #field) iss >> field.x >> field.y >> field.z

    if (false)
    {
    }
    LOAD_SCALAR(terrainHeightScale);
    LOAD_SCALAR(terrainHillsWeight);
    LOAD_SCALAR(terrainRidgesWeight);
    LOAD_SCALAR(terrainDetailAmplitude);
    LOAD_SCALAR(terrainBaseFreq);
    LOAD_SCALAR(terrainOctaves);
    LOAD_SCALAR(terrainPersistence);
    LOAD_SCALAR(terrainLacunarity);
    LOAD_SCALAR(terrainBiasPower);
    LOAD_SCALAR(terrainOceanCut);
    LOAD_SCALAR(terrainMountainMaskFreq);
    LOAD_SCALAR(terrainMountainMaskOffset);
    LOAD_SCALAR(terrainMountainMaskWidth);
    LOAD_SCALAR(terrainWarpAmp);
    LOAD_SCALAR(terrainWarpFreq);
    LOAD_SCALAR(terrainRidgedSharpness);
    LOAD_SCALAR(terrainHeightLow);
    LOAD_SCALAR(terrainHeightHigh);
    LOAD_SCALAR(terrainBlendSharpness);
    LOAD_SCALAR(terrainSlopeThreshold);
    LOAD_SCALAR(detailTilePeriod);
    LOAD_SCALAR(detailNormalStrength);
    LOAD_SCALAR(clipmapMorphWidth);
    LOAD_SCALAR(liveLevelsCount);
    LOAD_BOOL(debugClipmapLevels);
    LOAD_BOOL(showMorphAlpha);
    LOAD_BOOL(useClipmapTerrain);
    LOAD_VEC3(sunDirection);
    LOAD_VEC3(sunColor);
    LOAD_SCALAR(sunIntensity);
    LOAD_BOOL(enableShadows);
    LOAD_SCALAR(shadowOrthoHalfSize);
    LOAD_BOOL(fogEnabled);
    LOAD_SCALAR(fogDensity);
    LOAD_SCALAR(fogHeightFalloff);
    LOAD_SCALAR(fogGroundLevel);
    LOAD_SCALAR(fogScatterCoef);
    LOAD_SCALAR(fogExtinctionCoef);
    LOAD_SCALAR(fogPhaseG);
    LOAD_SCALAR(fogSteps);
    LOAD_SCALAR(fogMaxDistance);
    LOAD_SCALAR(fogNoiseScale);
    LOAD_SCALAR(fogNoiseStrength);
    else if (key == "fogWindDir") iss >> fogWindDir.x >> fogWindDir.y;
    LOAD_SCALAR(fogWindSpeed);
    LOAD_SCALAR(tonemapMode);
    LOAD_SCALAR(adaptationSpeed);
    LOAD_SCALAR(keyValue);
    LOAD_SCALAR(minExposure);
    LOAD_SCALAR(maxExposure);
    LOAD_SCALAR(debugMode);

#undef LOAD_SCALAR
#undef LOAD_BOOL
#undef LOAD_VEC3
  }

  spdlog::info("loadSettings: loaded '{}'", path.string());
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
    "clipmap_terrain_depth",
    {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "clipmap_terrain_depth.vert.spv",
     BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "clipmap_terrain_depth.frag.spv"});
  etna::create_program(
    "clipmap_fill", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "clipmap_fill.comp.spv"});
  etna::create_program(
    "clipmap_splat", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "clipmap_splat.comp.spv"});
  etna::create_program(
    "detail_gen", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "detail_gen.comp.spv"});
  etna::create_program("fog", {BINDLESS_AND_PBR_RENDERER_SHADERS_ROOT "fog.comp.spv"});
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
  cullCountPipeline = pipelineManager.createComputePipeline("cull_count", {});
  prefixSumPipeline = pipelineManager.createComputePipeline("prefix_sum", {});
  cullWritePipeline = pipelineManager.createComputePipeline("cull_write", {});
  perlinPipeline = pipelineManager.createComputePipeline("perlin", {});
  normalPipeline = pipelineManager.createComputePipeline("normal", {});
  clipmapFillPipeline = pipelineManager.createComputePipeline("clipmap_fill", {});
  clipmapSplatPipeline = pipelineManager.createComputePipeline("clipmap_splat", {});
  detailGenPipeline = pipelineManager.createComputePipeline("detail_gen", {});
  fogPipeline = pipelineManager.createComputePipeline("fog", {});

  clipmapTerrainPipeline =
    pipelineManager
      .createGraphicsPipeline(
        "clipmap_terrain",
        etna::GraphicsPipeline::CreateInfo{
          .vertexShaderInput =
            etna::VertexShaderInputDescription{
              .bindings = {etna::VertexShaderInputDescription::Binding{
                .byteStreamDescription =
                  etna::VertexByteStreamFormatDescription{
                    .stride = sizeof(glm::vec2),
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

  // Depth-only pipeline для shadow pass'а. Vertex input идентичен color path,
  // но: нет color attachments, включён pipeline-level depth bias (борьба с
  // shadow acne на наклонных поверхностях). blendingConfig.attachments = {}
  // нужно явно — иначе etna выставит 1 attachment по умолчанию, что не
  // совпадёт с пустым colorAttachmentFormats.
  terrainDepthPipeline =
    pipelineManager
      .createGraphicsPipeline(
        "clipmap_terrain_depth",
        etna::GraphicsPipeline::CreateInfo{
          .vertexShaderInput =
            etna::VertexShaderInputDescription{
              .bindings = {etna::VertexShaderInputDescription::Binding{
                .byteStreamDescription =
                  etna::VertexByteStreamFormatDescription{
                    .stride = sizeof(glm::vec2),
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
              .depthBiasEnable = vk::True,
              .depthBiasConstantFactor = 1.25f,
              .depthBiasSlopeFactor = 1.75f,
              .lineWidth = 1.f,
            },
          .blendingConfig =
            {
              .attachments = {},
              .logicOpEnable = false,
              .logicOp = vk::LogicOp::eClear,
              .blendConstants = {0.f, 0.f, 0.f, 0.f},
            },
          .fragmentShaderOutput =
            {
              .colorAttachmentFormats = {},
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
  shadowDebugQuad = std::make_unique<QuadRenderer>(QuadRenderer::CreateInfo{
    .format = swapchain_format,
    .rect = {{0, 0}, {256, 256}},
  });
}

void WorldRenderer::updateCascades(const FramePacket& packet)
{
  const float nearClip = findZNear();
  const float farClip = findZFar();
  const float clipRange = farClip - nearClip;
  const float minZ = nearClip;
  const float maxZ = nearClip + clipRange;
  const float range = maxZ - minZ;
  const float ratio = maxZ / minZ;

  std::array<float, CASCADE_COUNT> splits{};
  for (uint32_t i = 0; i < CASCADE_COUNT; ++i)
  {
    const float p = static_cast<float>(i + 1) / static_cast<float>(CASCADE_COUNT);
    const float logS = minZ * std::pow(ratio, p);
    const float uniS = minZ + range * p;
    const float d = cascadeSplitLambda * (logS - uniS) + uniS;
    splits[i] = (d - nearClip) / clipRange;
  }

  const float aspect = static_cast<float>(resolution.x) / static_cast<float>(resolution.y);
  const glm::mat4 invCam = glm::inverse(packet.mainCam.projTm(aspect) * packet.mainCam.viewTm());
  const glm::vec3 lightDir = glm::normalize(sunDirection);

  float lastSplitDist = 0.0f;
  for (uint32_t i = 0; i < CASCADE_COUNT; ++i)
  {
    const float splitDist = splits[i];

    glm::vec3 corners[8] = {
      glm::vec3(-1.f, 1.f, 0.f),
      glm::vec3(1.f, 1.f, 0.f),
      glm::vec3(1.f, -1.f, 0.f),
      glm::vec3(-1.f, -1.f, 0.f),
      glm::vec3(-1.f, 1.f, 1.f),
      glm::vec3(1.f, 1.f, 1.f),
      glm::vec3(1.f, -1.f, 1.f),
      glm::vec3(-1.f, -1.f, 1.f),
    };
    for (uint32_t j = 0; j < 8; ++j)
    {
      const glm::vec4 w = invCam * glm::vec4(corners[j], 1.0f);
      corners[j] = glm::vec3(w) / w.w;
    }

    for (uint32_t j = 0; j < 4; ++j)
    {
      const glm::vec3 dist = corners[j + 4] - corners[j];
      corners[j + 4] = corners[j] + dist * splitDist;
      corners[j] = corners[j] + dist * lastSplitDist;
    }

    glm::vec3 center{0.0f};
    for (uint32_t j = 0; j < 8; ++j)
      center += corners[j];
    center /= 8.0f;

    float radius = 0.0f;
    for (uint32_t j = 0; j < 8; ++j)
      radius = std::max(radius, glm::length(corners[j] - center));
    radius = std::ceil(radius * 16.0f) / 16.0f;

    const glm::vec3 maxExt = glm::vec3(radius);
    const glm::vec3 minExt = -maxExt;

    const glm::vec3 lightEye = center - lightDir * (-minExt.z);
    const glm::mat4 lightView = glm::lookAtLH(lightEye, center, glm::vec3(0.f, 1.f, 0.f));
    const glm::mat4 lightProj =
      glm::orthoLH_ZO(minExt.x, maxExt.x, minExt.y, maxExt.y, 0.f, maxExt.z - minExt.z);

    cascadeViewProj[i] = lightProj * lightView;
    cascadeSplitDepths[i] = nearClip + splitDist * clipRange; // positive view-space Z (LH)

    lastSplitDist = splitDist;
  }

  static int logFrame = 0;
  if (++logFrame % 60 == 0)
  {
    for (uint32_t i = 0; i < CASCADE_COUNT; ++i)
    {
      const glm::vec4 origin = cascadeViewProj[i] * glm::vec4(0.f, 0.f, 0.f, 1.f);
      spdlog::info(
        "Cascade {}: split={:.1f} det={:.3e} worldOriginClip=({:.2f}, {:.2f}, {:.2f}, {:.2f})",
        i,
        cascadeSplitDepths[i],
        glm::determinant(cascadeViewProj[i]),
        origin.x,
        origin.y,
        origin.z,
        origin.w);
    }
  }
}

void WorldRenderer::debugInput(const Keyboard&) {}

void WorldRenderer::update(const FramePacket& packet)
{
  ZoneScoped;
  {
    const float aspect = float(resolution.x) / float(resolution.y);
    worldViewProj = packet.mainCam.projTm(aspect) * packet.mainCam.viewTm();
    cameraWorldPos = packet.mainCam.position;
  }

  {
    const glm::vec3 lightDir = glm::normalize(sunDirection);

    const glm::vec3 target = glm::vec3(cameraWorldPos.x, 0.f, cameraWorldPos.z);

    const float boxHalf = shadowOrthoHalfSize;
    const float boxDepth = std::max(2000.f, terrainHeightScale * 5.f);

    const glm::vec3 lightEye = target + lightDir * (boxDepth * 0.5f);
    const glm::mat4 lightView = glm::lookAtLH(lightEye, target, glm::vec3(0, 1, 0));
    const glm::mat4 lightProj =
      glm::orthoLH_ZO(-boxHalf, +boxHalf, -boxHalf, +boxHalf, 0.f, boxDepth);

    lightViewProj = lightProj * lightView;

    const glm::vec4 worldOriginClip = lightViewProj * glm::vec4(0, 0, 0, 1);
    const glm::vec2 shadowTexels = glm::vec2(worldOriginClip) * (float(SHADOWMAP_DIM) * 0.5f);
    const glm::vec2 rounded = glm::round(shadowTexels);
    const glm::vec2 fracOffset = (rounded - shadowTexels) * (2.f / float(SHADOWMAP_DIM));

    lightViewProj[3][0] += fracOffset.x;
    lightViewProj[3][1] += fracOffset.y;
  }

  updateCascades(packet);

  if (previousTime <= 0.f)
    deltaTime = 1.f / 60.f;
  else
    deltaTime = std::max(packet.currentTime - previousTime, 0.f);
  previousTime = packet.currentTime;

  timeSec = packet.currentTime; // для анимации тумана ветром
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

  if (selectedScene == SceneType::Terrain && useClipmapTerrain)
  {
    updateClipmapLevels();
    updateClipmapHeightmaps(cmd_buf);
    if (liveLevelsCount < CLIPMAP_LEVELS)
      updateClipmapAlbedos(cmd_buf);

    if (enableShadows)
    {
      renderShadowPass(cmd_buf);
      etna::set_state(
        cmd_buf,
        shadowMap.get(),
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::ImageLayout::eDepthReadOnlyOptimal,
        vk::ImageAspectFlagBits::eDepth);
      etna::flush_barriers(cmd_buf);
    }
  }

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

  // Volumetric fog
  const bool fogActive =
    fogEnabled && enableShadows && selectedScene == SceneType::Terrain && useClipmapTerrain;
  if (fogActive)
  {
    renderFogPass(cmd_buf);
  }
  else
  {
    etna::set_state(
      cmd_buf,
      fogTarget.get(),
      vk::PipelineStageFlagBits2::eFragmentShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(cmd_buf);
  }

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
    auto fogBinding =
      fogTarget.genBinding(fogSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal);
    auto descSet = etna::create_descriptor_set(
      programInfo.getDescriptorLayoutId(0),
      cmd_buf,
      {etna::Binding{0, hdrBinding}, etna::Binding{1, statsBinding}, etna::Binding{2, fogBinding}});

    vk::DescriptorSet vkSet = descSet.getVkSet();
    const auto layout = postprocessPipeline.getVkPipelineLayout();
    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, postprocessPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, 1, &vkSet, 0, nullptr);

    struct PostPC
    {
      std::uint32_t tonemapMode;
      std::uint32_t fogEnabled;
    } postPc{
      static_cast<std::uint32_t>(tonemapMode),
      fogActive ? 1u : 0u,
    };
    cmd_buf.pushConstants<PostPC>(layout, vk::ShaderStageFlagBits::eFragment, 0, {postPc});

    cmd_buf.draw(3, 1, 0, 0);
  }

  if (drawShadowMapOverlay && shadowDebugQuad && enableShadows)
  {
    shadowDebugQuad->render(cmd_buf, target_image, target_image_view, shadowMap, perlinSampler);
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
      0,
      1,
      &vkSet,
      0,
      nullptr);
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
      info.getDescriptorLayoutId(0), cmd_buf, {etna::Binding{0, bind0}, etna::Binding{1, bind1}});
    vk::DescriptorSet vkSet = set.getVkSet();

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, normalPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      normalPipeline.getVkPipelineLayout(),
      0,
      1,
      &vkSet,
      0,
      nullptr);
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
    info.getDescriptorLayoutId(0), cmd_buf, {etna::Binding{0, bind0}, etna::Binding{1, bind1}});
  auto vkSet = descSet.getVkSet();
  auto layout = terrainPipeline.getVkPipelineLayout();

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, terrainPipeline.getVkPipeline());
  cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, 1, &vkSet, 0, nullptr);

  TerrainPushConst pc{
    worldViewProj,
    glm::vec4(cameraWorldPos, 0.f),
    glm::vec4(glm::normalize(sunDirection), 0.f),
    glm::vec4(sunColor, sunIntensity),
  };
  cmd_buf.pushConstants<TerrainPushConst>(
    layout,
    vk::ShaderStageFlagBits::eTessellationControl |
      vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment,
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
    sceneRelemCount = 0;
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

  if (ImGui::Button("Save settings"))
    saveSettings("terrain_settings.txt");
  ImGui::SameLine();
  if (ImGui::Button("Load settings"))
    loadSettings("terrain_settings.txt");
  ImGui::SameLine();
  ImGui::TextDisabled("(?)");
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip(
      "Reads/writes terrain_settings.txt in the working directory.\n"
      "The file is auto-loaded at startup if present.");
  ImGui::Separator();

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

  const char* scenes[] = {
    "SimpleMeshes", "Low Poly Dark Town", "Lovely Town", "Avocado", "Terrain"};
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
    ImGui::Checkbox("Use Clipmap", &useClipmapTerrain);
    if (useClipmapTerrain)
    {
      ImGui::Checkbox("Debug: show LOD levels", &debugClipmapLevels);
      ImGui::Checkbox("Debug: show morph alpha", &showMorphAlpha);
      ImGui::SliderFloat("Morph width (texels)", &clipmapMorphWidth, 1.f, 64.f, "%.1f");

      if (ImGui::CollapsingHeader("Terrain shape"))
      {
        ImGui::SliderFloat("Height scale (m)", &terrainHeightScale, 10.f, 800.f, "%.0f");
        ImGui::SliderFloat("Hills weight", &terrainHillsWeight, 0.f, 3.f, "%.2f");
        ImGui::SliderFloat("Ridges weight", &terrainRidgesWeight, 0.f, 3.f, "%.2f");
        ImGui::SliderFloat("Detail amplitude", &terrainDetailAmplitude, 0.f, 0.1f, "%.3f");
      }

      if (ImGui::CollapsingHeader("fBm"))
      {
        float period = terrainBaseFreq > 1e-7f ? 1.0f / terrainBaseFreq : 1024.f;
        if (ImGui::SliderFloat("Base period (m)", &period, 64.f, 8192.f, "%.0f"))
          terrainBaseFreq = 1.0f / period;
        ImGui::SliderInt("Octaves", &terrainOctaves, 1, 8);
        ImGui::SliderFloat("Persistence", &terrainPersistence, 0.1f, 1.0f, "%.2f");
        ImGui::SliderFloat("Lacunarity", &terrainLacunarity, 1.2f, 3.0f, "%.2f");
      }

      if (ImGui::CollapsingHeader("Shaping"))
      {
        ImGui::SliderFloat("Bias power", &terrainBiasPower, 0.2f, 4.0f, "%.2f");
        ImGui::SliderFloat("Ocean cut", &terrainOceanCut, 0.0f, 0.6f, "%.2f");
      }

      if (ImGui::CollapsingHeader("Mountain mask"))
      {
        float maskPeriod = terrainMountainMaskFreq > 1e-7f ? 1.0f / terrainMountainMaskFreq : 0.f;
        if (ImGui::SliderFloat("Mask period (m)", &maskPeriod, 512.f, 16384.f, "%.0f"))
          terrainMountainMaskFreq = maskPeriod > 0.f ? 1.0f / maskPeriod : terrainMountainMaskFreq;
        ImGui::SliderFloat("Mask offset", &terrainMountainMaskOffset, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Mask width", &terrainMountainMaskWidth, 0.05f, 1.0f, "%.2f");
      }

      if (ImGui::CollapsingHeader("Domain warp"))
      {
        ImGui::SliderFloat("Warp amplitude (m)", &terrainWarpAmp, 0.f, 400.f, "%.0f");
        float warpPeriod = terrainWarpFreq > 1e-7f ? 1.0f / terrainWarpFreq : 0.f;
        if (ImGui::SliderFloat("Warp period (m)", &warpPeriod, 64.f, 4096.f, "%.0f"))
          terrainWarpFreq = warpPeriod > 0.f ? 1.0f / warpPeriod : terrainWarpFreq;
      }

      if (ImGui::CollapsingHeader("Ridged multifractal"))
      {
        ImGui::SliderFloat("Ridge sharpness", &terrainRidgedSharpness, 1.0f, 6.0f, "%.1f");
      }

      if (ImGui::CollapsingHeader("Splatting"))
      {
        ImGui::SliderInt("Live levels (close-up)", &liveLevelsCount, 0, CLIPMAP_LEVELS);
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip(
            "Number of innermost clipmap levels that compute splatting per-fragment\n"
            "from full-resolution detail textures (sharp close-up).\n"
            "Remaining outer levels read from pre-baked 256x256 albedo cache (fast).\n"
            "0 = all baked (fastest, blurry close-up)\n"
            "%d = all live  (slowest, sharpest)\n"
            "3 = balanced (default): innermost 3 rings sharp, rest cached.",
            CLIPMAP_LEVELS);

        ImGui::SliderFloat("Ground-Grass height", &terrainHeightLow, -50.f, 100.f, "%.1f");
        ImGui::SliderFloat("Grass-Snow height", &terrainHeightHigh, 0.f, 150.f, "%.1f");
        ImGui::SliderFloat("Blend sharpness", &terrainBlendSharpness, 0.5f, 40.f, "%.1f");
        ImGui::SliderFloat("Rock slope", &terrainSlopeThreshold, 0.0f, 1.0f, "%.2f");
      }

      if (ImGui::CollapsingHeader("Detail textures"))
      {
        ImGui::SliderFloat("Tile period (m)", &detailTilePeriod, 2.f, 256.f, "%.0f");
        ImGui::SliderFloat("Normal strength", &detailNormalStrength, 0.f, 2.f, "%.2f");
      }
    }
  }

  const char* debugModes[] = {"Shaded", "BaseColor", "Normal (raw)", "MetalRough", "Occlusion"};
  ImGui::Combo("Debug view", &debugMode, debugModes, IM_ARRAYSIZE(debugModes));

  if (ImGui::CollapsingHeader("Shadows"))
  {
    ImGui::Checkbox("Enable shadows", &enableShadows);
    ImGui::SliderFloat("Ortho half size (m)", &shadowOrthoHalfSize, 200.f, 2000.f, "%.0f");
    ImGui::Checkbox("Show shadow map overlay", &drawShadowMapOverlay);
  }

  if (ImGui::CollapsingHeader("Volumetric Fog"))
  {
    ImGui::Checkbox("Enable fog", &fogEnabled);
    ImGui::TextDisabled("(требует Enable shadows + clipmap terrain)");
    ImGui::SliderFloat("Density", &fogDensity, 0.0f, 0.2f, "%.4f");
    ImGui::SliderFloat("Height falloff", &fogHeightFalloff, 1.0f, 500.f, "%.0f");
    ImGui::SliderFloat("Ground level", &fogGroundLevel, -100.f, 400.f, "%.0f");
    ImGui::SliderFloat("Scatter coef", &fogScatterCoef, 0.0f, 4.0f, "%.2f");
    ImGui::SliderFloat("Extinction coef", &fogExtinctionCoef, 0.0f, 4.0f, "%.2f");
    ImGui::SliderFloat("Phase g", &fogPhaseG, -0.9f, 0.9f, "%.2f");
    ImGui::SliderInt("Steps (quality)", &fogSteps, 8, 128);
    ImGui::SliderFloat("Max distance", &fogMaxDistance, 100.f, 5000.f, "%.0f");
    ImGui::SliderFloat("Noise scale", &fogNoiseScale, 0.0005f, 0.05f, "%.4f");
    ImGui::SliderFloat("Noise strength", &fogNoiseStrength, 0.0f, 1.0f, "%.2f");
    ImGui::Separator();
    ImGui::SliderFloat2("Wind dir", &fogWindDir.x, -1.f, 1.f, "%.2f");
    ImGui::SliderFloat("Wind speed", &fogWindSpeed, 0.0f, 20.f, "%.1f");
  }

  if (ImGui::CollapsingHeader("Adaptive exposure"))
  {
    const char* tonemaps[] = {"Reinhard", "ACES", "None (clamp)"};
    ImGui::Combo("Tonemap", &tonemapMode, tonemaps, IM_ARRAYSIZE(tonemaps));
    ImGui::SliderFloat("Adaptation speed", &adaptationSpeed, 0.1f, 10.f, "%.2f");
    ImGui::SliderFloat("Key value", &keyValue, 0.01f, 1.0f, "%.3f");
    ImGui::SliderFloat("Min exposure", &minExposure, 0.001f, 1.f, "%.3f");
    ImGui::SliderFloat("Max exposure", &maxExposure, 1.f, 100.f, "%.1f");
  }

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

void WorldRenderer::initDetailTexture(vk::CommandBuffer cmd_buf)
{
  if (detailTexInitialized)
    return;

  ETNA_PROFILE_GPU(cmd_buf, clipmapDetailGen);

  {
    vk::ImageMemoryBarrier2 toGeneral{
      .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
      .srcAccessMask = {},
      .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
      .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
      .oldLayout = vk::ImageLayout::eUndefined,
      .newLayout = vk::ImageLayout::eGeneral,
      .image = detailTex.get(),
      .subresourceRange =
        vk::ImageSubresourceRange{
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .baseMipLevel = 0,
          .levelCount = DETAIL_TEX_MIPS,
          .baseArrayLayer = 0,
          .layerCount = 1,
        },
    };
    cmd_buf.pipelineBarrier2(
      vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &toGeneral,
      });
  }

  {
    auto info = etna::get_shader_program("detail_gen");
    auto bind = detailTex.genBinding(
      perlinSampler.get(),
      vk::ImageLayout::eGeneral,
      etna::Image::ViewParams{.baseMip = 0, .levelCount = 1});
    auto descSet =
      etna::create_descriptor_set(info.getDescriptorLayoutId(0), cmd_buf, {etna::Binding{0, bind}});
    vk::DescriptorSet vkSet = descSet.getVkSet();

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, detailGenPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      detailGenPipeline.getVkPipelineLayout(),
      0,
      1,
      &vkSet,
      0,
      nullptr);
    // 512 / 16 = 32 groups
    cmd_buf.dispatch(DETAIL_TEX_SIZE / 16, DETAIL_TEX_SIZE / 16, 1);
  }

  {
    vk::ImageMemoryBarrier2 b{
      .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
      .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
      .dstStageMask = vk::PipelineStageFlagBits2::eBlit,
      .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
      .oldLayout = vk::ImageLayout::eGeneral,
      .newLayout = vk::ImageLayout::eGeneral,
      .image = detailTex.get(),
      .subresourceRange =
        vk::ImageSubresourceRange{
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0,
          .layerCount = 1,
        },
    };
    cmd_buf.pipelineBarrier2(
      vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &b,
      });
  }

  int srcW = static_cast<int>(DETAIL_TEX_SIZE);
  int srcH = static_cast<int>(DETAIL_TEX_SIZE);
  for (uint32_t i = 1; i < DETAIL_TEX_MIPS; ++i)
  {
    const int dstW = std::max(srcW / 2, 1);
    const int dstH = std::max(srcH / 2, 1);

    vk::ImageBlit blit{
      .srcSubresource =
        vk::ImageSubresourceLayers{
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .mipLevel = i - 1,
          .baseArrayLayer = 0,
          .layerCount = 1,
        },
      .srcOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{srcW, srcH, 1}},
      .dstSubresource =
        vk::ImageSubresourceLayers{
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .mipLevel = i,
          .baseArrayLayer = 0,
          .layerCount = 1,
        },
      .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{dstW, dstH, 1}},
    };
    cmd_buf.blitImage(
      detailTex.get(),
      vk::ImageLayout::eGeneral,
      detailTex.get(),
      vk::ImageLayout::eGeneral,
      1,
      &blit,
      vk::Filter::eLinear);

    if (i + 1 < DETAIL_TEX_MIPS)
    {
      vk::ImageMemoryBarrier2 b{
        .srcStageMask = vk::PipelineStageFlagBits2::eBlit,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eBlit,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = detailTex.get(),
        .subresourceRange =
          vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = i,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
          },
      };
      cmd_buf.pipelineBarrier2(
        vk::DependencyInfo{
          .imageMemoryBarrierCount = 1,
          .pImageMemoryBarriers = &b,
        });
    }

    srcW = dstW;
    srcH = dstH;
  }

  {
    vk::ImageMemoryBarrier2 toReadOnly{
      .srcStageMask = vk::PipelineStageFlagBits2::eBlit,
      .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
      .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
      .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
      .oldLayout = vk::ImageLayout::eGeneral,
      .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
      .image = detailTex.get(),
      .subresourceRange =
        vk::ImageSubresourceRange{
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .baseMipLevel = 0,
          .levelCount = DETAIL_TEX_MIPS,
          .baseArrayLayer = 0,
          .layerCount = 1,
        },
    };
    cmd_buf.pipelineBarrier2(
      vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &toReadOnly,
      });
  }

  detailTexInitialized = true;
}

void WorldRenderer::updateClipmapHeightmaps(vk::CommandBuffer cmd_buf)
{
  ETNA_PROFILE_GPU(cmd_buf, clipmapFill);

  initDetailTexture(cmd_buf);

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

  struct FillPC
  {
    glm::vec2 levelOrigin; // group 0
    float gridStep;
    int32_t levelIdx;
    float hillsWeight; // group 1
    float ridgesWeight;
    float detailAmplitude;
    float biasPower;
    float oceanCut; // group 2
    float mountainMaskFreq;
    float mountainMaskOffset;
    float mountainMaskWidth;
    float baseFreq; // group 3
    int32_t octaves;
    float persistence;
    float lacunarity;
    float warpAmp; // group 4
    float warpFreq;
    float ridgedSharpness;
    float _pad{0.f};
  };

  auto fillInfo = etna::get_shader_program("clipmap_fill");
  auto fillBind = clipmapHeightmapArray.genBinding(
    clipmapSampler.get(),
    vk::ImageLayout::eGeneral,
    etna::Image::ViewParams{.type = vk::ImageViewType::e2DArray});
  auto detailBind = detailTex.genBinding(
    clipmapSampler.get(), // eRepeat addressMode for tile wrapping
    vk::ImageLayout::eShaderReadOnlyOptimal);
  auto fillSet = etna::create_descriptor_set(
    fillInfo.getDescriptorLayoutId(0),
    cmd_buf,
    {etna::Binding{0, fillBind}, etna::Binding{1, detailBind}});
  vk::DescriptorSet fillVkSet = fillSet.getVkSet();

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, clipmapFillPipeline.getVkPipeline());
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eCompute,
    clipmapFillPipeline.getVkPipelineLayout(),
    0,
    1,
    &fillVkSet,
    0,
    nullptr);

  for (int level = CLIPMAP_LEVELS - 1; level >= 0; --level)
  {
    const float step = clipmapBaseStep * static_cast<float>(1 << level);
    const float snap = 2.f * step;
    const glm::vec2 center = glm::floor(camXZ / snap) * snap;
    const glm::vec2 origin = center - halfGrid * step;

    const int instanceIdx = CLIPMAP_LEVELS - 1 - level;
    const FillPC pc{
      origin,
      step,
      instanceIdx,
      terrainHillsWeight,
      terrainRidgesWeight,
      terrainDetailAmplitude,
      terrainBiasPower,
      terrainOceanCut,
      terrainMountainMaskFreq,
      terrainMountainMaskOffset,
      terrainMountainMaskWidth,
      terrainBaseFreq,
      terrainOctaves,
      terrainPersistence,
      terrainLacunarity,
      terrainWarpAmp,
      terrainWarpFreq,
      terrainRidgedSharpness,
    };
    cmd_buf.pushConstants<FillPC>(
      clipmapFillPipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {pc});

    cmd_buf.dispatch(16, 16, 1);
  }

  etna::set_state(
    cmd_buf,
    clipmapHeightmapArray.get(),
    vk::PipelineStageFlagBits2::eVertexShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmd_buf);
}

void WorldRenderer::updateClipmapAlbedos(vk::CommandBuffer cmd_buf)
{
  ETNA_PROFILE_GPU(cmd_buf, clipmapSplat);

  const float heightScale = terrainHeightScale;
  const float halfGrid = static_cast<float>(clipmapMesh->n() - 1) * 0.5f;
  const glm::vec2 camXZ{cameraWorldPos.x, cameraWorldPos.z};

  etna::set_state(
    cmd_buf,
    clipmapHeightmapArray.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  etna::flush_barriers(cmd_buf);
  {
    vk::ImageMemoryBarrier2 toGeneral{
      .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
      .srcAccessMask = {},
      .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
      .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
      .oldLayout = vk::ImageLayout::eUndefined,
      .newLayout = vk::ImageLayout::eGeneral,
      .image = clipmapAlbedoArray.get(),
      .subresourceRange =
        vk::ImageSubresourceRange{
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .baseMipLevel = 0,
          .levelCount = CLIPMAP_ALBEDO_MIPS,
          .baseArrayLayer = 0,
          .layerCount = CLIPMAP_LEVELS,
        },
    };
    cmd_buf.pipelineBarrier2(
      vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &toGeneral,
      });
  }

  struct SplatPC
  {
    glm::vec2 levelOrigin;
    float gridStep;
    int32_t levelIdx;
    float heightScale;
    float detailTilePeriod;
    uint32_t _pad0;
    uint32_t _pad1;
    glm::vec4 splatParams;
  }; // 48

  auto info = etna::get_shader_program("clipmap_splat");
  auto bindH = clipmapHeightmapArray.genBinding(
    perlinSampler.get(),
    vk::ImageLayout::eShaderReadOnlyOptimal,
    etna::Image::ViewParams{.type = vk::ImageViewType::e2DArray});
  // Storage-image view must be single-mip; bind only mip 0 for the compute write.
  auto bindA = clipmapAlbedoArray.genBinding(
    clipmapSampler.get(),
    vk::ImageLayout::eGeneral,
    etna::Image::ViewParams{
      .baseMip = 0,
      .levelCount = 1,
      .type = vk::ImageViewType::e2DArray,
    });
  auto bindDC = detailColorArray.genBinding(
    detailSampler.get(),
    vk::ImageLayout::eShaderReadOnlyOptimal,
    etna::Image::ViewParams{.type = vk::ImageViewType::e2DArray});
  auto bindDH = detailHeightArray.genBinding(
    detailSampler.get(),
    vk::ImageLayout::eShaderReadOnlyOptimal,
    etna::Image::ViewParams{.type = vk::ImageViewType::e2DArray});

  auto descSet = etna::create_descriptor_set(
    info.getDescriptorLayoutId(0),
    cmd_buf,
    {etna::Binding{0, bindH},
     etna::Binding{1, bindA},
     etna::Binding{2, bindDC},
     etna::Binding{3, bindDH}});
  vk::DescriptorSet vkSet = descSet.getVkSet();

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, clipmapSplatPipeline.getVkPipeline());
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eCompute,
    clipmapSplatPipeline.getVkPipelineLayout(),
    0,
    1,
    &vkSet,
    0,
    nullptr);

  const glm::vec4 splatParams4{
    terrainHeightLow,
    terrainHeightHigh,
    terrainBlendSharpness,
    terrainSlopeThreshold,
  };

  const int firstBakedLevel = std::clamp(liveLevelsCount, 0, CLIPMAP_LEVELS);
  for (int level = CLIPMAP_LEVELS - 1; level >= firstBakedLevel; --level)
  {
    const float step = clipmapBaseStep * static_cast<float>(1 << level);
    const float snap = 2.f * step;
    const glm::vec2 center = glm::floor(camXZ / snap) * snap;
    const glm::vec2 origin = center - halfGrid * step;

    const int32_t instanceIdx = CLIPMAP_LEVELS - 1 - level;
    const SplatPC pc{origin, step, instanceIdx, heightScale, detailTilePeriod, 0, 0, splatParams4};

    cmd_buf.pushConstants<SplatPC>(
      clipmapSplatPipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {pc});
    cmd_buf.dispatch(16, 16, 1);
  }

  {
    vk::ImageMemoryBarrier2 computeToBlit{
      .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
      .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
      .dstStageMask = vk::PipelineStageFlagBits2::eBlit,
      .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
      .oldLayout = vk::ImageLayout::eGeneral,
      .newLayout = vk::ImageLayout::eGeneral,
      .image = clipmapAlbedoArray.get(),
      .subresourceRange =
        vk::ImageSubresourceRange{
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0,
          .layerCount = CLIPMAP_LEVELS,
        },
    };
    cmd_buf.pipelineBarrier2(
      vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &computeToBlit,
      });
  }

  int srcW = 256, srcH = 256;
  for (uint32_t i = 1; i < CLIPMAP_ALBEDO_MIPS; ++i)
  {
    const int dstW = std::max(srcW / 2, 1);
    const int dstH = std::max(srcH / 2, 1);

    vk::ImageBlit blit{
      .srcSubresource =
        vk::ImageSubresourceLayers{
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .mipLevel = i - 1,
          .baseArrayLayer = 0,
          .layerCount = CLIPMAP_LEVELS,
        },
      .srcOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{srcW, srcH, 1}},
      .dstSubresource =
        vk::ImageSubresourceLayers{
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .mipLevel = i,
          .baseArrayLayer = 0,
          .layerCount = CLIPMAP_LEVELS,
        },
      .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{dstW, dstH, 1}},
    };
    cmd_buf.blitImage(
      clipmapAlbedoArray.get(),
      vk::ImageLayout::eGeneral,
      clipmapAlbedoArray.get(),
      vk::ImageLayout::eGeneral,
      1,
      &blit,
      vk::Filter::eLinear);

    if (i + 1 < CLIPMAP_ALBEDO_MIPS)
    {
      vk::ImageMemoryBarrier2 b{
        .srcStageMask = vk::PipelineStageFlagBits2::eBlit,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eBlit,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = clipmapAlbedoArray.get(),
        .subresourceRange =
          vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = i,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = CLIPMAP_LEVELS,
          },
      };
      cmd_buf.pipelineBarrier2(
        vk::DependencyInfo{
          .imageMemoryBarrierCount = 1,
          .pImageMemoryBarriers = &b,
        });
    }

    srcW = dstW;
    srcH = dstH;
  }

  // All mips: eGeneral -> eShaderReadOnlyOptimal for fragment trilinear sampling.
  {
    vk::ImageMemoryBarrier2 toReadOnly{
      .srcStageMask = vk::PipelineStageFlagBits2::eBlit,
      .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
      .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
      .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
      .oldLayout = vk::ImageLayout::eGeneral,
      .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
      .image = clipmapAlbedoArray.get(),
      .subresourceRange =
        vk::ImageSubresourceRange{
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .baseMipLevel = 0,
          .levelCount = CLIPMAP_ALBEDO_MIPS,
          .baseArrayLayer = 0,
          .layerCount = CLIPMAP_LEVELS,
        },
    };
    cmd_buf.pipelineBarrier2(
      vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &toReadOnly,
      });
  }
}

void WorldRenderer::updateClipmapLevels()
{
  const float halfGrid = static_cast<float>(clipmapMesh->n() - 1) * 0.5f;
  const glm::vec2 camXZ{cameraWorldPos.x, cameraWorldPos.z};
  auto* levelData = reinterpret_cast<glm::vec4*>(clipmapLevelsBuffer.data());
  for (int level = CLIPMAP_LEVELS - 1; level >= 0; --level)
  {
    const float step = clipmapBaseStep * static_cast<float>(1 << level);
    const float snap = 2.f * step;
    const glm::vec2 center = glm::floor(camXZ / snap) * snap;
    const glm::vec2 origin = center - halfGrid * step;
    levelData[CLIPMAP_LEVELS - 1 - level] = glm::vec4(origin, step, 0.f);
  }
}

void WorldRenderer::renderFogPass(vk::CommandBuffer cmd_buf)
{
  ETNA_PROFILE_GPU(cmd_buf, fogPass);

  // Подготовка layout'ов для compute-выборки:
  //  - scene depth: depth attachment → sampled read
  //  - shadow map: уже depthReadOnly, делаем видимой для compute-стейджа
  //  - fog target: → general (storage write)
  etna::set_state(
    cmd_buf,
    mainViewDepth.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eDepth);
  etna::set_state(
    cmd_buf,
    shadowMap.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eDepthReadOnlyOptimal,
    vk::ImageAspectFlagBits::eDepth);
  etna::set_state(
    cmd_buf,
    fogTarget.get(),
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderStorageWrite,
    vk::ImageLayout::eGeneral,
    vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmd_buf);

  struct FogPC
  {
    glm::mat4 invViewProj;
    glm::mat4 lightViewProj;
    glm::vec4 cameraPos;
    glm::vec4 sunDir;
    glm::vec4 sunColor;
    glm::vec4 fogParams0; // density, heightFalloff, groundLevel, scatterCoef
    glm::vec4 fogParams1; // extinctionCoef, phaseG, maxDistance, steps
    glm::vec4 fogParams2; // noiseScale, noiseStrength, time, windSpeed
    glm::vec4 windDir;    // xy = wind direction
  };

  auto info = etna::get_shader_program("fog");
  auto bind0 = fogTarget.genBinding(
    fogSampler.get(), vk::ImageLayout::eGeneral); // storage; sampler игнорируется
  auto bind1 = shadowMap.genBinding(shadowSampler.get(), vk::ImageLayout::eDepthReadOnlyOptimal);
  auto bind2 =
    mainViewDepth.genBinding(depthPointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal);

  auto descSet = etna::create_descriptor_set(
    info.getDescriptorLayoutId(0),
    cmd_buf,
    {etna::Binding{0, bind0}, etna::Binding{1, bind1}, etna::Binding{2, bind2}});
  vk::DescriptorSet vkSet = descSet.getVkSet();

  auto layout = fogPipeline.getVkPipelineLayout();
  cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, fogPipeline.getVkPipeline());
  cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, 1, &vkSet, 0, nullptr);

  const FogPC pc{
    glm::inverse(worldViewProj),
    lightViewProj,
    glm::vec4(cameraWorldPos, 0.f),
    glm::vec4(glm::normalize(sunDirection), 0.f),
    glm::vec4(sunColor, sunIntensity),
    glm::vec4(fogDensity, fogHeightFalloff, fogGroundLevel, fogScatterCoef),
    glm::vec4(fogExtinctionCoef, fogPhaseG, fogMaxDistance, static_cast<float>(fogSteps)),
    glm::vec4(fogNoiseScale, fogNoiseStrength, timeSec, fogWindSpeed),
    glm::vec4(fogWindDir, 0.f, 0.f),
  };
  cmd_buf.pushConstants<FogPC>(layout, vk::ShaderStageFlagBits::eCompute, 0, {pc});

  cmd_buf.dispatch((fogResolution.x + 7) / 8, (fogResolution.y + 7) / 8, 1);

  etna::set_state(
    cmd_buf,
    fogTarget.get(),
    vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmd_buf);
}

void WorldRenderer::renderShadowPass(vk::CommandBuffer cmd_buf)
{
  ETNA_PROFILE_GPU(cmd_buf, shadowMapPass);

  auto info = etna::get_shader_program("clipmap_terrain_depth");
  auto bind0 = clipmapHeightmapArray.genBinding(
    perlinSampler.get(),
    vk::ImageLayout::eShaderReadOnlyOptimal,
    etna::Image::ViewParams{.type = vk::ImageViewType::e2DArray});
  auto bind2 = clipmapLevelsBuffer.genBinding();

  auto descSet = etna::create_descriptor_set(
    info.getDescriptorLayoutId(0), cmd_buf, {etna::Binding{0, bind0}, etna::Binding{2, bind2}});

  etna::RenderTargetState rt(
    cmd_buf,
    {{0, 0}, {SHADOWMAP_DIM, SHADOWMAP_DIM}},
    {},
    {.image = shadowMap.get(), .view = shadowMap.getView({})});

  auto layout = terrainDepthPipeline.getVkPipelineLayout();
  cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, terrainDepthPipeline.getVkPipeline());
  vk::DescriptorSet vkSet = descSet.getVkSet();
  cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, 1, &vkSet, 0, nullptr);
  cmd_buf.bindVertexBuffers(0, {clipmapMesh->vertexBuffer()}, {vk::DeviceSize{0}});
  cmd_buf.bindIndexBuffer(clipmapMesh->indexBuffer(), 0, vk::IndexType::eUint32);

  struct DepthPC
  {
    glm::mat4 lightViewProj;
    glm::vec4 eyeAndScale; // .w (heightScale)
    glm::vec4 morphParams; // .x (morphWidth)
  };
  const DepthPC pc{
    lightViewProj,
    glm::vec4(cameraWorldPos, terrainHeightScale),
    glm::vec4(clipmapMorphWidth, 0.f, 0.f, 0.f),
  };
  cmd_buf.pushConstants<DepthPC>(layout, vk::ShaderStageFlagBits::eVertex, 0, {pc});

  constexpr uint32_t SHADOW_FIRST_INSTANCE = 4;
  constexpr uint32_t SHADOW_INSTANCE_COUNT = CLIPMAP_LEVELS - SHADOW_FIRST_INSTANCE;
  cmd_buf.drawIndexed(
    clipmapFootprint.indexCount,
    SHADOW_INSTANCE_COUNT,
    clipmapFootprint.firstIndex,
    clipmapFootprint.vertexOffset,
    SHADOW_FIRST_INSTANCE);
}

void WorldRenderer::renderClipmapTerrain(vk::CommandBuffer cmd_buf)
{
  ETNA_PROFILE_GPU(cmd_buf, clipmapTerrain);

  const float heightScale = terrainHeightScale;

  auto info = etna::get_shader_program("clipmap_terrain");
  auto bind0 = clipmapHeightmapArray.genBinding(
    perlinSampler.get(),
    vk::ImageLayout::eShaderReadOnlyOptimal,
    etna::Image::ViewParams{.type = vk::ImageViewType::e2DArray});
  auto bind2 = clipmapLevelsBuffer.genBinding();
  auto bind3 = clipmapAlbedoArray.genBinding(
    perlinSampler.get(),
    vk::ImageLayout::eShaderReadOnlyOptimal,
    etna::Image::ViewParams{.type = vk::ImageViewType::e2DArray});
  auto bind4 = detailColorArray.genBinding(
    detailSampler.get(),
    vk::ImageLayout::eShaderReadOnlyOptimal,
    etna::Image::ViewParams{.type = vk::ImageViewType::e2DArray});
  auto bind5 = detailHeightArray.genBinding(
    detailSampler.get(),
    vk::ImageLayout::eShaderReadOnlyOptimal,
    etna::Image::ViewParams{.type = vk::ImageViewType::e2DArray});
  auto bind6 = shadowMap.genBinding(shadowSampler.get(), vk::ImageLayout::eDepthReadOnlyOptimal);

  auto descSet = etna::create_descriptor_set(
    info.getDescriptorLayoutId(0),
    cmd_buf,
    {etna::Binding{0, bind0},
     etna::Binding{2, bind2},
     etna::Binding{3, bind3},
     etna::Binding{4, bind4},
     etna::Binding{5, bind5},
     etna::Binding{6, bind6}});
  auto layout = clipmapTerrainPipeline.getVkPipelineLayout();

  const float scaleSigned = debugClipmapLevels ? -heightScale : heightScale;
  const ClipmapPushConst pc{
    worldViewProj,
    glm::vec4(glm::normalize(sunDirection), detailNormalStrength),
    glm::vec4(sunColor, sunIntensity),
    glm::vec4(cameraWorldPos, scaleSigned),
    glm::vec4(
      clipmapMorphWidth,
      showMorphAlpha ? 1.0f : 0.0f,
      static_cast<float>(liveLevelsCount),
      detailTilePeriod),
    glm::vec4(terrainHeightLow, terrainHeightHigh, terrainBlendSharpness, terrainSlopeThreshold),
    lightViewProj,
    glm::vec4(enableShadows ? 1.f : 0.f, 0.f, 0.f, 0.f),
  };

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, clipmapTerrainPipeline.getVkPipeline());
  vk::DescriptorSet vkSet = descSet.getVkSet();
  cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, 1, &vkSet, 0, nullptr);
  cmd_buf.bindVertexBuffers(0, {clipmapMesh->vertexBuffer()}, {vk::DeviceSize{0}});
  cmd_buf.bindIndexBuffer(clipmapMesh->indexBuffer(), 0, vk::IndexType::eUint32);
  cmd_buf.pushConstants<ClipmapPushConst>(
    layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, {pc});

  cmd_buf.drawIndexed(
    clipmapFootprint.indexCount,
    CLIPMAP_LEVELS,
    clipmapFootprint.firstIndex,
    clipmapFootprint.vertexOffset,
    0);
}
