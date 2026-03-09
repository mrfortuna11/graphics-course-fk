#include "SceneManager.hpp"

#include <array>
#include <cctype>
#include <stack>
#include <bit>
#include <cstring>
#include <limits>

#include <spdlog/spdlog.h>
#include <fmt/std.h>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/OneShotCmdMgr.hpp>


SceneManager::SceneManager()
  : oneShotCommands{etna::get_context().createOneShotCmdMgr()}
  , transferHelper{etna::BlockingTransferHelper::CreateInfo{.stagingSize = 4096 * 4096 * 4}}
{
}

namespace
{

struct SceneFsUserData
{
  std::filesystem::path sceneDir;
};

bool is_image_extension(std::filesystem::path path)
{
  auto ext = path.extension().string();
  for (auto& c : ext)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

  return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tga" ||
         ext == ".gif" || ext == ".webp" || ext == ".ktx" || ext == ".ktx2";
}

std::string remap_texture_to_textures_subfolder(const std::string& filepath, void* user_data)
{
  const auto expanded = tinygltf::ExpandFilePath(filepath, nullptr);

  auto* fsUserData = static_cast<SceneFsUserData*>(user_data);
  if (!fsUserData)
    return expanded;

  const std::filesystem::path sourcePath = expanded;
  if (!is_image_extension(sourcePath))
    return expanded;

  const auto remapped = fsUserData->sceneDir / "textures" / sourcePath.filename();
  const auto remappedExpanded = tinygltf::ExpandFilePath(remapped.string(), nullptr);

  if (tinygltf::FileExists(remappedExpanded, nullptr))
    return remappedExpanded;

  return expanded;
}

bool scene_file_exists(const std::string& abs_filename, void* user_data)
{
  return tinygltf::FileExists(remap_texture_to_textures_subfolder(abs_filename, user_data), nullptr);
}

bool scene_read_whole_file(
  std::vector<unsigned char>* out,
  std::string* err,
  const std::string& filepath,
  void* user_data)
{
  return tinygltf::ReadWholeFile(
    out, err, remap_texture_to_textures_subfolder(filepath, user_data), nullptr);
}

bool scene_write_whole_file(
  std::string* err,
  const std::string& filepath,
  const std::vector<unsigned char>& contents,
  void*)
{
  return tinygltf::WriteWholeFile(err, filepath, contents, nullptr);
}

bool scene_get_file_size(
  size_t* filesize_out,
  std::string* err,
  const std::string& abs_filename,
  void* user_data)
{
  return tinygltf::GetFileSizeInBytes(
    filesize_out,
    err,
    remap_texture_to_textures_subfolder(abs_filename, user_data),
    nullptr);
}

} // namespace

std::optional<tinygltf::Model> SceneManager::loadModel(std::filesystem::path path)
{
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;

  SceneFsUserData fsUserData{.sceneDir = path.parent_path()};
  loader.SetFsCallbacks(
    tinygltf::FsCallbacks{
      .FileExists = scene_file_exists,
      .ExpandFilePath = tinygltf::ExpandFilePath,
      .ReadWholeFile = scene_read_whole_file,
      .WriteWholeFile = scene_write_whole_file,
      .GetFileSizeInBytes = scene_get_file_size,
      .user_data = &fsUserData,
    });

  std::string error;
  std::string warning;
  bool success = false;

  auto ext = path.extension();
  if (ext == ".gltf")
    success = loader.LoadASCIIFromFile(&model, &error, &warning, path.string());
  else if (ext == ".glb")
    success = loader.LoadBinaryFromFile(&model, &error, &warning, path.string());
  else
  {
    spdlog::error("glTF: Unknown glTF file extension: '{}'. Expected .gltf or .glb.", ext);
    return std::nullopt;
  }

  if (!success)
  {
    spdlog::error("glTF: Failed to load model!");
    if (!error.empty())
      spdlog::error("glTF: {}", error);
    return std::nullopt;
  }

  if (!warning.empty())
    spdlog::warn("glTF: {}", warning);

  if (
    !model.extensions.empty() || !model.extensionsRequired.empty() || !model.extensionsUsed.empty())
    spdlog::warn("glTF: No glTF extensions are currently implemented!");

  return model;
}

SceneManager::ProcessedInstances SceneManager::processInstances(const tinygltf::Model& model) const
{
  std::vector nodeTransforms(model.nodes.size(), glm::identity<glm::mat4x4>());

  for (std::size_t nodeIdx = 0; nodeIdx < model.nodes.size(); ++nodeIdx)
  {
    const auto& node = model.nodes[nodeIdx];
    auto& transform = nodeTransforms[nodeIdx];

    if (!node.matrix.empty())
    {
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          transform[i][j] = static_cast<float>(node.matrix[4 * i + j]);
    }
    else
    {
      if (!node.scale.empty())
        transform = scale(
          transform,
          glm::vec3(
            static_cast<float>(node.scale[0]),
            static_cast<float>(node.scale[1]),
            static_cast<float>(node.scale[2])));

      if (!node.rotation.empty())
        transform *= mat4_cast(glm::quat(
          static_cast<float>(node.rotation[3]),
          static_cast<float>(node.rotation[0]),
          static_cast<float>(node.rotation[1]),
          static_cast<float>(node.rotation[2])));

      if (!node.translation.empty())
        transform = translate(
          transform,
          glm::vec3(
            static_cast<float>(node.translation[0]),
            static_cast<float>(node.translation[1]),
            static_cast<float>(node.translation[2])));
    }
  }

  std::stack<std::size_t> vertices;
  for (auto vert : model.scenes[model.defaultScene].nodes)
    vertices.push(vert);

  while (!vertices.empty())
  {
    auto vert = vertices.top();
    vertices.pop();

    for (auto child : model.nodes[vert].children)
    {
      nodeTransforms[child] = nodeTransforms[vert] * nodeTransforms[child];
      vertices.push(child);
    }
  }

  ProcessedInstances result;

  // Don't overallocate matrices, they are pretty chonky.
  {
    std::size_t totalNodesWithMeshes = 0;
    for (std::size_t i = 0; i < model.nodes.size(); ++i)
      if (model.nodes[i].mesh >= 0)
        ++totalNodesWithMeshes;
    result.matrices.reserve(totalNodesWithMeshes);
    result.meshes.reserve(totalNodesWithMeshes);
  }

  for (std::size_t i = 0; i < model.nodes.size(); ++i)
    if (model.nodes[i].mesh >= 0)
    {
      result.matrices.push_back(nodeTransforms[i]);
      result.meshes.push_back(model.nodes[i].mesh);
    }

  return result;
}

static std::uint32_t encode_normal(glm::vec3 normal)
{
  const std::int32_t x = static_cast<std::int32_t>(normal.x * 32767.0f);
  const std::int32_t y = static_cast<std::int32_t>(normal.y * 32767.0f);

  const std::uint32_t sign = normal.z >= 0 ? 0 : 1;
  const std::uint32_t sx = static_cast<std::uint32_t>(x & 0xfffe) | sign;
  const std::uint32_t sy = static_cast<std::uint32_t>(y & 0xffff) << 16;

  return sx | sy;
}

static std::vector<std::uint8_t> image_to_rgba8(const tinygltf::Image& image)
{
  if (image.width <= 0 || image.height <= 0 || image.image.empty())
    return {};

  const std::size_t pixelCount = static_cast<std::size_t>(image.width) *
                                 static_cast<std::size_t>(image.height);

  if (image.bits != 8 && image.bits != 16)
    return {};

  const std::size_t bytesPerChannel = image.bits / 8;
  if (bytesPerChannel == 0)
    return {};

  const std::size_t srcChannels = static_cast<std::size_t>(image.component);
  if (srcChannels < 1 || srcChannels > 4)
    return {};

  const std::size_t srcStride = srcChannels * bytesPerChannel;
  if (image.image.size() < pixelCount * srcStride)
    return {};

  std::vector<std::uint8_t> rgba(pixelCount * 4u, 255u);

  auto read_channel_8bit = [&](const std::uint8_t* ptr) -> std::uint8_t {
    if (image.bits == 8)
      return ptr[0];

    std::uint16_t value = 0;
    std::memcpy(&value, ptr, sizeof(value));
    return static_cast<std::uint8_t>(value >> 8);
  };

  for (std::size_t i = 0; i < pixelCount; ++i)
  {
    const std::uint8_t* src = image.image.data() + i * srcStride;

    const std::uint8_t c0 = read_channel_8bit(src + 0 * bytesPerChannel);
    const std::uint8_t c1 = srcChannels >= 2 ? read_channel_8bit(src + 1 * bytesPerChannel) : c0;
    const std::uint8_t c2 = srcChannels >= 3 ? read_channel_8bit(src + 2 * bytesPerChannel) : c0;
    const std::uint8_t c3 = srcChannels >= 4 ? read_channel_8bit(src + 3 * bytesPerChannel)
                                             : static_cast<std::uint8_t>(255u);

    rgba[i * 4 + 0] = c0;
    rgba[i * 4 + 1] = srcChannels == 2 ? c0 : c1;
    rgba[i * 4 + 2] = srcChannels == 2 ? c0 : c2;
    rgba[i * 4 + 3] = c3;
  }

  return rgba;
}

std::vector<SceneManager::SceneTexture> SceneManager::processAlbedoTextures(
  const tinygltf::Model& model) const
{
  std::vector<SceneTexture> textures;
  textures.reserve(model.images.size() + 1u);

  textures.push_back(SceneTexture{
    .width = 1,
    .height = 1,
    .rgba8 = {255u, 255u, 255u, 255u},
  });

  for (std::size_t i = 0; i < model.images.size(); ++i)
  {
    const auto& img = model.images[i];
    auto rgba8 = image_to_rgba8(img);
    if (rgba8.empty())
    {
      spdlog::warn(
        "glTF: image #{} has unsupported format (bits={}, components={}), using white fallback",
        i,
        img.bits,
        img.component);
      textures.push_back(SceneTexture{
        .width = 1,
        .height = 1,
        .rgba8 = {255u, 255u, 255u, 255u},
      });
      continue;
    }

    textures.push_back(SceneTexture{
      .width = static_cast<std::uint32_t>(img.width),
      .height = static_cast<std::uint32_t>(img.height),
      .rgba8 = std::move(rgba8),
    });
  }

  return textures;
}

static std::uint32_t resolve_albedo_texture_idx(
  const tinygltf::Model& model, int primitiveMaterialIdx, std::uint32_t defaultTextureIdx)
{
  if (primitiveMaterialIdx < 0 || primitiveMaterialIdx >= static_cast<int>(model.materials.size()))
    return defaultTextureIdx;

  auto resolveImageIndexFromTexture = [&](int textureIdx) -> std::uint32_t {
    if (textureIdx < 0 || textureIdx >= static_cast<int>(model.textures.size()))
      return defaultTextureIdx;

    const auto& texture = model.textures[textureIdx];
    if (texture.source < 0 || texture.source >= static_cast<int>(model.images.size()))
      return defaultTextureIdx;

    return static_cast<std::uint32_t>(texture.source + 1);
  };

  const auto& material = model.materials[primitiveMaterialIdx];

  if (material.pbrMetallicRoughness.baseColorTexture.index >= 0)
    return resolveImageIndexFromTexture(material.pbrMetallicRoughness.baseColorTexture.index);

  auto extIt = material.extensions.find("KHR_materials_pbrSpecularGlossiness");
  if (extIt == material.extensions.end() || !extIt->second.IsObject())
    return defaultTextureIdx;

  const auto& extObj = extIt->second;
  if (!extObj.Has("diffuseTexture"))
    return defaultTextureIdx;

  const auto& diffuseTex = extObj.Get("diffuseTexture");
  if (!diffuseTex.IsObject() || !diffuseTex.Has("index"))
    return defaultTextureIdx;

  const auto& indexValue = diffuseTex.Get("index");
  if (!indexValue.IsInt())
    return defaultTextureIdx;

  return resolveImageIndexFromTexture(indexValue.Get<int>());
}

SceneManager::ProcessedMeshes SceneManager::processMeshes(const tinygltf::Model& model) const
{
  // NOTE: glTF assets can have pretty wonky data layouts which are not appropriate
  // for real-time rendering, so we have to press the data first. In serious engines
  // this is mitigated by storing assets on the disc in an engine-specific format that
  // is appropriate for GPU upload right after reading from disc.

  ProcessedMeshes result;

  // Pre-allocate enough memory so as not to hit the
  // allocator on the memcpy hotpath
  {
    std::size_t vertexBytes = 0;
    std::size_t indexBytes = 0;
    for (const auto& bufView : model.bufferViews)
    {
      switch (bufView.target)
      {
      case TINYGLTF_TARGET_ARRAY_BUFFER:
        vertexBytes += bufView.byteLength;
        break;
      case TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER:
        indexBytes += bufView.byteLength;
        break;
      default:
        break;
      }
    }
    result.vertices.reserve(vertexBytes / sizeof(Vertex));
    result.indices.reserve(indexBytes / sizeof(std::uint32_t));
  }

  {
    std::size_t totalPrimitives = 0;
    for (const auto& mesh : model.meshes)
      totalPrimitives += mesh.primitives.size();
    result.relems.reserve(totalPrimitives);
  }

  result.meshes.reserve(model.meshes.size());

  for (const auto& mesh : model.meshes)
  {
    result.meshes.push_back(Mesh{
      .firstRelem = static_cast<std::uint32_t>(result.relems.size()),
      .relemCount = static_cast<std::uint32_t>(mesh.primitives.size()),
    });

    for (const auto& prim : mesh.primitives)
    {
      if (prim.mode != TINYGLTF_MODE_TRIANGLES)
      {
        spdlog::warn(
          "Encountered a non-triangles primitive, these are not supported for now, skipping it!");
        --result.meshes.back().relemCount;
        continue;
      }

      const auto normalIt = prim.attributes.find("NORMAL");
      const auto tangentIt = prim.attributes.find("TANGENT");
      const auto texcoordIt = prim.attributes.find("TEXCOORD_0");

      const bool hasNormals = normalIt != prim.attributes.end();
      const bool hasTangents = tangentIt != prim.attributes.end();
      const bool hasTexcoord = texcoordIt != prim.attributes.end();
      std::array accessorIndices{
        prim.indices,
        prim.attributes.at("POSITION"),
        hasNormals ? normalIt->second : -1,
        hasTangents ? tangentIt->second : -1,
        hasTexcoord ? texcoordIt->second : -1,
      };

      std::array accessors{
        &model.accessors[prim.indices],
        &model.accessors[accessorIndices[1]],
        hasNormals ? &model.accessors[accessorIndices[2]] : nullptr,
        hasTangents ? &model.accessors[accessorIndices[3]] : nullptr,
        hasTexcoord ? &model.accessors[accessorIndices[4]] : nullptr,
      };

      std::array bufViews{
        &model.bufferViews[accessors[0]->bufferView],
        &model.bufferViews[accessors[1]->bufferView],
        hasNormals ? &model.bufferViews[accessors[2]->bufferView] : nullptr,
        hasTangents ? &model.bufferViews[accessors[3]->bufferView] : nullptr,
        hasTexcoord ? &model.bufferViews[accessors[4]->bufferView] : nullptr,
      };

      result.relems.push_back(RenderElement{
        .vertexOffset = static_cast<std::uint32_t>(result.vertices.size()),
        .indexOffset = static_cast<std::uint32_t>(result.indices.size()),
        .indexCount = static_cast<std::uint32_t>(accessors[0]->count),
        .albedoTextureIdx = resolve_albedo_texture_idx(model, prim.material, 0u),
      });

      const std::size_t vertexCount = accessors[1]->count;

      std::array ptrs{
        reinterpret_cast<const std::byte*>(model.buffers[bufViews[0]->buffer].data.data()) +
          bufViews[0]->byteOffset + accessors[0]->byteOffset,
        reinterpret_cast<const std::byte*>(model.buffers[bufViews[1]->buffer].data.data()) +
          bufViews[1]->byteOffset + accessors[1]->byteOffset,
        hasNormals
          ? reinterpret_cast<const std::byte*>(model.buffers[bufViews[2]->buffer].data.data()) +
            bufViews[2]->byteOffset + accessors[2]->byteOffset
          : nullptr,
        hasTangents
          ? reinterpret_cast<const std::byte*>(model.buffers[bufViews[3]->buffer].data.data()) +
            bufViews[3]->byteOffset + accessors[3]->byteOffset
          : nullptr,
        hasTexcoord
          ? reinterpret_cast<const std::byte*>(model.buffers[bufViews[4]->buffer].data.data()) +
            bufViews[4]->byteOffset + accessors[4]->byteOffset
          : nullptr,
      };

      std::array strides{
        bufViews[0]->byteStride != 0
          ? bufViews[0]->byteStride
          : tinygltf::GetComponentSizeInBytes(accessors[0]->componentType) *
            tinygltf::GetNumComponentsInType(accessors[0]->type),
        bufViews[1]->byteStride != 0
          ? bufViews[1]->byteStride
          : tinygltf::GetComponentSizeInBytes(accessors[1]->componentType) *
            tinygltf::GetNumComponentsInType(accessors[1]->type),
        hasNormals ? (bufViews[2]->byteStride != 0
                        ? bufViews[2]->byteStride
                        : tinygltf::GetComponentSizeInBytes(accessors[2]->componentType) *
                          tinygltf::GetNumComponentsInType(accessors[2]->type))
                   : 0,
        hasTangents ? (bufViews[3]->byteStride != 0
                         ? bufViews[3]->byteStride
                         : tinygltf::GetComponentSizeInBytes(accessors[3]->componentType) *
                           tinygltf::GetNumComponentsInType(accessors[3]->type))
                    : 0,
        hasTexcoord ? (bufViews[4]->byteStride != 0
                         ? bufViews[4]->byteStride
                         : tinygltf::GetComponentSizeInBytes(accessors[4]->componentType) *
                           tinygltf::GetNumComponentsInType(accessors[4]->type))
                    : 0,
      };

      for (std::size_t i = 0; i < vertexCount; ++i)
      {
        auto& vtx = result.vertices.emplace_back();
        glm::vec3 pos;
        // Fall back to 0 in case we don't have something.
        // NOTE: if tangents are not available, one could use http://mikktspace.com/
        // NOTE: if normals are not available, reconstructing them is possible but will look ugly
        glm::vec3 normal{0};
        glm::vec3 tangent{0};
        glm::vec2 texcoord{0};
        std::memcpy(&pos, ptrs[1], sizeof(pos));

        // NOTE: it's faster to do a template here with specializations for all combinations than to
        // do ifs at runtime. Also, SIMD should be used. Try implementing this!
        if (hasNormals)
          std::memcpy(&normal, ptrs[2], sizeof(normal));
        if (hasTangents)
          std::memcpy(&tangent, ptrs[3], sizeof(tangent));
        if (hasTexcoord)
          std::memcpy(&texcoord, ptrs[4], sizeof(texcoord));


        vtx.positionAndNormal = glm::vec4(pos, std::bit_cast<float>(encode_normal(normal)));
        vtx.texCoordAndTangentAndPadding =
          glm::vec4(texcoord, std::bit_cast<float>(encode_normal(tangent)), 0);

        ptrs[1] += strides[1];
        if (hasNormals)
          ptrs[2] += strides[2];
        if (hasTangents)
          ptrs[3] += strides[3];
        if (hasTexcoord)
          ptrs[4] += strides[4];
      }

      // Indices are guaranteed to have no stride
      ETNA_VERIFY(bufViews[0]->byteStride == 0);
      const std::size_t indexCount = accessors[0]->count;
      if (accessors[0]->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
      {
        for (std::size_t i = 0; i < indexCount; ++i)
        {
          std::uint16_t index;
          std::memcpy(&index, ptrs[0], sizeof(index));
          result.indices.push_back(index);
          ptrs[0] += 2;
        }
      }
      else if (accessors[0]->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
      {
        const std::size_t lastTotalIndices = result.indices.size();
        result.indices.resize(lastTotalIndices + indexCount);
        std::memcpy(
          result.indices.data() + lastTotalIndices,
          ptrs[0],
          sizeof(result.indices[0]) * indexCount);
      }
      // Compute AABB for this render element using the vertices we just appended
      {
        const std::uint32_t vOff = result.relems.back().vertexOffset;
        const std::size_t vCount = vertexCount;
        glm::vec3 mn{std::numeric_limits<float>::infinity()};
        glm::vec3 mx{-std::numeric_limits<float>::infinity()};
        for (std::size_t vi = 0; vi < vCount; ++vi)
        {
          const auto& v = result.vertices[vOff + vi];
          const glm::vec3 pos = glm::vec3(v.positionAndNormal);
          mn = glm::min(mn, pos);
          mx = glm::max(mx, pos);
        }
        result.aabbs.push_back(SceneManager::AABB{.min = mn, .max = mx});
      }
    }
  }

  return result;
}

void SceneManager::uploadData(
  std::span<const Vertex> vertices, std::span<const std::uint32_t> indices)
{
  unifiedVbuf = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = vertices.size_bytes(),
    .bufferUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "unifiedVbuf",
  });

  unifiedIbuf = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = indices.size_bytes(),
    .bufferUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "unifiedIbuf",
  });

  transferHelper.uploadBuffer<Vertex>(*oneShotCommands, unifiedVbuf, 0, vertices);
  transferHelper.uploadBuffer<std::uint32_t>(*oneShotCommands, unifiedIbuf, 0, indices);
}

SceneManager::ProcessedMeshes SceneManager::processBakedMeshes(const tinygltf::Model& model) const
{
  ProcessedMeshes result;
  std::vector<std::size_t> relemVertexCounts;

  std::size_t totalPrimitives = 0;
  for (const auto& mesh : model.meshes)
    totalPrimitives += mesh.primitives.size();
  result.relems.reserve(totalPrimitives);


  result.meshes.reserve(model.meshes.size());

  for (const auto& mesh : model.meshes)
  {
    result.meshes.push_back(Mesh{
      .firstRelem = static_cast<std::uint32_t>(result.relems.size()),
      .relemCount = static_cast<std::uint32_t>(mesh.primitives.size()),
    });

    for (const auto& prim : mesh.primitives)
    {
      if (prim.mode != TINYGLTF_MODE_TRIANGLES)
      {
        spdlog::warn("Not triangles primitive!");
        --result.meshes.back().relemCount;
        continue;
      }

      const tinygltf::Accessor& indAccessor = model.accessors[prim.indices];
      const tinygltf::Accessor& posAccessor = model.accessors[prim.attributes.at("POSITION")];

      result.relems.push_back(RenderElement{
        .vertexOffset = static_cast<std::uint32_t>(posAccessor.byteOffset / sizeof(Vertex)),
        .indexOffset = static_cast<std::uint32_t>(indAccessor.byteOffset / sizeof(std::uint32_t)),
        .indexCount = static_cast<std::uint32_t>(indAccessor.count),
        .albedoTextureIdx = resolve_albedo_texture_idx(model, prim.material, 0u),
      });
      relemVertexCounts.push_back(posAccessor.count);
    }
  }
  size_t vertex_count = model.bufferViews[0].byteLength / sizeof(Vertex);
  result.vertices.resize(vertex_count);
  memcpy(result.vertices.data(), model.buffers[0].data.data(), model.bufferViews[0].byteLength);

  size_t index_count = model.bufferViews[1].byteLength / sizeof(std::uint32_t);
  result.indices.resize(index_count);
  memcpy(
    result.indices.data(),
    model.buffers[0].data.data() + model.bufferViews[0].byteLength,
    model.bufferViews[1].byteLength);

  result.aabbs.reserve(result.relems.size());
  for (std::size_t i = 0; i < result.relems.size(); ++i)
  {
    const std::uint32_t vOff = result.relems[i].vertexOffset;
    const std::size_t vCount = relemVertexCounts[i];
    glm::vec3 mn{std::numeric_limits<float>::infinity()};
    glm::vec3 mx{-std::numeric_limits<float>::infinity()};
    for (std::size_t vi = 0; vi < vCount; ++vi)
    {
      const auto& v = result.vertices[vOff + vi];
      const glm::vec3 pos = glm::vec3(v.positionAndNormal);
      mn = glm::min(mn, pos);
      mx = glm::max(mx, pos);
    }
    result.aabbs.push_back(SceneManager::AABB{.min = mn, .max = mx});
  }

  return result;
}

void SceneManager::selectScene(std::filesystem::path path, bool baked)
{
  auto maybeModel = loadModel(path);
  if (!maybeModel.has_value())
    return;

  auto model = std::move(*maybeModel);

  albedoTextures = processAlbedoTextures(model);

  // By aggregating all SceneManager fields mutations here,
  // we guarantee that we don't forget to clear something
  // when re-loading a scene.

  // NOTE: you might want to store these on the GPU for GPU-driven rendering.
  auto [instMats, instMeshes] = processInstances(model);
  instanceMatrices = std::move(instMats);
  instanceMeshes = std::move(instMeshes);

  if (baked)
  {
    auto [verts, inds, relems, meshs, aabbs] = processBakedMeshes(model);
    renderElements = std::move(relems);
    meshes = std::move(meshs);
    renderElementAABBs = std::move(aabbs);
    uploadData(verts, inds);
  }
  else
  {
    auto [verts, inds, relems, meshs, aabbs] = processMeshes(model);
    renderElements = std::move(relems);
    meshes = std::move(meshs);
    renderElementAABBs = std::move(aabbs);

    uploadData(verts, inds);
  }
}

etna::VertexByteStreamFormatDescription SceneManager::getVertexFormatDescription()
{
  return etna::VertexByteStreamFormatDescription{
    .stride = sizeof(Vertex),
    .attributes = {
      etna::VertexByteStreamFormatDescription::Attribute{
        .format = vk::Format::eR32G32B32A32Sfloat,
        .offset = 0,
      },
      etna::VertexByteStreamFormatDescription::Attribute{
        .format = vk::Format::eR32G32B32A32Sfloat,
        .offset = sizeof(glm::vec4),
      },
    }};
}
