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

// Resolves a glTF texture index (into model.textures) to our TextureId, which
// indexes our flat textures[] array with the BuiltInCount fallback prefix.
// Returns TextureId::Invalid if the glTF index is missing or malformed.
static TextureId resolve_texture_id(const tinygltf::Model& model, int texture_idx)
{
  if (texture_idx < 0 || texture_idx >= static_cast<int>(model.textures.size()))
    return TextureId::Invalid;
  const auto& tex = model.textures[texture_idx];
  if (tex.source < 0 || tex.source >= static_cast<int>(model.images.size()))
    return TextureId::Invalid;
  return static_cast<TextureId>(
    static_cast<std::uint32_t>(tex.source) +
    static_cast<std::uint32_t>(TextureId::BuiltInCount));
}

// Maps a glTF primitive's `material` field to our MaterialId.
// Materials array layout: [0] = default material, [1..] = glTF materials.
// Primitives without an explicit material get MaterialId{0}.
static MaterialId resolve_material_id(int primitive_material_idx)
{
  if (primitive_material_idx < 0)
    return MaterialId{0};
  return static_cast<MaterialId>(static_cast<std::uint32_t>(primitive_material_idx) + 1u);
}

std::vector<Material> SceneManager::processMaterials(
  const tinygltf::Model& model, std::vector<bool>& out_is_srgb_image) const
{
  // Every image starts out assumed to be linear; we flip to sRGB below for
  // images sampled as baseColor (emissive would go here too, if we used it).
  out_is_srgb_image.assign(model.images.size(), false);

  auto markSrgb = [&](int texture_idx) {
    if (texture_idx < 0 || texture_idx >= static_cast<int>(model.textures.size()))
      return;
    const int imgIdx = model.textures[texture_idx].source;
    if (imgIdx >= 0 && imgIdx < static_cast<int>(out_is_srgb_image.size()))
      out_is_srgb_image[imgIdx] = true;
  };

  std::vector<Material> result;
  result.reserve(model.materials.size() + 1u);

  // [0] — the default material used by primitives without a material reference.
  // All fields already default to sensible values in the struct definition.
  result.push_back(Material{});

  for (const auto& gm : model.materials)
  {
    Material mat{};

    const auto& pbr = gm.pbrMetallicRoughness;

    // Base color factor (default {1,1,1,1} per glTF spec).
    if (pbr.baseColorFactor.size() == 4)
    {
      mat.baseColorFactor = glm::vec4(
        static_cast<float>(pbr.baseColorFactor[0]),
        static_cast<float>(pbr.baseColorFactor[1]),
        static_cast<float>(pbr.baseColorFactor[2]),
        static_cast<float>(pbr.baseColorFactor[3]));
    }

    auto extIt = gm.extensions.find("KHR_materials_pbrSpecularGlossiness");
    const bool hasSpecGloss = extIt != gm.extensions.end() && extIt->second.IsObject();

    if (hasSpecGloss)
    {
      const auto& ext = extIt->second;
      if (ext.Has("diffuseFactor"))
      {
        const auto& df = ext.Get("diffuseFactor");
        if (df.IsArray() && df.ArrayLen() == 4)
        {
          mat.baseColorFactor = glm::vec4(
            static_cast<float>(df.Get(0).GetNumberAsDouble()),
            static_cast<float>(df.Get(1).GetNumberAsDouble()),
            static_cast<float>(df.Get(2).GetNumberAsDouble()),
            static_cast<float>(df.Get(3).GetNumberAsDouble()));
        }
      }
    }

    const TextureId baseColor = resolve_texture_id(model, pbr.baseColorTexture.index);
    if (baseColor != TextureId::Invalid)
    {
      mat.baseColorTex = baseColor;
      markSrgb(pbr.baseColorTexture.index);
    }
    else if (hasSpecGloss)
    {
      const auto& ext = extIt->second;
      if (ext.Has("diffuseTexture"))
      {
        const auto& diffTex = ext.Get("diffuseTexture");
        if (diffTex.IsObject() && diffTex.Has("index"))
        {
          const auto& idxVal = diffTex.Get("index");
          if (idxVal.IsInt())
          {
            const TextureId diffuse = resolve_texture_id(model, idxVal.Get<int>());
            if (diffuse != TextureId::Invalid)
            {
              mat.baseColorTex = diffuse;
              markSrgb(idxVal.Get<int>());
            }
          }
        }
      }
    }

    mat.metallicFactor = static_cast<float>(pbr.metallicFactor);
    mat.roughnessFactor = static_cast<float>(pbr.roughnessFactor);

    const TextureId metalRough =
      resolve_texture_id(model, pbr.metallicRoughnessTexture.index);
    if (metalRough != TextureId::Invalid)
      mat.metallicRoughnessTex = metalRough;
    // Metallic-roughness stays linear — no markSrgb call.

    const TextureId normal = resolve_texture_id(model, gm.normalTexture.index);
    if (normal != TextureId::Invalid)
    {
      mat.normalTex = normal;
      mat.normalScale = static_cast<float>(gm.normalTexture.scale);
    }

    const TextureId occlusion = resolve_texture_id(model, gm.occlusionTexture.index);
    if (occlusion != TextureId::Invalid)
    {
      mat.occlusionTex = occlusion;
      mat.occlusionStrength = static_cast<float>(gm.occlusionTexture.strength);
    }

    mat.doubleSided = gm.doubleSided;

    result.push_back(mat);
  }

  return result;
}

std::vector<SceneManager::SceneTexture> SceneManager::processTextures(
  const tinygltf::Model& model, const std::vector<bool>& is_srgb_image) const
{
  std::vector<SceneTexture> result;
  result.reserve(static_cast<std::size_t>(TextureId::BuiltInCount) + model.images.size());

  // [0] DefaultBaseColor — white sRGB (baseColorFactor multiplies this).
  result.push_back(SceneTexture{
    .width = 1, .height = 1, .rgba8 = {255u, 255u, 255u, 255u}, .isSrgb = true});
  // [1] DefaultMetallicRoughness — linear. Roughness in G, metallic in B.
  //     (0, 255, 0, 255) ⇒ roughness=1, metallic=0 (fully rough dielectric).
  result.push_back(SceneTexture{
    .width = 1, .height = 1, .rgba8 = {0u, 255u, 0u, 255u}, .isSrgb = false});
  // [2] DefaultNormal — neutral tangent-space normal (0.5, 0.5, 1.0) linear.
  result.push_back(SceneTexture{
    .width = 1, .height = 1, .rgba8 = {128u, 128u, 255u, 255u}, .isSrgb = false});
  // [3] DefaultOcclusion — white linear (occlusion=1, i.e. no occlusion).
  result.push_back(SceneTexture{
    .width = 1, .height = 1, .rgba8 = {255u, 255u, 255u, 255u}, .isSrgb = false});

  for (std::size_t i = 0; i < model.images.size(); ++i)
  {
    const auto& img = model.images[i];
    auto rgba8 = image_to_rgba8(img);
    const bool isSrgb = (i < is_srgb_image.size()) ? is_srgb_image[i] : false;

    if (rgba8.empty())
    {
      spdlog::warn(
        "glTF: image #{} has unsupported format (bits={}, components={}), using white fallback",
        i,
        img.bits,
        img.component);
      result.push_back(SceneTexture{
        .width = 1,
        .height = 1,
        .rgba8 = {255u, 255u, 255u, 255u},
        .isSrgb = isSrgb,
      });
      continue;
    }

    result.push_back(SceneTexture{
      .width = static_cast<std::uint32_t>(img.width),
      .height = static_cast<std::uint32_t>(img.height),
      .rgba8 = std::move(rgba8),
      .isSrgb = isSrgb,
    });
  }

  return result;
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
        .vertexOffset = static_cast<std::int32_t>(result.vertices.size()),
        .indexOffset = static_cast<std::uint32_t>(result.indices.size()),
        .indexCount = static_cast<std::uint32_t>(accessors[0]->count),
        .materialId = resolve_material_id(prim.material),
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
        glm::vec4 tangent{0, 0, 0, 0}; // w==0 means "no tangent", shader will fall back to geometric normal
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
        vtx.texCoordAndTangentAndPadding = glm::vec4(
          texcoord, std::bit_cast<float>(encode_normal(glm::vec3(tangent))), tangent.w);

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
        .vertexOffset = static_cast<std::int32_t>(posAccessor.byteOffset / sizeof(Vertex)),
        .indexOffset = static_cast<std::uint32_t>(indAccessor.byteOffset / sizeof(std::uint32_t)),
        .indexCount = static_cast<std::uint32_t>(indAccessor.count),
        .materialId = resolve_material_id(prim.material),
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

  // Materials must be processed before textures so we know which images
  // are used as baseColor (and thus need sRGB encoding).
  std::vector<bool> isSrgbImage;
  materials = processMaterials(model, isSrgbImage);
  textures = processTextures(model, isSrgbImage);

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
