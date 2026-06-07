#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <tiny_gltf.h>
#include <etna/Buffer.hpp>
#include <etna/BlockingTransferHelper.hpp>
#include <etna/VertexInput.hpp>

enum class TextureId : std::uint32_t
{
  // Built-in 1x1 fallback textures
  DefaultBaseColor = 0,
  DefaultMetallicRoughness = 1,
  DefaultNormal = 2,
  DefaultOcclusion = 3,
  DefaultEmissive = 4,
  BuiltInCount = 5,

  Invalid = ~std::uint32_t{0},
};

enum class MaterialId : std::uint32_t
{
  Invalid = ~std::uint32_t{0},
};

// A glTF PBR metallic-roughness material
struct Material
{
  TextureId baseColorTex = TextureId::DefaultBaseColor;
  TextureId metallicRoughnessTex = TextureId::DefaultMetallicRoughness;
  TextureId normalTex = TextureId::DefaultNormal;
  TextureId occlusionTex = TextureId::DefaultOcclusion;
  TextureId emissiveTex = TextureId::DefaultEmissive;

  glm::vec4 baseColorFactor = glm::vec4(1.0f);
  glm::vec3 emissiveFactor = glm::vec3(0.0f);
  float metallicFactor = 1.0f;
  float roughnessFactor = 1.0f;
  float normalScale = 1.0f;
  float occlusionStrength = 1.0f;

  bool doubleSided = false;
};

// A single render element (relem) corresponds to a single draw call
// of a certain pipeline with specific bindings (including material data)
struct RenderElement
{
  std::int32_t vertexOffset;
  std::uint32_t indexOffset;
  std::uint32_t indexCount;
  MaterialId materialId;
};

// A mesh is a collection of relems. A scene may have the same mesh
// located in several different places, so a scene consists of **instances**,
// not meshes.
struct Mesh
{
  std::uint32_t firstRelem;
  std::uint32_t relemCount;
};

class SceneManager
{
public:
  struct SceneTexture
  {
    std::uint32_t width;
    std::uint32_t height;
    std::vector<std::uint8_t> rgba8;
    // true  => data is sRGB-encoded (baseColor, emissive) - R8G8B8A8_SRGB
    // false => data is linear (metallicRoughness, normal, occlusion) -R8G8B8A8_UNORM
    bool isSrgb = false;
  };

  SceneManager();

  void selectScene(std::filesystem::path path, bool baked);

  // Every instance is a mesh drawn with a certain transform
  std::span<const glm::mat4x4> getInstanceMatrices() { return instanceMatrices; }
  std::span<const std::uint32_t> getInstanceMeshes() { return instanceMeshes; }

  // Every mesh is a collection of relems
  std::span<const Mesh> getMeshes() { return meshes; }

  // Every relem is a single draw call
  std::span<const RenderElement> getRenderElements() { return renderElements; }

  // Every material is a set of factors and texture ids, referenced by relems.
  std::span<const Material> getMaterials() { return materials; }

  // All textures used by scene materials. The first TextureId::BuiltInCount
  // entries are 1x1 fallbacks, the rest come from the glTF file
  std::span<const SceneTexture> getTextures() { return textures; }

  struct AABB
  {
    glm::vec3 min;
    glm::vec3 max;
  };

  // Axis-aligned bounding box for every RenderElement in model/local space
  std::span<const AABB> getRenderElementAABBs() { return renderElementAABBs; }

  vk::Buffer getVertexBuffer() { return unifiedVbuf.get(); }
  vk::Buffer getIndexBuffer() { return unifiedIbuf.get(); }

  etna::VertexByteStreamFormatDescription getVertexFormatDescription();

private:
  std::optional<tinygltf::Model> loadModel(std::filesystem::path path);

  std::vector<SceneTexture> processTextures(
    const tinygltf::Model& model, const std::vector<bool>& is_srgb_image) const;

  std::vector<Material> processMaterials(
    const tinygltf::Model& model, std::vector<bool>& out_is_srgb_image) const;

  struct ProcessedInstances
  {
    std::vector<glm::mat4x4> matrices;
    std::vector<std::uint32_t> meshes;
  };

  ProcessedInstances processInstances(const tinygltf::Model& model) const;

  struct Vertex
  {
    glm::vec4 positionAndNormal;
    glm::vec4 texCoordAndTangentAndPadding;
  };

  static_assert(sizeof(Vertex) == sizeof(float) * 8);

  struct ProcessedMeshes
  {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<RenderElement> relems;
    std::vector<Mesh> meshes;
    std::vector<AABB> aabbs;
  };

  ProcessedMeshes processMeshes(const tinygltf::Model& model) const;
  ProcessedMeshes processBakedMeshes(const tinygltf::Model& model) const;
  void uploadData(std::span<const Vertex> vertices, std::span<const std::uint32_t>);

private:
  std::unique_ptr<etna::OneShotCmdMgr> oneShotCommands;
  etna::BlockingTransferHelper transferHelper;

  std::vector<RenderElement> renderElements;
  std::vector<Mesh> meshes;
  std::vector<AABB> renderElementAABBs;
  std::vector<glm::mat4x4> instanceMatrices;
  std::vector<std::uint32_t> instanceMeshes;
  std::vector<Material> materials;
  std::vector<SceneTexture> textures;

  etna::Buffer unifiedVbuf;
  etna::Buffer unifiedIbuf;
};
