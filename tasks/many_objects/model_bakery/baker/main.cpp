#include <cmath>
#include <array>
#include <iostream>
#include <filesystem>
#include <optional>
#include <vector>
#include <glm/glm.hpp>
#include <tiny_gltf.h>
#include <cstdlib>
#include <spdlog/spdlog.h>
#include "scene/SceneManager.hpp"
#include "Quant.hpp"

class ModelBakery {
public:
  void bake(std::filesystem::path path) {
    auto maybeModel = loadModel(path);
    if (!maybeModel.has_value())
      return;
    auto model = *maybeModel;
    auto [verts, inds, relems, meshes] = processMeshes(model);

    model.buffers.clear();

    tinygltf::Buffer buffer;
    auto name = path.parent_path() / path.stem();
    buffer.name = name.stem().string();
    buffer.uri = name.filename().string() + "_baked.bin";

    std::size_t indsSize = inds.size() * sizeof(uint32_t);
    std::size_t vertsSize = verts.size() * sizeof(Vertex);
    buffer.data.resize(indsSize + vertsSize);
    std::memcpy(buffer.data.data(), inds.data(), indsSize);
    std::memcpy(buffer.data.data() + indsSize, verts.data(), vertsSize);
    model.buffers.push_back(buffer);

    model.bufferViews.clear();

    auto& bufferViewInds = model.bufferViews.emplace_back();
    bufferViewInds.buffer = 0;
    bufferViewInds.byteLength = indsSize;
    bufferViewInds.byteOffset = 0;
    bufferViewInds.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;

    auto& bufferViewVerts = model.bufferViews.emplace_back();
    bufferViewVerts.buffer = 0;
    bufferViewVerts.byteLength = vertsSize;
    bufferViewVerts.byteOffset = indsSize;
    bufferViewVerts.byteStride = sizeof(Vertex);
    bufferViewVerts.target = TINYGLTF_TARGET_ARRAY_BUFFER;

    bakeGLTF(model, relems, meshes);

    auto output = name.string() + "_baked.gltf";
    loader.WriteGltfSceneToFile(&model, output, false, false, true, false);
  };
private:
  struct RenderElementExtention {
    std::uint32_t vertexOffset;
    std::uint32_t vertexCount;
    std::uint32_t indexOffset;
    std::uint32_t indexCount;
    std::array<std::vector<double>, 2> positionBound;
    std::optional<std::array<std::vector<double>, 2>> texcoordBound;
  };

  void bakeGLTF(tinygltf::Model& model, const std::vector<RenderElementExtention>& relems, const std::vector<Mesh>& meshes) {
    tinygltf::Accessor indices;
    indices.bufferView = 0;
    indices.byteOffset = 0;
    indices.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
    indices.type = TINYGLTF_TYPE_SCALAR;

    tinygltf::Accessor position;
    position.bufferView = 1;
    position.byteOffset = 0;
    position.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    position.type = TINYGLTF_TYPE_VEC3;

    tinygltf::Accessor normal;
    normal.bufferView = 1;
    normal.byteOffset = 12;
    normal.normalized = true;
    normal.componentType = TINYGLTF_COMPONENT_TYPE_BYTE;
    normal.type = TINYGLTF_TYPE_VEC3;

    tinygltf::Accessor tangent;
    tangent.bufferView = 1;
    tangent.byteOffset = 24;
    tangent.normalized = true;
    tangent.componentType = TINYGLTF_COMPONENT_TYPE_BYTE;
    tangent.type = TINYGLTF_TYPE_VEC4;

    tinygltf::Accessor texcoord_0;
    texcoord_0.bufferView = 1;
    texcoord_0.byteOffset = 16;
    texcoord_0.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    texcoord_0.type = TINYGLTF_TYPE_VEC2;

    std::map<std::string, tinygltf::Accessor> accessors = {
      {"POSITION", position},
      {"NORMAL", normal},
      {"TANGENT", tangent},
      {"TEXCOORD_0", texcoord_0},
    };

    model.accessors.clear();

    for (size_t i = 0; i < model.meshes.size(); i++) {
      auto& mesh = model.meshes[i];
      for (size_t j = 0; j < mesh.primitives.size(); j++) {
        auto& prim = mesh.primitives[j];
        if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
          continue;
        }

        std::erase_if(prim.attributes, [&](const auto& attribute) {
          for (const auto& accessor : accessors) {
            if (attribute.first == accessor.first) {
              return false;
            }
          }
          return true;
        });

        auto& relem = relems[meshes[i].firstRelem + j];

        prim.indices = static_cast<int>(model.accessors.size());
        auto& new_ind_accessor = model.accessors.emplace_back(indices);
        new_ind_accessor.byteOffset += relem.indexOffset * sizeof(uint32_t);
        new_ind_accessor.count = relem.indexCount;

        accessors["POSITION"].minValues = relem.positionBound[0];
        accessors["POSITION"].maxValues = relem.positionBound[1];
        if (relem.texcoordBound.has_value()) {
          accessors["TEXCOORD_0"].minValues = relem.texcoordBound->at(0);
          accessors["TEXCOORD_0"].maxValues = relem.texcoordBound->at(1);
        }

        for (const auto& accessor : accessors) {
          if (!prim.attributes.contains(accessor.first)) {
            continue;
          }
          prim.attributes[accessor.first] = static_cast<int>(model.accessors.size());
          auto& new_accessor = model.accessors.emplace_back(accessor.second);
          new_accessor.byteOffset += relem.vertexOffset * sizeof(Vertex);
          new_accessor.count = relem.vertexCount;
        }
      }
    }
  };

  tinygltf::TinyGLTF loader;
  struct Vertex
  {
    glm::vec4 positionAndNormal;
    glm::vec4 texCoordAndTangentAndPadding;
  };

  struct ProcessedMeshes
  {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<RenderElementExtention> relems;
    std::vector<Mesh> meshes;
  };

  std::optional<tinygltf::Model> loadModel(std::filesystem::path path)
  {
    tinygltf::Model model;

    std::string error;
    std::string warning;
    bool success = false;

    auto ext = path.extension();
    if (ext == ".gltf")
      success = loader.LoadASCIIFromFile(&model, &error, &warning, path.string());
    else if (ext == ".glb")
      success = loader.LoadBinaryFromFile(&model, &error, &warning, path.string());
    else
      return std::nullopt;

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

    model.extensionsUsed.push_back("KHR_mesh_quantization");
    model.extensionsRequired.push_back("KHR_mesh_quantization");

    return model;
  };

  ProcessedMeshes processMeshes(const tinygltf::Model& model) const
  {

    ProcessedMeshes result;

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
            "NOT TRIANGLE PRIMITIVES!");
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

        result.relems.push_back(RenderElementExtention{
          .vertexOffset = static_cast<std::uint32_t>(result.vertices.size()),
          .vertexCount = static_cast<std::uint32_t>(accessors[1]->count),
          .indexOffset = static_cast<std::uint32_t>(result.indices.size()),
          .indexCount = static_cast<std::uint32_t>(accessors[0]->count),
          .positionBound = std::array{accessors[1]->minValues, accessors[1]->maxValues},
          .texcoordBound = hasTexcoord ? std::optional(std::array{accessors[4]->minValues, accessors[4]->maxValues}) : std::nullopt
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
          glm::vec3 normal{0};
          glm::vec4 tangent{0};
          glm::vec2 texcoord{0};
          std::memcpy(&pos, ptrs[1], sizeof(pos));

          if (hasNormals)
            std::memcpy(&normal, ptrs[2], sizeof(normal));
          if (hasTangents)
            std::memcpy(&tangent, ptrs[3], sizeof(tangent));
          if (hasTexcoord)
            std::memcpy(&texcoord, ptrs[4], sizeof(texcoord));


          vtx.positionAndNormal = glm::vec4(pos, std::bit_cast<float>(quantize4fnorm(glm::vec4(normal, 0))));
          vtx.texCoordAndTangentAndPadding =
            glm::vec4(texcoord, std::bit_cast<float>(quantize4fnorm(tangent)), 0);

          ptrs[1] += strides[1];
          if (hasNormals)
            ptrs[2] += strides[2];
          if (hasTangents)
            ptrs[3] += strides[3];
          if (hasTexcoord)
            ptrs[4] += strides[4];
        }

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
      }
    }

    return result;
  }
};

int main(int argc, char* argv[]) {
  auto path = argc == 2 ? std::filesystem::path(argv[1])
                        : std::filesystem::path(GRAPHICS_COURSE_RESOURCES_ROOT "/scenes/Avocado/Avocado.gltf");
  if (!std::filesystem::exists(path) || path.extension() != ".gltf") {
    spdlog::error("No .gltf file found");
    return EXIT_FAILURE;
  }

  ModelBakery b;
  b.bake(path);

  return EXIT_SUCCESS;
}