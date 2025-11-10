#define GLM_ENABLE_EXPERIMENTAL

#include <tiny_gltf.h>
#include <glm/glm.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <glm/geometric.hpp>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <stdexcept>
#include "Quant.hpp"

inline uint32_t best_fit_normal(glm::vec3& normal)
{
  normal = glm::normalize(normal);

  constexpr float LENGTH_STEP = 0.1f;
  constexpr float LENGTH_BASE = 0.1f;
  constexpr size_t STEPS_COUNT = 64;

  std::array<float, STEPS_COUNT> errors{};
  std::fill(errors.begin(), errors.end(), 0.f);

  auto quantizeScaled = [&](size_t step_id) -> uint32_t {
    float coeff = LENGTH_BASE + step_id * LENGTH_STEP;
    glm::vec3 scaled = normal * coeff;
    return quantize4fnorm(glm::vec4(scaled, 0.f));
  };

  // Angle error calculation
  std::for_each(errors.begin(), errors.end(), [&](float& out) {
    size_t id = &out - errors.data();
    glm::vec3 dequantized = dequantize3fnorm(quantizeScaled(id));
    dequantized = glm::normalize(dequantized);
    out = glm::angle(normal, dequantized);
  });

  // Step with minimal error
  auto it = std::min_element(errors.begin(), errors.end());
  size_t bestStep = std::distance(errors.begin(), it);


  glm::vec3 finalDeq = dequantize3fnorm(quantizeScaled(bestStep));
  finalDeq = glm::normalize(finalDeq);

  return quantize4fnorm(glm::vec4(finalDeq, 0.f));
}

inline void check(bool condition, const std::string& message)
{
  if (!condition)
  {
    spdlog::error(message);
    throw std::runtime_error(message);
  }
}


struct Vertex
{
  glm::vec3 pos{};
  int32_t norm = 0;
  glm::vec2 texcoord{};
  int32_t tang = 0;
  int32_t pad_ = 0;
};


static_assert(sizeof(Vertex) == 32);
static_assert(sizeof(uint32_t) == 4);


int main(int argc, char** argv)
{
  try
  {
    check(argc >= 2, "Invalid number of args: specify a gltf asset to bake");
    std::filesystem::path path{argv[1]};

    bool bestFitNormal = false;

    for (int i = 2; i < argc; ++i)
    {
      if (std::strcmp(argv[i], "-bfn") == 0)
        bestFitNormal = true;
      else
        throw std::invalid_argument("Unknown argument: " + std::string(argv[i]));
    }


    tinygltf::TinyGLTF api;
    tinygltf::Model model;
    std::string error, warning;
    bool success = false;

    if (path.extension() == ".gltf")
      success = api.LoadASCIIFromFile(&model, &error, &warning, path.string());
    else if (path.extension() == ".glb")
      success = api.LoadBinaryFromFile(&model, &error, &warning, path.string());
    else
      throw std::runtime_error("glTF: Unknown glTF file extension, expected .gltf or .glb.");

    if (!success)
    {
      if (!error.empty())
        spdlog::error("glTF: {}", error);
      throw std::runtime_error("Error! Can't load glTF model");
    }

    if (!warning.empty())
      spdlog::warn("glTF: {}", warning);


    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    size_t vertexBytes = 0, indexBytes = 0;
    for (const auto& view : model.bufferViews)
    {
      switch (view.target)
      {
      case TINYGLTF_TARGET_ARRAY_BUFFER:
        vertexBytes += view.byteLength;
        break;
      case TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER:
        indexBytes += view.byteLength;
        break;
      default:
        break;
      }
    }

    vertices.reserve(vertexBytes / sizeof(Vertex));
    indices.reserve(indexBytes / sizeof(uint32_t));

    auto byteSize = []<typename T>(const std::vector<T>& v) { return sizeof(T) * v.size(); };

    std::vector<tinygltf::Accessor> combinedAccessors;

    for (auto& mesh : model.meshes)
    {
      for (auto& prim : mesh.primitives)
      {
        if (prim.mode != TINYGLTF_MODE_TRIANGLES)
        {
          spdlog::warn("Not triangle primitives!");
          continue;
        }

        const auto normIt = prim.attributes.find("NORMAL");
        const auto tangIt = prim.attributes.find("TANGENT");
        const auto texIt = prim.attributes.find("TEXCOORD_0");

        bool hasNormals = normIt != prim.attributes.end();
        bool hasTangents = tangIt != prim.attributes.end();
        bool hasTex = texIt != prim.attributes.end();

        std::array<int, 5> accIdx{
          prim.indices,
          prim.attributes.at("POSITION"),
          hasNormals ? normIt->second : -1,
          hasTangents ? tangIt->second : -1,
          hasTex ? texIt->second : -1};

        std::array<const tinygltf::Accessor*, 5> acc{
          &model.accessors[accIdx[0]],
          &model.accessors[accIdx[1]],
          hasNormals ? &model.accessors[accIdx[2]] : nullptr,
          hasTangents ? &model.accessors[accIdx[3]] : nullptr,
          hasTex ? &model.accessors[accIdx[4]] : nullptr};

        std::array<const tinygltf::BufferView*, 5> views{
          &model.bufferViews[acc[0]->bufferView],
          &model.bufferViews[acc[1]->bufferView],
          hasNormals ? &model.bufferViews[acc[2]->bufferView] : nullptr,
          hasTangents ? &model.bufferViews[acc[3]->bufferView] : nullptr,
          hasTex ? &model.bufferViews[acc[4]->bufferView] : nullptr};

        const size_t vertexCount = acc[1]->count;
        const size_t vertsOffset = byteSize(vertices);

        auto bufferData = [&](const tinygltf::BufferView* v, const tinygltf::Accessor* a) {
          return reinterpret_cast<const uint8_t*>(model.buffers[v->buffer].data.data()) +
            v->byteOffset + a->byteOffset;
        };

        std::array<const uint8_t*, 5> ptrs{
          bufferData(views[0], acc[0]),
          bufferData(views[1], acc[1]),
          hasNormals ? bufferData(views[2], acc[2]) : nullptr,
          hasTangents ? bufferData(views[3], acc[3]) : nullptr,
          hasTex ? bufferData(views[4], acc[4]) : nullptr};

        auto stride = [&](const tinygltf::BufferView* v, const tinygltf::Accessor* a) {
          return v->byteStride != 0 ? v->byteStride
                                    : tinygltf::GetComponentSizeInBytes(a->componentType) *
              tinygltf::GetNumComponentsInType(a->type);
        };

        std::array<size_t, 5> strides{
          stride(views[0], acc[0]),
          stride(views[1], acc[1]),
          hasNormals ? stride(views[2], acc[2]) : 0,
          hasTangents ? stride(views[3], acc[3]) : 0,
          hasTex ? stride(views[4], acc[4]) : 0};

        for (size_t i = 0; i < vertexCount; ++i)
        {
          auto& vtx = vertices.emplace_back();
          glm::vec3 normal(0.f);
          glm::vec4 tangent(0.f);

          std::memcpy(&vtx.pos, ptrs[1], sizeof(vtx.pos));
          if (hasNormals)
            std::memcpy(&normal, ptrs[2], sizeof(normal));
          if (hasTangents)
            std::memcpy(&tangent, ptrs[3], sizeof(tangent));
          if (hasTex)
            std::memcpy(&vtx.texcoord, ptrs[4], sizeof(vtx.texcoord));

          vtx.norm =
            bestFitNormal ? best_fit_normal(normal) : quantize4fnorm(glm::vec4{normal, 0.f});
          vtx.tang = quantize4fnorm(tangent);

          ptrs[1] += strides[1];
          if (hasNormals)
            ptrs[2] += strides[2];
          if (hasTangents)
            ptrs[3] += strides[3];
          if (hasTex)
            ptrs[4] += strides[4];
        }

        check(views[0]->byteStride == 0, "Indexes can't have stride!");
        const size_t indexCount = acc[0]->count;
        const size_t indOffset = byteSize(indices);

        if (acc[0]->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
        {
          for (size_t i = 0; i < indexCount; ++i)
          {
            uint16_t idx;
            std::memcpy(&idx, ptrs[0], sizeof(idx));
            indices.push_back(idx);
            ptrs[0] += 2;
          }
        }
        else if (acc[0]->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
        {
          const size_t oldSize = indices.size();
          indices.resize(oldSize + indexCount);
          std::memcpy(indices.data() + oldSize, ptrs[0], indexCount * sizeof(uint32_t));
        }

        tinygltf::Accessor indAcc{};
        indAcc.bufferView = 1;
        indAcc.byteOffset = indOffset;
        indAcc.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
        indAcc.count = indexCount;
        indAcc.type = TINYGLTF_TYPE_SCALAR;
        indAcc.minValues = acc[0]->minValues;
        indAcc.maxValues = acc[0]->maxValues;

        tinygltf::Accessor posAcc{};
        posAcc.bufferView = 0;
        posAcc.byteOffset = vertsOffset;
        posAcc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        posAcc.count = vertexCount;
        posAcc.type = TINYGLTF_TYPE_VEC3;
        posAcc.minValues = acc[1]->minValues;
        posAcc.maxValues = acc[1]->maxValues;


        tinygltf::Accessor normAcc{};
        normAcc.bufferView = 0;
        normAcc.byteOffset = vertsOffset + 12;
        normAcc.componentType = TINYGLTF_COMPONENT_TYPE_BYTE;
        normAcc.count = vertexCount;
        normAcc.type = TINYGLTF_TYPE_VEC3;
        normAcc.normalized = true;

        tinygltf::Accessor tcAcc{};
        tcAcc.bufferView = 0;
        tcAcc.byteOffset = vertsOffset + 16;
        tcAcc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        tcAcc.count = vertexCount;
        tcAcc.type = TINYGLTF_TYPE_VEC2;

        tinygltf::Accessor tangAcc{};
        tangAcc.bufferView = 0;
        tangAcc.byteOffset = vertsOffset + 24;
        tangAcc.componentType = TINYGLTF_COMPONENT_TYPE_BYTE;
        tangAcc.count = vertexCount;
        tangAcc.type = TINYGLTF_TYPE_VEC4;
        tangAcc.normalized = true;

        size_t base = combinedAccessors.size();
        combinedAccessors.push_back(indAcc);
        combinedAccessors.push_back(posAcc);
        if (hasNormals)
          combinedAccessors.push_back(normAcc);
        if (hasTex)
          combinedAccessors.push_back(tcAcc);
        if (hasTangents)
          combinedAccessors.push_back(tangAcc);

        prim.attributes.clear();
        prim.indices = static_cast<int>(base);
        prim.attributes["POSITION"] = static_cast<int>(base + 1);
        if (hasNormals)
          prim.attributes["NORMAL"] = static_cast<int>(base + 2);
        if (hasTex)
          prim.attributes["TEXCOORD_0"] = static_cast<int>(base + 3);
        if (hasTangents)
          prim.attributes["TANGENT"] = static_cast<int>(base + 4);
      }
    }


    const size_t vertsBytes = byteSize(vertices);
    const size_t indBytes = byteSize(indices);

    tinygltf::Buffer combinedBuf{};
    combinedBuf.data.resize(vertsBytes + indBytes);
    std::memcpy(combinedBuf.data.data(), vertices.data(), vertsBytes);
    std::memcpy(combinedBuf.data.data() + vertsBytes, indices.data(), indBytes);

    tinygltf::BufferView vertView{};
    vertView.buffer = 0;
    vertView.byteOffset = 0;
    vertView.byteLength = vertsBytes;
    vertView.byteStride = sizeof(Vertex);
    vertView.target = TINYGLTF_TARGET_ARRAY_BUFFER;

    tinygltf::BufferView indView{};
    indView.buffer = 0;
    indView.byteOffset = vertsBytes;
    indView.byteLength = indBytes;
    indView.byteStride = 0;
    indView.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;

    model.buffers = {std::move(combinedBuf)};
    model.bufferViews = {vertView, indView};
    model.accessors = std::move(combinedAccessors);

    model.extensionsRequired.emplace_back("KHR_mesh_quantization");
    model.extensionsUsed.emplace_back("KHR_mesh_quantization");

    path.replace_extension("");
    path.replace_filename("baked/" + path.filename().string());
    path.replace_extension(".gltf");

    if (!std::filesystem::exists(path.parent_path()))
      std::filesystem::create_directories(path.parent_path());

    bool written = api.WriteGltfSceneToFile(&model, path.string(), false, false, true, false);
    check(written, "Failed to write baked scene: " + path.string());

    spdlog::info("Model saved to {}", path.string());
  }
  catch (const std::exception& e)
  {
    spdlog::critical("Error: {}", e.what());
    return 1;
  }

  return 0;
}
