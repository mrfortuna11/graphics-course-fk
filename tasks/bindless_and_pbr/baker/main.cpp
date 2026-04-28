#include <iostream>
#include <filesystem>
#include <vector>
#include <array>
#include <span>
#include <optional>
#include <cstring>
#include <cstdint>
#include <bit>
#include <algorithm>

#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <tiny_gltf.h>
#include <stb_image.h>

#include <spdlog/spdlog.h>
#include <fmt/std.h>
#include <render_utils/shaders/quant.h>

struct RenderElement
{
    std::uint32_t vertexOffset;
    std::uint32_t vertexCount;
    std::uint32_t indexOffset;
    std::uint32_t indexCount;
    std::array<double, 3> posMax;
    std::array<double, 3> posMin;
};

struct Mesh
{
    std::uint32_t firstRelem;
    std::uint32_t relemCount;
};

struct Vertex
{
    // First 3 floats are position, 4th float is a packed normal
    glm::vec4 positionAndNormal;
    // First 2 floats are tex coords, 3rd is a packed tangent, 4th is padding
    glm::vec4 texCoordAndTangentAndPadding;
};
static_assert(sizeof(Vertex) == sizeof(float) * 8);

struct ProcessedMeshes
{
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<RenderElement> relems;
    std::vector<Mesh> meshes;
};

static void updateMinMax(RenderElement& relem, glm::vec3 curr)
{
    for (uint32_t i = 0; i < 3; ++i)
    {
        relem.posMin[i] = std::min(relem.posMin[i], static_cast<double>(curr[i]));
        relem.posMax[i] = std::max(relem.posMax[i], static_cast<double>(curr[i]));
    }
}

static std::optional<tinygltf::Model> loadModel(const std::filesystem::path& path)
{
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err, warn;
    bool success = false;

    auto ext = path.extension();
    if (ext == ".gltf")
        success = loader.LoadASCIIFromFile(&model, &err, &warn, path.string());
    else if (ext == ".glb")
        success = loader.LoadBinaryFromFile(&model, &err, &warn, path.string());
    else
    {
        spdlog::error("glTF: Unknown extension: '{}'. Expected .gltf or .glb.", ext);
        return std::nullopt;
    }

    if (!success)
    {
        spdlog::error("glTF: Failed to load model!");
        if (!err.empty()) spdlog::error("glTF: {}", err);
        return std::nullopt;
    }
    if (!warn.empty()) spdlog::warn("glTF: {}", warn);

    if (!model.extensions.empty() || !model.extensionsRequired.empty() || !model.extensionsUsed.empty())
        spdlog::warn("glTF: Extensions present – none are implemented!");

    return model;
}

static void ProcessAttribute(
    const tinygltf::Model& model,
    int accessor_ind,
    std::span<Vertex> vertices,
    auto setter)
{
    const auto& accessor = model.accessors[accessor_ind];
    const auto& bufView = model.bufferViews[accessor.bufferView];
    const std::byte* ptr = reinterpret_cast<const std::byte*>(model.buffers[bufView.buffer].data.data())
        + bufView.byteOffset + accessor.byteOffset;

    const size_t stride = bufView.byteStride != 0
        ? bufView.byteStride
        : tinygltf::GetComponentSizeInBytes(accessor.componentType) *
          tinygltf::GetNumComponentsInType(accessor.type);

    for (auto& vtx : vertices)
    {
        setter(vtx, ptr);
        ptr += stride;
    }
}

static ProcessedMeshes processMeshes(const tinygltf::Model& model)
{
    ProcessedMeshes result;

    // Pre-allocate
    {
        std::size_t vertexBytes = 0, indexBytes = 0;
        for (const auto& bv : model.bufferViews)
        {
            if (bv.target == TINYGLTF_TARGET_ARRAY_BUFFER)      vertexBytes += bv.byteLength;
            if (bv.target == TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER) indexBytes += bv.byteLength;
        }
        result.vertices.reserve(vertexBytes / sizeof(Vertex));
        result.indices.reserve(indexBytes / sizeof(std::uint32_t));
    }
    {
        std::size_t totalPrims = 0;
        for (const auto& m : model.meshes) totalPrims += m.primitives.size();
        result.relems.reserve(totalPrims);
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
                spdlog::warn("Non-triangle primitive skipped");
                --result.meshes.back().relemCount;
                continue;
            }

            auto posIt = prim.attributes.find("POSITION");
            if (posIt == prim.attributes.end())
            {
                spdlog::warn("Primitive without POSITION skipped");
                --result.meshes.back().relemCount;
                continue;
            }

            const size_t vertexCount = model.accessors[posIt->second].count;

            result.relems.push_back(RenderElement{
                .vertexOffset = static_cast<std::uint32_t>(result.vertices.size()),
                .vertexCount = static_cast<std::uint32_t>(vertexCount),
                .indexOffset = static_cast<std::uint32_t>(result.indices.size()),
                .indexCount = static_cast<std::uint32_t>(model.accessors[prim.indices].count),
                .posMax = {std::numeric_limits<double>::lowest(),
                           std::numeric_limits<double>::lowest(),
                           std::numeric_limits<double>::lowest()},
                .posMin = {std::numeric_limits<double>::max(),
                           std::numeric_limits<double>::max(),
                           std::numeric_limits<double>::max()}
            });

            result.vertices.resize(result.vertices.size() + vertexCount);
            auto vertexSpan = std::span(result.vertices).last(vertexCount);

            for (const auto& [attr_name, acc_idx] : prim.attributes)
            {
                if (attr_name == "POSITION")
                {
                    ProcessAttribute(model, acc_idx, vertexSpan, [&](Vertex& v, const std::byte* ptr)
                    {
                        glm::vec3 pos;
                        std::memcpy(&pos, ptr, sizeof(pos));
                        v.positionAndNormal.x = pos.x;
                        v.positionAndNormal.y = pos.y;
                        v.positionAndNormal.z = pos.z;
                        updateMinMax(result.relems.back(), pos);
                    });
                }
                else if (attr_name == "NORMAL")
                {
                    ProcessAttribute(model, acc_idx, vertexSpan, [](Vertex& v, const std::byte* ptr)
                    {
                        glm::vec3 n;
                        std::memcpy(&n, ptr, sizeof(n));
                        v.positionAndNormal.w = std::bit_cast<float>(quantize4fnorm(n));
                    });
                }
                else if (attr_name == "TANGENT")
                {
                    ProcessAttribute(model, acc_idx, vertexSpan, [](Vertex& v, const std::byte* ptr)
                    {
                        glm::vec4 t;
                        std::memcpy(&t, ptr, sizeof(t));
                        v.texCoordAndTangentAndPadding.z = std::bit_cast<float>(quantize4fnorm(t));
                    });
                }
                else if (attr_name == "TEXCOORD_0")
                {
                    ProcessAttribute(model, acc_idx, vertexSpan, [](Vertex& v, const std::byte* ptr)
                    {
                        glm::vec2 uv;
                        std::memcpy(&uv, ptr, sizeof(uv));
                        v.texCoordAndTangentAndPadding.x = uv.x;
                        v.texCoordAndTangentAndPadding.y = uv.y;
                    });
                }
                else
                {
                    spdlog::warn("Unsupported attribute: {}", attr_name);
                }
            }

            // Indices
            const auto& idxAcc = model.accessors[prim.indices];
            const auto& idxBv  = model.bufferViews[idxAcc.bufferView];
            const std::byte* idxPtr = reinterpret_cast<const std::byte*>(model.buffers[idxBv.buffer].data.data())
                + idxBv.byteOffset + idxAcc.byteOffset;

            if (idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
            {
                for (size_t i = 0; i < idxAcc.count; ++i)
                {
                    std::uint16_t idx;
                    std::memcpy(&idx, idxPtr, sizeof(idx));
                    result.indices.push_back(idx);
                    idxPtr += 2;
                }
            }
            else
            {
                const size_t oldSize = result.indices.size();
                result.indices.resize(oldSize + idxAcc.count);
                std::memcpy(result.indices.data() + oldSize, idxPtr,
                            idxAcc.count * sizeof(std::uint32_t));
            }
        }
    }
    return result;
}

static void bakeScene(const std::filesystem::path& path)
{
    auto maybeModel = loadModel(path);
    if (!maybeModel) return;

    tinygltf::Model model = std::move(*maybeModel);
    tinygltf::TinyGLTF loader;

    auto [verts, inds, relems, meshes] = processMeshes(model);

    // Prepare single buffer
    model.extensionsRequired.emplace_back("KHR_mesh_quantization");
    model.extensionsUsed.emplace_back("KHR_mesh_quantization");

    if (model.buffers.empty()) model.buffers.emplace_back();
    auto& buf = model.buffers[0];
    buf.data.resize(verts.size() * sizeof(Vertex) + inds.size() * sizeof(std::uint32_t));
    std::memcpy(buf.data.data(), verts.data(), verts.size() * sizeof(Vertex));
    std::memcpy(buf.data.data() + verts.size() * sizeof(Vertex),
                inds.data(), inds.size() * sizeof(std::uint32_t));
    buf.uri = (path.stem().string() + "_baked.bin");

    // BufferViews
    model.bufferViews.clear();
    {
        auto& bv = model.bufferViews.emplace_back();
        bv.buffer = 0;
        bv.byteLength = verts.size() * sizeof(Vertex);
        bv.byteOffset = 0;
        bv.byteStride = sizeof(Vertex);
        bv.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    }
    {
        auto& bv = model.bufferViews.emplace_back();
        bv.buffer = 0;
        bv.byteLength = inds.size() * sizeof(std::uint32_t);
        bv.byteOffset = verts.size() * sizeof(Vertex);
        bv.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
    }

    // Accessors templates
tinygltf::Accessor posAcc, normAcc, texAcc, tanAcc, indAcc;

    indAcc.bufferView = 1;
    indAcc.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
    indAcc.type = TINYGLTF_TYPE_SCALAR;

    posAcc.bufferView = 0;
    posAcc.byteOffset = 0;
    posAcc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    posAcc.type = TINYGLTF_TYPE_VEC3;

    normAcc.bufferView = 0;
    normAcc.byteOffset = 12;
    normAcc.componentType = TINYGLTF_COMPONENT_TYPE_BYTE;
    normAcc.normalized = true;
    normAcc.type = TINYGLTF_TYPE_VEC3;

    texAcc.bufferView = 0;
    texAcc.byteOffset = 16;
    texAcc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    texAcc.type = TINYGLTF_TYPE_VEC2;

    tanAcc.bufferView = 0;
    tanAcc.byteOffset = 24;
    tanAcc.componentType = TINYGLTF_COMPONENT_TYPE_BYTE;
    tanAcc.normalized = true;
    tanAcc.type = TINYGLTF_TYPE_VEC4;

    std::array<tinygltf::Accessor, 4> vertexAccessors = {posAcc, normAcc, texAcc, tanAcc};
    const std::array<std::string, 4> attrNames = {"POSITION", "NORMAL", "TEXCOORD_0", "TANGENT"};

    model.accessors.clear();

    for (size_t i = 0; i < model.meshes.size(); ++i)
    {
        auto& mesh = model.meshes[i];
        std::erase_if(mesh.primitives, [](const auto& p){
            return p.mode != TINYGLTF_MODE_TRIANGLES || !p.attributes.contains("POSITION");
        });

        for (size_t j = 0; j < mesh.primitives.size(); ++j)
        {
            auto& prim = mesh.primitives[j];
            const auto& relem = relems[meshes[i].firstRelem + j];

            // Index accessor
            prim.indices = static_cast<int>(model.accessors.size());
            auto& currInd = model.accessors.emplace_back(indAcc);
            currInd.byteOffset = relem.indexOffset * sizeof(std::uint32_t);
            currInd.count = relem.indexCount;

            // Vertex accessors
            vertexAccessors[0].minValues.assign(relem.posMin.begin(), relem.posMin.end());
            vertexAccessors[0].maxValues.assign(relem.posMax.begin(), relem.posMax.end());

            std::erase_if(prim.attributes, [&](const auto& kv){
                for (const auto& name : attrNames)
                    if (kv.first == name) return false;
                return true;
            });

            for (size_t k = 0; k < attrNames.size(); ++k)
            {
                if (!prim.attributes.contains(attrNames[k])) continue;

                prim.attributes[attrNames[k]] = static_cast<int>(model.accessors.size());

                tinygltf::Accessor curr = vertexAccessors[k];
                curr.byteOffset += relem.vertexOffset * sizeof(Vertex);
                curr.count = relem.vertexCount;

                model.accessors.push_back(curr);
            }
        }
    }

    const std::filesystem::path outPath = path.parent_path() / (path.stem().string() + "_baked.gltf");
    loader.WriteGltfSceneToFile(&model, outPath.string(), false, false, true, false);
    spdlog::info("Baked glTF saved to: {}", outPath);
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path_to_gltf_or_glb>\n";
        return 1;
    }

    std::filesystem::path inputPath = argv[1];
    if (!std::filesystem::exists(inputPath))
    {
        std::cerr << "File not found: " << inputPath << "\n";
        return 1;
    }

    bakeScene(inputPath);
    return 0;
} 
