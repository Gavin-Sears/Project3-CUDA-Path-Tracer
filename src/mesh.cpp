#include "mesh.h"
#include "tiny_gltf_v3.h"
#include <iostream>

// Helper functions for working with the binary blobs in the gltf 2.0 and glb formats
namespace
{
    const uint8_t* accessorElementPtr(const tg3_model* model, const tg3_accessor& accessor, uint32_t index)
    {
        const tg3_buffer_view& bufferView = model->buffer_views[accessor.buffer_view];
        const tg3_buffer& buffer = model->buffers[bufferView.buffer];
        int32_t stride = tg3_accessor_byte_stride(&accessor, &bufferView);
        return buffer.data.data + bufferView.byte_offset + accessor.byte_offset + (uint64_t)index * stride;
    }

    glm::vec3 readVec3(const tg3_model* model, const tg3_accessor& accessor, uint32_t index)
    {
        const float* p = reinterpret_cast<const float*>(accessorElementPtr(model, accessor, index));
        return glm::vec3(p[0], p[1], p[2]);
    }

    uint32_t readIndex(const tg3_model* model, const tg3_accessor& accessor, uint32_t index)
    {
        const uint8_t* p = accessorElementPtr(model, accessor, index);
        switch (accessor.component_type)
        {
            case TG3_COMPONENT_TYPE_UNSIGNED_BYTE:  return *p;
            case TG3_COMPONENT_TYPE_UNSIGNED_SHORT: return *reinterpret_cast<const uint16_t*>(p);
            case TG3_COMPONENT_TYPE_UNSIGNED_INT:   return *reinterpret_cast<const uint32_t*>(p);
            default:                                return 0;
        }
    }

    int32_t findAttribute(const tg3_primitive& primitive, const char* name)
    {
        for (uint32_t i = 0; i < primitive.attributes_count; ++i)
        {
            if (tg3_str_equals_cstr(primitive.attributes[i].key, name))
            {
                return primitive.attributes[i].value;
            }
        }
        return -1;
    }
}

bool loadMeshTriangles(const std::string& filepath, std::vector<Triangle>& outTriangles)
{
    tinygltf3::Model model;
    tinygltf3::ErrorStack errors;
    tg3_error_code err = tinygltf3::parse_file(model, errors, filepath.c_str());
    if (err != TG3_OK)
    {
        std::cout << errors.count() << " error(s) loading " << filepath << ":" << std::endl;
        for (uint32_t i = 0; i < errors.count(); ++i)
        {
            const tg3_error_entry* entry = errors.entry(i);
            std::cout << "  " << entry->message << std::endl;
        }
        return false;
    }

    for (uint32_t m = 0; m < model->meshes_count; ++m)
    {
        const tg3_mesh& mesh = model->meshes[m];
        for (uint32_t p = 0; p < mesh.primitives_count; ++p)
        {
            const tg3_primitive& primitive = mesh.primitives[p];
            if (primitive.mode != -1 && primitive.mode != TG3_MODE_TRIANGLES)
            {
                // This skips lines and points for now
                continue;
            }

            int32_t posIdx = findAttribute(primitive, "POSITION");
            if (posIdx < 0)
            {
                // If we somehow don't find positions for our vertices
                continue;
            }
            // position accessor
            const tg3_accessor& posAccessor = model->accessors[posIdx];

            int32_t normIdx = findAttribute(primitive, "NORMAL");
            const tg3_accessor* normAccessor = (normIdx >= 0) ? &model->accessors[normIdx] : nullptr;

            if (primitive.indices < 0)
            {
                // If we don't have an index for a vertex, we don't load it 
                // (will probably change this in the future)
                continue;
            }
            const tg3_accessor& idxAccessor = model->accessors[primitive.indices];

            for (uint64_t i = 0; i + 2 < idxAccessor.count; i += 3)
            {
                uint32_t i0 = readIndex(model.get(), idxAccessor, (uint32_t)i + 0);
                uint32_t i1 = readIndex(model.get(), idxAccessor, (uint32_t)i + 1);
                uint32_t i2 = readIndex(model.get(), idxAccessor, (uint32_t)i + 2);

                Triangle tri;
                tri.v0 = readVec3(model.get(), posAccessor, i0);
                tri.v1 = readVec3(model.get(), posAccessor, i1);
                tri.v2 = readVec3(model.get(), posAccessor, i2);

                if (normAccessor)
                {
                    tri.n0 = readVec3(model.get(), *normAccessor, i0);
                    tri.n1 = readVec3(model.get(), *normAccessor, i1);
                    tri.n2 = readVec3(model.get(), *normAccessor, i2);
                }
                else
                {
                    glm::vec3 faceNormal = glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                    tri.n0 = tri.n1 = tri.n2 = faceNormal;
                }

                outTriangles.push_back(tri);
            }
        }
    }

    return true;
}
