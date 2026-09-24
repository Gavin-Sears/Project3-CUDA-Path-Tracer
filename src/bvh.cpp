#include "bvh.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cfloat>
#include <vector>

#define BVH_MAX_LEAF_TRIANGLES 4
#define BVH_MAX_DEPTH 32

namespace
{
    glm::vec3 triangleCentroid(const Triangle &tri)
    {
        return (tri.v0 + tri.v1 + tri.v2) / 3.0f;
    }

    void triangleBounds(const Triangle &tri, glm::vec3 &boundsMin, glm::vec3 &boundsMax)
    {
        boundsMin = glm::min(tri.v0, glm::min(tri.v1, tri.v2));
        boundsMax = glm::max(tri.v0, glm::max(tri.v1, tri.v2));
    }

    // Builds the bvh subtree node for triangles in the range start to start + count - 1 (inclusive).
    // Resulting node is sent to BVHNodes, and index of current subtree node is returned.
    size_t buildRange(std::vector<Triangle> &triangles, std::vector<BVHNode> &bvhNodes,
        size_t start, size_t count, int depth)
    {
        size_t nodeIndex = bvhNodes.size();
        bvhNodes.push_back(BVHNode{});

        // Finding bounds of triangle range
        glm::vec3 boundsMin(FLT_MAX);
        glm::vec3 boundsMax(-FLT_MAX);
        for (size_t i = start; i < start + count; ++i)
        {
            glm::vec3 triMin, triMax;
            triangleBounds(triangles[i], triMin, triMax);
            boundsMin = glm::min(boundsMin, triMin);
            boundsMax = glm::max(boundsMax, triMax);
        }

        // If we hit our limit for BVH parameters, simply record the bounds, triangles,
        // then return. This is one base case
        if (count <= BVH_MAX_LEAF_TRIANGLES || depth >= BVH_MAX_DEPTH)
        {
            bvhNodes[nodeIndex].boundsMin = boundsMin;
            bvhNodes[nodeIndex].boundsMax = boundsMax;
            bvhNodes[nodeIndex].triangleStart = start;
            bvhNodes[nodeIndex].triangleCount = count;
            bvhNodes[nodeIndex].leftChild = 0;
            bvhNodes[nodeIndex].rightChild = 0;
            return nodeIndex;
        }

        // find the largest axis to split on.
        glm::vec3 extent = boundsMax - boundsMin;
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent[axis]) axis = 2;

        size_t mid = start + count / 2;
        std::nth_element(
            triangles.begin() + start, triangles.begin() + mid, triangles.begin() + start + count,
            // axis gets accessed outside of the scope of this lambda.
            // sort triangles on the largest axis.
            [axis](const Triangle &a, const Triangle &b)
            {
                return triangleCentroid(a)[axis] < triangleCentroid(b)[axis];
            });

        // Recurse
        size_t leftChild = buildRange(triangles, bvhNodes, start, mid - start, depth + 1);
        size_t rightChild = buildRange(triangles, bvhNodes, mid, start + count - mid, depth + 1);

        // This node will not contain triangles, since it has child nodes
        bvhNodes[nodeIndex].boundsMin = boundsMin;
        bvhNodes[nodeIndex].boundsMax = boundsMax;
        bvhNodes[nodeIndex].leftChild = leftChild;
        bvhNodes[nodeIndex].rightChild = rightChild;
        bvhNodes[nodeIndex].triangleStart = 0;
        bvhNodes[nodeIndex].triangleCount = 0;

        return nodeIndex;
    }
}

int constructBVH(std::vector<Triangle> &triangles, std::vector<BVHNode> &bvhNodes)
{
    if (triangles.empty())
    {
        return -1;
    }

    return (int)buildRange(triangles, bvhNodes, 0, triangles.size(), 0);
}
