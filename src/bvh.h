#pragma once

#include "sceneStructs.h"

#include <vector>

// Builds a BVH over triangles[0, triangles.size()), reordering triangles in
// place so that each leaf's triangles end up contiguous. New nodes are
// appended to bvhNodes (so multiple meshes can share one node array).
// Returns the index of this tree's root node in bvhNodes, or -1 if
// triangles is empty.
int constructBVH(std::vector<Triangle> &triangles, std::vector<BVHNode> &bvhNodes);
