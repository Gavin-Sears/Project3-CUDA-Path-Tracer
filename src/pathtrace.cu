#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <cub/device/device_radix_sort.cuh>
#include <utility>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define BVH_STACK_SIZE 64

#define BVH_HEAT_MAX 96.0f
#define BVH_OUTLINE_PIXELS 1.5f

#define SORT_MIN_PATHS 16384
#define SORT_MIN_DEPTH 2

#define ERRORCHECK 0

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

// indicator to pass into remove_if
struct max_reached {
    __host__ __device__ bool operator()(const PathSegment p) {
        return p.remainingBounces <= 0;
    }
};

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static Triangle* dev_triangles = NULL;
static BVHNode* dev_bvhNodes = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

// Timing for the GUI
static cudaEvent_t evStart = NULL;
static cudaEvent_t evStop = NULL;

// Arrays that we use for final gather with radix sort.
static PathSegment* dev_paths_sorted = NULL;
static ShadeableIntersection* dev_intersections_sorted = NULL;

// These are unsorted (in) and sorted (out) arrays of all material indices in intersections.
static unsigned int* dev_sort_keys_in = NULL;
static unsigned int* dev_sort_keys_out = NULL;

// These arrays will hold unsorted and sorted indices intersection and path arrays.
static int* dev_sort_idx_in = NULL;
static int* dev_sort_idx_out = NULL;

// Scratch space for radix sort.
static void* dev_sort_temp = NULL;
static size_t sort_temp_bytes = 0;
static int sort_end_bit = 1;

// Key 0 is reserved for paths that missed everything (materialId == -1)
__global__ void buildMaterialSortKeys(int n, const ShadeableIntersection* intersections,
    unsigned int* keys, int* indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        keys[idx] = (unsigned int)(intersections[idx].materialId + 1);
        indices[idx] = idx;
    }
}

__global__ void gatherSortedPaths(int n, const int* sortedIndices,
    const PathSegment* pathsIn, PathSegment* pathsOut,
    const ShadeableIntersection* intersectionsIn, ShadeableIntersection* intersectionsOut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        int src = sortedIndices[idx];
        pathsOut[idx] = pathsIn[src];
        intersectionsOut[idx] = intersectionsIn[src];
    }
}

static void sortPathsByMaterial(int num_paths, int blockSize1d)
{
    const int blocks = (num_paths + blockSize1d - 1) / blockSize1d;

    // Much like naive boids, we build an array of indices that we sort with a temp array
    buildMaterialSortKeys<<<blocks, blockSize1d>>>(
        num_paths, dev_intersections, dev_sort_keys_in, dev_sort_idx_in);

    size_t temp_bytes = sort_temp_bytes;
    // This function call is very confusing, so here is a guide:
    // In dev_sort_keys_out and dev_sort_idx_out (which have length num_paths) we store values of
    // dev_sort_keys_in and dev_sort_idx_in that are sorted based on bits zero (0, second to last argument) 
    // through sort_end_bit (an integer) of the values in dev_sort_keys_in (which are materialId).
    // temp_bytes number of bytes are used for scratch space, in the array dev_sort_temp 
    // (which is initialized in pathtraceinit). The two "out" arrays are sorted at the end, but the sorted
    // key array is useless right now because it is just a bunch of sorted material IDs, which we can find using the
    // indices anyways.
    cub::DeviceRadixSort::SortPairs(dev_sort_temp, temp_bytes,
        dev_sort_keys_in, dev_sort_keys_out, dev_sort_idx_in, dev_sort_idx_out,
        num_paths, 0, sort_end_bit);

    // Gather ala naive boids, but with two arrays
    gatherSortedPaths<<<blocks, blockSize1d>>>(
        num_paths, dev_sort_idx_out,
        dev_paths, dev_paths_sorted,
        dev_intersections, dev_intersections_sorted);

    // Swap sorted arrays with original now that we are done.
    std::swap(dev_paths, dev_paths_sorted);
    std::swap(dev_intersections, dev_intersections_sorted);
}

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    
    cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, -1, pixelcount * sizeof(ShadeableIntersection));

    // Sort buffers always allocated so we can toggle material sorting
    cudaMalloc(&dev_paths_sorted, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersections_sorted, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_sort_keys_in, pixelcount * sizeof(unsigned int));
    cudaMalloc(&dev_sort_keys_out, pixelcount * sizeof(unsigned int));
    cudaMalloc(&dev_sort_idx_in, pixelcount * sizeof(int));
    cudaMalloc(&dev_sort_idx_out, pixelcount * sizeof(int));

    // Since materialId is such a small data type (uint) with few values, we
    // calculate the minimum number of bits needed to do radix sort on the value.
    sort_end_bit = 1;
    while ((1u << sort_end_bit) < scene->materials.size() + 1)
    {
        sort_end_bit++;
    }

    // This CUB (CUDA UnBound) radix sort function requires scratch space.
    // To get the correct amount into sort_temp_bytes, we must run the function itself without any scratch space initialized.
    // All of these parameters will be the same every time we sort, except for num_paths, so we run the function
    // with the max number of paths (pixelcount) so that we always have enough scratch space.
    // I feel like this is super unintuitive, so I figure I will appreciate having this note in the future.
    sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, sort_temp_bytes,
        dev_sort_keys_in, dev_sort_keys_out, dev_sort_idx_in, dev_sort_idx_out,
        pixelcount, 0, sort_end_bit);
    cudaMalloc(&dev_sort_temp, sort_temp_bytes);

    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_triangles);
    cudaFree(dev_bvhNodes);
    cudaFree(dev_intersections);
    cudaFree(dev_paths_sorted);
    cudaFree(dev_intersections_sorted);
    cudaFree(dev_sort_keys_in);
    cudaFree(dev_sort_keys_out);
    cudaFree(dev_sort_idx_in);
    cudaFree(dev_sort_idx_out);
    cudaFree(dev_sort_temp);

    // reset event objects
    if (evStart) cudaEventDestroy(evStart);
    if (evStop) cudaEventDestroy(evStop);
    evStart = evStop = NULL;

    checkCUDAError("pathtraceFree");
}

void pathtraceReset()
{
    if (!hst_scene || !dev_image)
    {
        return;
    }

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    checkCUDAError("pathtraceReset");
}

void pathtraceCopyImageToHost()
{
    if (!hst_scene || !dev_image)
    {
        return;
    }

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtraceCopyImageToHost");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, segment.pixelIndex, segment.remainingBounces);
        thrust::uniform_real_distribution<float> u01(-0.5, 0.5);

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng))
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng))
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    Triangle* triangles,
    BVHNode* bvhNodes,
    int geoms_size,
    ShadeableIntersection* intersections,
    bool useBVH)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];
        if (pathSegment.remainingBounces <= 0) return;

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH)
            {
                glm::vec3 minInter;
                glm::vec3 minNorm;

                t = FLT_MAX;

                // This ray used for triangle intersection is transformed into mesh space,
                // as opposed to the triangles being transformed.
                Ray localRay;
                localRay.origin = multiplyMV(geom.inverseTransform, glm::vec4(pathSegment.ray.origin, 1.0f));
                localRay.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(pathSegment.ray.direction, 0.0f)));

                if (useBVH)
                {
                  if (geom.bvhRoot >= 0)
                  {
                    // aabbintersectionTest returns a value in object space, so we need a way to convert it to world
                    // space before comparing it to regular 
                    float toWorld = glm::length(multiplyMV(geom.transform, glm::vec4(localRay.direction, 0.0f)));

                    size_t stack[BVH_STACK_SIZE];
                    int stackPtr = 0;
                    stack[stackPtr++] = (size_t)geom.bvhRoot;

                    while (stackPtr > 0)
                    {
                        size_t nodeIdx = stack[--stackPtr];
                        BVHNode node = bvhNodes[nodeIdx];

                        float tNear;
                        // No intersection with BVH
                        if (!aabbIntersectionTest(node.boundsMin, node.boundsMax, localRay, tNear))
                        {
                            continue;
                        }
                        // Intersection is farther than nearest intersection already found
                        if (tNear * toWorld > t)
                        {
                            continue;
                        }

                        // Otheriwise, we:
                        // loop through triangles if we hit a leaf node
                        if (node.triangleCount > 0)
                        {
                            for (size_t tIdx = node.triangleStart; tIdx < node.triangleStart + node.triangleCount; ++tIdx)
                            {
                                Triangle tri = triangles[tIdx];
                                float curt = triangleIntersectionTest(geom, tri, pathSegment.ray, localRay, minInter, minNorm, outside);
                                if (curt < t && curt > 0)
                                {
                                    t = curt;
                                    tmp_intersect = minInter;
                                    tmp_normal = minNorm;
                                }
                            }
                        }
                        // or continue to children if we are not in a leaf node
                        else
                        {
                            stack[stackPtr++] = node.leftChild;
                            stack[stackPtr++] = node.rightChild;
                        }
                    }
                  }
                }
                else
                {
                    for (size_t tIdx = geom.triangleStart; tIdx < geom.triangleStart + geom.triangleCount; ++tIdx) {
                        Triangle tri = triangles[tIdx];
                        float curt = triangleIntersectionTest(geom, tri, pathSegment.ray, localRay, minInter, minNorm, outside);
                        if (curt < t && curt > 0)
                        {
                            t = curt;
                            tmp_intersect = minInter;
                            tmp_normal = minNorm;
                        }
                    }
                }
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// BVH debug view
// Each camera ray does a traversal through the BVH
// mode 0 colors triangles by the closest leaf node
// mode 1 is a heat map of how many BVH nodes the ray popped off the stack
// outlines can also be toggled
__global__ void bvhDebugKernel(
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Triangle* triangles,
    BVHNode* bvhNodes,
    glm::vec3* image,
    int mode,
    bool outlines,
    float pixelAngle)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment path = pathSegments[idx];

    float t = FLT_MAX;      // closest triangle hit (world units)
    float tEdge = FLT_MAX;  // closest leaf-box edge (world units)
    int visited = 0;
    long long hitLeaf = -1;

    glm::vec3 tmpIntersect;
    glm::vec3 tmpNormal;
    bool outside;

    for (int g = 0; g < geoms_size; g++)
    {
        Geom& geom = geoms[g];
        if (geom.type != MESH || geom.bvhRoot < 0) continue;

        Ray localRay;
        localRay.origin = multiplyMV(geom.inverseTransform, glm::vec4(path.ray.origin, 1.0f));
        localRay.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(path.ray.direction, 0.0f)));
        float toWorld = glm::length(multiplyMV(geom.transform, glm::vec4(localRay.direction, 0.0f)));

        // create traversal stack, and load root
        size_t stack[BVH_STACK_SIZE];
        int stackPtr = 0;
        stack[stackPtr++] = (size_t)geom.bvhRoot;

        while (stackPtr > 0)
        {
            // pop node before processing
            size_t nodeIdx = stack[--stackPtr];
            BVHNode node = bvhNodes[nodeIdx];
            // value for heat map
            ++visited;

            // skip if no intersection or not in front of everything else so far.
            float tNear, tFar;
            if (!aabbIntersectionTest(node.boundsMin, node.boundsMax, localRay, tNear, tFar))
                continue;
            if (tNear * toWorld > t)
                continue;

            // if leaf node
            if (node.triangleCount > 0)
            {
                // check triangle intersections
                for (size_t tIdx = node.triangleStart; tIdx < node.triangleStart + node.triangleCount; ++tIdx)
                {
                    float curt = triangleIntersectionTest(geom, triangles[tIdx], path.ray, localRay,
                        tmpIntersect, tmpNormal, outside);
                    // if hit and in front, update shortest distance and create seed for random color
                    if (curt < t && curt > 0)
                    {
                        t = curt;
                        hitLeaf = (long long)nodeIdx;
                    }
                }

                if (outlines)
                {
                    // A box edge is where the ray enters or exits the box within a pixel or so of two faces at once.
                    // Distances stay in object space; the outline width is one pixel's footprint at that distance.
                    float hits[2] = { tNear, tFar };
                    for (int k = 0; k < 2; ++k)
                    {
                        float tc = hits[k];
                        // if we didn't hit anything or an edge is in front, continue
                        if (tc <= 0.0f || tc * toWorld >= tEdge) continue;

                        // intersect point
                        glm::vec3 p = localRay.origin + tc * localRay.direction;
                        // distance from face of slab
                        glm::vec3 faceDist = glm::min(p - node.boundsMin, node.boundsMax - p);
                        // BVH_OUTLINE_PIXELS * pixelAngle gives us a world space
                        // distance equal to BVH_OUTLINE_PIXELS number of pixels.
                        // we scale this value by the hit distance to make sure the lines don't disappear.
                        float width = BVH_OUTLINE_PIXELS * pixelAngle * tc;
                        // Cheaper to do this than a logical statement, we just are checking if we intersect with 2 or more faces
                        int nearFaces = (faceDist.x < width) + (faceDist.y < width) + (faceDist.z < width);
                        if (nearFaces >= 2) tEdge = tc * toWorld;
                    }
                }
            }
            // if not a leaf node, continue to children
            else
            {
                stack[stackPtr++] = node.leftChild;
                stack[stackPtr++] = node.rightChild;
            }
        }
    }

    glm::vec3 color(0.05f);
    if (mode == 0)
    {
        if (hitLeaf >= 0)
        {
            unsigned int h = utilhash((unsigned int)hitLeaf);
            color = 0.25f + 0.75f * glm::vec3(h & 255, (h >> 8) & 255, (h >> 16) & 255) / 255.0f;
        }
    }
    else
    {
        float x = glm::min(visited / BVH_HEAT_MAX, 1.0f);
        color = glm::vec3(x, 4.0f * x * (1.0f - x), 1.0f - x);  // blue (cheap) -> green -> red (expensive)
    }

    // Only edges in front of the surface or empty space are visible
    if (outlines && tEdge < t) color = glm::vec3(1.0f);

    image[path.pixelIndex] += color;
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    glm::vec3* image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        if (pathSegments[idx].remainingBounces <= 0) return;
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, pathSegments[idx].pixelIndex, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
                // Will need to be changed once there is more than one path per pixel.
                image[pathSegments[idx].pixelIndex] += pathSegments[idx].color;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                //pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                //pathSegments[idx].color *= u01(rng); // apply some noise because why not
                scatterRay(
                    pathSegments[idx], 
                    getPointOnRay(pathSegments[idx].ray, intersection.t), 
                    intersection.surfaceNormal, material, rng
                );
                // A path that runs out of bounces without reaching a light adds nothing.
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // Runtime toggles for the GUI
    const bool useBVH = guiData ? guiData->useBVH : true;
    const bool sortByMaterial = guiData ? guiData->sortByMaterial : true;
    const bool visualizeBVH = guiData ? guiData->visualizeBVH : false;

    if (evStart) cudaEventRecord(evStart);

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // If we visualize the BVH, we skip regular rendering
    // and do one iteration of this debug view
    if (visualizeBVH)
    {
        bvhDebugKernel<<<(pixelcount + blockSize1d - 1) / blockSize1d, blockSize1d>>>(
            pixelcount, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_triangles, dev_bvhNodes,
            dev_image, guiData->bvhVizMode, guiData->bvhOutlines, cam.pixelLength.y);
        checkCUDAError("bvh debug view");
        guiData->TracedDepth = 0;
    }

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = visualizeBVH;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, -1, num_paths * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            dev_triangles,
            dev_bvhNodes,
            hst_scene->geoms.size(),
            dev_intersections,
            useBVH
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        if (sortByMaterial && depth >= SORT_MIN_DEPTH && num_paths >= SORT_MIN_PATHS)
        {
            sortPathsByMaterial(num_paths, blockSize1d);
        }

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_image
        );

        // in dev_paths: remove_if path has terminated
        PathSegment* new_end = thrust::remove_if(thrust::device, dev_paths, dev_paths + num_paths, max_reached());
        // then lower length of num_paths to match new array
        num_paths = new_end - dev_paths;
        //iterationComplete = depth >= traceDepth; // TODO: should be based off stream compaction results.
        iterationComplete = num_paths <= 0;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    // Not needed anymore because of stream compaction
    /*dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);*/

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    checkCUDAError("pathtrace");

    // GPU time for this iteration which we pass to GUI
    if (evStop && guiData)
    {
        cudaEventRecord(evStop);
        cudaEventSynchronize(evStop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, evStart, evStop);
        guiData->iterationMs = (guiData->iterationMs == 0.0f) ? ms : 0.9f * guiData->iterationMs + 0.1f * ms;
    }
}
