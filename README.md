CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Stephen Gavin Sears
  * [LinkedIn](https://www.linkedin.com/in/gavin-sears-536a1b285), [personal website](https://gavin-sears.github.io/sgavinsears/index.html)
* Tested on: Windows 11, i9-14900HX @ 2.20GHz 32GB, RTX 4090 Laptop 16GB, Compute Capability 8.9 (Personal Computer)

### Basic Pathtracer

![A basic ray traced cornell box scene with a sphere](img/base_pathtracer.png)

### Extra Features

### Mesh Loading

![A cornell box with the stanford bunny inside](img/bunny.png)

### BVH

<table border="0">
  <tr>
    <td><img src="img/nobvh.png" width="300" alt="bunny render without bvh"></td>
    <td><img src="img/bvh.png" width="300" alt="bunny render using bvh"></td>
  </tr>
  <tr align="center">
    <td><b>No BVH</b></td>
    <td><b>BVH</b></td>
  </tr>
</table>

I ran 50 iterations on this scene with a 69660 triangle stanford bunny just to show that using bvh does not change the output. I record how long it took to render for each of them. Without a BVH, it took 29 minutes and 54.503 seconds (1794.503s). With a bvh, that was shaved down to 1.869 seconds for 50 iterations. Doing the math, that is a 99.896% speed up, which should show just how essential using a bvh is when rendering meshes.

### Fun Bloopers/Outtakes

![A cornell box with the stanford bunny inside. This bunny has a very strange shadow.](img/funnybunnyshadow.png)

This scene appears to be normal, except that the shadow of the bunny has strange holes in it. For a while I was thinking this could be an issue with my BVH (I ended up running my non-bvh version overnight, and it also had this strange hole). Once I determined that was not the case, I thought I would test more objects in the scene. I threw the bunny into Blender to decimate it or subdivide it to see if the geometry density was the issue, and... there were tons of holes in the mesh, making it non manifold. It turns out that rays were shooting out of the camera, hitting the ground, bouncing up, going through the bunny, and then going through the backfaces (I use glm::intersectRayTriangle, which backface culls). From there the rays were finding the light, and illuminating the floor. I just filled in the holes in Blender, and that solved it. This is nothing crazy, but I thought it would be fun to share, because it perplexed me for a moment.

### CHANGED CmakeLists.txt:

I added mesh.h to headers and mesh.cpp to sources for mesh loading.
I also added bvh.h and bvh.cpp to headers and sources for bvh.
Additionally, I added C compile options (for tiny gltf) and changed the preprocessor.