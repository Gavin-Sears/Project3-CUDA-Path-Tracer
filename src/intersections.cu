#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(
    Geom mesh,
    Triangle tri,
    Ray r,
    Ray localRay,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    glm::vec3 baryPosition;
    if (!glm::intersectRayTriangle(localRay.origin, localRay.direction, tri.v0, tri.v1, tri.v2, baryPosition))
    {
        return -1;
    }

    float t = baryPosition.z;
    if (t <= 0.0f)
    {
        return -1;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(localRay, t);

    float u = baryPosition.x;
    float v = baryPosition.y;
    glm::vec3 objspaceNormal = glm::normalize((1.0f - u - v) * tri.n0 + u * tri.n1 + v * tri.n2);

    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(objspaceNormal, 0.f)));
    // glm::intersectRayTriangle already back-face culls
    outside = true;

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool aabbIntersectionTest(
    glm::vec3 boundsMin,
    glm::vec3 boundsMax,
    Ray r,
    float &tNear)
{
    float tmin = -1e38f;
    float tmax = 1e38f;
    for (int axis = 0; axis < 3; ++axis)
    {
        float invD = 1.0f / r.direction[axis];
        float t0 = (boundsMin[axis] - r.origin[axis]) * invD;
        float t1 = (boundsMax[axis] - r.origin[axis]) * invD;
        tmin = glm::max(tmin, glm::min(t0, t1));
        tmax = glm::min(tmax, glm::max(t0, t1));
    }

    tNear = tmin;
    return tmax >= tmin && tmax > 0.0f;
}
