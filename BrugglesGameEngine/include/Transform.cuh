#pragma once

#include "Vector2.cuh"

namespace bruggles {
    /**
     * Describes the position, rotation, and scale of an object in 2D space
    */
    struct Transform {
        __host__ __device__ Transform();
        __host__ __device__ Transform(Vector2 i_position, float i_rotation, Vector2 i_scale);

        Vector2 Position{0, 0};
        float Rotation{0};
        Vector2 Scale{1, 1};

        __host__ __device__ Transform& operator=(const Transform& other);
    };
}