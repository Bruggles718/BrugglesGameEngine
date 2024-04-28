#pragma once

#include "Vector2.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace bruggles {
    namespace physics {
        /**
         * Stores the response information about a collision
        */
        struct CollisionPoints {
            Vector2 A{ 0, 0 }; // Furthest point of A into B
            Vector2 B{ 0, 0 }; // Furthest point of B into A
            Vector2 Normal{ 0, 0 }; // A - B normalized
            float Depth{ 0 }; // Length of A - B
            bool HasCollision{ false };

            __host__ __device__ CollisionPoints Flip();
        };
    }
}