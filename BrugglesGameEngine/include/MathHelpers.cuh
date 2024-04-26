#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace bruggles {
    namespace math {
        /**
         * Linearly interpolates between two values
        */
        __host__ __device__ float Lerp(float a, float b, float t);

        __host__ __device__ int Min(int a, int b);
    }
}