#pragma once

#include <array>
#include "Vector2.cuh"
#include <stddef.h>
#include "TArray.cuh"

namespace bruggles {
    namespace physics {
        /**
         * The simplest shape possible for capturing an area in a 2D space.
        */
        struct Simplex {
            TArray<Vector2> Vertices = TArray<Vector2>(3);

            __host__ __device__ void Push_Front(Vector2 vertex);

            __host__ __device__ Vector2& operator[](int i) const;

            __host__ __device__ size_t Size() const;

            __host__ __device__ Simplex& operator=(const TArray<Vector2>& list);

            size_t m_size = 0;
        };
    }
}