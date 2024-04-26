#pragma once

#include "Serializable.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace bruggles {
    struct Transform;

    struct Vector2 : Serializable {
        float x = 0;
        float y = 0;

        __host__ __device__ Vector2();

        __host__ __device__ Vector2(const float i_x, const float i_y);
        
        /**
         * Gets the magnitude of this Vector2
        */
        __host__ __device__ float Magnitude() const;

        /**
         * Normalizes this Vector2
        */
        __host__ __device__ void Normalize();

        /**
         * Returns a new normalized Vector2 from this Vector2
        */
        __host__ __device__ Vector2 Normalized() const;

        __host__ __device__ void ApplyTransform(const Transform* i_tf);

        __host__ __device__ Vector2 Transformed(const Transform* i_tf) const;

        /**
         * Gets the distance between two Vector2s
        */
        __host__ __device__ static float Distance(const Vector2& a, const Vector2& b);

        /**
         * Gets the dot product of two Vector2s
        */
        __host__ __device__ static float Dot(const Vector2& a, const Vector2& b);
        
        /**
         * Gets the angle between two Vector2s
        */
        __host__ __device__ static float Angle(const Vector2& a, const Vector2& b);

        /**
         * Linearly interpolates between Vector2 a and Vector2 B
        */
        __host__ __device__ static Vector2 Lerp(const Vector2& a, const Vector2& b, float t);

        /**
         * Returns a new zeroed Vector2
        */
        __host__ __device__ static Vector2 Zero();

        /**
         * Vector where X is 1 and Y is 0
        */
        __host__ __device__ static Vector2 UnitX();

        /**
         * Vector where X is 0 and Y is 1
        */
        __host__ __device__ static Vector2 UnitY();

        __host__ __device__ friend Vector2 operator-(const Vector2& a, const Vector2& b);

        __host__ __device__ friend Vector2 operator+(const Vector2& a, const Vector2& b);

        __host__ __device__ friend Vector2 operator/(const Vector2& v, const float& f);

        __host__ __device__ friend Vector2 operator*(const Vector2& v, const float& f);

        __host__ __device__ Vector2& operator=(const Vector2& other);

        __host__ __device__ friend inline bool operator==(const Vector2& lhs, const Vector2& rhs);

        __host__ __device__ Vector2 operator-() const;

        __host__ std::string Serialize() override;
    };
}