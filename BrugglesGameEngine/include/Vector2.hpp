#pragma once

#include "Serializable.hpp"
#include <cmath>

namespace bruggles {
    struct Transform;

    struct Vector2 : Serializable {
        float x = 0;
        float y = 0;

        Vector2();

        Vector2(const float i_x, const float i_y);
        
        /**
         * Gets the magnitude of this Vector2
        */
        float Magnitude() const;

        /**
         * Normalizes this Vector2
        */
        void Normalize();

        /**
         * Returns a new normalized Vector2 from this Vector2
        */
        Vector2 Normalized() const;

        void ApplyTransform(const Transform* i_tf);

        Vector2 Transformed(const Transform* i_tf) const;

        /**
         * Gets the distance between two Vector2s
        */
        static float Distance(const Vector2& a, const Vector2& b);

        /**
         * Gets the dot product of two Vector2s
        */
        static float Dot(const Vector2& a, const Vector2& b);
        
        /**
         * Gets the angle between two Vector2s
        */
        static float Angle(const Vector2& a, const Vector2& b);

        /**
         * Linearly interpolates between Vector2 a and Vector2 B
        */
        static Vector2 Lerp(const Vector2& a, const Vector2& b, float t);

        /**
         * Returns a new zeroed Vector2
        */
        static Vector2 Zero();

        /**
         * Vector where X is 1 and Y is 0
        */
        static Vector2 UnitX();

        /**
         * Vector where X is 0 and Y is 1
        */
        static Vector2 UnitY();

        friend Vector2 operator-(const Vector2& a, const Vector2& b);

        friend Vector2 operator+(const Vector2& a, const Vector2& b);

        friend Vector2 operator/(const Vector2& v, const float& f);

        friend Vector2 operator*(const Vector2& v, const float& f);

        Vector2& operator=(const Vector2& other);

        friend inline bool operator==(const Vector2& lhs, const Vector2& rhs);

        Vector2 operator-() const;

        std::string Serialize() override;
    };
}