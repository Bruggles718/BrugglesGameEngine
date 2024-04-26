#include "Vector2.cuh"
#include "Transform.cuh"
#include "math.h"
#include <SDL.h>
#include "MathHelpers.cuh"

namespace bruggles {
    __host__ __device__ Vector2::Vector2() {
        this->x = 0;
        this->y = 0;
    }

    __host__ __device__ Vector2::Vector2(const float i_x, const float i_y) {
        this->x = i_x;
        this->y = i_y;
    }

    __host__ __device__ float Vector2::Magnitude() const {
        return sqrtf(x * x + y * y);
    }

    __host__ __device__ Vector2 Vector2::Normalized() const {
        return Vector2(this->x, this->y) / Magnitude();
    }

    __host__ __device__ void Vector2::Normalize() {
        Vector2 normalizedVector = Normalized();
        this->x = normalizedVector.x;
        this->y = normalizedVector.y;
    }

    __host__ __device__ void Vector2::ApplyTransform(const Transform* i_tf) {
        float scaledX = this->x * i_tf->Scale.x;
        float scaledY = this->y * i_tf->Scale.y;
        float rad = i_tf->Rotation * M_PI / 180.0f;
        float rotatedX = scaledX * cosf(rad) - scaledY * sinf(rad);
        float rotatedY = scaledX * sinf(rad) + scaledY * cosf(rad);
        this->x = rotatedX + i_tf->Position.x;
        this->y = rotatedY + i_tf->Position.y;
    }

    __host__ __device__ Vector2 Vector2::Transformed(const Transform* i_tf) const {
        Vector2 result{this->x, this->y};
        result.ApplyTransform(i_tf);
        return result;
    }

    __host__ __device__ float Vector2::Distance(const Vector2& a, const Vector2& b) {
        float deltaX = a.x - b.x;
        float deltaY = a.y - b.y;
        return sqrtf((deltaX * deltaX + deltaY * deltaY));
    }

    __host__ __device__ float Vector2::Dot(const Vector2& a, const Vector2& b) {
        return a.x * b.x + a.y * b.y;
    }

    __host__ __device__ float Vector2::Angle(const Vector2& a, const Vector2& b) {
        return acos(Vector2::Dot(a, b) / (a.Magnitude() * b.Magnitude()));
    }

    __host__ __device__ Vector2 Vector2::Lerp(const Vector2& a, const Vector2& b, float t) {
        return Vector2(math::Lerp(a.x, b.x, t), math::Lerp(a.y, b.y, t));
    }

    __host__ __device__ Vector2 Vector2::Zero() {
        return Vector2(0, 0);
    }

    __host__ __device__ Vector2 Vector2::UnitX() {
        return Vector2(1, 0);
    }

    __host__ __device__ Vector2 Vector2::UnitY() {
        return Vector2(0, 1);
    }

    __host__ __device__ Vector2 operator-(const Vector2& a, const Vector2& b) {
        float newX = a.x - b.x;
        float newY = a.y - b.y;
        return Vector2(newX, newY);
    }

    __host__ __device__ Vector2 operator+(const Vector2& a, const Vector2& b) {
        float newX = a.x + b.x;
        float newY = a.y + b.y;
        return Vector2(newX, newY);
    }

    __host__ __device__ Vector2 operator*(const Vector2& v, const float& f) {
        float newX = v.x * f;
        float newY = v.y * f;
        return Vector2(newX, newY);
    }

    __host__ __device__ Vector2 operator/(const Vector2& v, const float& f) {
        float newX = v.x / f;
        float newY = v.y / f;
        return Vector2(newX, newY);
    }

    __host__ __device__ Vector2& Vector2::operator=(const Vector2& other) {
        if (this == &other) {
            return *this;
        }
        this->x = other.x;
        this->y = other.y;
        return *this;
    }

    __host__ __device__ inline bool operator==(const Vector2& lhs, const Vector2& rhs) {
        return lhs.x == rhs.x && lhs.y == rhs.y;
    }

    __host__ __device__ Vector2 Vector2::operator-() const {
        return Vector2(-this->x, -this->y);
    }

    std::string Vector2::Serialize() {
        return "{" SERIALIZE(X, float, "\"" + std::to_string(x) + "\"") + ", " + SERIALIZE(Y, float, "\"" + std::to_string(y) + "\"") "}";
    }
}