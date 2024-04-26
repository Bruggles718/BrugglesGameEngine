#include "Transform.cuh"

namespace bruggles {
    __host__ __device__ Transform::Transform() {
        Position = Vector2(0, 0);
        Rotation = 0;
        Scale = Vector2(1, 1);
    }

    __host__ __device__ Transform::Transform(Vector2 i_position, float i_rotation, Vector2 i_scale) {
        Position = i_position;
        Rotation = i_rotation;
        Scale = i_scale;
    }

    __host__ __device__ Transform& Transform::operator=(const Transform& other) {
        this->Position = other.Position;
        this->Rotation = other.Rotation;
        this->Scale = other.Scale;
        return *this;
    }
}