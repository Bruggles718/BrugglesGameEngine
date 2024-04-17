#include "Transform.hpp"

namespace bruggles {
    Transform& Transform::operator=(const Transform& other) {
        this->Position = other.Position;
        this->Rotation = other.Rotation;
        this->Scale = other.Scale;
        return *this;
    }
}