#pragma once

#include "Vector2.hpp"

namespace bruggles {
    /**
     * Describes the position, rotation, and scale of an object in 2D space
    */
    struct Transform {
        Vector2 Position{0, 0};
        float Rotation{0};
        Vector2 Scale{1, 1};

        Transform& operator=(const Transform& other);
    };
}