#pragma once

#include "Vector2.hpp"

namespace bruggles {
    namespace physics {
        /**
         * Stores the response information about a collision
        */
        struct CollisionPoints {
            Vector2 A; // Furthest point of A into B
            Vector2 B; // Furthest point of B into A
            Vector2 Normal; // A - B normalized
            float Depth; // Length of A - B
            bool HasCollision;

            CollisionPoints();

            CollisionPoints(Vector2& i_A, Vector2& i_B);

            CollisionPoints Flip();
        };
    }
}