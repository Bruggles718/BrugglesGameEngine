#pragma once

#include "physics/CollisionPoints.hpp"

typedef uint64_t Uint64;

namespace bruggles {
    namespace physics {
        struct CollisionObject;

        /**
         * Stores the related objects and response information for a collision
        */
        struct Collision {
            CollisionObject* A;
            CollisionObject* B;
            CollisionPoints Points;

            Collision();

            Collision(
                CollisionObject* i_A, 
                CollisionObject* i_B, 
                CollisionPoints i_Points
            );
        };
    }
}