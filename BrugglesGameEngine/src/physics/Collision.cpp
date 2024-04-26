#include "physics/Collision.hpp"
#include "physics/CollisionObject.cuh"

namespace bruggles {
    namespace physics {
        Collision::Collision() {}

        Collision::Collision(
            CollisionObject* i_A, 
            CollisionObject* i_B, 
            CollisionPoints i_Points
        ) {
            A = i_A;
            B = i_B;
            Points = i_Points;
        }
    }
}