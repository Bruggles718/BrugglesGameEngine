#pragma once

#include <vector>
#include "physics/Collision.hpp"
#include "Transform.hpp"

namespace bruggles {
    namespace physics {
        /**
         * Represents an function-like object used for computing constraints in a physics world
        */
        struct Solver {
            virtual void Solve(
                std::vector<Collision>& i_collisions,
                float i_deltaTime
            ) = 0;
        };
    }
}