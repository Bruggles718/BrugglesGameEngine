#pragma once

#include <vector>
#include "physics/Solver.hpp"

namespace bruggles {
    namespace physics {
        /**
         * Updates the velocities of two colliding physics objects
        */
        struct ImpulseSolver : Solver {
            void Solve(
                std::vector<Collision>& i_collisions,
                float i_deltaTime
            ) override;
        };
    }
}