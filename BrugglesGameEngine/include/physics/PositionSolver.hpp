#pragma once

#include <vector>
#include "physics/Solver.hpp"

namespace bruggles {
    namespace physics {
        /**
         * Updates the positions of two colliding physics objects
        */
        struct PositionSolver : Solver {
            void Solve(
                std::vector<Collision>& i_collisions,
                float i_deltaTime
            ) override;
        };
    }
}