#include <vector>
#include "physics/PositionSolver.hpp"
#include "physics/CollisionObject.cuh"
#include "physics/Rigidbody.hpp"
#include "math.h"

namespace bruggles {
    namespace physics {
        void PositionSolver::Solve(
            std::vector<Collision>& i_collisions,
            float i_deltaTime
        ) {
            std::vector<std::pair<Vector2, Vector2>> deltas;
            for (Collision& collision : i_collisions) {

                Rigidbody* a = collision.A->IsDynamic ? static_cast<Rigidbody*>(collision.A) : nullptr;
                Rigidbody* b = collision.B->IsDynamic ? static_cast<Rigidbody*>(collision.B) : nullptr;

                float aInverseMass = a ? 1.0f/a->Mass : 0;
                float bInverseMass = b ? 1.0f/b->Mass : 0;

                const float percent = 0.8f;

                const float slop = 0.01f;

                Vector2 resolution = collision.Points.Normal * percent * std::max(collision.Points.Depth - slop, 0.0f) / (aInverseMass + bInverseMass);

                Vector2 deltaA = Vector2(0, 0);
                Vector2 deltaB = Vector2(0, 0);

                if (a ? a->IsSimulated : false) {
                    deltaA = resolution * aInverseMass;
                }

                if (b ? b->IsSimulated : false) {
                    deltaB = resolution * bInverseMass;
                }

                deltas.emplace_back(deltaA, deltaB);
            }

            for (int i = 0; i < deltas.size(); i++) {
                Rigidbody* a = i_collisions[i].A->IsDynamic ? static_cast<Rigidbody*>(i_collisions[i].A) : nullptr;
                Rigidbody* b = i_collisions[i].B->IsDynamic ? static_cast<Rigidbody*>(i_collisions[i].B) : nullptr;

                if (a ? a->IsSimulated : false) {
                    a->GetTransform().Position = a->GetTransform().Position + deltas[i].first;
                }

                if (b ? b->IsSimulated : false) {
                    b->GetTransform().Position = b->GetTransform().Position - deltas[i].second;
                }
            }
        }
    }
}