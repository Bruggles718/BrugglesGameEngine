#include <vector>
#include "physics/ImpulseSolver.hpp"
#include "physics/Rigidbody.hpp"
#include "math.h"
#include <iostream>

namespace bruggles {
    namespace physics {
        void ImpulseSolver::Solve(
            std::vector<Collision>& i_collisions,
            float i_deltaTime
        ) {
            for (Collision& collision : i_collisions) {

                // Init values for calculations
                Rigidbody* a = collision.A->IsDynamic ? static_cast<Rigidbody*>(collision.A) : nullptr;
                Rigidbody* b = collision.B->IsDynamic ? static_cast<Rigidbody*>(collision.B) : nullptr;

                Vector2 aVelocity = a ? a->Velocity : Vector2::Zero();
                Vector2 bVelocity = b ? b->Velocity : Vector2::Zero();

                Vector2 diff = bVelocity - aVelocity;
                float speed = Vector2::Dot(diff, collision.Points.Normal);

                float aInverseMass = a ? 1.0f/a->Mass : 0;
                float bInverseMass = b ? 1.0f/b->Mass : 0;

                // skip if negative impulse
                if (speed <= 0) continue;

                // Calculate Impulse

                float restitutionCoefficient = (a ? a->Restitution : 1.0f) * (b ? b->Restitution : 1.0f);

                float impulseMagnitude = (-(1.0f + restitutionCoefficient)) * speed / (aInverseMass + bInverseMass);

                Vector2 impulse = collision.Points.Normal * impulseMagnitude;

                if (a ? a->IsSimulated : false) {
                    aVelocity = aVelocity - (impulse * aInverseMass);
                }

                if (b ? b->IsSimulated : false) {
                    bVelocity = bVelocity + (impulse * bInverseMass);
                }

                // Calculate Friction
                
                diff = bVelocity - aVelocity;
                speed = Vector2::Dot(diff, collision.Points.Normal);
                Vector2 tangent = diff - (collision.Points.Normal * speed);

                if (tangent.Magnitude() > 0.0001f) { // safe normalize
                    tangent.Normalize();
                }

                float frictionVelocity = Vector2::Dot(diff, tangent);

                float aSFriction = a ? a->StaticFriction : 0.0f;
                float bSFriction = b ? b->StaticFriction : 0.0f;
                float aDFriction = a ? a->DynamicFriction : 0.0f;
                float bDFriction = b ? b->DynamicFriction : 0.0f;

                float mu = -Vector2(aSFriction, bSFriction).Magnitude();

                float threshold = -frictionVelocity / (aInverseMass + bInverseMass);

                Vector2 friction;
                if (std::abs(threshold) < impulseMagnitude * mu) {
                    friction = tangent * threshold;
                } else {
                    mu = -Vector2(aDFriction, bDFriction).Magnitude();
                    friction = tangent * -impulseMagnitude * mu;
                }

                if (a ? a->IsSimulated : false) {
                    aVelocity = aVelocity - (friction * aInverseMass);
                }

                if (b ? b->IsSimulated : false) {
                    bVelocity = bVelocity + (friction * bInverseMass);
                }

                if (a) {
                    a->Velocity = aVelocity;
                }
                
                if (b) {
                    b->Velocity = bVelocity;
                }
            }
        }
    }
}