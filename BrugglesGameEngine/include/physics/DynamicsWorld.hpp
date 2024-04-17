#pragma once

#include <vector>
#include "physics/CollisionWorld.hpp"
#include "physics/Rigidbody.hpp"
#include "physics/Solver.hpp"

namespace bruggles {
    namespace physics {
        /**
         * Represents a physics world with forces
        */
        class DynamicsWorld : public CollisionWorld {
        public:
            void AddRigidbody(Rigidbody* i_object);

            void RemoveRigidbody(Rigidbody* i_object);

            void Step(float i_deltaTime);

            void SetGravity(Vector2 i_gravity);

            Vector2 GetGravity();
        private:
            Vector2 m_gravity = Vector2(0, 50.0f);
        };
    }
}