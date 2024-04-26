#pragma once

#include "physics/CollisionObject.cuh"
#include "Vector2.cuh"

namespace bruggles {
    namespace physics {
        /**
         * Represents an object in a physics world that can respond to forces
        */
        struct Rigidbody : CollisionObject {
            bool IsSimulated = true; /**< whether or not this object should be considered when doing any physics calculations.*/

            Vector2 Velocity = Vector2(0, 0);
            Vector2 Force = Vector2(0, 0);
            float Mass = 1;

            Vector2 Gravity = Vector2(0, 0);
            bool TakesGravity = true; /**< If true, this Rigidbody will have the world gravity applied to it. If false, it will have its Gravity applied to it.*/

            float StaticFriction = 0.5f;
            float DynamicFriction = 0.5f;
            float Restitution = 0.5f; /**< Zero for completely inelastic collisions, one for completely elastic collisions.*/

            float Drag = 0.05f;
        };
    }
}