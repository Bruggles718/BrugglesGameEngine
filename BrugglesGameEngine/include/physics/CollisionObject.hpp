#pragma once

#include <functional>
#include "Vector2.hpp"
#include "physics/Collider.hpp"
#include "Transform.hpp"
#include "physics/Collision.hpp"
#include "GameObject.hpp"
#include <unordered_map>

namespace bruggles {
    namespace physics {
        /**
         * Represents a static object in a physics world.
        */
        struct CollisionObject {
            Uint64 m_uniqueID;

            Collider* collider;
            Transform* transform;

            bool IsDynamic = true; /**< Whether or not this object has forces or velocity applied to it.*/
            bool IsTrigger = false; /**< Whether or not this object is a trigger. Triggers do not have dynamic solvers applied to them.*/

            std::function<void(Collision, float)> OnCollision; /**< The callback to be fired when a collision is detected for this object*/

            GameObject* m_gameObject;
        };
    }
}