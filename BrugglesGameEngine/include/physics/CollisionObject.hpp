#pragma once

#include <functional>
#include "Vector2.hpp"
#include "physics/Collider.hpp"
#include "Transform.hpp"
#include "physics/Collision.hpp"
#include <unordered_map>
#include "physics/EndPoint.hpp"

namespace bruggles {
    class GameObject;
    namespace physics {
        /**
         * Represents a static object in a physics world.
        */
        struct CollisionObject {
            Uint64 m_uniqueID;

            Collider* collider;

            bool IsDynamic = true; /**< Whether or not this object has forces or velocity applied to it.*/
            bool IsTrigger = false; /**< Whether or not this object is a trigger. Triggers do not have dynamic solvers applied to them.*/

            std::function<void(Collision, float)> OnCollision; /**< The callback to be fired when a collision is detected for this object*/

            GameObject* m_gameObject;

            Transform& GetTransform();

            Transform& GetLastTransform();

            void SetTransform(Transform* tf);

            void UpdateLastTransform();

            void UpdateTopLeftBottomRightAABB();

            EndPoint* GetTop();
            EndPoint* GetLeft();
            EndPoint* GetBottom();
            EndPoint* GetRight();

            bool m_addedEndPoints = false;

        protected:
            

            Transform m_transform{
                Vector2(0, 0),
                0.0f,
                Vector2(1, 1)
            };
            Transform m_lastTransform{
                Vector2(0, 0),
                0.0f,
                Vector2(1, 1)
            };

            std::shared_ptr<EndPoint> m_top;
            std::shared_ptr<EndPoint> m_left;
            std::shared_ptr<EndPoint> m_bottom;
            std::shared_ptr<EndPoint> m_right;

            void SetEndPoint(EndPoint* i_e, float i_value, bool isMin);
        };
    }
}