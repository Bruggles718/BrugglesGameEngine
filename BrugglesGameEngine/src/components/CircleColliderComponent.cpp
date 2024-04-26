#include "components/CircleColliderComponent.hpp"
#include "physics/CircleCollider.cuh"
#include "components/RigidbodyComponent.hpp"
#include "physics/Rigidbody.hpp"

namespace bruggles {
    namespace components {
        CircleColliderComponent::CircleColliderComponent(
            Vector2 i_center, float i_radius
        ) {
            m_collider = std::make_shared<physics::CircleCollider>(i_center, i_radius);
            m_collider->m_component = this;
        }

        CircleColliderComponent::CircleColliderComponent() {
            m_collider = std::make_shared<physics::CircleCollider>(Vector2(0, 0), 20.0f);
            m_collider->m_component = this;
            SetCenter(Vector2(0, 0));
            SetRadius(20.0f);
        }

        Vector2 CircleColliderComponent::Center() {
            return dynamic_cast<physics::CircleCollider*>(m_collider.get())->Center;
        }
        float CircleColliderComponent::Radius() {
            return dynamic_cast<physics::CircleCollider*>(m_collider.get())->Radius;
        }

        void CircleColliderComponent::SetCenter(Vector2 i_center) {
            dynamic_cast<physics::CircleCollider*>(m_collider.get())->Center = i_center;
        }
        void CircleColliderComponent::SetRadius(float i_radius) {
            dynamic_cast<physics::CircleCollider*>(m_collider.get())->Radius = i_radius;
        }

        std::string CircleColliderComponent::Serialize() {
            return "{" SERIALIZE(Center, Vector2, "" + Center().Serialize() + "") 
                + ", " + SERIALIZE(Radius, float, "\"" + std::to_string(Radius()) + "\"") "}";
        }
    }
}