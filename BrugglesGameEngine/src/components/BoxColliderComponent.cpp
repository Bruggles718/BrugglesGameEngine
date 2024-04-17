#include "components/BoxColliderComponent.hpp"
#include "physics/BoxCollider.hpp"
#include "components/RigidbodyComponent.hpp"
#include "physics/Rigidbody.hpp"

namespace bruggles {
    namespace components {
        BoxColliderComponent::BoxColliderComponent(
            float i_width, float i_height
        ) {
            m_collider = std::make_shared<physics::BoxCollider>(i_width, i_height);
            m_width = i_width;
            m_height = i_height;
            m_collider->m_component = this;
        }

        BoxColliderComponent::BoxColliderComponent() {
            float i_width = 40.0f;
            float i_height = 40.0f;
            m_collider = std::make_shared<physics::BoxCollider>(i_width, i_height);
            m_width = i_width;
            m_height = i_height;
            m_collider->m_component = this;
        }

        float BoxColliderComponent::Width() {
            return m_width;
        }
        float BoxColliderComponent::Height() {
            return m_height;
        }
        void BoxColliderComponent::SetWidth(float i_width) {
            m_width = i_width;
            dynamic_cast<physics::BoxCollider*>(m_collider.get())->SetDimensions(m_width, m_height);
        }
        void BoxColliderComponent::SetHeight(float i_height) {
            m_height = i_height;
            dynamic_cast<physics::BoxCollider*>(m_collider.get())->SetDimensions(m_width, m_height);
        }

        std::string BoxColliderComponent::Serialize() {
            return "{" SERIALIZE(Width, float, "\"" + std::to_string(Width()) + "\"") 
                + ", " + SERIALIZE(Height, float, "\"" + std::to_string(Height()) + "\"") "}";
        }
    }
}