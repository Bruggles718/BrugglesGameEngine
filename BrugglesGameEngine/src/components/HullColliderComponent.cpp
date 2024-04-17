#include "components/HullColliderComponent.hpp"
#include "physics/HullCollider.hpp"
#include "components/RigidbodyComponent.hpp"
#include "physics/Rigidbody.hpp"

namespace bruggles {
    namespace components {
        HullColliderComponent::HullColliderComponent(
            std::vector<Vector2> i_vertices
        ) {
            m_collider = std::make_shared<physics::HullCollider>(i_vertices);
            m_collider->m_component = this;
        }

        HullColliderComponent::HullColliderComponent() {
            m_collider = std::make_shared<physics::HullCollider>(std::vector<Vector2>{});
            m_collider->m_component = this;
        }

        std::vector<Vector2> HullColliderComponent::Vertices() {
            physics::HullCollider* hull = dynamic_cast<physics::HullCollider*>(m_collider.get());
            return hull->Vertices;
        }
        void HullColliderComponent::SetVertices(std::vector<Vector2> i_vertices) {
            physics::HullCollider* hull = dynamic_cast<physics::HullCollider*>(m_collider.get());
            hull->Vertices = i_vertices;
        }

        std::string HullColliderComponent::Serialize() {
            std::string result = "";
            physics::HullCollider* hull = dynamic_cast<physics::HullCollider*>(m_collider.get());
            for (int i = 0; i < hull->Vertices.size(); i++) {
                result += SERIALIZE_ROOT(Vector2, "" + hull->Vertices[i].Serialize() + "");
                if (i + 1 < hull->Vertices.size()) {
                    result += ",";
                }
            }
            return "{" SERIALIZE(Vertices, list, "[" + result + "]") "}";
        }
    }
}