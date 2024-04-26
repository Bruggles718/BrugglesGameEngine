#include "components/HullColliderComponent.hpp"
#include "physics/HullCollider.cuh"
#include "components/RigidbodyComponent.hpp"
#include "physics/Rigidbody.hpp"

namespace bruggles {
    namespace components {
        HullColliderComponent::HullColliderComponent(
            std::vector<Vector2> i_vertices
        ) {
            TDynamicArray<Vector2> verts{};
            for (int i = 0; i < i_vertices.size(); i++) {
                verts.PushBack(i_vertices[i]);
            }
            m_collider = std::make_shared<physics::HullCollider>(verts);
            m_collider->m_component = this;
        }

        HullColliderComponent::HullColliderComponent() {
            TDynamicArray<Vector2> verts{};
            m_collider = std::make_shared<physics::HullCollider>(verts);
            m_collider->m_component = this;
        }

        std::vector<Vector2> HullColliderComponent::Vertices() {
            physics::HullCollider* hull = dynamic_cast<physics::HullCollider*>(m_collider.get());
            std::vector<Vector2> verts{};
            for (int i = 0; i < hull->Vertices.Size(); i++) {
                verts.push_back(hull->Vertices[i]);
            }
            return verts;
        }
        void HullColliderComponent::SetVertices(std::vector<Vector2> i_vertices) {
            physics::HullCollider* hull = dynamic_cast<physics::HullCollider*>(m_collider.get());
            hull->Vertices.Clear();
            for (int i = 0; i < i_vertices.size(); i++) {
                hull->Vertices.PushBack(i_vertices[i]);
            }
        }

        std::string HullColliderComponent::Serialize() {
            std::string result = "";
            physics::HullCollider* hull = dynamic_cast<physics::HullCollider*>(m_collider.get());
            for (int i = 0; i < hull->Vertices.Size(); i++) {
                result += SERIALIZE_ROOT(Vector2, "" + hull->Vertices[i].Serialize() + "");
                if (i + 1 < hull->Vertices.Size()) {
                    result += ",";
                }
            }
            return "{" SERIALIZE(Vertices, list, "[" + result + "]") "}";
        }
    }
}