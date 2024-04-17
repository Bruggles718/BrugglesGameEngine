#include "components/TransformComponent.hpp"
#include <pybind11/embed.h>

namespace py = pybind11;

namespace bruggles {
    namespace components {
        TransformComponent::TransformComponent() {
            m_transform = std::make_shared<Transform>();
            SetPosition(Vector2(0, 0));
            SetRotation(0);
            SetScale(Vector2(1, 1));
        }

        TransformComponent::~TransformComponent() {
            
        }

        Vector2 TransformComponent::Position() {
            return m_transform->Position;
        }

        float TransformComponent::Rotation() {
            return m_transform->Rotation;
        }

        Vector2 TransformComponent::Scale() {
            return m_transform->Scale;
        }

        void TransformComponent::SetPosition(Vector2 i_position) {
            m_transform->Position = i_position;
        }

        void TransformComponent::SetRotation(float i_rotation) {
            m_transform->Rotation = i_rotation;
        }

        void TransformComponent::SetScale(Vector2 i_scale) {
            m_transform->Scale = i_scale;
        }

        std::shared_ptr<Transform> TransformComponent::GetTransform() {
            return m_transform;
        }

        std::string TransformComponent::Serialize() {
            return "{" SERIALIZE(Position, Vector2, "" + Position().Serialize() + "") 
                + ", " + SERIALIZE(Rotation, float, "\"" + std::to_string(Rotation()) + "\"")
                + ", " + SERIALIZE(Scale, Vector2, "" + Scale().Serialize() + "") "}";
        }
    }
}