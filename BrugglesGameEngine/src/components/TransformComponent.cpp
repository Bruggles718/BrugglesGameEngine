#include "components/TransformComponent.hpp"
#include <pybind11/embed.h>
#include "components/RigidbodyComponent.hpp"
#include <iostream>

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
            if (!m_gameObject) return;
            pybind11::object PyRb = pybind11::module::import("bruggles").attr("RigidbodyComponent");
            pybind11::object pyrbComp = m_gameObject->GetComponent(PyRb);
            if (pyrbComp.ptr() == pybind11::cast<pybind11::none>(Py_None).ptr()) {
                return;
            }
            RigidbodyComponent rbComp = pyrbComp.cast<bruggles::components::RigidbodyComponent>();
            std::shared_ptr<physics::Rigidbody> rb = rbComp.GetRigidbody();
            rb->SetTransform(m_transform.get());
        }

        void TransformComponent::SetRotation(float i_rotation) {
            m_transform->Rotation = i_rotation;
            if (!m_gameObject) return;
            pybind11::object PyRb = pybind11::module::import("bruggles").attr("RigidbodyComponent");
            pybind11::object pyrbComp = m_gameObject->GetComponent(PyRb);
            if (pyrbComp.ptr() == pybind11::cast<pybind11::none>(Py_None).ptr()) {
                return;
            }
            RigidbodyComponent rbComp = pyrbComp.cast<bruggles::components::RigidbodyComponent>();
            std::shared_ptr<physics::Rigidbody> rb = rbComp.GetRigidbody();
            rb->SetTransform(m_transform.get());
        }

        void TransformComponent::SetScale(Vector2 i_scale) {
            m_transform->Scale = i_scale;
            if (!m_gameObject) return;
            pybind11::object PyRb = pybind11::module::import("bruggles").attr("RigidbodyComponent");
            pybind11::object pyrbComp = m_gameObject->GetComponent(PyRb);
            if (pyrbComp.ptr() == pybind11::cast<pybind11::none>(Py_None).ptr()) {
                return;
            }
            RigidbodyComponent rbComp = pyrbComp.cast<bruggles::components::RigidbodyComponent>();
            std::shared_ptr<physics::Rigidbody> rb = rbComp.GetRigidbody();
            rb->SetTransform(m_transform.get());
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