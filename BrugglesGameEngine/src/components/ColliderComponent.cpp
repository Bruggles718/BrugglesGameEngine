#include "components/ColliderComponent.hpp"
#include "physics/Collider.cuh"
#include "components/RigidbodyComponent.hpp"
#include "physics/Rigidbody.hpp"

namespace bruggles {
    namespace components {

        void ColliderComponent::OnComponentRemoved() {
            pybind11::object PyRb = pybind11::module::import("bruggles").attr("RigidbodyComponent");
            pybind11::object pyrbComp = GetGameObject()->GetComponent(PyRb);
            if (pyrbComp.ptr() == pybind11::cast<pybind11::none>(Py_None).ptr()) {
                return;
            }
            std::shared_ptr<physics::Rigidbody> rb = pyrbComp.cast<bruggles::components::RigidbodyComponent>().GetRigidbody();
            rb->collider = nullptr;
        }

        void ColliderComponent::_SetGameObject(GameObject* i_object) {
            pybind11::object PyRb = pybind11::module::import("bruggles").attr("RigidbodyComponent");
            pybind11::object pyrbComp = i_object->GetComponent(PyRb);
            if (!pybind11::isinstance(pyrbComp, PyRb)) {
                return;
            }
            std::shared_ptr<physics::Rigidbody> rb = pyrbComp.cast<bruggles::components::RigidbodyComponent>().GetRigidbody();
            rb->collider = m_collider.get();
        }
    }
}