#include "components/Component.hpp"
#include <SDL.h>

namespace bruggles {
    namespace components {
        pybind11::object Component::GetComponent(pybind11::object i_type) {
            return m_gameObject->GetComponent(i_type);
        }

        void Component::SetGameObject(GameObject* i_object) {
            m_gameObject = i_object;
            _SetGameObject(i_object);
        }

        void Component::_SetGameObject(GameObject* i_object) {
            
        }

        void Component::OnComponentRemoved() {

        }

        GameObject* Component::GetGameObject() {
            return m_gameObject;
        }
    }
}