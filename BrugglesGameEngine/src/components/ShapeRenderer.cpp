#include "components/ShapeRenderer.hpp"
#include "components/TransformComponent.hpp"

namespace bruggles {
    namespace components {
        void ShapeRenderer::SetColor(std::array<int, 4> i_color) {
            this->m_color = i_color;
        }

        std::array<int, 4> ShapeRenderer::Color() {
            return m_color;
        }

        void ShapeRenderer::_SetGameObject(GameObject* i_object) {
            pybind11::object PyTf = pybind11::module::import("bruggles").attr("TransformComponent");
            pybind11::object pytfComp = i_object->GetComponent(PyTf);
            std::shared_ptr<Transform> tf = pytfComp.cast<bruggles::components::TransformComponent>().GetTransform();
            m_transform = tf.get();
        }
    }
}