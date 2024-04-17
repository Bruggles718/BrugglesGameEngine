#pragma once

#include "components/Renderer.hpp"
#include <vector>
#include <array>

namespace bruggles {
    namespace components {
        /**
         * Renders a shape for the Game Object this Shape Renderer is attached to
        */
        class ShapeRenderer : public Renderer {
        public:
            /**
             * Set the color of this shape renderer
            */
            void SetColor(std::array<int, 4> i_color);
            /**
             * Get the color of this shape renderer
            */
            std::array<int, 4> Color();
        protected:
            void _SetGameObject(GameObject* i_object) override;
            Transform* m_transform;
            std::array<int, 4> m_color;
        };
    }
}