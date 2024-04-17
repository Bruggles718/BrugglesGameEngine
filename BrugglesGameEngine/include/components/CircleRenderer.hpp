#pragma once

#include "components/ShapeRenderer.hpp"
#include <vector>
#include <array>
#include "Serializable.hpp"

namespace bruggles {
    namespace components {
        /**
         * Used to render a circle for the Game Object this Renderer is attached to.
        */
        class CircleRenderer : public ShapeRenderer, public Serializable {
        public:
            CircleRenderer();
            CircleRenderer(std::array<int, 4> i_color, Vector2 i_center, float i_radius);
            void Render() override;
            void RenderToCamera(Camera* camera) override;

            /**
             * Get the relative center of this circle renderer
            */
            Vector2 Center();

            /**
             * Get the radius of this circle renderer
            */
            float Radius();

            /**
             * Set the relative center of this circle renderer
            */
            void SetCenter(Vector2 i_center);
            
            /**
             * Set the radius of this circle renderer
            */
            void SetRadius(float i_radius);

            std::string Serialize() override;
        private:
            Vector2 m_center;
            float m_radius;
        };
    }
}