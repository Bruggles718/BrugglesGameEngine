#pragma once

#include "components/ShapeRenderer.hpp"
#include <vector>
#include <array>
#include "Serializable.hpp"

namespace bruggles {
    namespace components {
        /**
         * Used to render a Polygon for the Game Object this Renderer is attached to.
        */
        class PolygonRenderer : public ShapeRenderer, public Serializable {
        public:
            PolygonRenderer();
            PolygonRenderer(std::array<int, 4> i_color, std::vector<Vector2> i_vertices);
            void Render() override;
            void RenderToCamera(Camera* camera) override;

            /**
             * Get the relative vertices of this polygon renderer
            */
            std::vector<Vector2> Vertices();

            /**
             * Set the relative vertices of this polygon renderer
            */
            void SetVertices(std::vector<Vector2> i_vertices);

            std::string Serialize() override;
        private:
            std::vector<Vector2> m_vertices;
        };
    }
}