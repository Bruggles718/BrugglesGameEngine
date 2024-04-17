#include "components/PolygonRenderer.hpp"
#include "Application.hpp"
#include "components/TransformComponent.hpp"

namespace bruggles {
    namespace components {
        PolygonRenderer::PolygonRenderer(std::array<int, 4> i_color, std::vector<Vector2> i_vertices) {
            this->m_vertices = i_vertices;
            this->m_color = i_color;
        }

        PolygonRenderer::PolygonRenderer() {
            m_color = std::array<int, 4>{255, 255, 255, 255};
            m_vertices = std::vector<Vector2>{};
        }

        void PolygonRenderer::Render() {
            GetGameObject()->GetApplication()->RenderToCamera(this);
        }
        
        void PolygonRenderer::RenderToCamera(Camera* i_camera) {
            SDL_SetRenderDrawColor(i_camera->m_renderer, m_color[0], m_color[1], m_color[2], m_color[3]);
            for (int i = 0; i < this->m_vertices.size(); i++) {
                Vector2 p1 = this->m_vertices[i];
                Vector2 p2 = this->m_vertices[(i + 1) % this->m_vertices.size()];
                p1.ApplyTransform(m_transform);
                p2.ApplyTransform(m_transform);
                Transform invTf = i_camera->GetInverseTransform();
                p1.ApplyTransform(&invTf);
                p2.ApplyTransform(&invTf);

                SDL_RenderDrawLine(
                    i_camera->m_renderer,
                    (int)p1.x, (int)p1.y,
                    (int)p2.x, (int)p2.y
                );
            }
        }

        std::vector<Vector2> PolygonRenderer::Vertices() {
            return this->m_vertices;
        }
        void PolygonRenderer::SetVertices(std::vector<Vector2> i_vertices) {
            this->m_vertices = i_vertices;
        }

        std::string PolygonRenderer::Serialize() {
            std::string vertResult = "";
            for (int i = 0; i < this->m_vertices.size(); i++) {
                vertResult += SERIALIZE_ROOT(Vector2, "" + this->m_vertices[i].Serialize() + "");
                if (i + 1 < this->m_vertices.size()) {
                    vertResult += ",";
                }
            }
            std::string colResult = "";
            for (int i = 0; i < this->m_color.size(); i++) {
                colResult += SERIALIZE_ROOT(float, "\"" + std::to_string(this->m_color[i]) + "\"");
                if (i + 1 < this->m_color.size()) {
                    colResult += ",";
                }
            }
            return "{" SERIALIZE(Vertices, list, "[" + vertResult + "]") 
                + ", " SERIALIZE(Color, list, "[" + colResult + "]") "}";
        }
    }
}