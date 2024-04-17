#include "components/CircleRenderer.hpp"
#include "Application.hpp"
#include "components/TransformComponent.hpp"

namespace bruggles {
    namespace components {
        CircleRenderer::CircleRenderer(std::array<int, 4> i_color, Vector2 i_center, float i_radius) {
            this->m_color = i_color;
            this->m_center = i_center;
            this->m_radius = i_radius;
        }

        CircleRenderer::CircleRenderer() {
            this->m_color = std::array<int, 4>{255, 255, 255, 255};
            this->m_center = Vector2(0, 0);
            this->m_radius = 20;
        }

        void CircleRenderer::Render() {
            GetGameObject()->GetApplication()->RenderToCamera(this);
        }
        
        void CircleRenderer::RenderToCamera(Camera* i_camera) {
            SDL_SetRenderDrawColor(i_camera->m_renderer, m_color[0], m_color[1], m_color[2], m_color[3]);
            float w = m_transform->Rotation * M_PI/180;

            Transform invTf = i_camera->GetInverseTransform();

            for (int i = 0; i < 360; i++) {
                Vector2 result{};
                Vector2 center{
                    m_center.x,
                    m_center.y
                };
                center.ApplyTransform(m_transform);
                float px = center.x;
                float py = center.y;
                float sx = m_transform->Scale.x * m_radius;
                float sy = m_transform->Scale.y * m_radius;
                float dx = std::cos(i * M_PI/180);
                float dy = std::sin(i * M_PI/180);
                float drnx = dx*std::cos(-w) - dy*std::sin(-w);
                float drny = dx*std::sin(-w) + dy*std::cos(-w);
                float nA2 = std::atan2(drny/sy, drnx/sx);
                result.x = sx*std::cos(nA2)*std::cos(w) - sy*std::sin(nA2)*std::sin(w) + px;
                result.y = sx*std::cos(nA2)*std::sin(w) + sy*std::sin(nA2)*std::cos(w) + py;
                result.ApplyTransform(&invTf);
                SDL_RenderDrawPoint(i_camera->m_renderer, (int)result.x, (int)result.y);
            }
        }

        Vector2 CircleRenderer::Center() {
            return m_center;
        }
        float CircleRenderer::Radius() {
            return m_radius;
        }

        void CircleRenderer::SetCenter(Vector2 i_center) {
            m_center = i_center;
        }
        void CircleRenderer::SetRadius(float i_radius) {
            m_radius = i_radius;
        }

        std::string CircleRenderer::Serialize() {
            std::string colResult = "";
            for (int i = 0; i < this->m_color.size(); i++) {
                colResult += SERIALIZE_ROOT(float, "\"" + std::to_string(this->m_color[i]) + "\"");
                if (i + 1 < this->m_color.size()) {
                    colResult += ",";
                }
            }
            return "{" SERIALIZE(Center, Vector2, "" + Center().Serialize() + "") 
                + ", " + SERIALIZE(Radius, float, "\"" + std::to_string(Radius()) + "\"")
                + ", " SERIALIZE(Color, list, "[" + colResult + "]") "}";
        }
    }
}