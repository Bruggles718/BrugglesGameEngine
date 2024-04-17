#include "components/SpriteRenderer.hpp"
#include "Application.hpp"
#include "components/TransformComponent.hpp"

namespace bruggles {
    namespace components {
        SpriteRenderer::SpriteRenderer(std::string i_filePath) {
            m_filePath = i_filePath;
        }

        SpriteRenderer::SpriteRenderer() {
            m_filePath = "";
        }

        void SpriteRenderer::_SetGameObject(GameObject* i_object) {
            m_texture = i_object->GetApplication()->LoadTexture(m_filePath);
            pybind11::object PyTf = pybind11::module::import("bruggles").attr("TransformComponent");
            pybind11::object pytfComp = i_object->GetComponent(PyTf);
            std::shared_ptr<Transform> tf = pytfComp.cast<bruggles::components::TransformComponent>().GetTransform();
            m_transform = tf.get();
        }

        void SpriteRenderer::Render() {
            GetGameObject()->GetApplication()->RenderToCamera(this);
        }

        void SpriteRenderer::RenderToCamera(Camera* i_camera) {
            if (m_texture == nullptr) {
                return;
            }
            int w = -1;
            int h = -1;

            int centerX = 0;
            int centerY = 0;

            SDL_QueryTexture(m_texture.get(), nullptr, nullptr, &w, &h);
            w = w <= 0 ? 32 : w;
            h = h <= 0 ? 32 : h;

            Vector2 wh{(float)w, (float)h};
            Vector2 xy{(float)(0), (float)(0)};

            xy.ApplyTransform(m_transform);
            Transform invTf = i_camera->GetInverseTransform();
            xy.ApplyTransform(&invTf);

            wh.x = std::abs(wh.x * m_transform->Scale.x) * invTf.Scale.x;
            wh.y = std::abs(wh.y * m_transform->Scale.y) * invTf.Scale.y;

            SDL_Rect rect{(int)xy.x - (int)wh.x/2, (int)xy.y - (int)wh.y/2, (int)wh.x, (int)wh.y};

            float rot = m_transform->Rotation;
            rot += invTf.Rotation;

            SDL_RendererFlip flipHorizontal = SDL_FLIP_NONE;
            SDL_RendererFlip flipVertical = SDL_FLIP_NONE;
            
            if (m_transform->Scale.x < 0) {
                flipHorizontal = SDL_FLIP_HORIZONTAL;
            }

            if (m_transform->Scale.y < 0) {
                flipVertical = SDL_FLIP_VERTICAL;
            }

            SDL_RenderCopyEx(
                i_camera->m_renderer,
                m_texture.get(),
                nullptr,
                &rect,
                rot,
                nullptr,
                static_cast<SDL_RendererFlip>(flipHorizontal | flipVertical));
        }

        std::string SpriteRenderer::FilePath() {
            return m_filePath;
        }

        void SpriteRenderer::SetFilePath(std::string i_filePath) {
            m_filePath = i_filePath;
            m_texture = GetGameObject()->GetApplication()->LoadTexture(m_filePath);
        }

        std::string SpriteRenderer::Serialize() {
            return "{" SERIALIZE(FilePath, str, "\"" + FilePath() + "\"") "}";
        }
    }
}