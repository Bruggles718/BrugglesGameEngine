#include "Camera.hpp"

namespace bruggles {
    Camera::Camera() {}

    Camera::Camera(SDL_Renderer* i_renderer, float i_width, float i_height) {
        this->transform = std::make_shared<Transform>();
        this->m_renderer = i_renderer;
        this->m_width = i_width;
        this->m_height = i_height;
    }

    Transform Camera::GetInverseTransform() const {
        Transform result;
        result.Position = -(this->transform->Position) + Vector2(m_width/2, m_height/2);
        result.Rotation = -this->transform->Rotation;
        result.Scale = this->transform->Scale;
        result.Scale.x = 1/result.Scale.x;
        result.Scale.y = 1/result.Scale.y;
        return result;
    }
}