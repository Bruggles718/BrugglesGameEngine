#pragma once

#include <memory>
#include "Transform.hpp"
#include "Vector2.hpp"
#include <SDL.h>

namespace bruggles {
    /**
     * Represents a camera to be rendered to.
    */
    struct Camera {
        SDL_Renderer* m_renderer;
        std::shared_ptr<Transform> transform;
        float m_width;
        float m_height;

        Camera();

        Camera(SDL_Renderer* i_renderer, float i_width, float i_height);

        Transform GetInverseTransform() const;
    };
}