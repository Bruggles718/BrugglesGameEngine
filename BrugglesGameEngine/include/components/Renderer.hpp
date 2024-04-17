#pragma once

#include "components/Component.hpp"
#include "Camera.hpp"
#include <SDL.h>

namespace bruggles {
    namespace components {
        /**
         * Provides the ability to display a Game Object in 2D space
        */
        class Renderer : public Component {
        public:
            /**
             * Used to render this Renderer. The Renderer will determine whether to render itself to world space or screen space.
            */
            virtual void Render();
            
            /**
             * Renders this Renderer to World Space relative to the given Camera.
            */
            virtual void RenderToCamera(Camera* camera);
            
            /**
             * Renders this Renderer to Screen Space.
            */
            virtual void RenderToScreen(SDL_Renderer* i_renderer);
            bool IsEnabled = true;
        };
    }
}