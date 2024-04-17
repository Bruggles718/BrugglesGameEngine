#pragma once

#include "components/Renderer.hpp"
#include "Serializable.hpp"

namespace bruggles {
    namespace components {
        /**
         * Renders a Sprite for the Game Object this Sprite Renderer is attached to
        */
        class SpriteRenderer : public Renderer, public Serializable {
        public:
            SpriteRenderer();
            SpriteRenderer(std::string i_filePath);
            void Render() override;
            void RenderToCamera(Camera* camera) override;
            void _SetGameObject(GameObject* i_object) override;

            /**
             * Get the file path of the image to be loaded for this sprite
            */
            std::string FilePath();

            /**
             * Set the file path of the image to be loaded for this sprite
            */
            void SetFilePath(std::string i_filePath);

            std::string Serialize() override;
        private:
            std::string m_filePath = "";
            std::shared_ptr<SDL_Texture> m_texture;
            Transform* m_transform;
        };
    }
}