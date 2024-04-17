#pragma once

#include <SDL.h>
#include <memory>
#include <unordered_map>
#include <string>
namespace bruggles {
    struct TextureFunctorDeleter {
        void operator()(SDL_Texture* texture) const;
    };

    std::shared_ptr<SDL_Texture> make_shared_texture(SDL_Renderer* i_renderer, SDL_Surface* i_pixels);

    /**
     * Loads textures from images and stores them for re-use.
    */
    class TextureManager {
    public:
        TextureManager() {}

        std::shared_ptr<SDL_Texture> LoadTexture(SDL_Renderer* i_renderer, std::string i_filepath);

    private:
        
        SDL_Renderer* m_renderer;
        std::unordered_map<std::string, std::shared_ptr<SDL_Texture>> m_textureResources;
    };
}