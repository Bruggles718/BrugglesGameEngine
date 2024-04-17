#include "TextureManager.hpp"
#include <SDL_image.h>

namespace bruggles {
    void TextureFunctorDeleter::operator()(SDL_Texture* texture) const {
        SDL_DestroyTexture(texture);
    }

    std::shared_ptr<SDL_Texture> make_shared_texture(SDL_Renderer* i_renderer, SDL_Surface* i_pixels) {
        SDL_Texture* tex = SDL_CreateTextureFromSurface(i_renderer, i_pixels);
        return std::shared_ptr<SDL_Texture>(tex, TextureFunctorDeleter());
    }

    std::shared_ptr<SDL_Texture> TextureManager::LoadTexture(SDL_Renderer* i_renderer, std::string i_filepath) {
        if (m_textureResources.find(i_filepath) == m_textureResources.end()) {
            SDL_Surface* pixels = IMG_Load(i_filepath.c_str());
            if (pixels == nullptr) {
                return nullptr;
            }
            //SDL_SetColorKey(pixels, SDL_TRUE, SDL_MapRGB(pixels->format, 255, 0, 255));

            std::shared_ptr<SDL_Texture> tex = make_shared_texture(i_renderer, pixels);

            if (tex == nullptr) {
                SDL_Log("Could not load texture");
                return nullptr;
            }

            m_textureResources.insert({ i_filepath, tex });

            SDL_FreeSurface(pixels);

            //SDL_Log("Created new resource %s", i_filepath.c_str());
        }
        else {
            //SDL_Log("Reused resource %s", i_filepath);
        }

        return m_textureResources[i_filepath];
    }
}