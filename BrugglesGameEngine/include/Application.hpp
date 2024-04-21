#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include "GameObject.hpp"
#include "physics/DynamicsWorld.hpp"
#include "components/Renderer.hpp"
#include "TextureManager.hpp"
#include "physics/Rigidbody.hpp"
#include <SDL.h>
#include <pybind11/pybind11.h>

typedef uint64_t Uint64;

namespace bruggles {
    /**
     * Represents an application created using this game engine
    */
    class Application {
    public:
        Application(int i_width, int i_height, bool i_editorMode = false);

        Application(std::string i_windowName, int i_width, int i_height, bool i_editorMode = false);

        ~Application();

        void SetWindowTitle(std::string i_windowTitle);

        /**
         * Main Application Loop when not running in editor mode
        */
        void Loop();

        /**
         * Unsafely adds a game object to this Application. Use with caution.
        */
        std::shared_ptr<bruggles::GameObject> AddGameObject();

        /**
         * Unsafely removes a game object from this Application. Use with caution.
        */
        void RemoveGameObjectAtIdx(int i_idx);

        /**
         * Returns whether or not the given key is being pressed
        */
        bool GetKey(const std::string key);

        /**
         * Gets the intended display-width of this Application
        */
        float GetWidth();

        /**
         * Gets the intended display-height of this Application
        */
        float GetHeight();

        /**
         * Sets the gravity of the physics world of this application
        */
        void SetGravity(Vector2 i_gravity);


        /**
         * Gets the gravity of the physics world of this application
        */
        Vector2 GetGravity();

        /**
         * Returns whether or not the given was pressed on this frame.
        */
        bool GetKeyDown(const std::string key);

        /**
         * Returns whether or not the given was released on this frame.
        */
        bool GetKeyUp(const std::string key);

        /**
         * Sets the position of the camera
        */
        void SetCameraPosition(Vector2 i_position);

        /**
         * Gets the position of the camera
        */
        Vector2 GetCameraPosition();

        void RenderToCamera(components::Renderer* i_renderer);

        void RenderToScreen(components::Renderer* i_renderer);

        /**
         * Loads and stores a texture at the the given filepath
        */
        std::shared_ptr<SDL_Texture> LoadTexture(std::string i_filePath);

        /**
         * Safely loads a scene when running the main application loop
        */
        void QueueSceneLoad(std::string i_filePath);

        /**
         * Unsafely loads a scene. Use with caution.
        */
        void LoadSceneData(std::string i_filePath);

        /**
         * Gets the filepath of the last scene that was queued to load
        */
        std::string GetCurrentScene();

        /**
         * Returns the list of game objects stored by this application
        */
        std::vector<std::shared_ptr<bruggles::GameObject>> GetGameObjects();

        /**
         * Adds a rigidbody to this application's physics world
        */
        void AddRigidbody(physics::Rigidbody* i_rigidbody);

        /**
         * Removes a rigidbody from this application's physics world
        */
        void RemoveRigidbody(physics::Rigidbody* i_rigidbody);

        /**
         * Generates a new unique id to be assigned to a game object
        */
        Uint64 GenerateUniqueID();

        /**
         * Duplicates the game object at the given index and unsafely adds it to this Application
        */
        void DuplicateGameObjectAtIdx(int i_idx);

        /**
         * Deserializes a serialized game object.
        */
        void DeserializeGameObjectHelper(pybind11::handle gSerialized);

        /**
         * Swaps the index in this Applications Game Object list of the game objects at the given indicies.
        */
        void SwapGameObjectPosition(int aIdx, int bIdx);

        /**
         * Gets the total elapsed time since this application was started.
        */
        Uint64 Time();

        /**
         * Renders and polls SDL input. Only used to view the application window while using the editor.
        */
        void EditorUpdate();

        /**
         * Safely adds a game object to this Application.
        */
        std::shared_ptr<bruggles::GameObject> InstantiateGameObject();

        /**
         * Safely removes a game object at the given index from this Application. Keep in mind that this operation can be expensive.
        */
        void DestroyGameObjectAtIdx(int i_idx);

    private:

        void Render();

        void Input();

        void Update(float i_deltaTime);

        void FixedUpdate(float i_fixedDeltaTime);

        void LateUpdate(float i_deltaTime);

        Camera m_camera;

        bool m_editorMode = false;
        bool m_run = true;
        SDL_Window* m_window;
        SDL_Renderer* m_renderer;

        TextureManager m_textureManager;

        std::string m_sceneToLoad = "";
        bool m_loadScene = false;

        std::unordered_map<std::string, int> m_lastKeyboardState;

        int m_width = 640;
        int m_height = 480;

        std::vector<std::shared_ptr<GameObject>> m_gameObjects;
        physics::DynamicsWorld m_physicsWorld;

        Uint64 m_nextUniqueID = 0;

        std::vector<std::shared_ptr<GameObject>> m_gameObjectsToInstantiate;
        std::vector<int> m_gameObjectsAtIdxToDestroy;
    };
}