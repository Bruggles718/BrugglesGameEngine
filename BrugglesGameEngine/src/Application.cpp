#include "Application.hpp"
#include "physics/ImpulseSolver.hpp"
#include "physics/PositionSolver.hpp"
#include "components/RigidbodyComponent.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include "Deserializer.hpp"
#include <SDL_image.h>
#include "components/TransformComponent.hpp"
#include <iostream>

namespace py = pybind11;

namespace bruggles {

    Application::Application(std::string i_windowName, int i_width, int i_height, bool i_editorMode) {
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) != 0) {
            SDL_Log("Unable to initialize SDL: %s",
                SDL_GetError());
            return;
        }

        if (IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG) == 0) {
            SDL_Log("Unable to initialize IMG: %s",
                IMG_GetError());
            return;
        } 

        m_width = i_width;
        m_height = i_height;
        m_editorMode = i_editorMode;

        m_window = SDL_CreateWindow(
            i_windowName.c_str(),
            50,
            50,
            m_width,
            m_height,
            SDL_WINDOW_OPENGL
        );

        SDL_SetWindowResizable(m_window, i_editorMode ? SDL_TRUE : SDL_FALSE);

        m_renderer = SDL_CreateRenderer(m_window, -1, SDL_RENDERER_ACCELERATED);

        if (m_renderer == nullptr) {
            SDL_Log("Error creating renderer");
        }

        m_run = true;

        m_camera = Camera(m_renderer, i_width, i_height);
    }

    Application::Application(int i_width, int i_height, bool i_editorMode) : Application::Application("My Application", i_width, i_height, i_editorMode) {}
    
    Application::~Application() {
        SDL_DestroyWindow(m_window);
        SDL_DestroyRenderer(m_renderer);
        SDL_Quit();
    }

    void Application::SetWindowTitle(std::string i_windowTitle) {
        SDL_SetWindowTitle(m_window, i_windowTitle.c_str());
    }

    Uint64 Application::Time() {
        return SDL_GetTicks64();
    }

    std::vector<std::shared_ptr<bruggles::GameObject>> Application::GetGameObjects() {
        return m_gameObjects;
    }

    void Application::QueueSceneLoad(std::string i_filePath) {
        m_loadScene = true;
        m_sceneToLoad = i_filePath;
    }

    std::string Application::GetCurrentScene() {
        return m_sceneToLoad;
    }

    void Application::DeserializeGameObjectHelper(py::handle gSerialized) {
        py::exec("import SerializeField");
        py::object bruggles = py::module_::import("bruggles");
        std::shared_ptr<GameObject> g = AddGameObject();
        g->SetName(gSerialized["Name"]["Value"].cast<std::string>());
        py::list components = gSerialized["Components"]["Value"];
        for (py::handle cSerialized : components) {
            std::string typeStr = cSerialized["Type"].cast<std::string>();
            std::string moduleStr = "bruggles";
            if (!py::hasattr(bruggles, typeStr.c_str())) {
                moduleStr = typeStr;
            }
            py::exec("import " + moduleStr);
            py::object c = g->GetComponent(py::eval(moduleStr + "." + typeStr));
            if (c.ptr() == pybind11::cast<pybind11::none>(Py_None).ptr()) {
                py::object cToAdd = py::eval(moduleStr + "." + typeStr + "()");
                g->AddComponent(cToAdd);
            }
            c = g->GetComponent(py::eval(moduleStr + "." + typeStr));
            if (py::hasattr(c, "Serialize")) {
                for (py::handle field : cSerialized["Value"]) {
                    if (Deserializer::CanDeserialize(cSerialized["Value"][field]["Type"].cast<std::string>())) {
                        std::string deserializedString = Deserializer::Deserialize(cSerialized["Value"][field]["Type"].cast<std::string>(), cSerialized["Value"][field]);
                        py::setattr(c, field.cast<std::string>().c_str(), py::eval(deserializedString));
                    }
                }
            }
        }
    }

    void Application::LoadSceneData(std::string i_filePath) {
        this->m_gameObjects.clear();
        this->m_physicsWorld.RemoveAllObjects();
        this->m_nextUniqueID = 0;
        Deserializer::Init();
        py::object json = py::module_::import("json");
        py::object f = py::eval("open(\"" + i_filePath + "\", \"r\")");
        py::dict sceneDict = json.attr("load")(f);

        py::dict worldDict = sceneDict["world"];
        this->SetCameraPosition(Vector2(worldDict["CameraPosition"]["X"].cast<float>(), worldDict["CameraPosition"]["Y"].cast<float>()));
        this->SetGravity(Vector2(worldDict["Gravity"]["X"].cast<float>(), worldDict["Gravity"]["Y"].cast<float>()));

        py::list sceneList = sceneDict["scene"];

        for (py::handle gSerialized : sceneList) {
            DeserializeGameObjectHelper(gSerialized);
        }
    }

    void Application::DuplicateGameObjectAtIdx(int i_idx) {
        std::shared_ptr<GameObject> g = GetGameObjects()[i_idx];
        py::object json = py::module_::import("json");
        py::dict gSerialized = json.attr("loads")(g->Serialize());
        DeserializeGameObjectHelper(gSerialized);
    }

    float Application::GetWidth() {
        return (float)m_width;
    }

    float Application::GetHeight() {
        return (float)m_height;
    }

    void Application::SetGravity(Vector2 i_gravity) {
        m_physicsWorld.SetGravity(i_gravity);
    }

    Vector2 Application::GetGravity() {
        return m_physicsWorld.GetGravity();
    }

    std::shared_ptr<bruggles::GameObject> Application::AddGameObject() {
        std::shared_ptr<bruggles::GameObject> g = std::make_shared<bruggles::GameObject>();
        g->SetApplication(this);
        this->m_gameObjects.push_back(g);
        return g;
    }

    void Application::SwapGameObjectPosition(int aIdx, int bIdx) {
        std::shared_ptr<GameObject> objAtAIdx = m_gameObjects[aIdx];
        m_gameObjects[aIdx] = m_gameObjects[bIdx];
        m_gameObjects[bIdx] = objAtAIdx;
    }

    void Application::RemoveGameObjectAtIdx(int i_idx) {
        m_gameObjects[i_idx]->OnGameObjectRemoved();
        m_gameObjects.erase(m_gameObjects.begin() + i_idx);
    }

    bool Application::GetKey(const std::string key) {
        const Uint8* state = SDL_GetKeyboardState(nullptr);

        if (state[SDL_GetScancodeFromName(key.c_str())]) {
            return true;
        }
        return false;
    }

    bool Application::GetKeyDown(const std::string key) {
        if (GetKey(key) && m_lastKeyboardState[key] == 0) {
            return true;
        }
        return false;
    }

    bool Application::GetKeyUp(const std::string key) {
        if (!GetKey(key) && m_lastKeyboardState[key] == 1) {
            return true;
        }
        return false;
    }

    Vector2 Application::GetCameraPosition() {
        return this->m_camera.transform->Position;
    }

    void Application::SetCameraPosition(Vector2 i_position) {
        this->m_camera.transform->Position = i_position;
    }

    void Application::RenderToCamera(components::Renderer* i_renderer) {
        i_renderer->RenderToCamera(&m_camera);
    }

    void Application::RenderToScreen(components::Renderer* i_renderer) {
        i_renderer->RenderToScreen(m_renderer);
    }

    std::shared_ptr<SDL_Texture> Application::LoadTexture(std::string i_filePath) {
        return m_textureManager.LoadTexture(m_renderer, i_filePath.c_str());
    }

    void Application::Input() {
        SDL_Event e;
         //Handle events on queue
		while(SDL_PollEvent( &e ) != 0){
        	// User posts an event to quit
	        // An example is hitting the "x" in the corner of the window.
    	    if(e.type == SDL_QUIT){
        		m_run = false;
	        }
      	} // End SDL_PollEvent loop.
        const Uint8* state = SDL_GetKeyboardState(nullptr);
    }

    void Application::Update(float i_deltaTime) {
        for (std::shared_ptr<bruggles::GameObject>& object : m_gameObjects) {
            if (!object->IsActive()) continue;
            object->Update(i_deltaTime);
        }
    }

    void Application::LateUpdate(float i_deltaTime) {
        for (std::shared_ptr<bruggles::GameObject>& object : m_gameObjects) {
            if (!object->IsActive()) continue;
            object->LateUpdate(i_deltaTime);
        }
    }

    void Application::FixedUpdate(float i_fixedDeltaTime) {
        for (std::shared_ptr<bruggles::GameObject>& object : m_gameObjects) {
            if (!object->IsActive()) continue;
            object->FixedUpdate(i_fixedDeltaTime);
        }
        m_physicsWorld.Step(i_fixedDeltaTime);
    }

    void Application::Render() {
        SDL_SetRenderDrawColor(m_renderer, 0, 0, 0, 0xFF);
        SDL_RenderClear(m_renderer);

        for (std::shared_ptr<bruggles::GameObject>& object : m_gameObjects) {
            if (!object->IsActive()) continue;
            object->Render();
        }

        if (m_editorMode) {
            m_physicsWorld.RenderColliders(&m_camera);
        }

        m_physicsWorld.RenderQuadTree(&m_camera);

        SDL_RenderPresent(m_renderer);
    }

    void Application::AddRigidbody(physics::Rigidbody* i_rigidbody) {
        m_physicsWorld.AddRigidbody(i_rigidbody);
    }

    void Application::RemoveRigidbody(physics::Rigidbody* i_rigidbody) {
        m_physicsWorld.RemoveRigidbody(i_rigidbody);
    }

    void Application::EditorUpdate() {
        Input();

        Render();
        for (const auto & [ key, value ] : m_lastKeyboardState) {
            m_lastKeyboardState[key] = GetKey(key);
        }
    }

    
    std::shared_ptr<bruggles::GameObject> Application::InstantiateGameObject() {
        std::shared_ptr<bruggles::GameObject> g = std::make_shared<bruggles::GameObject>();
        g->SetApplication(this);
        this->m_gameObjectsToInstantiate.push_back(g);
        return g;
    }

    void Application::DestroyGameObjectAtIdx(int i_idx) {
        this->m_gameObjectsAtIdxToDestroy.push_back(i_idx);
    }

    void Application::Loop() {
        std::shared_ptr<physics::Solver> posSolver = std::make_shared<physics::PositionSolver>();
        std::shared_ptr<physics::Solver> impulseSolver = std::make_shared<bruggles::physics::ImpulseSolver>();
        m_physicsWorld.AddSolver(impulseSolver.get());
        m_physicsWorld.AddSolver(posSolver.get());

        float deltaTime = 0.01f;
        float fixedDeltaTime = 0.016f;

        float newTime = SDL_GetTicks64() / 1000.0f;
        float accumulator = 0.0f;
        float currentTime = newTime;

        while(m_run) {
            newTime = SDL_GetTicks64() / 1000.0f;
            deltaTime = newTime - currentTime;
            currentTime = newTime;

            if (m_loadScene) {
                LoadSceneData(m_sceneToLoad);
                m_loadScene = false;
            }

            Input();

            if (!m_editorMode) {
                
                Update(deltaTime);

                accumulator += deltaTime;

                while (accumulator >= fixedDeltaTime) {
                    FixedUpdate(fixedDeltaTime);
                    accumulator -= fixedDeltaTime;
                }

                LateUpdate(deltaTime);

                for (std::shared_ptr<bruggles::GameObject>& object : m_gameObjects) {
                    if (!object->IsActive()) {
                        continue;
                    }
                    pybind11::object PyRb = pybind11::module::import("bruggles").attr("RigidbodyComponent");
                    pybind11::object pyrbComp = object->GetComponent(PyRb);
                    if (pyrbComp.ptr() == pybind11::cast<pybind11::none>(Py_None).ptr()) {
                        continue;
                    }
                    components::RigidbodyComponent rbComp = pyrbComp.cast<bruggles::components::RigidbodyComponent>();
                    std::shared_ptr<physics::Rigidbody> rb = rbComp.GetRigidbody();

                    if (!rb->IsDynamic || !rb->IsSimulated) {
                        continue;
                    }

                    if (!object->m_transform) {
                        continue;
                    }
                    Transform* tf = object->m_transform;
                    tf->Position = Vector2::Lerp(rb->GetLastTransform().Position, rb->GetTransform().Position, accumulator / fixedDeltaTime);
                    tf->Position = rb->GetTransform().Position;
                }
            }

            Render();
            for (const auto & [ key, value ] : m_lastKeyboardState) {
                m_lastKeyboardState[key] = GetKey(key);
            }

            if (m_gameObjectsAtIdxToDestroy.size() > 0) {
                std::vector<std::shared_ptr<GameObject>> newList;
                for (int i = 0; i < m_gameObjects.size(); i++) {
                    if (std::find(m_gameObjectsAtIdxToDestroy.begin(), m_gameObjectsAtIdxToDestroy.end(), i) != m_gameObjectsAtIdxToDestroy.end()) {
                        newList.push_back(m_gameObjects[i]);
                    }
                }
                m_gameObjectsAtIdxToDestroy.clear();
                m_gameObjects = newList;
            }

            if (m_gameObjectsToInstantiate.size() > 0) {
                for (int i = 0; i < m_gameObjectsToInstantiate.size(); i++) {
                    m_gameObjects.push_back(m_gameObjectsToInstantiate[i]);
                }
                m_gameObjectsToInstantiate.clear();
            }

            SDL_Delay(1);
        }

        SDL_Log("Program ended");
    }

    Uint64 Application::GenerateUniqueID() {
        Uint64 idToReturn = m_nextUniqueID;
        m_nextUniqueID++;
        return idToReturn;
    }
}