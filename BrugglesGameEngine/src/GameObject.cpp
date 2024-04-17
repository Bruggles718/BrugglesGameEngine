#include "GameObject.hpp"
#include "components/Behavior.hpp"
#include <pybind11/embed.h>
#include "Application.hpp"
#include "components/RigidbodyComponent.hpp"

namespace bruggles {
    GameObject::GameObject() {
        m_isActive = true;
        pybind11::object PyTransformComponent = pybind11::module::import("bruggles").attr("TransformComponent");
        pybind11::object tf = PyTransformComponent();
        AddComponent(tf);
    }

    void GameObject::AddComponent(pybind11::object i_component) {
        pybind11::object PyComp = pybind11::module::import("bruggles").attr("Component");
        assert(pybind11::isinstance(i_component, PyComp));
        m_components.push_back(i_component);
        pybind11::object PyBehavior = pybind11::module::import("bruggles").attr("Behavior");
        if (pybind11::isinstance(i_component, PyBehavior)) {
            m_behaviorIndicies.push_back((int)(m_components.size() - 1));
        } else {
            pybind11::object PyRenderer = pybind11::module::import("bruggles").attr("Renderer");
            if (pybind11::isinstance(i_component, PyRenderer)) {
                m_rendererIndicies.push_back((int)(m_components.size() - 1));
            }
        }
        i_component.attr("SetGameObject")(this);
    }

    void GameObject::RemoveComponentAtIdx(int i_idx) {
        m_components[i_idx].attr("OnComponentRemoved")();
        m_components.erase(m_components.begin() + i_idx);
        m_behaviorIndicies.clear();
        m_rendererIndicies.clear();
        for (int i = 0; i < m_components.size(); i++) {
            pybind11::object c = m_components[i];
            pybind11::object PyBehavior = pybind11::module::import("bruggles").attr("Behavior");
            if (pybind11::isinstance(c, PyBehavior)) {
                m_behaviorIndicies.push_back(i);
            } else {
                pybind11::object PyRenderer = pybind11::module::import("bruggles").attr("Renderer");
                if (pybind11::isinstance(c, PyRenderer)) {
                    m_rendererIndicies.push_back(i);
                }
            }
        }
    }

    pybind11::object GameObject::GetComponent(pybind11::object i_type) {
        for (int i = 0; i < m_components.size(); i++) {
            if (pybind11::isinstance(m_components[i], i_type)) {
                return m_components[i];
            }
        }
        return pybind11::cast<pybind11::none>(Py_None);
    }

    void GameObject::SetActive(bool i_active) {
        m_isActive = i_active;
    }

    bool GameObject::IsActive() {
        return m_isActive;
    }

    void GameObject::SetName(std::string i_name) {
        this->m_name = i_name;
    }

    std::string GameObject::Name() {
        return this->m_name;
    }

    void GameObject::SetApplication(Application* i_application) {
        this->m_application = i_application;
        this->m_uniqueID = i_application->GenerateUniqueID();
    }

    Application* GameObject::GetApplication() {
        return this->m_application;
    }

    void GameObject::Update(float i_deltaTime) {
        for (int behaviorIdx : m_behaviorIndicies) {
            pybind11::object behavior = m_components[behaviorIdx];
            if (pybind11::hasattr(behavior, "Start")) {
                if (!behavior.attr("HasStarted")().cast<bool>() && behavior.attr("IsEnabled").cast<bool>()) {
                    behavior.attr("Start")();
                    behavior.attr("SetHasStarted")();
                }
            }
            if (pybind11::hasattr(behavior, "Update")) {
                if (behavior.attr("IsEnabled").cast<bool>()) {
                    behavior.attr("Update")(i_deltaTime);
                }
            }
        }
    }

    void GameObject::LateUpdate(float i_deltaTime) {
        for (int behaviorIdx : m_behaviorIndicies) {
            pybind11::object behavior = m_components[behaviorIdx];
            if (pybind11::hasattr(behavior, "LateUpdate")) {
                if (behavior.attr("IsEnabled").cast<bool>()) {
                    behavior.attr("LateUpdate")(i_deltaTime);
                }
            }
        }
    }

    void GameObject::FixedUpdate(float i_fixedDeltaTime) {
        for (int behaviorIdx : m_behaviorIndicies) {
            pybind11::object behavior = m_components[behaviorIdx];
            if (pybind11::hasattr(behavior, "FixedUpdate")) {
                if (behavior.attr("IsEnabled").cast<bool>()) {
                    behavior.attr("FixedUpdate")(i_fixedDeltaTime);
                }
            }
        }
    }

    void GameObject::Render() {
        for (int rendererIdx : m_rendererIndicies) {
            pybind11::object renderer = m_components[rendererIdx];
            if (pybind11::hasattr(renderer, "Render")) {
                if (renderer.attr("IsEnabled").cast<bool>()) {
                    renderer.attr("Render")();
                }
            }
        }
    }

    void GameObject::OnCollision(physics::CollisionEvent i_event, float i_deltaTime) {
        for (int behaviorIdx : m_behaviorIndicies) {
            pybind11::object behavior = m_components[behaviorIdx];
            if (pybind11::hasattr(behavior, "OnCollision")) {
                if (behavior.attr("IsEnabled").cast<bool>()) {
                    behavior.attr("OnCollision")(i_event, i_deltaTime);
                }
            }
        }
    }

    Uint64 GameObject::GetUniqueID() {
        return m_uniqueID;
    }

    std::vector<pybind11::object> GameObject::GetComponents() {
        return m_components;
    }

    std::string GameObject::Serialize() {
        std::string result = "";
        for (int i = 0; i < m_components.size(); i++) {
            std::string serializedComponent = "\"\"";
            if (pybind11::hasattr(m_components[i], "Serialize")) {
                serializedComponent = m_components[i].attr("Serialize")().cast<std::string>();
            }
            result += 
                "{\"Type\": " "\"" + m_components[i].attr("__class__").attr("__name__").cast<std::string>() 
                + "\"" ", \"Value\": " + serializedComponent + "}";
            if (i + 1 < m_components.size()) {
                result += ",";
            }
        }
        return "{" SERIALIZE(Name, str, "\"" + Name() + "\"") 
        + "," + SERIALIZE(Components, list, "[" + result + "]") "}";
    }

    std::shared_ptr<GameObject> GameObject::getptr() {
        return shared_from_this();
    }

    void GameObject::OnGameObjectRemoved() {
        for (int i = 0; i < m_components.size(); i++) {
            m_components[i].attr("OnComponentRemoved")();
        }
        m_components.clear();
        return;
    }
}