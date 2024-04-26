#pragma once

#include <memory>
#include "Transform.cuh"
#include "Vector2.cuh"
#include "pybind11/pybind11.h"
#include "physics/Collision.hpp"
#include "physics/CollisionEvent.hpp"
#include "Serializable.hpp"
#include "physics/Rigidbody.hpp"

typedef uint64_t Uint64;

namespace bruggles {
    class Application;
    /**
     * Represents an object in 2D space.
    */
    class GameObject : public Serializable, public std::enable_shared_from_this<GameObject> {
    public:
        GameObject();

        void AddComponent(pybind11::object i_component);

        pybind11::object GetComponent(pybind11::object i_type);

        void SetApplication(Application* i_application);

        Application* GetApplication();
        
        void SetActive(bool i_active);

        bool IsActive();

        void Update(float i_deltaTime);

        void FixedUpdate(float i_fixedDeltaTime);

        void LateUpdate(float i_deltaTime);

        void Render();

        void OnCollision(physics::CollisionEvent i_event, float i_deltaTime);

        void SetName(std::string i_name);
        std::string Name();

        std::vector<pybind11::object> GetComponents();

        std::string Serialize() override;

        std::shared_ptr<GameObject> getptr();

        void RemoveComponentAtIdx(int i_idx);

        void OnGameObjectRemoved();

        Uint64 GetUniqueID();

        Transform* m_transform;

    private:
        std::string m_name;
        Uint64 m_uniqueID;
        bool m_isActive = true;
        std::vector<pybind11::object> m_components;
        std::vector<int> m_behaviorIndicies;
        std::vector<int> m_rendererIndicies;

        Application* m_application;
    };
}