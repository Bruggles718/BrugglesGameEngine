#ifndef SDLGRAPHICSPROGRAM
#define SDLGRAPHICSPROGRAM

// ==================== Libraries ==================
// Depending on the operating system we use
// The paths to SDL are actually different.
// The #define statement should be passed in
// when compiling using the -D argument.
// This gives an example of how a programmer
// may support multiple platforms with different
// dependencies.
#include <SDL.h>

// The glad library helps setup OpenGL extensions.
//#include <glad/glad.h>

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include "Bruggles.hpp"

// Include the pybindings
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Creates a macro function that will be called
// whenever the module is imported into python
// 'bruggles' is what we 'import' into python.
// 'm' is the interface (creates a py::module object)
//      for which the bindings are created.
//  The magic here is in 'template metaprogramming'
PYBIND11_MODULE(bruggles, m){
    m.doc() = "our game engine as a library"; // Optional docstring

    m.def("Lerp", &bruggles::math::Lerp);

    py::class_<bruggles::Application>(m, "Application")
        .def(py::init<int,int>(), py::arg("i_w"), py::arg("i_h"))   // our constructor
        .def(py::init<int,int,bool>(), py::arg("i_w"), py::arg("i_h"), py::arg("i_editorMode"))   // our constructor
        .def(py::init<std::string, int,int,bool>(), py::arg("i_windowName"), py::arg("i_w"), py::arg("i_h"), py::arg("i_editorMode"))
        .def(py::init<std::string, int,int>(), py::arg("i_windowName"), py::arg("i_w"), py::arg("i_h"))
        .def("Loop", &bruggles::Application::Loop)
        .def("EditorUpdate", &bruggles::Application::EditorUpdate)
        .def("AddGameObject", &bruggles::Application::AddGameObject)
        .def("GetKey", &bruggles::Application::GetKey)
        .def("GetKeyDown", &bruggles::Application::GetKeyDown)
        .def("GetKeyUp", &bruggles::Application::GetKeyUp)
        .def("SetGravity", &bruggles::Application::SetGravity)
        .def("GetGravity", &bruggles::Application::GetGravity)
        .def_property("CameraPosition", &bruggles::Application::GetCameraPosition, &bruggles::Application::SetCameraPosition)
        .def("GetWidth", &bruggles::Application::GetWidth)
        .def("GetHeight", &bruggles::Application::GetHeight)
        .def("QueueSceneLoad", &bruggles::Application::QueueSceneLoad)
        .def("LoadSceneData", &bruggles::Application::LoadSceneData)
        .def("GetGameObjects", &bruggles::Application::GetGameObjects)
        .def("RemoveGameObjectAtIdx", &bruggles::Application::RemoveGameObjectAtIdx)
        .def("GetCurrentScene", &bruggles::Application::GetCurrentScene)
        .def("DuplicateGameObjectAtIdx", &bruggles::Application::DuplicateGameObjectAtIdx)
        .def("Time", &bruggles::Application::Time)
        .def("SwapGameObjectPosition", &bruggles::Application::SwapGameObjectPosition)
        .def("InstantiateGameObject", &bruggles::Application::InstantiateGameObject)
        .def("DestroyGameObjectAtIdx", &bruggles::Application::DestroyGameObjectAtIdx);
// We do not need to expose everything to our users!
//            .def("getSDLWindow", &SDLGraphicsProgram::getSDLWindow, py::return_value_policy::reference)
    py::class_<bruggles::GameObject, std::shared_ptr<bruggles::GameObject>>GameObject(m, "GameObject");
    GameObject
        .def(py::init<>())
        .def("AddComponent", &bruggles::GameObject::AddComponent)
        .def("GetComponent", &bruggles::GameObject::GetComponent)
        .def("GetComponents", &bruggles::GameObject::GetComponents)
        .def("SetActive", &bruggles::GameObject::SetActive)
        .def("IsActive", &bruggles::GameObject::IsActive)
        .def("GetApplication", &bruggles::GameObject::GetApplication)
        .def("Serialize", &bruggles::GameObject::Serialize)
        .def("RemoveComponentAtIdx", &bruggles::GameObject::RemoveComponentAtIdx)
        .def_property("Name", &bruggles::GameObject::Name, &bruggles::GameObject::SetName);

    py::class_<bruggles::components::Component>Component(m, "Component");
    Component
        .def(py::init<>())
        .def("GetGameObject", &bruggles::components::Component::GetGameObject)
        .def("SetGameObject", &bruggles::components::Component::SetGameObject)
        .def("GetComponent", &bruggles::components::Component::GetComponent)
        .def("OnComponentRemoved", &bruggles::components::Component::OnComponentRemoved);

    py::class_<bruggles::components::Behavior, bruggles::components::Component>(m, "Behavior")
        .def(py::init<>())
        .def("HasStarted", &bruggles::components::Behavior::HasStarted)
        .def("SetHasStarted", &bruggles::components::Behavior::SetHasStarted)
        .def_readwrite("IsEnabled", &bruggles::components::Behavior::IsEnabled);
    
    py::class_<bruggles::components::Renderer, bruggles::components::Component>Renderer(m, "Renderer");
        Renderer.def(py::init<>())
        .def("Render", &bruggles::components::Renderer::Render)
        .def_readwrite("IsEnabled", &bruggles::components::Renderer::IsEnabled);

    py::class_<bruggles::components::ShapeRenderer, bruggles::components::Renderer>(m, "ShapeRenderer")
        .def(py::init<>())
        .def_property("Color", &bruggles::components::ShapeRenderer::Color, &bruggles::components::ShapeRenderer::SetColor);

    py::class_<bruggles::components::PolygonRenderer, bruggles::components::ShapeRenderer>(m, "PolygonRenderer")
        .def(py::init<>())
        .def(py::init<std::array<int, 4>, std::vector<bruggles::Vector2>>(), py::arg("i_color"), py::arg("i_vertices"))
        .def_property("Vertices", &bruggles::components::PolygonRenderer::Vertices, &bruggles::components::PolygonRenderer::SetVertices)
        .def("Serialize", &bruggles::components::PolygonRenderer::Serialize);

    py::class_<bruggles::components::CircleRenderer, bruggles::components::ShapeRenderer>(m, "CircleRenderer")
        .def(py::init<>())
        .def(py::init<std::array<int, 4>, bruggles::Vector2, float>(), py::arg("i_color"), py::arg("i_center"), py::arg("i_radius"))
        .def_property("Center", &bruggles::components::CircleRenderer::Center, &bruggles::components::CircleRenderer::SetCenter)
        .def_property("Radius", &bruggles::components::CircleRenderer::Radius, &bruggles::components::CircleRenderer::SetRadius)
        .def("Serialize", &bruggles::components::CircleRenderer::Serialize);

    py::class_<bruggles::components::SpriteRenderer, bruggles::components::Renderer>(m, "SpriteRenderer")
        .def(py::init<>())
        .def(py::init<std::string>(), py::arg("i_filePath"))
        .def_property("FilePath", &bruggles::components::SpriteRenderer::FilePath, &bruggles::components::SpriteRenderer::SetFilePath)
        .def("Serialize", &bruggles::components::SpriteRenderer::Serialize);
    
    py::class_<bruggles::components::RigidbodyComponent, bruggles::components::Component>(m, "RigidbodyComponent")
        .def(py::init<>())
        .def_property("IsDynamic", &bruggles::components::RigidbodyComponent::IsDynamic, &bruggles::components::RigidbodyComponent::SetIsDynamic)
        .def_property("IsTrigger", &bruggles::components::RigidbodyComponent::IsTrigger, &bruggles::components::RigidbodyComponent::SetIsTrigger)
        .def_property("IsSimulated", &bruggles::components::RigidbodyComponent::IsSimulated, &bruggles::components::RigidbodyComponent::SetIsSimulated)
        .def_property("Velocity", &bruggles::components::RigidbodyComponent::Velocity, &bruggles::components::RigidbodyComponent::SetVelocity)
        .def_property("Force", &bruggles::components::RigidbodyComponent::Force, &bruggles::components::RigidbodyComponent::SetForce)
        .def_property("Mass", &bruggles::components::RigidbodyComponent::Mass, &bruggles::components::RigidbodyComponent::SetMass)
        .def_property("Gravity", &bruggles::components::RigidbodyComponent::Gravity, &bruggles::components::RigidbodyComponent::SetGravity)
        .def_property("TakesGravity", &bruggles::components::RigidbodyComponent::TakesGravity, &bruggles::components::RigidbodyComponent::SetTakesGravity)
        .def_property("StaticFriction", &bruggles::components::RigidbodyComponent::StaticFriction, &bruggles::components::RigidbodyComponent::SetStaticFriction)
        .def_property("DynamicFriction", &bruggles::components::RigidbodyComponent::DynamicFriction, &bruggles::components::RigidbodyComponent::SetDynamicFriction)
        .def_property("Restitution", &bruggles::components::RigidbodyComponent::Restitution, &bruggles::components::RigidbodyComponent::SetRestitution)
        .def_property("Drag", &bruggles::components::RigidbodyComponent::Drag, &bruggles::components::RigidbodyComponent::SetDrag)
        .def("Serialize", &bruggles::components::RigidbodyComponent::Serialize);

    py::class_<bruggles::Transform>(m, "Transform")
        .def(py::init<>())
        .def_readwrite("Position", &bruggles::Transform::Position)
        .def_readwrite("Rotation", &bruggles::Transform::Rotation)
        .def_readwrite("Scale", &bruggles::Transform::Scale);

    py::class_<bruggles::Vector2>(m, "Vector2")
        .def(py::init<>())
        .def(py::init<float, float>(), py::arg("i_x"), py::arg("i_y"))
        .def_readwrite("X", &bruggles::Vector2::x)
        .def_readwrite("Y", &bruggles::Vector2::y)
        .def("Magnitude", &bruggles::Vector2::Magnitude)
        .def("Normalize", &bruggles::Vector2::Normalize)
        .def("Normalized", &bruggles::Vector2::Normalized)
        .def("Transformed", &bruggles::Vector2::Transformed)
        .def("ApplyTransform", &bruggles::Vector2::ApplyTransform)
        .def_static("Distance", &bruggles::Vector2::Distance)
        .def_static("Dot", &bruggles::Vector2::Dot)
        .def_static("Angle", &bruggles::Vector2::Angle)
        .def_static("Zero", &bruggles::Vector2::Zero)
        .def_static("UnitX", &bruggles::Vector2::UnitX)
        .def_static("UnitY", &bruggles::Vector2::UnitY)
        .def("Serialize", &bruggles::Vector2::Serialize)
        .def("Lerp", &bruggles::Vector2::Lerp);

    py::class_<bruggles::components::TransformComponent, bruggles::components::Component>(m, "TransformComponent")
        .def(py::init<>())
        .def_property("Position", &bruggles::components::TransformComponent::Position, &bruggles::components::TransformComponent::SetPosition)
        .def_property("Rotation", &bruggles::components::TransformComponent::Rotation, &bruggles::components::TransformComponent::SetRotation)
        .def_property("Scale", &bruggles::components::TransformComponent::Scale, &bruggles::components::TransformComponent::SetScale)
        .def("Serialize", &bruggles::components::TransformComponent::Serialize);

    py::class_<bruggles::components::ColliderComponent, bruggles::components::Component>(m, "ColliderComponent")
        .def(py::init<>());

    py::class_<bruggles::components::BoxColliderComponent, bruggles::components::ColliderComponent>(m, "BoxColliderComponent")
        .def(py::init<>())
        .def(py::init<float, float>(), py::arg("i_width"), py::arg("i_height"))
        .def_property("Width", &bruggles::components::BoxColliderComponent::Width, &bruggles::components::BoxColliderComponent::SetWidth)
        .def_property("Height", &bruggles::components::BoxColliderComponent::Height, &bruggles::components::BoxColliderComponent::SetHeight)
        .def("Serialize", &bruggles::components::BoxColliderComponent::Serialize);

    py::class_<bruggles::components::CircleColliderComponent, bruggles::components::ColliderComponent>(m, "CircleColliderComponent")
        .def(py::init<>())
        .def(py::init<bruggles::Vector2, float>(), py::arg("i_center"), py::arg("i_radius"))
        .def_property("Center", &bruggles::components::CircleColliderComponent::Center, &bruggles::components::CircleColliderComponent::SetCenter)
        .def_property("Radius", &bruggles::components::CircleColliderComponent::Radius, &bruggles::components::CircleColliderComponent::SetRadius)
        .def("Serialize", &bruggles::components::CircleColliderComponent::Serialize);

    py::class_<bruggles::components::HullColliderComponent, bruggles::components::ColliderComponent>(m, "HullColliderComponent")
        .def(py::init<>())
        .def(py::init<std::vector<bruggles::Vector2>>(), py::arg("i_vertices"))
        .def_property("Vertices", &bruggles::components::HullColliderComponent::Vertices, &bruggles::components::HullColliderComponent::SetVertices)
        .def("Serialize", &bruggles::components::HullColliderComponent::Serialize);

    py::class_<bruggles::physics::CollisionPoints>(m, "CollisionPoints")
        .def_readonly("HasCollision", &bruggles::physics::CollisionPoints::HasCollision)
        .def_readonly("Normal", &bruggles::physics::CollisionPoints::Normal)
        .def_readonly("Depth", &bruggles::physics::CollisionPoints::Depth);

    py::class_<bruggles::physics::CollisionEvent>(m, "CollisionEvent")
        .def_readonly("OtherCollider", &bruggles::physics::CollisionEvent::OtherCollider)
        .def_readonly("Points", &bruggles::physics::CollisionEvent::Points);

    py::class_<bruggles::Deserializer>(m, "Deserializer")
        .def(py::init<>())
        .def_static("Init", &bruggles::Deserializer::Init)
        .def_static("Deserialize", &bruggles::Deserializer::Deserialize);
}

#endif
