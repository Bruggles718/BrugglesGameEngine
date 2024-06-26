#include "physics/DynamicsWorld.hpp"
#include <math.h>
#include "components/TransformComponent.hpp"
#include <pybind11/pybind11.h>
#include <iostream>

namespace bruggles {
    namespace physics {
        void DynamicsWorld::AddRigidbody(Rigidbody* i_object) {
            /*i_object->m_uniqueID = GenerateUniqueID();
            this->m_objects.push_back(i_object);*/
            AddCollisionObject(i_object);
        }

        void DynamicsWorld::RemoveRigidbody(Rigidbody* i_object) {
            /*auto itr = std::find(m_objects.begin(), m_objects.end(), i_object);
            m_objects.erase(itr);*/
            RemoveCollisionObject(i_object);
        }

        void DynamicsWorld::Step(float i_deltaTime) {
            //auto t1 = std::chrono::high_resolution_clock::now();
            for (CollisionObject* obj : m_objects) {
                if (obj->m_gameObject && !obj->m_gameObject->IsActive()) continue;
                if (!obj->IsDynamic) continue;
                Rigidbody* body = static_cast<Rigidbody*>(obj);
                if (!body->IsSimulated) continue;
                body->UpdateLastTransform();
                if (body->TakesGravity) {
                    body->Force = body->Force + ((m_gravity * body->Mass));
                } else {
                    body->Force = body->Force + ((body->Gravity * body->Mass));
                }

                body->Velocity = body->Velocity + ((body->Force / body->Mass) * i_deltaTime);
                body->GetTransform().Position = body->GetTransform().Position + (body->Velocity * i_deltaTime);
                body->Velocity = body->Velocity * std::max(1 - i_deltaTime * body->Drag, 0.0f);

                body->Force = Vector2::Zero();

                /*if (body->m_gameObject) {
                    pybind11::object PyTf = pybind11::module::import("bruggles").attr("TransformComponent");
                    pybind11::object pytfComp = body->m_gameObject->GetComponent(PyTf);
                    components::TransformComponent tfComp = pytfComp.cast<bruggles::components::TransformComponent>();
                    std::shared_ptr<Transform> tf = tfComp.GetTransform();
                    tf->Position = body->GetTransform().Position;

                }*/
            }
            //auto t2 = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double, std::milli> ms_double = t2 - t1;
            //std::cout << "update objects: " << ms_double.count() << "ms\n";

            //t1 = std::chrono::high_resolution_clock::now();
            ResolveCollisions(i_deltaTime);
            //ResolveCollisions(i_deltaTime);
            //t2 = std::chrono::high_resolution_clock::now();
            //ms_double = t2 - t1;
            //std::cout << "resovle collisions twice: " << ms_double.count() << "ms\n";
        }

        void DynamicsWorld::SetGravity(Vector2 i_gravity) {
            m_gravity = i_gravity;
        }

        Vector2 DynamicsWorld::GetGravity() {
            return m_gravity;
        }
    }
}