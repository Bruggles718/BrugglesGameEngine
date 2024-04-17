#include "physics/CollisionWorld.hpp"
#include <iostream>

namespace bruggles {
    namespace physics {
        void CollisionWorld::AddCollisionObject(CollisionObject* i_object) {
            i_object->m_uniqueID = GenerateUniqueID();
            this->m_objects.push_back(i_object);
        }

        void CollisionWorld::RemoveCollisionObject(CollisionObject* i_object) {
            auto itr = std::find(m_objects.begin(), m_objects.end(), i_object);
            m_objects.erase(itr);
        }

        void CollisionWorld::AddSolver(Solver* solver) {
            this->m_solvers.push_back(solver);
        }

        void CollisionWorld::RemoveSolver(Solver* solver) {
            auto itr = std::find(m_solvers.begin(), m_solvers.end(), solver);
            m_solvers.erase(itr);
        }

        void CollisionWorld::SolveCollisions(
            std::vector<Collision>& collisions,
            float i_deltaTime
        ) {
            for (Solver* solver : m_solvers) {
                solver->Solve(collisions, i_deltaTime);
            }
        }

        void CollisionWorld::SendCollisionCallbacks(
            std::vector<Collision>& collisions,
            float i_deltaTime
        ) {
            for (Collision& collision : collisions) {
                auto& a = collision.A->OnCollision;

                if (a) {
                    a(collision, i_deltaTime);
                }
            }
        }

        void CollisionWorld::ResolveCollisions(float i_deltaTime) {
            std::vector<Collision> collisions;
            std::vector<Collision> triggers;

            for (CollisionObject* a : m_objects) {
                for (CollisionObject* b: m_objects) {
                    if (a == b) continue;
                    if (a->m_gameObject && !a->m_gameObject->IsActive()) continue;
                    if (b->m_gameObject && !b->m_gameObject->IsActive()) continue;

                    if (!a->collider || !b->collider) continue;

                    CollisionPoints points = a->collider->CheckCollision(
                        a->transform,
                        b->collider,
                        b->transform
                    );

                    Collision collision = Collision(a, b, points);

                    if (points.HasCollision) {
                        bool trigger = a->IsTrigger || b->IsTrigger;

                        if (trigger) {
                            triggers.emplace_back(a, b, points);
                        } else {
                            collisions.emplace_back(a, b, points);
                        }
                    }
                }
            }

            SolveCollisions(collisions, i_deltaTime);
            SendCollisionCallbacks(collisions, i_deltaTime);
            SendCollisionCallbacks(triggers, i_deltaTime);
        }

        void CollisionWorld::RenderColliders(Camera* i_camera) {
            for (CollisionObject* obj : m_objects) {
                if (obj) {
                    if (obj->collider) {
                        obj->collider->Render(obj->transform, i_camera);
                    }
                }
            }
        }

        void CollisionWorld::RemoveAllObjects() {
            m_objects.clear();
            m_nextUniqueID = 0;
        }

        Uint64 CollisionWorld::GenerateUniqueID() {
            Uint64 idToReturn = m_nextUniqueID;
            m_nextUniqueID++;
            return idToReturn;
        }
    }
}