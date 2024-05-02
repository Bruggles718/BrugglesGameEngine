#include "physics/CollisionWorld.hpp"
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include "GameObject.hpp"
#include "physics/QuadTree.hpp"
#include "physics/EndPoint.hpp"
#include <chrono>

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
                auto& b = collision.B->OnCollision;

                if (a) {
                    a(collision, i_deltaTime);
                }

                if (b) {
                    Collision bCollision{};
                    bCollision.A = collision.B;
                    bCollision.B = collision.A;
                    CollisionPoints bCollisionPoints{};
                    bCollisionPoints.HasCollision = collision.Points.HasCollision;
                    bCollisionPoints.Depth = collision.Points.Depth;
                    bCollisionPoints.Normal = -collision.Points.Normal;
                    bCollision.Points = bCollisionPoints;
                    b(bCollision, i_deltaTime);
                }
            }
        }

        void Insert(std::vector<EndPoint>& points, EndPoint& pointToInsert) {
            for (int i = 0; i < points.size(); i++) {
                EndPoint& currentPoint = points[i];
                if (currentPoint.value > pointToInsert.value) {
                    points.insert(points.begin() + i, pointToInsert);
                    return;
                }
            }
            points.insert(points.begin() + points.size(), pointToInsert);
        }

        std::vector<std::pair<CollisionObject*, CollisionObject*>> CollisionWorld::GetSweepAndPrunePairs() {

            std::vector<EndPoint> xPoints{};
            xPoints.reserve(m_objects.size() * 2);
            std::vector<EndPoint> yPoints{};
            yPoints.reserve(m_objects.size() * 2);

            std::unordered_map<Uint64, std::pair<Vector2, Vector2>> tlbrs{};
            float minX = FLT_MAX;
            float maxX = FLT_MIN;
            float minY = FLT_MAX;
            float maxY = FLT_MIN;

            for (CollisionObject* object : m_objects) {
                if (!object->collider) {
                    continue;
                }

                std::pair<Vector2, Vector2> topLeftBottomRight = object->TopLeftBottomRightAABB();

                tlbrs[object->m_uniqueID] = topLeftBottomRight;

                // Get x min and max
                // Insert into list
                // Get y min and max
                // Insert into second list

                EndPoint minXEP{
                    object,
                    object->m_uniqueID,
                    topLeftBottomRight.first.x,
                    true
                };

                if (minX > minXEP.value) {
                    minX = minXEP.value;
                }

                EndPoint maxXEP{
                    object,
                    object->m_uniqueID,
                    topLeftBottomRight.second.x,
                    false
                };
                if (maxX < maxXEP.value) {
                    maxX = maxXEP.value;
                }
                EndPoint minYEP{
                    object,
                    object->m_uniqueID,
                    topLeftBottomRight.first.y,
                    true
                };
                if (minY > minYEP.value) {
                    minY = minYEP.value;
                }
                EndPoint maxYEP{
                    object,
                    object->m_uniqueID,
                    topLeftBottomRight.second.y,
                    false
                };
                if (maxY < maxYEP.value) {
                    maxY = maxYEP.value;
                }

                /*BinaryInsert(xPoints, minX);
                BinaryInsert(xPoints, maxX);
                BinaryInsert(yPoints, minY);
                BinaryInsert(yPoints, maxY);*/
            }



            m_quadTree = QuadTree(minX, minY, maxX - minX, maxY - minY, 8);
            for (CollisionObject* object : m_objects) {
                m_quadTree.Insert(object, tlbrs);
            }

            return m_quadTree.GetSweepAndPrunePairs(tlbrs);
        }

        void CollisionWorld::RenderQuadTree(Camera* i_camera) {
            m_quadTree.Render(i_camera);
        }

        void CollisionWorld::ResolveCollisions(float i_deltaTime) {
            // Get pairs
            // calculate collisions for pairs
            //auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<CollisionObject*, CollisionObject*>> result = GetSweepAndPrunePairs();
            //auto t2 = std::chrono::high_resolution_clock::now();

            //std::chrono::duration<double, std::milli> ms_double = t2 - t1;
            //std::cout << "sweep and prune: " << ms_double.count() << "ms\n";

            int iterations = 2;
            for (int i = 0; i < iterations; i++) {
                std::vector<Collision> collisions;
                std::vector<Collision> triggers;
                //t1 = std::chrono::high_resolution_clock::now();
                std::unordered_map<Uint64, std::unordered_set<Uint64>> computedCollisions{};
                for (std::pair<CollisionObject*, CollisionObject*>& pair : result) {
                    auto a = pair.first;
                    auto b = pair.second;
                    if (a == b) continue;
                    if (computedCollisions[a->m_uniqueID].contains(b->m_uniqueID) || computedCollisions[b->m_uniqueID].contains(a->m_uniqueID)) continue;
                    computedCollisions[a->m_uniqueID].insert(b->m_uniqueID);
                    computedCollisions[b->m_uniqueID].insert(a->m_uniqueID);
                    if (a->m_gameObject && !a->m_gameObject->IsActive()) continue;
                    if (b->m_gameObject && !b->m_gameObject->IsActive()) continue;

                    if (!a->collider || !b->collider) continue;

                    CollisionPoints points = a->collider->CheckCollision(
                        &a->GetTransform(),
                        b->collider,
                        &b->GetTransform()
                    );

                    Collision collision = Collision(a, b, points);

                    if (points.HasCollision) {
                        bool trigger = a->IsTrigger || b->IsTrigger;

                        if (trigger) {
                            triggers.emplace_back(a, b, points);
                        }
                        else {
                            collisions.emplace_back(a, b, points);
                        }
                    }
                }
                //t2 = std::chrono::high_resolution_clock::now();
                //ms_double = t2 - t1;
                //std::cout << "gjk + epa: " << ms_double.count() << "ms\n";

                //t1 = std::chrono::high_resolution_clock::now();
                SolveCollisions(collisions, i_deltaTime);
                //t2 = std::chrono::high_resolution_clock::now();
                //ms_double = t2 - t1;
                //std::cout << "solve collisions: " << ms_double.count() << "ms\n";
                SendCollisionCallbacks(collisions, i_deltaTime);
                SendCollisionCallbacks(triggers, i_deltaTime);
            }
        }

        void CollisionWorld::RenderColliders(Camera* i_camera) {
            for (CollisionObject* obj : m_objects) {
                if (obj) {
                    if (obj->collider) {
                        obj->collider->Render(&obj->GetTransform(), i_camera);
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