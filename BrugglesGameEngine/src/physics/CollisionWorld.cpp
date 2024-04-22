#include "physics/CollisionWorld.hpp"
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include "GameObject.hpp"

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

        void BinaryInsert(std::vector<EndPoint>& points, EndPoint& pointToInsert) {
            int low = 0;
            int high = points.size() - 1;

            while (low <= high) {
                int mid = low + ((high - low) / 2);
                if (pointToInsert.value == points[mid].value) {
                    low = mid + 1;
                    break;
                }
                else if (pointToInsert.value > points[mid].value) {
                    low = mid + 1;
                }
                else {
                    high = mid - 1;
                }
            }

            points.insert(points.begin() + low, pointToInsert);
        }

        std::vector<std::pair<CollisionObject*, CollisionObject*>> CollisionWorld::GetSweepAndPrunePairs() {

            std::vector<EndPoint> xPoints{};
            xPoints.reserve(m_objects.size() * 2);
            std::vector<EndPoint> yPoints{};
            yPoints.reserve(m_objects.size() * 2);

            for (CollisionObject* object : m_objects) {
                if (!object->collider) {
                    continue;
                }

                std::pair<Vector2, Vector2> topLeftBottomRight = object->TopLeftBottomRightAABB();

                // Get x min and max
                // Insert into list
                // Get y min and max
                // Insert into second list

                EndPoint minX{
                    object,
                    object->m_uniqueID,
                    topLeftBottomRight.first.x,
                    true
                };
                EndPoint maxX{
                    object,
                    object->m_uniqueID,
                    topLeftBottomRight.second.x,
                    false
                };
                EndPoint minY{
                    object,
                    object->m_uniqueID,
                    topLeftBottomRight.first.y,
                    true
                };
                EndPoint maxY{
                    object,
                    object->m_uniqueID,
                    topLeftBottomRight.second.y,
                    false
                };

                BinaryInsert(xPoints, minX);
                BinaryInsert(xPoints, maxX);
                BinaryInsert(yPoints, minY);
                BinaryInsert(yPoints, maxY);
            }

            // go through both lists, and compute pairs
            // store map of id to list of ids
            // for id, if id is in both lists, we have a collision

            std::unordered_map<Uint64, std::vector<EndPoint>> xPairs;
            std::unordered_map<Uint64, std::vector<EndPoint>> yPairs;

            std::vector<Uint64> insideListX{};

            for (int i = 0; i < xPoints.size(); i++) {
                EndPoint& xPoint = xPoints[i];
                if (xPoint.isMin) {
                    for (Uint64 insideID : insideListX) {
                        if (insideID != xPoint.id) {
                            xPairs[insideID].push_back(xPoint);
                        }
                    }
                    insideListX.push_back(xPoint.id);

                }
                else {
                    auto itr = std::find(insideListX.begin(), insideListX.end(), xPoint.id);
                    insideListX.erase(itr);
                }
            }

            for (EndPoint& e : xPoints) {
                auto& pairsWithE = xPairs[e.id];
                for (EndPoint& pairWithE : pairsWithE) {
                    xPairs[pairWithE.id].push_back(e);
                }
            }

            std::vector<Uint64> insideListY{};

            for (int i = 0; i < yPoints.size(); i++) {
                EndPoint& yPoint = yPoints[i];
                if (yPoint.isMin) {
                    for (Uint64 insideID : insideListY) {
                        if (insideID != yPoint.id) {
                            yPairs[insideID].push_back(yPoint);
                        }
                    }
                    insideListY.push_back(yPoint.id);
                }
                else {
                    auto itr = std::find(insideListY.begin(), insideListY.end(), yPoint.id);
                    insideListY.erase(itr);
                }
            }

            std::vector<std::pair<CollisionObject*, CollisionObject*>> result{};
            for (CollisionObject* object : m_objects) {
                auto& xPointsPotential = xPairs[object->m_uniqueID];
                auto& yPointsPotential = yPairs[object->m_uniqueID];
                std::vector<EndPoint> pairs{};
                std::unordered_set<Uint64> pairIDs{};
                for (auto& xE : xPointsPotential) {
                    for (auto& yE : yPointsPotential) {
                        if (xE.id == yE.id) {
                            pairs.push_back(xE);
                            break;
                        }
                    }
                }
                for (auto& pairObject : pairs) {
                    if (!pairIDs.contains(pairObject.id)) {
                        result.emplace_back(object, pairObject.object);
                        pairIDs.insert(pairObject.id);
                    }
                }
            }


            return result;
        }

        void CollisionWorld::ResolveCollisions(float i_deltaTime) {
            std::vector<Collision> collisions;
            std::vector<Collision> triggers;

            // Get pairs
            // calculate collisions for pairs
            std::vector<std::pair<CollisionObject*, CollisionObject*>> result = GetSweepAndPrunePairs();

            for (std::pair<CollisionObject*, CollisionObject*>& pair : result) {
                auto a = pair.first;
                auto b = pair.second;
                if (a == b) continue;
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

            SolveCollisions(collisions, i_deltaTime);
            SendCollisionCallbacks(collisions, i_deltaTime);
            SendCollisionCallbacks(triggers, i_deltaTime);
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