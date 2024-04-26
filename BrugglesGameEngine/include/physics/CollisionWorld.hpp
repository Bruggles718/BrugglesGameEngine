#pragma once

#include <vector>
#include "physics/CollisionObject.cuh"
#include "physics/Solver.hpp"
#include "SDL.h"
#include <unordered_set>

typedef uint64_t Uint64;

namespace bruggles {
    namespace physics {
        /**
         * Represents a physics world with no forces
        */
        class CollisionWorld {
        public:
            void AddCollisionObject(CollisionObject* i_object);

            void RemoveCollisionObject(CollisionObject* i_object);

            void AddSolver(Solver* solver);

            void RemoveSolver(Solver* solver);

            void SolveCollisions(
                std::vector<Collision>& collisions,
                float i_deltaTime
            );

            void SendCollisionCallbacks(
                std::vector<Collision>& collisions,
                float i_deltaTime
            );

            void ResolveCollisions(float i_deltaTime);

            void RenderColliders(Camera* i_camera);

            void RemoveAllObjects();

            Uint64 GenerateUniqueID();

            std::vector<std::pair<CollisionObject*, CollisionObject*>> GetSweepAndPrunePairs();

        protected:
            std::vector<CollisionObject*> m_objects;
            std::vector<Solver*> m_solvers;

            Uint64 m_nextUniqueID = 0;
        };

        struct EndPoint {
            CollisionObject* object;
            Uint64 id = 0;
            float value = 0.0f;
            bool isMin = false;

            bool operator==(const EndPoint& other) const;
        };
    }
}