#include "physics/CollisionHelpers.cuh"
#include "physics/CollisionPoints.cuh"
#include "physics/CollisionObject.cuh"
#include <iostream>
#include "TDynamicArray.cuh"
#include "Transform.cuh"
#include "physics/CircleCollider.cuh"
#include "physics/Simplex.cuh"
#include "physics/PhysicsHelpers.cuh"
#include "math.h"
#include <chrono>

namespace bruggles {
	namespace physics {
		__global__ void ComputeCollision(Transform* i_at, Transform* i_bt, Collider** i_ac, Collider** i_bc, CollisionPoints* i_points) {
			int i = threadIdx.x;

            Transform at = i_at[i];
            Transform bt = i_bt[i];

            /*CircleCollider* cc1 = new CircleCollider(((CircleCollider*)i_ac[i])->Center, ((CircleCollider*)i_ac[i])->Radius );
            CircleCollider* cc2 = new CircleCollider(((CircleCollider*)i_bc[i])->Center, ((CircleCollider*)i_bc[i])->Radius);*/

            /*printf("a tf pos: (%f, %f)\n", at.Position.x, at.Position.y);
            printf("b tf pos: (%f, %f)\n", bt.Position.x, bt.Position.y);
            printf("a radius: %f\n", ((CircleCollider*)i_ac[i])->Radius);
            printf("b radius: %f\n", ((CircleCollider*)i_bc[i])->Radius);*/

            auto result = GJK(i_bc[i], &bt, i_ac[i], &at);

            //printf("%s\n", result.first ? "true" : "false");

            if (!result.first) {
                CollisionPoints* c = new CollisionPoints();
                i_points[i] = *c;
                return;
            }

            i_points[i] = EPA(result.second, i_bc[i], &bt, i_ac[i], &at);

            //i_points[i] = cc2->CheckCollisionWithCircleCollider(&bt, cc1, &at);

            //i_points[i] = i_ac[i]->CheckCollision(&at, i_bc[i], &bt);

            //i_hasCollision[i] = result.first;
            //i_simplexes[i] = result.second;
            //i_hasCollision[i] = false;
            /*Simplex s{};
            Vector2 vec{ 1, 1 };
            s.Push_Front(vec);
            s.Push_Front(vec);
            s.Push_Front(vec);*/

            //i_vertices = s.Vertices.m_data;
            //i_vertices = result.second.Vertices.m_data;

            /*for (int i = 0; i < s.Size(); i++) {
                printf("(%f, %f)\n", s[i].x, s[i].y);
            }*/
		}

		void GPUComputeCollisions(std::vector<std::pair<CollisionObject*, CollisionObject*>>& pairs, std::vector<CollisionPoints>& i_result) {
            if (pairs.size() < 1) {
                return;
            }
            auto t1 = std::chrono::high_resolution_clock::now();

            // Extract pointers from the vector of pairs
            i_result.resize(pairs.size());
            std::vector<Transform> firstList{};
            std::vector<Transform> secondList{};
            firstList.reserve(pairs.size());
            secondList.reserve(pairs.size());

            for (int i = 0; i < pairs.size(); i++) {
                firstList.push_back(pairs[i].first->GetTransform());
                secondList.push_back(pairs[i].second->GetTransform());
            }

            // Allocate device memory and copy data
            Transform* d_firstList = 0;
            Transform* d_secondList = 0;
            CollisionPoints* d_result = 0;

            Collider** d_firstColliderList = 0;
            Collider** d_secondColliderList = 0;

            cudaError_t e;

            e = cudaMalloc(&d_firstList, pairs.size() * sizeof(Transform));
            e = cudaMalloc(&d_secondList, pairs.size() * sizeof(Transform));
            e = cudaMalloc(&d_firstColliderList, pairs.size() * sizeof(Collider*));
            e = cudaMalloc(&d_secondColliderList, pairs.size() * sizeof(Collider*));

            std::vector<Collider*> firstColliderList;
            std::vector<Collider*> secondColliderList;
            firstColliderList.reserve(pairs.size());
            secondColliderList.reserve(pairs.size());

            for (int i = 0; i < pairs.size(); i++) {
                Collider* a = pairs[i].first->collider->GetDeviceCopy();
                Collider* b = pairs[i].second->collider->GetDeviceCopy();
                firstColliderList.push_back(a);
                secondColliderList.push_back(b);
            }

            e = cudaMemcpy(d_firstColliderList, firstColliderList.data(), firstColliderList.size() * sizeof(Collider*), cudaMemcpyHostToDevice);
            e = cudaMemcpy(d_secondColliderList, secondColliderList.data(), secondColliderList.size() * sizeof(Collider*), cudaMemcpyHostToDevice);

            e = cudaMalloc((void**)&d_result, pairs.size() * sizeof(CollisionPoints));
            e = cudaMemcpy(d_firstList, firstList.data(), firstList.size() * sizeof(Transform), cudaMemcpyHostToDevice);
            e = cudaMemcpy(d_secondList, secondList.data(), secondList.size() * sizeof(Transform), cudaMemcpyHostToDevice);
            auto t2 = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> ms_double = t2 - t1;
            std::cout << "memory copy: " << ms_double.count() << "ms\n";

            t1 = std::chrono::high_resolution_clock::now();
            // Launch kernel
            int numCollisions = pairs.size();
            ComputeCollision << <1, numCollisions >> > (d_firstList, d_secondList, d_firstColliderList, d_secondColliderList, d_result);
            t2 = std::chrono::high_resolution_clock::now();
            ms_double = t2 - t1;
            std::cout << "compute collisions: " << ms_double.count() << "ms\n";
            cudaDeviceSynchronize();
            //std::cout << "got here1\n";
            e = cudaGetLastError();
            if (e != cudaSuccess) {
                std::cout << "compute collision: " << cudaGetErrorString(e) << std::endl;
                return;
            }
            // Copy result back to host
            cudaMemcpy(i_result.data(), d_result, pairs.size() * sizeof(CollisionPoints), cudaMemcpyDeviceToHost);

            // Free device memory
            for (int i = 0; i < firstColliderList.size(); i++) {
                cudaFree(firstColliderList[i]);
                cudaFree(secondColliderList[i]);
            }
            cudaFree(d_firstList);
            cudaFree(d_secondList);
            cudaFree(d_result);
            cudaFree(d_firstColliderList);
            cudaFree(d_secondColliderList);
		}
	}
}