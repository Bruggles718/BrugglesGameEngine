#include "physics/CollisionHelpers.cuh"
#include "physics/CollisionPoints.cuh"
#include "physics/CollisionObject.cuh"
#include <iostream>
#include "TDynamicArray.cuh"
#include "Transform.cuh"
#include "physics/CircleCollider.cuh"
#include "physics/Simplex.cuh"
#include "physics/PhysicsHelpers.cuh"

namespace bruggles {
	namespace physics {
		__global__ void ComputeCollision(Transform* i_at, Transform* i_bt, Collider** i_ac, Collider** i_bc, bool* i_hasCollision, Simplex* i_simplexes) {
			int i = threadIdx.x;

            auto result = GJK(i_bc[i], &i_bt[i], i_ac[i], &i_at[i]);

            i_hasCollision[i] = result.first;
            i_simplexes[i] = result.second;
		}

		void GPUComputeCollisions(std::vector<std::pair<CollisionObject*, CollisionObject*>>& pairs, std::vector<CollisionPoints>& i_result) {
            if (pairs.size() < 1) {
                return;
            }

            cudaDeviceSetLimit(cudaLimitMallocHeapSize, 100000);

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

            float* d_radii = 0;

            cudaError_t e;

            e = cudaMalloc(&d_firstList, pairs.size() * sizeof(Transform));
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << std::endl;
            }
            e = cudaMalloc(&d_secondList, pairs.size() * sizeof(Transform));
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << std::endl;
            }
            e = cudaMalloc(&d_firstColliderList, pairs.size() * sizeof(Collider*));
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << std::endl;
            }
            e = cudaMalloc(&d_secondColliderList, pairs.size() * sizeof(Collider*));
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << std::endl;
            }

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
            e = cudaGetLastError();
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << std::endl;
            }

            e = cudaMemcpy(d_firstColliderList, firstColliderList.data(), firstColliderList.size() * sizeof(Collider*), cudaMemcpyHostToDevice);
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << std::endl;
            }
            e = cudaMemcpy(d_secondColliderList, secondColliderList.data(), secondColliderList.size() * sizeof(Collider*), cudaMemcpyHostToDevice);
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << std::endl;
            }

            e = cudaMalloc((void**)&d_result, pairs.size() * sizeof(CollisionPoints));
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << std::endl;
            }
            e = cudaMalloc((void**)&d_radii, pairs.size() * sizeof(float));
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << std::endl;
            }
            e = cudaMemcpy(d_firstList, firstList.data(), firstList.size() * sizeof(Transform), cudaMemcpyHostToDevice);
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << std::endl;
            }
            e = cudaMemcpy(d_secondList, secondList.data(), secondList.size() * sizeof(Transform), cudaMemcpyHostToDevice);
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << std::endl;
            }

            bool* d_hasCollision = 0;
            e = cudaMalloc(&d_hasCollision, pairs.size() * sizeof(bool));
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << std::endl;
            }

            Simplex* d_simplexes = 0;
            e = cudaMalloc(&d_simplexes, pairs.size() * sizeof(Simplex));
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << std::endl;
            }

            // Launch kernel
            int numCollisions = pairs.size();
            ComputeCollision << <1, numCollisions >> > (d_firstList, d_secondList, d_firstColliderList, d_secondColliderList, d_hasCollision, d_simplexes);
            cudaDeviceSynchronize();
            /*e = cudaGetLastError();
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << std::endl;
            }*/
            // Copy result back to host
            /*std::vector<float> i_radii{};
            i_radii.resize(pairs.size());*/
            bool* i_hasCollision = new bool[pairs.size()];
            //cudaMemcpy(i_result.data(), d_result, i_result.size() * sizeof(CollisionPoints), cudaMemcpyDeviceToHost);
            Vector2* simplex = new Vector2[3];
            cudaMemcpy(i_hasCollision, d_hasCollision, pairs.size() * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(simplex, &d_simplexes[0].Vertices, sizeof(Vector2) * 3, cudaMemcpyDeviceToHost);

            //std::cout << std::boolalpha;

            for (int i = 0; i < i_result.size(); i++) {
                std::cout << i << " result: \nHasCollision: " << i_hasCollision[i] << std::endl;
                if (i_hasCollision) {
                    for (int i = 0; i < 3; i++) {
                        std::cout << simplex[i].x << ", " << simplex[i].y << std::endl;
                    }
                }
            }

            // Free device memory
            cudaFree(d_firstList);
            cudaFree(d_secondList);
            cudaFree(d_result);
            cudaFree(d_firstColliderList);
            cudaFree(d_secondColliderList);
            cudaFree(d_hasCollision);
            cudaFree(d_simplexes);
            delete[] i_hasCollision;
            delete[] simplex;
		}
	}
}