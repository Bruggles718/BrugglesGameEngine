#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "physics/CollisionObject.cuh"
#include "physics/CollisionPoints.cuh"
#include <vector>

namespace bruggles {
	namespace physics {
		void GPUComputeCollisions(std::vector<std::pair<CollisionObject*, CollisionObject*>>& pairs, std::vector<CollisionPoints>& i_result);
	}
}