#pragma once

namespace bruggles {
	template<typename T, typename U>
	struct TPair {
		T first;
		U second;

		__host__ __device__ TPair();
		__host__ __device__ TPair(T i_first, U i_second);
	};
}

#include "TPair_impl.cuh"