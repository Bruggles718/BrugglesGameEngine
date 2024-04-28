#include "TPair.cuh"

namespace bruggles {
	template<typename T, typename U>
	__host__ __device__ TPair<T, U>::TPair() {
		first = T();
		second = U();
	}

	template<typename T, typename U>
	__host__ __device__ TPair<T, U>::TPair(T i_first, U i_second) {
		first = i_first;
		second = i_second;
	}
}