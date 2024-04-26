#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace bruggles {
	template<typename T>
	struct TArray {
		int m_size = 0;

		T* m_data;

		__host__ __device__ TArray(int i_size);
		__host__ __device__ TArray(const TArray& i_arr);
		__host__ __device__ ~TArray();
		__host__ __device__ TArray(T i_arr[], int i_size);
		__host__ __device__ int Size() const;
		__host__ __device__ T& operator[](int i_idx) const;
		__host__ __device__ TArray& operator= (const TArray& i_array);
	};
}