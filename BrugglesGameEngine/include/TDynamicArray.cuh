#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "TArray.cuh"

namespace bruggles {
	template<typename T>
	struct TDynamicArray {
		int m_size = 0;
		int m_capacity = 2;

		T* m_data;

		__host__ __device__ TDynamicArray();
		__host__ __device__ TDynamicArray(const TDynamicArray& i_arr);
		__host__ __device__ ~TDynamicArray();
		__host__ __device__ int Size() const;
		__host__ __device__ T& operator[](int i_idx) const;
		__host__ __device__ void PushBack(const T& i_value);
		__host__ __device__ void PopBack();
		__host__ __device__ void Insert(int i_idx, const T& i_value);
		__host__ __device__ void Clear();
		__host__ __device__ TDynamicArray& operator= (const TDynamicArray& i_dynamicArray);
		__host__ __device__ void Resize(int i_newCapacity);
	};
}