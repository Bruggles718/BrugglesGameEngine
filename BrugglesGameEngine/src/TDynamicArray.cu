#include "TDynamicArray.cuh"
#include "Vector2.cuh"
#include <assert.h>
#include <iostream>
#include "physics/CollisionObject.cuh"
#include "physics/CollisionPoints.cuh"

namespace bruggles {
	template<typename T>
	__host__ __device__ TDynamicArray<T>::TDynamicArray() {
		int defaultCapacity = 2;
		m_data = new T[defaultCapacity];
		m_size = 0;
		m_capacity = defaultCapacity;
		memset(m_data, 0, sizeof(T) * m_capacity);
	}

	template<typename T>
	__host__ __device__ TDynamicArray<T>::TDynamicArray<T>(const TDynamicArray<T>& i_arr) {
		m_size = i_arr.m_size;
		m_capacity = i_arr.m_capacity;
		m_data = new T[m_capacity];
		for (int i = 0; i < m_size; ++i) {
			m_data[i] = i_arr.m_data[i];
		}
	}

	template<typename T>
	__host__ __device__ TDynamicArray<T>::~TDynamicArray() {
		delete[] m_data;
	}

	template<typename T>
	__host__ __device__ void TDynamicArray<T>::Resize(int i_newCapacity) {
		// Allocate new memory with the new capacity
		T* new_data = new T[i_newCapacity];

		// Copy existing elements to the new memory
		for (int i = 0; i < m_size; ++i) {
			new_data[i] = m_data[i];
		}

		// Deallocate the old memory
		delete[] m_data;

		// Update the capacity and data pointer
		m_capacity = i_newCapacity;
		m_data = new_data;
	}

	template<typename T>
	__host__ __device__ int TDynamicArray<T>::Size() const {
		return m_size;
	}

	template<typename T>
	__host__ __device__ T& TDynamicArray<T>::operator[](int i_idx) const {
		#ifdef  __CUDA_ARCH__
		assert(i_idx >= 0);
		assert(i_idx < m_size);
		#else
		if (i_idx < 0 || i_idx >= m_size) {
			throw std::out_of_range("i_idx out of bounds");
		}
		#endif
		return m_data[i_idx];
	}

	template<typename T>
	__host__ __device__ void TDynamicArray<T>::PushBack(const T& i_value) {
		if (m_size == m_capacity) {
			Resize(m_capacity * 2);
		}
		m_data[m_size] = i_value;
		m_size++;
	}

	template<typename T>
	__host__ __device__ void TDynamicArray<T>::PopBack() {
		assert(m_size > 0);
		m_size--;
	}

	template<typename T>
	__host__ __device__ void TDynamicArray<T>::Insert(int i_idx, const T& i_value) {
		assert(i_idx >= 0);
		assert(i_idx <= Size());
		if (m_size == m_capacity) {
			Resize(m_capacity * 2);
		}
		for (int i = Size(); i > i_idx; i--) {
			m_data[i] = m_data[i - 1];
		}
		m_data[i_idx] = i_value;
		m_size++;
	}

	template<typename T>
	__host__ __device__ void TDynamicArray<T>::Clear() {
		m_size = 0;
	}

	template<typename T>
	__host__ __device__ TDynamicArray<T>& TDynamicArray<T>::operator= (const TDynamicArray<T>& i_dynamicArray) {
		if (this != &i_dynamicArray) {
			// Deallocate existing memory
			delete[] m_data;

			// Copy size and capacity
			m_size = i_dynamicArray.m_size;
			m_capacity = i_dynamicArray.m_capacity;

			// Allocate new memory
			m_data = new T[m_capacity];

			// Copy elements
			for (int i = 0; i < m_size; ++i) {
				m_data[i] = i_dynamicArray.m_data[i];
			}
		}
		return *this;
	}

	template TDynamicArray<Vector2>::TDynamicArray();
	template TDynamicArray<Vector2>::TDynamicArray(const TDynamicArray<Vector2>& i_arr);
	template TDynamicArray<Vector2>::~TDynamicArray();
	template int TDynamicArray<Vector2>::Size() const;
	template Vector2& TDynamicArray<Vector2>::operator[](int i_idx) const;
	template void TDynamicArray<Vector2>::PushBack(const Vector2& i_value);
	template void TDynamicArray<Vector2>::PopBack();
	template void TDynamicArray<Vector2>::Insert(int i_idx, const Vector2& i_value);
	template void TDynamicArray<Vector2>::Clear();
	template void TDynamicArray<Vector2>::Resize(int i_newCapacity);
	template TDynamicArray<Vector2>& TDynamicArray<Vector2>::operator=(const TDynamicArray<Vector2>& i_array);
	template TDynamicArray<int>::TDynamicArray();
	template TDynamicArray<int>::TDynamicArray(const TDynamicArray<int>& i_arr);
	template TDynamicArray<int>::~TDynamicArray();
	template int TDynamicArray<int>::Size() const;
	template int& TDynamicArray<int>::operator[](int i_idx) const;
	template void TDynamicArray<int>::PushBack(const int& i_value);
	template void TDynamicArray<int>::PopBack();
	template void TDynamicArray<int>::Insert(int i_idx, const int& i_value);
	template void TDynamicArray<int>::Clear();
	template void TDynamicArray<int>::Resize(int i_newCapacity);
	template TDynamicArray<int>& TDynamicArray<int>::operator=(const TDynamicArray<int>& i_array);
	template TDynamicArray<physics::CollisionObject>::TDynamicArray();
	template TDynamicArray<physics::CollisionObject>::TDynamicArray(const TDynamicArray<physics::CollisionObject>& i_arr);
	template TDynamicArray<physics::CollisionObject>::~TDynamicArray();
	template int TDynamicArray<physics::CollisionObject>::Size() const;
	template physics::CollisionObject& TDynamicArray<physics::CollisionObject>::operator[](int i_idx) const;
	template void TDynamicArray<physics::CollisionObject>::PushBack(const physics::CollisionObject& i_value);
	template void TDynamicArray<physics::CollisionObject>::PopBack();
	template void TDynamicArray<physics::CollisionObject>::Insert(int i_idx, const physics::CollisionObject& i_value);
	template void TDynamicArray<physics::CollisionObject>::Clear();
	template void TDynamicArray<physics::CollisionObject>::Resize(int i_newCapacity);
	template TDynamicArray<physics::CollisionPoints>::TDynamicArray();
	template TDynamicArray<physics::CollisionPoints>::~TDynamicArray();
	template int TDynamicArray<physics::CollisionPoints>::Size() const;
	template physics::CollisionPoints& TDynamicArray<physics::CollisionPoints>::operator[](int i_idx) const;
	template void TDynamicArray<physics::CollisionPoints>::PushBack(const physics::CollisionPoints& i_value);
	template void TDynamicArray<physics::CollisionPoints>::PopBack();
	template void TDynamicArray<physics::CollisionPoints>::Insert(int i_idx, const physics::CollisionPoints& i_value);
	template void TDynamicArray<physics::CollisionPoints>::Clear();
	template void TDynamicArray<physics::CollisionPoints>::Resize(int i_newCapacity);
	template TDynamicArray<physics::CollisionPoints>& TDynamicArray<physics::CollisionPoints>::operator=(const TDynamicArray<physics::CollisionPoints>& i_array);


}