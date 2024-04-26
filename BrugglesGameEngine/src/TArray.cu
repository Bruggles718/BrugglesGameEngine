#include "TArray.cuh"
#include "Vector2.cuh"
#include <assert.h>
#include "physics/CollisionPoints.cuh"

namespace bruggles {
	template<typename T> 
	__host__ __device__ TArray<T>::TArray(int i_size) {
		m_data = new T[i_size];
		m_size = i_size;
	}

	template<typename T>
	__host__ __device__ TArray<T>::TArray(const TArray<T>& i_arr) : m_size(i_arr.m_size) {
		m_data = new T[m_size];
		for (int i = 0; i < m_size; ++i) {
			m_data[i] = i_arr.m_data[i];
		}
	}

	template<typename T>
	__host__ __device__ TArray<T>::~TArray() {
		delete[] m_data;
	}
	
	template<typename T> 
	__host__ __device__ TArray<T>::TArray(T i_arr[], int i_size) {
		m_data = new T[i_size];
		m_size = i_size;
		memcpy(m_data, i_arr, sizeof(T) * i_size);
	}

	template<typename T>
	__host__ __device__ int TArray<T>::Size() const {
		return m_size;
	}

	template<typename T>
	__host__ __device__ T& TArray<T>::operator[](int i_idx) const {
		assert(i_idx >= 0);
		assert(i_idx < m_size);
		return m_data[i_idx];
	}

	template<typename T>
	__host__ __device__ TArray<T>& TArray<T>::operator=(const TArray<T>& i_array) {
		if (this != &i_array) {
			// Deallocate existing memory
			delete[] m_data;

			// Copy size
			m_size = i_array.m_size;

			// Allocate new memory
			m_data = new T[m_size];

			// Copy elements
			for (int i = 0; i < m_size; ++i) {
				m_data[i] = i_array.m_data[i];
			}
		}
		return *this;
	}

	template TArray<Vector2>::TArray(int i_size);
	template TArray<Vector2>::TArray(const TArray<Vector2>& i_arr);
	template TArray<Vector2>::~TArray();
	template TArray<Vector2>::TArray(Vector2 i_arr[], int i_size);
	template int TArray<Vector2>::Size() const;
	template Vector2& TArray<Vector2>::operator[](int i_idx) const;
	template TArray<Vector2>& TArray<Vector2>::operator=(const TArray<Vector2>& i_array);
	template TArray<physics::CollisionPoints>::TArray(int i_size);
	template TArray<physics::CollisionPoints>::TArray(const TArray<physics::CollisionPoints>& i_arr);
	template TArray<physics::CollisionPoints>::~TArray();
	template TArray<physics::CollisionPoints>::TArray(physics::CollisionPoints i_arr[], int i_size);
	template int TArray<physics::CollisionPoints>::Size() const;
	template physics::CollisionPoints& TArray<physics::CollisionPoints>::operator[](int i_idx) const;
	template TArray<physics::CollisionPoints>& TArray<physics::CollisionPoints>::operator=(const TArray<physics::CollisionPoints>& i_array);
	template TArray<int>::TArray(int i_size);
	template TArray<int>::TArray(const TArray<int>& i_arr);
	template TArray<int>::~TArray();
	template TArray<int>::TArray(int i_arr[], int i_size);
	template int TArray<int>::Size() const;
	template int& TArray<int>::operator[](int i_idx) const;
	template TArray<int>& TArray<int>::operator=(const TArray<int>& i_array);

}