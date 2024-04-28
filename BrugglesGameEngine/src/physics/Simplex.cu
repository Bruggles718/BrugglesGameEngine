#include "physics/Simplex.cuh"
#include "MathHelpers.cuh"
#include <iostream>

namespace bruggles {
    namespace physics {
        void Simplex::Push_Front(Vector2 vertex) {
            Vector2 arr[] = { vertex, Vertices[0], Vertices[1] };
            Vertices = TArray<Vector2>(arr, 3);
            m_size = math::Min((int)(m_size + 1), 3);
        }

        Vector2& Simplex::operator[](int i) const {
            return Vertices[i];
        }

        size_t Simplex::Size() const {
            return this->m_size;
        }

        Simplex& Simplex::operator=(const TArray<Vector2>& list) {
            for (int i = 0; i < list.Size(); i++) {
                Vertices[i] = list[i];
            }
            this->m_size = list.Size();

            return *this;
        }
    }
}