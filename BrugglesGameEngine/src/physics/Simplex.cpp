#include "physics/Simplex.hpp"

namespace bruggles {
    namespace physics {
        void Simplex::Push_Front(Vector2& vertex) {
            Vertices = {vertex, Vertices[0], Vertices[1]};
            m_size = std::min<size_t>(m_size + 1, 3);
        }

        Vector2& Simplex::operator[](unsigned i) {
            return Vertices[i];
        }

        size_t Simplex::Size() {
            return this->m_size;
        }

        Simplex& Simplex::operator=(std::initializer_list<Vector2> list) {
            for (auto v = list.begin(); v != list.end(); v++) {
                Vertices[std::distance(list.begin(), v)] = *v;
            }
            this->m_size = list.size();

            return *this;
        }
    }
}