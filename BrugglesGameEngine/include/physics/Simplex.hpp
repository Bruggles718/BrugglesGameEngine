#pragma once

#include <array>
#include "Vector2.hpp"
#include <stddef.h>

namespace bruggles {
    namespace physics {
        /**
         * The simplest shape possible for capturing an area in a 2D space.
        */
        struct Simplex {
            std::array<Vector2, 3> Vertices;

            void Push_Front(Vector2& vertex);

            Vector2& operator[](unsigned i);

            size_t Size();

            Simplex& operator=(std::initializer_list<Vector2> list);

        private:
            size_t m_size = 0;
        };
    }
}