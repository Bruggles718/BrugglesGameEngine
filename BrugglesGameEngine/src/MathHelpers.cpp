#include "MathHelpers.hpp"

namespace bruggles {
    namespace math {
        float Lerp(float a, float b, float t) {
            return a + (b - a) * t;
        }
    }
}