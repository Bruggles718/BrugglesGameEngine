#include "MathHelpers.cuh"

namespace bruggles {
    namespace math {
        float Lerp(float a, float b, float t) {
            return a + (b - a) * t;
        }

        int Min(int a, int b) {
            if (a < b) {
                return a;
            }
            else {
                return b;
            }
        }
    }
}