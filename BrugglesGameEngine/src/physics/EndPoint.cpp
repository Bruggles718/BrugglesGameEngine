#include "physics/EndPoint.hpp"

namespace bruggles {
	namespace physics {
        void BinaryInsert(std::vector<EndPoint>& points, EndPoint& pointToInsert) {
            int low = 0;
            int high = points.size() - 1;

            while (low <= high) {
                int mid = low + ((high - low) / 2);
                if (pointToInsert.value == points[mid].value) {
                    low = mid + 1;
                    break;
                }
                else if (pointToInsert.value > points[mid].value) {
                    low = mid + 1;
                }
                else {
                    high = mid - 1;
                }
            }

            points.insert(points.begin() + low, pointToInsert);
        }
	}
}