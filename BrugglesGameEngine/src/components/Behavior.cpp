#include "components/Behavior.hpp"

namespace bruggles {
    namespace components {
        void Behavior::SetHasStarted() {
            if (m_hasStarted) return;
            m_hasStarted = true;
        }

        bool Behavior::HasStarted() {
            return m_hasStarted;
        }
    }
}