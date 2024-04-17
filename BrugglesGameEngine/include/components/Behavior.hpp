#pragma once

#include "components/Component.hpp"

namespace bruggles {
    namespace components {
        /**
         * Represents a scriptable behavior to be extended in python
        */
        class Behavior : public Component {
        public:
            bool HasStarted();
            void SetHasStarted();
            bool IsEnabled = true;
        private:
            bool m_hasStarted = false;
        };
    }
}