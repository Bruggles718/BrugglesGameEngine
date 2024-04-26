#pragma once

#include <memory>
#include "components/Component.hpp"
#include "physics/Collider.cuh"

namespace bruggles {
    namespace components {
        /**
         * Represents a wrapper around a physics collider that can be attached to a Game Object.
         * Only one collider should be attached per game object.
        */
        class ColliderComponent : public Component {
        public:
            void OnComponentRemoved() override;
        protected:
            void _SetGameObject(GameObject* i_object) override;
            std::shared_ptr<physics::Collider> m_collider;
        };
    }
}