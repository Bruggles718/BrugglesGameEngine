#pragma once

#include "components/ColliderComponent.hpp"
#include <pybind11/pybind11.h>
#include "Serializable.hpp"

namespace bruggles {
    namespace components {
        /**
         * Represents a wrapper around a physics circle collider to be attached to a Game Object
        */
        class CircleColliderComponent : public ColliderComponent, public Serializable {
        public:
            CircleColliderComponent();
            CircleColliderComponent(Vector2 i_center, float i_radius);

            /**
             * Get the relative center of this circle collider component
            */
            Vector2 Center();

            /**
             * Get the radius of this circle collider component
            */
            float Radius();

            /**
             * Set the relative center of this circle collider component
            */
            void SetCenter(Vector2 i_center);

            /**
             * Set the radius of this circle collider component
            */
            void SetRadius(float i_radius);

            std::string Serialize() override;
        };
    }
}
