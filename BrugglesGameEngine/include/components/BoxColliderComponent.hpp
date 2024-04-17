#pragma once

#include "components/ColliderComponent.hpp"
#include <pybind11/pybind11.h>
#include "Serializable.hpp"

namespace bruggles {
    namespace components {
        /**
         * Represents a wrapper around a physics box collider to be attached to a Game Object
        */
        class BoxColliderComponent : public ColliderComponent, public Serializable {
        public:
            BoxColliderComponent();
            BoxColliderComponent(float i_width, float i_height);
            /**
             * Get the width of this box collider component
            */
            float Width();
            /**
             * Get the height of this box collider component
            */
            float Height();

            /**
             * Set the width of this box collider component
            */
            void SetWidth(float i_width);

            /**
             * Set the height of this box collider component
            */
            void SetHeight(float i_height);

            std::string Serialize() override;
        private:
            float m_width;
            float m_height;
        };
    }
}

