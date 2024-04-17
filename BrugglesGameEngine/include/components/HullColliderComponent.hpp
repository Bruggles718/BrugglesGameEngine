#pragma once

#include "components/ColliderComponent.hpp"
#include <pybind11/pybind11.h>
#include "Serializable.hpp"

namespace bruggles {
    namespace components {
        /**
         * Represents a wrapper around a physics hull collider to be attached to a Game Object
        */
        class HullColliderComponent : public ColliderComponent, public Serializable {
        public:
            HullColliderComponent();
            HullColliderComponent(std::vector<Vector2> i_vertices);
            
            /**
             * Get the relative vertices of this hull collider component
            */
            std::vector<Vector2> Vertices();

            /**
             * Set the relative vertices of this hull collider component
            */
            void SetVertices(std::vector<Vector2> i_vertices);

            std::string Serialize() override;
        };
    }
}