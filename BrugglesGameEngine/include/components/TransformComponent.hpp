#pragma once

#include <memory>

#include "Transform.hpp"
#include "Vector2.hpp"
#include "components/Component.hpp"
#include "Serializable.hpp"
#include <physics/Rigidbody.hpp>

namespace bruggles {
    namespace components {
        /**
         * Represents a wrapper around a Transform that can be attached to a Game Object
        */
        class TransformComponent : public Component, public Serializable {
        public:
            TransformComponent();
            ~TransformComponent();

            /**
             * Get the position of this transform component
            */
            Vector2 Position();

            /**
             * Get the rotation of this transform component
            */
            float Rotation();

            /**
             * Get the scale of this transform component
            */
            Vector2 Scale();

            /**
             * Set the position of this transform component
            */
            void SetPosition(Vector2 i_position);

            /**
             * Set the rotation of this transform component
            */
            void SetRotation(float i_rotation);

            /**
             * Set the scale of this transform component
            */
            void SetScale(Vector2 i_scale);

            /**
             * Get a shared_ptr to the transform this component is wrapped around.
            */
            std::shared_ptr<Transform> GetTransform();

            std::string Serialize() override;

        private:
            std::shared_ptr<Transform> m_transform;
        };
    }
}