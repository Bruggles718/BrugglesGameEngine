#pragma once

#include <memory>
#include "physics/Rigidbody.hpp"
#include "components/Component.hpp"
#include "Serializable.hpp"

namespace bruggles {
    namespace components {
        /**
         * Represents a wrapper around a physics rigidbody that can be attached to a Game Object.
         * Only one should be attached per Game Object
        */
        class RigidbodyComponent : public Component, public Serializable {
        public:
            RigidbodyComponent();
            ~RigidbodyComponent();

            void OnComponentRemoved() override;

            /**
             * Get whether or not this Rigidbody Component is dynamic
            */
            bool IsDynamic();

            /**
             * Get whether or not this Rigidbody Component is a trigger
            */
            bool IsTrigger();

            /**
             * Get whether or not this Rigidbody Component should be simulated
            */
            bool IsSimulated();

            /**
             * Get the velocity of this Rigidbody Component
            */
            Vector2 Velocity();

            /**
             * Get the net force of this Rigidbody Component
            */
            Vector2 Force();

            /**
             * Get the mass of this Rigidbody Component
            */
            float Mass();

            /**
             * Get the individual unique gravity of this Rigidbody Component
            */
            Vector2 Gravity();

            /**
             * Get whether or not this Rigibody Component is using its Gravity or the physics world's gravity
            */
            bool TakesGravity();

            /**
             * Get the coefficient of static friction of this Rigidbody Component
            */
            float StaticFriction();

            /**
             * Get the coefficient of dynamic friction of this Rigidbody Component
            */
            float DynamicFriction();

            /**
             * Get the coefficient of restitution of this Rigidbody Component
            */
            float Restitution();

            /**
             * Get the drag factor of this Rigidbody Component
            */
            float Drag();


            /**
             * Set whether or not this Rigidbody Component is dynamic
            */
            void SetIsDynamic(bool i_isDynamic);

            /**
             * Set whether or not this Rigidbody Component is a trigger
            */
            void SetIsTrigger(bool i_isTrigger);

            /**
             * Set whether or not this Rigidbody Component should be simulated
            */
            void SetIsSimulated(bool i_isSimulated);

            /**
             * Set the velocity of this Rigidbody Component
            */
            void SetVelocity(Vector2 i_velocity);

            /**
             * Set the net force of this Rigidbody Component
            */
            void SetForce(Vector2 i_force);

            /**
             * Set the mass of this Rigidbody Component
            */
            void SetMass(float i_mass);

            /**
             * Set the individual unique gravity of this Rigidbody Component
            */
            void SetGravity(Vector2 i_gravity);

            /**
             * Set whether or not this Rigibody Component is using its Gravity or the physics world's gravity
            */
            void SetTakesGravity(bool i_takesGravity);

            /**
             * Set the coefficient of static friction of this Rigidbody Component
            */
            void SetStaticFriction(float i_staticFriction);

            /**
             * Set the coefficient of dynamic friction of this Rigidbody Component
            */
            void SetDynamicFriction(float i_dynamicFriction);

            /**
             * Set the coefficient of restitution of this Rigidbody Component
            */
            void SetRestitution(float i_restitution);

            /**
             * Set the drag factor of this Rigidbody Component
            */
            void SetDrag(float i_drag);

            /**
             * Get a shared_ptr to the physics Rigidbody this Rigidbody Component is wrapped around
            */
            std::shared_ptr<physics::Rigidbody> GetRigidbody();

            std::string Serialize() override;
        protected:
            void _SetGameObject(GameObject *i_object) override;
        private:
            std::shared_ptr<physics::Rigidbody> m_rigidbody;
        };
    }
}