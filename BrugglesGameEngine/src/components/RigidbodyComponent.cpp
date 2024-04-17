#include "components/RigidbodyComponent.hpp"
#include "components/TransformComponent.hpp"
#include "physics/Collision.hpp"
#include "physics/CollisionEvent.hpp"
#include "Application.hpp"

namespace bruggles {
    namespace components {
        RigidbodyComponent::RigidbodyComponent() {
            m_rigidbody = std::make_shared<physics::Rigidbody>();
        }

        RigidbodyComponent::~RigidbodyComponent() {
            
        }

        void RigidbodyComponent::OnComponentRemoved() {
            GetGameObject()->GetApplication()->RemoveRigidbody(m_rigidbody.get());
        }

        void RigidbodyComponent::_SetGameObject(GameObject* i_object) {
            pybind11::object PyTf = pybind11::module::import("bruggles").attr("TransformComponent");
            pybind11::object pytfComp = i_object->GetComponent(PyTf);
            std::shared_ptr<Transform> tf = pytfComp.cast<bruggles::components::TransformComponent>().GetTransform();
            m_rigidbody->m_gameObject = i_object;
            m_rigidbody->transform = tf.get();

            m_rigidbody->OnCollision = [i_object](physics::Collision i_c, float i_deltaTime) {
                physics::CollisionEvent e;
                e.OtherCollider = i_c.B->collider->m_component;
                e.Points = i_c.Points;
                i_object->OnCollision(e, i_deltaTime);
            };

            i_object->GetApplication()->AddRigidbody(m_rigidbody.get());

            pybind11::object PyColl = pybind11::module::import("bruggles").attr("ColliderComponent");
            pybind11::object pyCollComp = i_object->GetComponent(PyColl);
            if (pyCollComp.ptr() != pybind11::cast<pybind11::none>(Py_None).ptr()) {
                pyCollComp.attr("SetGameObject")(i_object);
            }

            SetIsDynamic(true);
            SetIsTrigger(false);

            SetIsSimulated(true);

            SetVelocity(Vector2(0, 0));
            SetForce(Vector2(0, 0));
            SetMass(1);

            SetGravity(Vector2(0, 0));
            SetTakesGravity(true);

            SetStaticFriction(0.5f);
            SetDynamicFriction(0.5f);
            SetRestitution(0.5f);

            SetDrag(0.05f);

        }

        bool RigidbodyComponent::IsDynamic() {
            return m_rigidbody->IsDynamic;
        }
        bool RigidbodyComponent::IsTrigger() {
            return m_rigidbody->IsTrigger;
        }

        bool RigidbodyComponent::IsSimulated() {
            return m_rigidbody->IsSimulated;
        }

        Vector2 RigidbodyComponent::Velocity() {
            return m_rigidbody->Velocity;
        }

        Vector2 RigidbodyComponent::Force() {
            return m_rigidbody->Force;
        }

        float RigidbodyComponent::Mass() {
            return m_rigidbody->Mass;
        }

        Vector2 RigidbodyComponent::Gravity() {
            return m_rigidbody->Gravity;
        }

        bool RigidbodyComponent::TakesGravity() {
            return m_rigidbody->TakesGravity;
        }

        float RigidbodyComponent::StaticFriction() {
            return m_rigidbody->StaticFriction;
        }

        float RigidbodyComponent::DynamicFriction() {
            return m_rigidbody->DynamicFriction;
        }

        float RigidbodyComponent::Restitution() {
            return m_rigidbody->Restitution;
        }

        float RigidbodyComponent::Drag() {
            return m_rigidbody->Drag;
        }

        void RigidbodyComponent::SetIsDynamic(bool i_isDynamic) {
            m_rigidbody->IsDynamic = i_isDynamic;
        }

        void RigidbodyComponent::SetIsTrigger(bool i_isTrigger) {
            m_rigidbody->IsTrigger = i_isTrigger;
        }

        void RigidbodyComponent::SetIsSimulated(bool i_isSimulated) {
            m_rigidbody->IsSimulated = i_isSimulated;
        }

        void RigidbodyComponent::SetVelocity(Vector2 i_velocity) {
            m_rigidbody->Velocity = i_velocity;
        }

        void RigidbodyComponent::SetForce(Vector2 i_force) {
            m_rigidbody->Force = i_force;
        }

        void RigidbodyComponent::SetMass(float i_mass) {
            m_rigidbody->Mass = i_mass;
        }

        void RigidbodyComponent::SetGravity(Vector2 i_gravity) {
            m_rigidbody->Gravity = i_gravity;
        }

        void RigidbodyComponent::SetTakesGravity(bool i_takesGravity) {
            m_rigidbody->TakesGravity = i_takesGravity;
        }

        void RigidbodyComponent::SetStaticFriction(float i_staticFriction) {
            m_rigidbody->StaticFriction = i_staticFriction;
        }

        void RigidbodyComponent::SetDynamicFriction(float i_dynamicFriction) {
            m_rigidbody->DynamicFriction = i_dynamicFriction;
        }

        void RigidbodyComponent::SetRestitution(float i_restitution) {
            m_rigidbody->Restitution = i_restitution;
        }

        void RigidbodyComponent::SetDrag(float i_drag) {
            m_rigidbody->Drag = i_drag;
        }

        std::shared_ptr<physics::Rigidbody> RigidbodyComponent::GetRigidbody() {
            return m_rigidbody;
        }

        std::string RigidbodyComponent::Serialize() {
            return "{" SERIALIZE(IsDynamic, bool, "\"" + std::to_string(IsDynamic()) + "\"")
                + ", " + SERIALIZE(IsTrigger, bool, "\"" + std::to_string(IsTrigger()) + "\"")
                + ", " + SERIALIZE(IsSimulated, bool, "\"" + std::to_string(IsSimulated()) + "\"")
                + ", " + SERIALIZE(Velocity, Vector2, "" + Velocity().Serialize() + "")
                + ", " + SERIALIZE(Force, Vector2, "" + Force().Serialize() + "")
                + ", " + SERIALIZE(Mass, float, "\"" + std::to_string(Mass()) + "\"")
                + ", " + SERIALIZE(Gravity, Vector2, "" + Gravity().Serialize() + "")
                + ", " + SERIALIZE(TakesGravity, bool, "\"" + std::to_string(TakesGravity()) + "\"")
                + ", " + SERIALIZE(StaticFriction, float, "\"" + std::to_string(StaticFriction()) + "\"")
                + ", " + SERIALIZE(DynamicFriction, float, "\"" + std::to_string(DynamicFriction()) + "\"")
                + ", " + SERIALIZE(Restitution, float, "\"" + std::to_string(Restitution()) + "\"")
                + ", " + SERIALIZE(Drag, float, "\"" + std::to_string(Drag()) + "\"") "}";
        }
    }
}