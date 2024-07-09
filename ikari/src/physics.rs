use crate::scene::GameNodeId;

use std::collections::HashMap;

use rapier3d_f64::prelude::*;

pub use rapier3d_f64;

pub struct PhysicsState {
    pub gravity: nalgebra::Vector3<f64>,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    pub ccd_solver: CCDSolver,

    pub query_pipeline: QueryPipeline,

    pub static_box_set: HashMap<GameNodeId, Vec<ColliderHandle>>,
}

impl PhysicsState {
    pub fn new() -> Self {
        Self {
            gravity: vector![0.0, -9.8, 0.0],
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),

            query_pipeline: QueryPipeline::new(),

            static_box_set: HashMap::new(),
        }
    }

    #[profiling::function]
    pub fn step(&mut self) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            None,
            &(),
            &(),
        );

        self.query_pipeline.update(&self.collider_set);
    }

    pub fn remove_rigid_body(&mut self, rigid_body_handle: RigidBodyHandle) {
        self.rigid_body_set.remove(
            rigid_body_handle,
            &mut self.island_manager,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            true,
        );
    }

    pub fn set_gravity_is_enabled(&mut self, is_enabled: bool) {
        self.gravity = vector![0.0, if is_enabled { -9.8 } else { 0.0 }, 0.0];
    }
}

impl Default for PhysicsState {
    fn default() -> Self {
        Self::new()
    }
}
