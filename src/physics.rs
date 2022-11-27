use std::collections::HashMap;

use cgmath::{Vector3, Vector4};
use rapier3d::prelude::*;

use super::*;

pub struct PhysicsState {
    pub gravity: nalgebra::Vector3<f32>,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: BroadPhase,
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
            broad_phase: BroadPhase::new(),
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

    pub fn step(&mut self) {
        puffin::profile_function!();

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
            &(),
            &(),
        );

        self.query_pipeline.update(
            &self.island_manager,
            &self.rigid_body_set,
            &self.collider_set,
        );
    }

    pub fn add_static_box(
        &mut self,
        scene: &Scene,
        renderer_state: &RendererState,
        node_id: GameNodeId,
    ) {
        #[allow(clippy::or_fun_call)]
        let collider_handles = self.static_box_set.entry(node_id).or_insert(vec![]);
        if let Some(node) = scene.get_node(node_id) {
            if let Some(mesh) = node.mesh.as_ref() {
                let transform: crate::transform::Transform =
                    scene.get_global_transform_for_node(node_id).into();
                let transform_decomposed = transform.decompose();
                for mesh_index in mesh.mesh_indices.iter() {
                    let bounding_box = match mesh.mesh_type {
                        GameNodeMeshType::Pbr { .. } => {
                            renderer_state.buffers.binded_pbr_meshes[*mesh_index]
                                .geometry_buffers
                                .bounding_box
                        }
                        GameNodeMeshType::Unlit { .. } => {
                            renderer_state.buffers.binded_unlit_meshes[*mesh_index].bounding_box
                        }
                    };
                    let bounding_box = (bounding_box.0, bounding_box.1);
                    let base_scale = (bounding_box.1 - bounding_box.0) / 2.0;
                    let base_position = (bounding_box.1 + bounding_box.0) / 2.0;
                    let scale = Vector3::new(
                        base_scale.x * transform_decomposed.scale.x,
                        base_scale.y * transform_decomposed.scale.y,
                        base_scale.z * transform_decomposed.scale.z,
                    );
                    let position_rotated = {
                        let rotated = make_rotation_matrix(transform_decomposed.rotation)
                            * Vector4::new(base_position.x, base_position.y, base_position.z, 1.0);
                        Vector3::new(rotated.x, rotated.y, rotated.z)
                    };
                    let position = Vector3::new(
                        position_rotated.x + transform_decomposed.position.x,
                        position_rotated.y + transform_decomposed.position.y,
                        position_rotated.z + transform_decomposed.position.z,
                    );
                    let rotation = transform_decomposed.rotation;
                    let mut collider = ColliderBuilder::cuboid(scale.x, scale.y, scale.z)
                        .collision_groups(
                            InteractionGroups::all()
                                .with_memberships(!COLLISION_GROUP_PLAYER_UNSHOOTABLE),
                        )
                        .friction(1.0)
                        .restitution(1.0)
                        .build();
                    collider.set_position(Isometry::from_parts(
                        nalgebra::Translation3::new(position.x, position.y, position.z),
                        nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                            rotation.s,
                            rotation.v.x,
                            rotation.v.y,
                            rotation.v.z,
                        )),
                    ));
                    collider_handles.push(self.collider_set.insert(collider));
                }
            }
        }
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
}
