use glam::f32::Vec3;
use ikari::physics::PhysicsState;
use ikari::scene::{GameNodeDescBuilder, GameNodeId, GameNodeVisual, Scene};

use ikari::physics::rapier3d_f64::prelude::*;
use ikari::transform::TransformBuilder;

use crate::game::{ARENA_SIDE_LENGTH, COLLISION_GROUP_PLAYER_UNSHOOTABLE};

const RESTITUTION: f64 = 0.1;

#[derive(Clone, Debug)]
pub struct PhysicsBall {
    node_id: GameNodeId,
    rigid_body_handle: RigidBodyHandle,
}

impl PhysicsBall {
    pub fn new(
        scene: &mut Scene,
        physics_state: &mut PhysicsState,
        mesh: GameNodeVisual,
        position: Vec3,
        radius: f32,
    ) -> Self {
        let transform = TransformBuilder::new()
            .position(Vec3::new(position.x, position.y, position.z))
            .scale(Vec3::new(radius, radius, radius))
            .build();

        let node = scene.add_node(
            GameNodeDescBuilder::new()
                .visual(Some(mesh))
                .transform(transform)
                .name(Some("physics_ball".to_string()))
                .build(),
        );

        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(vector![
                position.x as f64,
                position.y as f64,
                position.z as f64
            ])
            .build();
        let collider = ColliderBuilder::ball(radius as f64)
            .collision_groups(
                InteractionGroups::all().with_memberships(!COLLISION_GROUP_PLAYER_UNSHOOTABLE),
            )
            .restitution(RESTITUTION)
            .friction(1.0)
            .density(1.0)
            .build();
        let rigid_body_handle = physics_state.rigid_body_set.insert(rigid_body);

        physics_state.collider_set.insert_with_parent(
            collider,
            rigid_body_handle,
            &mut physics_state.rigid_body_set,
        );

        Self {
            node_id: node.id(),
            rigid_body_handle,
        }
    }

    pub fn new_random(
        scene: &mut Scene,
        physics_state: &mut PhysicsState,
        mesh: GameNodeVisual,
    ) -> Self {
        let radius = 0.2 + (rand::random::<f32>() * 0.4);
        let position = Vec3::new(
            ARENA_SIDE_LENGTH * (rand::random::<f32>() * 2.0 - 1.0) / 4.0,
            radius * 2.0 + rand::random::<f32>() * 15.0 + 5.0,
            ARENA_SIDE_LENGTH * (rand::random::<f32>() * 2.0 - 1.0) / 4.0,
        );
        Self::new(scene, physics_state, mesh, position, radius)
    }

    pub fn update(&self, scene: &mut Scene, physics_state: &mut PhysicsState) {
        if let Some(node) = scene.get_node_mut(self.node_id) {
            let rigid_body = &mut physics_state.rigid_body_set[self.rigid_body_handle];
            node.transform.apply_isometry(*rigid_body.position());
            if node.transform.position().y < -1.0 {
                self.destroy(scene, physics_state);
            }
        }
    }

    pub fn destroy(&self, scene: &mut Scene, physics_state: &mut PhysicsState) {
        scene.remove_node(self.node_id);
        physics_state.remove_rigid_body(self.rigid_body_handle);
    }

    pub fn _toggle_wireframe(&self, scene: &mut Scene) {
        if let Some(node) = scene.get_node_mut(self.node_id) {
            if let Some(mesh) = node.visual.as_mut() {
                mesh.wireframe = !mesh.wireframe;
            }
        }
    }

    pub fn rigid_body_handle(&self) -> RigidBodyHandle {
        self.rigid_body_handle
    }
}
