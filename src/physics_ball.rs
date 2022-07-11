use cgmath::Vector3;

use super::*;

const RESTITUTION: f32 = 0.9;

#[derive(Clone, Debug)]
pub struct PhysicsBall {
    node_index: usize,
    rigid_body_handle: RigidBodyHandle,
}

impl PhysicsBall {
    pub fn new(
        game_scene: &mut GameScene,
        physics_state: &mut PhysicsState,
        mesh: GameNodeMesh,
        position: Vector3<f32>,
        radius: f32,
    ) -> Self {
        let transform = TransformBuilder::new()
            .position(Vector3::new(position.x, position.y, position.z))
            .scale(Vector3::new(radius, radius, radius))
            .build();

        game_scene.nodes.push(
            GameNodeBuilder::new()
                .mesh(Some(mesh))
                .transform(transform)
                .build(),
        );
        let node_index = game_scene.nodes.len() - 1;

        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(vector![position.x, position.y, position.z])
            .build();
        let collider = ColliderBuilder::ball(radius)
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
            node_index,
            rigid_body_handle,
        }
    }

    pub fn new_random(
        game_scene: &mut GameScene,
        physics_state: &mut PhysicsState,
        mesh: GameNodeMesh,
    ) -> Self {
        let radius = 0.1 + (rand::random::<f32>() * 0.5);
        let position = Vector3::new(
            ARENA_SIDE_LENGTH * (rand::random::<f32>() * 2.0 - 1.0),
            radius * 2.0 + rand::random::<f32>() * 30.0 + 5.0,
            ARENA_SIDE_LENGTH * (rand::random::<f32>() * 2.0 - 1.0),
        );
        Self::new(game_scene, physics_state, mesh, position, radius)
    }

    pub fn update(&self, game_scene: &mut GameScene, physics_state: &PhysicsState) {
        let rigid_body = &physics_state.rigid_body_set[self.rigid_body_handle];
        let transform = &mut game_scene.nodes[self.node_index].transform;
        transform.apply_isometry(*rigid_body.position());
    }
}
