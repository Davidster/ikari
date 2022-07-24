use std::time::Instant;

use cgmath::{Deg, Quaternion, Rad, Vector3};

use super::*;

// (0, 1], higher means it syncs with the camera more quickly
const CAMERA_FOLLOW_LERP_FACTOR: f32 = 0.8;
// (0, 1], higher means it sways for a shorter time
const WEAPON_SWAY_RESET_LERP_FACTOR: f32 = 0.3;
const MAX_SWAY_DEG: Deg<f32> = Deg(3.0);

#[derive(Debug)]
pub struct Revolver {
    animation_index: usize,
    cooldown: f32,
    last_fired_instant: Option<Instant>,

    pub node_id: GameNodeId,
    hand_node_id: GameNodeId,
    camera_node_id: GameNodeId,
    current_hand_transform: Option<crate::transform::Transform>,
    last_camera_horizontal_rotation: Option<Rad<f32>>,
    base_rotation: Quaternion<f32>,
    sway: Rad<f32>,
}

impl Revolver {
    pub fn new(
        scene: &mut Scene,
        camera_node_id: GameNodeId,
        model_node_id: GameNodeId,
        animation_index: usize,
        transform: crate::transform::Transform,
    ) -> Self {
        let node_id = scene
            .add_node(GameNodeDescBuilder::new().transform(transform).build())
            .id();
        let hand_node_id = scene.add_node(GameNodeDescBuilder::new().build()).id();
        scene.set_node_parent(node_id, hand_node_id);
        scene.set_node_parent(model_node_id, node_id);

        // let cooldown = scene.animations[animation_index].length_seconds;
        let cooldown = scene.animations[animation_index].length_seconds + 0.1;

        Self {
            animation_index,
            cooldown,
            last_fired_instant: None,

            node_id,
            hand_node_id,
            camera_node_id,
            current_hand_transform: None,
            last_camera_horizontal_rotation: None,
            sway: Rad(0.0),
            base_rotation: transform.rotation(),
        }
    }

    pub fn update(&mut self, player_view_direction: ControlledViewDirection, scene: &mut Scene) {
        let camera_transform = scene.get_node(self.camera_node_id).unwrap().transform;

        // update
        let new_hand_transform = match self.current_hand_transform {
            Some(current_hand_transform) => {
                let mut new_hand_transform = current_hand_transform;
                new_hand_transform.set_scale(camera_transform.scale());
                new_hand_transform.set_position(camera_transform.position());

                new_hand_transform.set_rotation(
                    current_hand_transform
                        .rotation()
                        .nlerp(camera_transform.rotation(), CAMERA_FOLLOW_LERP_FACTOR),
                );

                new_hand_transform
            }
            None => camera_transform,
        };

        // update sway
        let last_camera_horizontal_rotation = self
            .last_camera_horizontal_rotation
            .unwrap_or(player_view_direction.horizontal);
        let max_sway: Rad<f32> = MAX_SWAY_DEG.into();
        self.sway += Rad(
            (player_view_direction.horizontal - last_camera_horizontal_rotation)
                .0
                .min(max_sway.0)
                .max(-max_sway.0),
        );
        self.sway = Rad(lerp(self.sway.0, 0.0, WEAPON_SWAY_RESET_LERP_FACTOR));

        self.last_camera_horizontal_rotation = Some(player_view_direction.horizontal);
        self.current_hand_transform = Some(new_hand_transform);

        if let Some(hand_node) = scene.get_node_mut(self.hand_node_id) {
            hand_node.transform = new_hand_transform;
        }

        if let Some(node) = scene.get_node_mut(self.node_id) {
            node.transform.set_rotation(
                make_quat_from_axis_angle(Vector3::new(0.0, 0.0, 1.0), self.sway)
                    * self.base_rotation,
            )
        }
    }

    pub fn fire(&mut self, scene: &mut Scene) -> bool {
        if let Some(last_fired_instant) = self.last_fired_instant {
            if last_fired_instant.elapsed().as_secs_f32() < self.cooldown {
                return false;
            }
        }
        // if self.last_fired_instant.is_none()
        self.last_fired_instant = Some(Instant::now());
        scene.animations[self.animation_index].state.is_playing = true;
        scene.animations[self.animation_index]
            .state
            .current_time_seconds = 0.0;
        true
    }
}
