use std::time::Instant;

use super::*;

#[derive(Debug)]
pub struct Revolver {
    animation_index: usize,
    cooldown: f32,
    last_fired_instant: Option<Instant>,
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
        scene.set_node_parent(node_id, camera_node_id);
        scene.set_node_parent(model_node_id, node_id);

        let cooldown = scene.animations[animation_index].length_seconds;

        Self {
            animation_index,
            cooldown,
            last_fired_instant: None,
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
