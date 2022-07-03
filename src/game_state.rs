use cgmath::{Vector2, Vector3};

use super::*;

#[derive(Debug)]
pub struct GameState {
    pub scene: GameScene,
    pub time_tracker: Option<TimeTracker>,
    pub state_update_time_accumulator: f32,

    pub point_lights: Vec<PointLightComponent>,
    pub directional_lights: Vec<DirectionalLightComponent>,

    // store the previous state and next state and interpolate between them
    pub next_balls: Vec<BallComponent>,
    pub prev_balls: Vec<BallComponent>,
    pub actual_balls: Vec<BallComponent>,

    pub test_object_instances: Vec<MeshInstance>,
    pub plane_instances: Vec<MeshInstance>,
}

impl GameState {
    // TODO: move all this init code to somewhere else, it should happen on the first frame or something
    pub fn new(scene: GameScene) -> Self {
        let balls: Vec<_> = (0..500)
            .into_iter()
            .map(|_| {
                BallComponent::new(
                    Vector2::new(
                        -10.0 + rand::random::<f32>() * 20.0,
                        -10.0 + rand::random::<f32>() * 20.0,
                    ),
                    Vector2::new(
                        -1.0 + rand::random::<f32>() * 2.0,
                        -1.0 + rand::random::<f32>() * 2.0,
                    ),
                    0.5 + (rand::random::<f32>() * 0.75),
                    1.0 + (rand::random::<f32>() * 15.0),
                )
            })
            .collect();

        let directional_lights = vec![DirectionalLightComponent {
            position: Vector3::new(10.0, 5.0, 0.0) * 10.0,
            direction: Vector3::new(-1.0, -0.7, 0.0).normalize(),
            color: LIGHT_COLOR_A,
            intensity: 1.0,
        }];
        // let directional_lights: Vec<DirectionalLightComponent> = vec![];

        let mut point_lights = vec![
            PointLightComponent {
                transform: crate::transform::Transform::new(),
                color: LIGHT_COLOR_A,
                intensity: 1.0,
            },
            PointLightComponent {
                transform: crate::transform::Transform::new(),
                color: LIGHT_COLOR_B,
                intensity: 1.0,
            },
        ];
        // let mut point_lights: Vec<PointLightComponent> = vec![];
        if let Some(point_light_0) = point_lights.get_mut(0) {
            point_light_0
                .transform
                .set_scale(Vector3::new(0.05, 0.05, 0.05));
            point_light_0
                .transform
                .set_position(Vector3::new(0.0, 12.0, 0.0));
        }
        if let Some(point_light_1) = point_lights.get_mut(1) {
            point_light_1
                .transform
                .set_scale(Vector3::new(0.1, 0.1, 0.1));
            point_light_1
                .transform
                .set_position(Vector3::new(0.0, 15.0, 0.0));
        }

        let mut test_object_instances = vec![MeshInstance::new()];
        test_object_instances[0]
            .transform
            .set_position(Vector3::new(4.0, 10.0, 4.0));

        let mut plane_instances = vec![MeshInstance::new()];
        plane_instances[0].transform.set_scale(Vector3::new(
            ARENA_SIDE_LENGTH,
            1.0,
            ARENA_SIDE_LENGTH,
        ));

        Self {
            scene,
            time_tracker: None,
            state_update_time_accumulator: 0.0,

            point_lights,
            directional_lights,

            next_balls: balls.clone(),
            prev_balls: balls.clone(),
            actual_balls: balls,

            test_object_instances,
            plane_instances,
        }
    }

    pub fn on_frame_started(&mut self) {
        self.time_tracker = self.time_tracker.or_else(|| TimeTracker::new().into());
        if let Some(time_tracker) = &mut self.time_tracker {
            time_tracker.on_frame_started();
        }
    }

    pub fn time(&self) -> TimeTracker {
        self.time_tracker.unwrap_or_else(|| {
            panic!("Must call GameState::on_frame_started at least once before getting the time")
        })
    }
}
