use std::sync::{Arc, Mutex};

use crate::{
    audio::{AudioManager, AudioStreams},
    physics::PhysicsState,
    renderer::DirectionalLight,
    renderer::PointLight,
    scene::Scene,
    time_tracker::TimeTracker,
};

pub struct EngineState {
    pub scene: Scene,
    pub(crate) time_tracker: Option<TimeTracker>,
    pub physics_state: PhysicsState,
    pub audio_streams: AudioStreams,
    pub audio_manager: Arc<Mutex<AudioManager>>,
    pub point_lights: Vec<PointLight>,
    pub directional_lights: Vec<DirectionalLight>,
}

impl EngineState {
    pub fn new() -> anyhow::Result<Self> {
        let (audio_manager, audio_streams) = AudioManager::new()?;

        let audio_manager_mutex = Arc::new(Mutex::new(audio_manager));

        Ok(EngineState {
            scene: Scene::default(),
            audio_streams,
            audio_manager: audio_manager_mutex,
            point_lights: vec![],
            directional_lights: vec![],
            time_tracker: None,
            physics_state: PhysicsState::new(),
        })
    }

    pub(crate) fn on_frame_started(&mut self) {
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
