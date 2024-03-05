use std::sync::{Arc, Mutex};

use crate::{
    audio::{AudioManager, AudioStreams},
    physics::PhysicsState,
    scene::Scene,
    time_tracker::TimeTracker,
};

pub struct EngineState {
    pub scene: Scene,
    pub(crate) time_tracker: TimeTracker,
    pub physics_state: PhysicsState,
    pub audio_streams: AudioStreams,
    pub audio_manager: Arc<Mutex<AudioManager>>,
}

impl EngineState {
    pub fn new() -> anyhow::Result<Self> {
        let (audio_manager, audio_streams) = AudioManager::new()?;

        let audio_manager_mutex = Arc::new(Mutex::new(audio_manager));

        Ok(EngineState {
            scene: Scene::default(),
            audio_streams,
            audio_manager: audio_manager_mutex,
            time_tracker: Default::default(),
            physics_state: PhysicsState::new(),
        })
    }

    pub fn time(&self) -> TimeTracker {
        self.time_tracker
    }
}
