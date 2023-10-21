use std::sync::{Arc, Mutex};

// TODO: don't leak iced outside of the ui.rs file?

use crate::{
    audio::{AudioManager, AudioStreams},
    light::DirectionalLightComponent,
    light::PointLightComponent,
    physics::PhysicsState,
    player_controller::PlayerController,
    renderer::{Renderer, SurfaceData},
    scene::{GameNodeDesc, GameNodeId, Scene},
    time_tracker::TimeTracker,
};

pub struct EngineState {
    pub scene: Scene,
    pub(crate) time_tracker: Option<TimeTracker>,
    pub player_node_id: GameNodeId,
    pub player_controller: PlayerController, // TODO: move into game_state.rs, but leave the implementation in ikari
    pub physics_state: PhysicsState,
    pub audio_streams: AudioStreams,
    pub audio_manager: Arc<Mutex<AudioManager>>,
    pub point_lights: Vec<PointLightComponent>,
    pub directional_lights: Vec<DirectionalLightComponent>,
}

impl EngineState {
    pub fn new<F>(
        init_player_controller: F,
        _renderer: &Renderer,
        _surface_data: &SurfaceData,
        _window: &winit::window::Window,
    ) -> anyhow::Result<Self>
    where
        F: FnOnce(&mut PhysicsState) -> PlayerController,
    {
        let mut scene = Scene::default();
        let mut physics_state = PhysicsState::new();

        let player_node_id = scene.add_node(GameNodeDesc::default()).id();
        let player_controller = init_player_controller(&mut physics_state);

        let (audio_manager, audio_streams) = AudioManager::new()?;

        let audio_manager_mutex = Arc::new(Mutex::new(audio_manager));

        Ok(EngineState {
            scene,
            audio_streams,
            audio_manager: audio_manager_mutex,
            point_lights: vec![],
            directional_lights: vec![],
            time_tracker: None,
            physics_state,
            player_node_id,
            player_controller,
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
