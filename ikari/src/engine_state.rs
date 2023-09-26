use std::sync::{Arc, Mutex};

use crate::{
    audio::{AudioManager, AudioStreams},
    light::DirectionalLightComponent,
    light::PointLightComponent,
    physics::PhysicsState,
    player_controller::PlayerController,
    renderer::{Renderer, SurfaceData},
    scene::{GameNodeDesc, GameNodeId, Scene},
    time_tracker::TimeTracker,
    ui_overlay::IkariUiOverlay,
};

pub struct EngineState {
    pub scene: Scene,
    pub time_tracker: Option<TimeTracker>,
    pub ui_overlay: IkariUiOverlay,
    pub player_node_id: GameNodeId,
    pub player_controller: PlayerController,
    pub physics_state: PhysicsState,
    pub audio_streams: AudioStreams,
    pub audio_manager: Arc<Mutex<AudioManager>>,
    pub point_lights: Vec<PointLightComponent>,
    pub directional_lights: Vec<DirectionalLightComponent>,
}

impl EngineState {
    pub fn new<F>(
        init_player_controller: F,
        renderer: &Renderer,
        surface_data: &SurfaceData,
        window: &winit::window::Window,
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

        let ui_overlay = {
            let surface_format = surface_data.surface_config.format;
            IkariUiOverlay::new(
                window,
                &renderer.base.device,
                &renderer.base.queue,
                // TODO: can I just pass surface_format here? seems it should be ok even if the surface is not srgb,
                // the renderer will take care of that contingency..? this code would be really bad for the user.
                if surface_format.is_srgb() {
                    surface_format
                } else {
                    wgpu::TextureFormat::Rgba16Float
                },
            )
        };

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
            ui_overlay,
        })
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
