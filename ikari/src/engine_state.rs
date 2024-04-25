use crate::{
    asset_loader::{AssetBinder, AssetLoader},
    audio::{AudioManager, AudioStreams, IkariAudioStreams},
    framerate_limiter::FramerateLimiter,
    mutex::Mutex,
    physics::PhysicsState,
    scene::Scene,
    time_tracker::TimeTracker,
    wasm_not_sync::WasmNotArc,
};

use std::sync::Arc;

pub struct EngineState {
    pub scene: Scene,
    pub time_tracker: TimeTracker,
    pub framerate_limiter: FramerateLimiter,
    pub physics_state: PhysicsState,
    // TODO: maybe instead of box dyn trait, it could be good to just have the real audio manager have a 'disable' flag, which could allow it to respond to
    //       audio devices being disconnected later on.
    _audio_streams: Box<dyn AudioStreams>,
    pub audio_manager: Arc<Mutex<AudioManager>>,
    pub asset_loader: Arc<AssetLoader>,
    pub asset_binder: WasmNotArc<AssetBinder>,
}

impl EngineState {
    #[profiling::function]
    pub fn new() -> anyhow::Result<Self> {
        let (audio_manager, _audio_streams) = AudioManager::new()?;
        let audio_manager = Arc::new(Mutex::new(audio_manager));
        let asset_loader = Arc::new(AssetLoader::new(audio_manager.clone()));
        let asset_binder = WasmNotArc::new(AssetBinder::new());

        Ok(EngineState {
            scene: Scene::default(),
            _audio_streams,
            audio_manager,
            time_tracker: Default::default(),
            framerate_limiter: Default::default(),
            physics_state: PhysicsState::new(),
            asset_loader,
            asset_binder,
        })
    }
}
