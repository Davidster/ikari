use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::*;

pub struct AssetLoader {
    pub pending_assets: Arc<Mutex<Vec<String>>>,
    pub loaded_assets: Arc<Mutex<HashMap<String, (Scene, RenderBuffers)>>>,
    pub renderer_base: Arc<Mutex<BaseRendererState>>,
    // pub renderer_data: Arc<Mutex<RendererStatePublicData>>,
}

impl AssetLoader {
    pub fn new(renderer_base: Arc<Mutex<BaseRendererState>>) -> Self {
        Self {
            pending_assets: Arc::new(Mutex::new(Vec::new())),
            loaded_assets: Arc::new(Mutex::new(HashMap::new())),
            renderer_base,
        }
    }

    pub fn load_asset(&self, path: &str) {
        let pending_assets_clone = self.pending_assets.clone();
        let mut pending_assets_clone_guard = pending_assets_clone.lock().unwrap();
        pending_assets_clone_guard.push(path.to_string());

        if pending_assets_clone_guard.len() == 1 {
            let pending_assets = self.pending_assets.clone();
            let loaded_assets = self.loaded_assets.clone();
            let renderer_base = self.renderer_base.clone();
            // let renderer_data = self.renderer_data.clone();

            // start a thread to load the assets
            std::thread::spawn(move || {
                while pending_assets.lock().unwrap().len() > 0 {
                    let next_asset = pending_assets.lock().unwrap().remove(0);

                    let do_load = || {
                        std::thread::sleep(std::time::Duration::from_secs_f32(5.0));
                        let (document, buffers, images) = gltf::import(&next_asset)?;
                        let (other_scene, other_render_buffers) = build_scene(
                            &mut renderer_base.lock().unwrap(),
                            (&document, &buffers, &images),
                        )?;
                        anyhow::Ok((other_scene, other_render_buffers))
                    };
                    match do_load() {
                        Ok(result) => {
                            let _replaced_ignored =
                                loaded_assets.lock().unwrap().insert(next_asset, result);
                        }
                        Err(err) => {
                            logger_log(&format!("Error loading asset {:?}: {:?}", next_asset, err));
                        }
                    }
                }
            });
        }
    }
}

pub struct GameState {
    pub scene: Scene,
    pub time_tracker: Option<TimeTracker>,
    pub state_update_time_accumulator: f32,
    pub is_playing_animations: bool,

    pub audio_manager: Option<AudioManager>,
    pub bgm_sound_index: usize,
    pub gunshot_sound_index: usize,
    pub gunshot_sound_data: SoundData,

    pub player_node_id: GameNodeId,

    pub point_lights: Vec<PointLightComponent>,
    pub point_light_node_ids: Vec<GameNodeId>,
    pub directional_lights: Vec<DirectionalLightComponent>,

    // store the previous state and next state and interpolate between them
    pub next_balls: Vec<BallComponent>,
    pub prev_balls: Vec<BallComponent>,
    pub actual_balls: Vec<BallComponent>,
    pub ball_node_ids: Vec<GameNodeId>,
    pub ball_pbr_mesh_index: usize,

    pub ball_spawner_acc: f32,

    pub test_object_node_id: GameNodeId,
    pub crosshair_node_id: Option<GameNodeId>,
    pub revolver: Option<Revolver>,

    pub bouncing_ball_node_id: GameNodeId,
    pub bouncing_ball_body_handle: RigidBodyHandle,

    pub physics_state: PhysicsState,

    pub physics_balls: Vec<PhysicsBall>,
    pub mouse_button_pressed: bool,

    pub character: Option<Character>,
    pub player_controller: PlayerController,

    pub asset_loader: AssetLoader,
}

impl GameState {
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
