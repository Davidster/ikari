use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::*;

pub struct AssetLoader {
    pub renderer_base: Arc<BaseRendererState>,

    pub pending_gltf_scenes: Arc<Mutex<Vec<String>>>,
    pub loaded_gltf_scenes: Arc<Mutex<HashMap<String, (Scene, RenderBuffers)>>>,

    pub pending_audio: Arc<Mutex<Vec<String>>>,
    pub loaded_audio: Arc<Mutex<HashMap<String, SoundData>>>,

    pub device_sample_rate: u32,
}

impl AssetLoader {
    pub fn new(renderer_base: Arc<BaseRendererState>, device_sample_rate: u32) -> Self {
        Self {
            renderer_base,

            pending_gltf_scenes: Arc::new(Mutex::new(Vec::new())),
            loaded_gltf_scenes: Arc::new(Mutex::new(HashMap::new())),

            pending_audio: Arc::new(Mutex::new(Vec::new())),
            loaded_audio: Arc::new(Mutex::new(HashMap::new())),
            device_sample_rate,
        }
    }

    pub fn load_gltf_asset(&self, path: &str) {
        let pending_assets_clone = self.pending_gltf_scenes.clone();
        let mut pending_assets_clone_guard = pending_assets_clone.lock().unwrap();
        pending_assets_clone_guard.push(path.to_string());

        if pending_assets_clone_guard.len() == 1 {
            let pending_assets = self.pending_gltf_scenes.clone();
            let loaded_assets = self.loaded_gltf_scenes.clone();
            let renderer_base = self.renderer_base.clone();

            std::thread::spawn(move || {
                while pending_assets.lock().unwrap().len() > 0 {
                    let next_scene_path = pending_assets.lock().unwrap().remove(0);

                    let do_load = || {
                        let (document, buffers, images) = gltf::import(&next_scene_path)?;
                        let (other_scene, other_render_buffers) =
                            build_scene(&renderer_base, (&document, &buffers, &images))?;
                        anyhow::Ok((other_scene, other_render_buffers))
                    };
                    match do_load() {
                        Ok(result) => {
                            let _replaced_ignored = loaded_assets
                                .lock()
                                .unwrap()
                                .insert(next_scene_path, result);
                        }
                        Err(err) => {
                            logger_log(&format!(
                                "Error loading asset {:?}: {:?}",
                                next_scene_path, err
                            ));
                        }
                    }
                }
            });
        }
    }

    pub fn load_audio(&self, path: &str, format: AudioFileFormat) {
        let pending_audio_clone = self.pending_audio.clone();
        let mut pending_audio_clone_guard = pending_audio_clone.lock().unwrap();
        pending_audio_clone_guard.push(path.to_string());

        if pending_audio_clone_guard.len() == 1 {
            let pending_audio = self.pending_audio.clone();
            let loaded_audio = self.loaded_audio.clone();

            std::thread::spawn(move || {
                while pending_audio.lock().unwrap().len() > 0 {
                    let next_audio_path = pending_audio.lock().unwrap().remove(0);

                    let do_load = || {
                        let sound_data = match format {
                            AudioFileFormat::MP3 => {
                                AudioManager::decode_mp3(self.device_sample_rate, &next_audio_path)?
                            }
                            AudioFileFormat::WAV => {
                                AudioManager::decode_wav(self.device_sample_rate, &next_audio_path)?
                            }
                        };
                        anyhow::Ok(sound_data)
                    };
                    match do_load() {
                        Ok(result) => {
                            let _replaced_ignored =
                                loaded_audio.lock().unwrap().insert(next_audio_path, result);
                        }
                        Err(err) => {
                            logger_log(&format!(
                                "Error loading asset {:?}: {:?}",
                                next_audio_path, err
                            ));
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

    pub cube_mesh: BasicMesh,

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
