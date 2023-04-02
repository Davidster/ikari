use crate::audio::*;
use crate::gltf_loader::*;
use crate::logger::*;
use crate::renderer::*;
use crate::scene::*;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub struct AssetLoader {
    pub renderer_base: Arc<BaseRenderer>,
    pub pending_gltf_scenes: Arc<Mutex<Vec<String>>>,
    pub loaded_gltf_scenes: Arc<Mutex<HashMap<String, (Scene, RenderBuffers)>>>,

    pub audio_manager: Arc<Mutex<AudioManager>>,
    pub pending_audio: Arc<Mutex<Vec<(String, AudioFileFormat, SoundParams)>>>,
    pub loaded_audio: Arc<Mutex<HashMap<String, usize>>>,
}

impl AssetLoader {
    pub fn new(renderer_base: Arc<BaseRenderer>, audio_manager: Arc<Mutex<AudioManager>>) -> Self {
        Self {
            renderer_base,
            pending_gltf_scenes: Arc::new(Mutex::new(Vec::new())),
            loaded_gltf_scenes: Arc::new(Mutex::new(HashMap::new())),

            audio_manager,
            pending_audio: Arc::new(Mutex::new(Vec::new())),
            loaded_audio: Arc::new(Mutex::new(HashMap::new())),
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
                        let (other_scene, other_render_buffers) = build_scene(
                            &renderer_base,
                            (&document, &buffers, &images),
                            Path::new(&next_scene_path),
                        )?;
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

    pub fn load_audio(&self, path: &str, format: AudioFileFormat, params: SoundParams) {
        let pending_audio_clone = self.pending_audio.clone();
        let mut pending_audio_clone_guard = pending_audio_clone.lock().unwrap();
        pending_audio_clone_guard.push((path.to_string(), format, params));

        if pending_audio_clone_guard.len() == 1 {
            let pending_audio = self.pending_audio.clone();
            let loaded_audio = self.loaded_audio.clone();
            let audio_manager = self.audio_manager.clone();

            std::thread::spawn(move || {
                while pending_audio.lock().unwrap().len() > 0 {
                    let (next_audio_path, next_audio_format, next_audio_params) =
                        pending_audio.lock().unwrap().remove(0);

                    let do_load = || {
                        let device_sample_rate = audio_manager.lock().unwrap().device_sample_rate();
                        // let sound_data = match next_audio_format {
                        //     AudioFileFormat::Mp3 => {
                        //         AudioManager::decode_mp3(device_sample_rate, &next_audio_path)?
                        //     }
                        //     AudioFileFormat::Wav => {
                        //         AudioManager::decode_wav(device_sample_rate, &next_audio_path)?
                        //     }
                        // };
                        let sound_data =
                            AudioManager::decode_audio_file(device_sample_rate, &next_audio_path)?;
                        // let sound = Sound::new(self, sound_data, params);
                        let signal = AudioManager::get_signal(
                            &sound_data,
                            next_audio_params.clone(),
                            device_sample_rate,
                        );
                        let sound_index = audio_manager.lock().unwrap().add_sound(
                            sound_data,
                            next_audio_params,
                            signal,
                        );
                        anyhow::Ok(sound_index)
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
