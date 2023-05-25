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
                                "Error loading gltf asset {}: {}\n{}",
                                next_scene_path,
                                err,
                                err.backtrace()
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
                        let mut audio_file_streamer = AudioFileStreamer::new(
                            device_sample_rate,
                            next_audio_path.clone(),
                            Some(next_audio_format),
                        )?;
                        let sound_data = if !next_audio_params.stream {
                            audio_file_streamer.read_chunk(0)?.0
                        } else {
                            Default::default()
                        };
                        let signal = AudioManager::get_signal(
                            &sound_data,
                            next_audio_params.clone(),
                            device_sample_rate,
                        );
                        let sound_index = audio_manager.lock().unwrap().add_sound(
                            sound_data,
                            next_audio_params.clone(),
                            signal,
                        );

                        if next_audio_params.stream {
                            Self::spawn_audio_streaming_thread(
                                audio_manager.clone(),
                                sound_index,
                                audio_file_streamer,
                            );
                        }

                        anyhow::Ok(sound_index)
                    };
                    match do_load() {
                        Ok(result) => {
                            let _replaced_ignored =
                                loaded_audio.lock().unwrap().insert(next_audio_path, result);
                        }
                        Err(err) => {
                            logger_log(&format!(
                                "Error loading audio asset {}: {}\n{}",
                                next_audio_path,
                                err,
                                err.backtrace()
                            ));
                        }
                    }
                }
            });
        }
    }

    fn spawn_audio_streaming_thread(
        audio_manager: Arc<Mutex<AudioManager>>,
        sound_index: usize,
        mut audio_file_streamer: AudioFileStreamer,
    ) {
        let device_sample_rate = audio_manager.lock().unwrap().device_sample_rate();
        let mut is_first_chunk = true;
        std::thread::spawn(move || loop {
            let chunk_size_seconds = if is_first_chunk {
                AUDIO_STREAM_BUFFER_LENGTH_SECONDS * 0.75
            } else {
                AUDIO_STREAM_BUFFER_LENGTH_SECONDS * 0.5
            };
            is_first_chunk = false;
            match audio_file_streamer
                .read_chunk((device_sample_rate as f32 * chunk_size_seconds) as usize)
            {
                Ok((sound_data, reached_end_of_stream)) => {
                    let sample_count = sound_data.0.len();
                    audio_manager
                        .lock()
                        .unwrap()
                        .write_stream_data(sound_index, sound_data);
                    // logger_log(&format!(
                    //     "Streamed in {:?} samples ({:?} seconds) from file: {}",
                    //     sample_count,
                    //     sample_count as f32 / device_sample_rate as f32,
                    //     audio_file_streamer.file_path(),
                    // ));
                    if reached_end_of_stream {
                        logger_log(&format!(
                            "Reached end of stream for file: {}",
                            audio_file_streamer.file_path(),
                        ));
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_secs_f32(
                        AUDIO_STREAM_BUFFER_LENGTH_SECONDS * 0.5,
                    ));
                }
                Err(err) => {
                    logger_log(&format!(
                        "Error loading audio asset {}: {}\n{}",
                        audio_file_streamer.file_path(),
                        err,
                        err.backtrace()
                    ));
                }
            }
        });
    }
}
