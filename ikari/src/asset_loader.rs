use crate::audio::*;
use crate::gltf_loader::*;
use crate::logger::*;
use crate::renderer::*;
use crate::scene::*;
use crate::time::*;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

const DEBUG_AUDIO_STREAMING: bool = false;

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
                pollster::block_on(async {
                    while pending_assets.lock().unwrap().len() > 0 {
                        let next_scene_path = pending_assets.lock().unwrap().remove(0);

                        let do_load = || async {
                            profiling::scope!("Load asset", &next_scene_path);
                            let gltf_slice = crate::file_loader::read(&next_scene_path).await?;
                            let (document, buffers, images) = gltf::import_slice(&gltf_slice)?;
                            let (other_scene, other_render_buffers) = build_scene(
                                &renderer_base,
                                (&document, &buffers, &images),
                                Path::new(&next_scene_path),
                            )
                            .await?;
                            anyhow::Ok((other_scene, other_render_buffers))
                        };
                        match do_load().await {
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
                pollster::block_on(async {
                    while pending_audio.lock().unwrap().len() > 0 {
                        let (next_audio_path, next_audio_format, next_audio_params) =
                            pending_audio.lock().unwrap().remove(0);

                        let do_load = || async {
                            let device_sample_rate =
                                audio_manager.lock().unwrap().device_sample_rate();
                            let mut audio_file_streamer = AudioFileStreamer::new(
                                device_sample_rate,
                                &next_audio_path,
                                Some(next_audio_format),
                            )
                            .await?;
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
                        match do_load().await {
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
        let mut last_buffer_fill_time: Option<Instant> = None;
        let target_max_buffer_length_seconds = AUDIO_STREAM_BUFFER_LENGTH_SECONDS * 0.75;
        let mut buffered_amount_seconds = 0.0;
        std::thread::spawn(move || loop {
            let requested_chunk_size_seconds = if is_first_chunk {
                target_max_buffer_length_seconds
            } else {
                let deficit_seconds: f32 =
                    target_max_buffer_length_seconds - buffered_amount_seconds;
                if DEBUG_AUDIO_STREAMING {
                    logger_log(&format!(
                        "buffered_amount_seconds={:?}, deficit_seconds={:?}",
                        buffered_amount_seconds, deficit_seconds
                    ));
                }
                (AUDIO_STREAM_BUFFER_LENGTH_SECONDS * 0.5 + deficit_seconds).max(0.0)
            };
            if DEBUG_AUDIO_STREAMING {
                logger_log(&format!(
                    "requested_chunk_size_seconds={:?}",
                    requested_chunk_size_seconds
                ));
            }
            is_first_chunk = false;
            match audio_file_streamer
                .read_chunk((device_sample_rate as f32 * requested_chunk_size_seconds) as usize)
            {
                Ok((sound_data, reached_end_of_stream)) => {
                    let sample_count = sound_data.0.len();

                    let added_buffer_seconds = sample_count as f32 / device_sample_rate as f32;
                    let removed_buffer_seconds = last_buffer_fill_time
                        .map(|last_buffer_fill_time| last_buffer_fill_time.elapsed().as_secs_f32())
                        .unwrap_or(0.0);
                    last_buffer_fill_time = Some(now());
                    buffered_amount_seconds += added_buffer_seconds - removed_buffer_seconds;

                    if DEBUG_AUDIO_STREAMING {
                        logger_log(&format!(
                            "Streamed in {:?} samples ({:?} seconds) from file: {}",
                            sample_count,
                            sample_count as f32 / device_sample_rate as f32,
                            audio_file_streamer.file_path(),
                        ));
                    }

                    audio_manager
                        .lock()
                        .unwrap()
                        .write_stream_data(sound_index, sound_data);

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
