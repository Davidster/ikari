use crate::audio::*;
use crate::buffer::*;
use crate::file_manager::FileManager;
use crate::file_manager::GameFilePath;
use crate::gltf_loader::*;
use crate::mesh::*;
use crate::raw_image::RawImage;
use crate::raw_image::RawImageDepthJoiner;
use crate::renderer::*;
use crate::sampler_cache::*;
use crate::texture::*;
use crate::texture_compression::TextureCompressor;
use crate::time::*;
use crate::wasm_not_sync::WasmNotArc;
use crate::wasm_not_sync::WasmNotMutex;
use crate::wasm_not_sync::{WasmNotSend, WasmNotSync};

use anyhow::bail;
use anyhow::Result;
use image::Pixel;
use std::collections::HashMap;

use std::collections::hash_map::Entry;
use std::sync::{Arc, Mutex};

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AssetId(u64);

#[derive(Clone, Debug)]
pub struct SceneAssetLoadParams {
    pub path: GameFilePath,
    /// generates wireframe counterparts, making all meshes renderable in wireframe mode
    pub generate_wireframe_meshes: bool,
}

#[derive(Clone, Debug)]
pub struct AudioAssetLoadParams {
    pub path: GameFilePath,
    pub format: AudioFileFormat,
    pub sound_params: SoundParams,
}

pub struct AssetLoader {
    next_asset_id: Arc<Mutex<AssetId>>,

    pending_audio: Arc<Mutex<Vec<(AssetId, AudioAssetLoadParams)>>>,
    pub loaded_audio: Arc<Mutex<HashMap<AssetId, usize>>>,

    audio_manager: Arc<Mutex<AudioManager>>,
    pending_scenes: Arc<Mutex<Vec<(AssetId, SceneAssetLoadParams)>>>,
    bindable_scenes: Arc<Mutex<HashMap<AssetId, BindableScene>>>,

    pending_skyboxes: Arc<Mutex<Vec<(AssetId, SkyboxPaths)>>>,
    bindable_skyboxes: Arc<Mutex<HashMap<AssetId, BindableSkybox>>>,
}

pub struct AssetBinder {
    scene_binder: WasmNotArc<Box<dyn BindScene>>,
    skybox_binder: WasmNotArc<Box<dyn BindSkybox>>,
}

/// loading must be split into two phases:
/// 1) Do all CPU work of loading the asset into memory and preprare it for upload to GPU (fill bindable_scenes)
/// 2) Upload to GPU (fill loaded_scenes)
///
/// This is because step 2) can only be done on the main thread on the web due to the fact that the webgpu device
/// can't be used on a thread other than the one where it was created
trait BindScene: WasmNotSend + WasmNotSync {
    fn update(
        &self,
        base_renderer: WasmNotArc<BaseRenderer>,
        renderer_constant_data: WasmNotArc<RendererConstantData>,
        asset_loader: Arc<AssetLoader>,
    );
    fn loaded_scenes(&self) -> WasmNotArc<WasmNotMutex<HashMap<AssetId, BindedScene>>>;
}

type BindGroupCache = HashMap<IndexedPbrTextures, WasmNotArc<wgpu::BindGroup>>;

struct TimeSlicedSceneBinder {
    staged_scenes: WasmNotMutex<HashMap<AssetId, BindedScene>>,
    loaded_scenes: WasmNotArc<WasmNotMutex<HashMap<AssetId, BindedScene>>>,
    bind_group_caches: WasmNotArc<WasmNotMutex<HashMap<AssetId, BindGroupCache>>>,
}

trait BindSkybox: WasmNotSend + WasmNotSync {
    fn update(
        &self,
        base_renderer: WasmNotArc<BaseRenderer>,
        renderer_constant_data: WasmNotArc<RendererConstantData>,
        asset_loader: Arc<AssetLoader>,
    );
    fn loaded_skyboxes(&self) -> WasmNotArc<WasmNotMutex<HashMap<AssetId, BindedSkybox>>>;
}

struct TimeSlicedSkyboxBinder {
    loaded_skyboxes: WasmNotArc<WasmNotMutex<HashMap<AssetId, BindedSkybox>>>,
}

impl AssetLoader {
    pub fn new(audio_manager: Arc<Mutex<AudioManager>>) -> Self {
        Self {
            next_asset_id: Arc::new(Mutex::new(AssetId(0))),

            pending_scenes: Arc::new(Mutex::new(Vec::new())),
            bindable_scenes: Arc::new(Mutex::new(HashMap::new())),
            audio_manager,
            pending_audio: Arc::new(Mutex::new(Vec::new())),
            loaded_audio: Arc::new(Mutex::new(HashMap::new())),

            pending_skyboxes: Arc::new(Mutex::new(Vec::new())),
            bindable_skyboxes: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn next_asset_id(&self) -> AssetId {
        let mut next_asset_id_guard = self.next_asset_id.lock().unwrap();
        let next_asset_id: AssetId = *next_asset_id_guard;

        next_asset_id_guard.0 += 1;

        next_asset_id
    }

    pub fn load_gltf_scene(&self, params: SceneAssetLoadParams) -> AssetId {
        let asset_id = self.next_asset_id();

        let pending_scenes = self.pending_scenes.clone();
        let mut pending_scenes_guard = pending_scenes.lock().unwrap();
        pending_scenes_guard.push((asset_id, params));

        if pending_scenes_guard.len() == 1 {
            let pending_scenes = self.pending_scenes.clone();
            let bindable_scenes = self.bindable_scenes.clone();

            crate::thread::spawn(move || {
                profiling::register_thread!("GLTF loader");
                crate::block_on(async move {
                    while pending_scenes.lock().unwrap().len() > 0 {
                        let (next_scene_id, next_scene_params) =
                            pending_scenes.lock().unwrap().remove(0);

                        match load_scene(next_scene_params.clone()).await {
                            Ok(bindable_scene) => {
                                let _replaced_ignored = bindable_scenes
                                    .lock()
                                    .unwrap()
                                    .insert(next_scene_id, bindable_scene);
                            }
                            Err(err) => {
                                log::error!(
                                    "Error loading scene asset {:?}: {}\n{}",
                                    next_scene_params.path,
                                    err,
                                    err.backtrace()
                                );
                            }
                        }
                    }
                });
            });
        }

        asset_id
    }

    pub fn load_audio(&self, params: AudioAssetLoadParams) -> AssetId {
        let asset_id = self.next_asset_id();

        let pending_audio_clone = self.pending_audio.clone();
        let mut pending_audio_clone_guard = pending_audio_clone.lock().unwrap();
        pending_audio_clone_guard.push((asset_id, params));

        if pending_audio_clone_guard.len() == 1 {
            let pending_audio = self.pending_audio.clone();
            let loaded_audio = self.loaded_audio.clone();
            let audio_manager = self.audio_manager.clone();

            crate::thread::spawn(move || {
                profiling::register_thread!("Audio loader");
                crate::block_on(async move {
                    while pending_audio.lock().unwrap().len() > 0 {
                        let (next_audio_id, next_audio_params) =
                            pending_audio.lock().unwrap().remove(0);

                        let do_load = || async {
                            let device_sample_rate =
                                audio_manager.lock().unwrap().device_sample_rate();
                            let mut audio_file_streamer = AudioFileStreamer::new(
                                device_sample_rate,
                                next_audio_params.path.clone(),
                                Some(next_audio_params.format),
                            )
                            .await?;
                            let sound_data = if !next_audio_params.sound_params.stream {
                                audio_file_streamer.read_chunk(0)?.0
                            } else {
                                Default::default()
                            };
                            let signal = AudioManager::get_signal(
                                &sound_data,
                                next_audio_params.sound_params.clone(),
                                device_sample_rate,
                            );
                            let sound_index = audio_manager.lock().unwrap().add_sound(
                                next_audio_params.path.clone(),
                                sound_data,
                                audio_file_streamer.track_length_seconds(),
                                next_audio_params.sound_params.clone(),
                                signal,
                            );

                            if next_audio_params.sound_params.stream {
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
                                    loaded_audio.lock().unwrap().insert(next_audio_id, result);
                            }
                            Err(err) => {
                                log::error!(
                                    "Error loading audio asset {:?}: {}\n{}",
                                    next_audio_params.path,
                                    err,
                                    err.backtrace()
                                );
                            }
                        }
                    }
                });
            });
        }

        asset_id
    }

    fn spawn_audio_streaming_thread(
        audio_manager: Arc<Mutex<AudioManager>>,
        sound_index: usize,
        mut audio_file_streamer: AudioFileStreamer,
    ) {
        let device_sample_rate = audio_manager.lock().unwrap().device_sample_rate();
        let mut is_first_chunk = true;
        let mut last_buffer_fill_time: Option<Instant> = None;
        let target_max_buffer_length_seconds = AUDIO_STREAM_BUFFER_LENGTH_SECONDS * 0.4;
        let max_chunk_size_length_seconds = AUDIO_STREAM_BUFFER_LENGTH_SECONDS * 0.3;
        let mut buffered_amount_seconds = 0.0;
        crate::thread::spawn(move || loop {
            profiling::register_thread!("Audio streamer");
            let requested_chunk_size_seconds = if is_first_chunk {
                target_max_buffer_length_seconds
            } else {
                let deficit_seconds: f32 =
                    target_max_buffer_length_seconds - buffered_amount_seconds;
                log::debug!(
                        "buffered_amount_seconds={buffered_amount_seconds:?}, deficit_seconds={deficit_seconds:?}",
                    );
                (max_chunk_size_length_seconds + deficit_seconds).max(0.0)
            };
            log::debug!("requested_chunk_size_seconds={requested_chunk_size_seconds:?}");
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
                    buffered_amount_seconds += added_buffer_seconds - removed_buffer_seconds;

                    log::debug!(
                        "Streamed in {:?} samples ({:?} seconds) from file: {:?}",
                        sample_count,
                        sample_count as f32 / device_sample_rate as f32,
                        audio_file_streamer.file_path().relative_path,
                    );

                    audio_manager
                        .lock()
                        .unwrap()
                        .write_stream_data(sound_index, sound_data);
                    loop {
                        if audio_manager.lock().unwrap().sound_is_playing(sound_index) {
                            break;
                        } else {
                            #[cfg(not(target_arch = "wasm32"))]
                            crate::thread::sleep(Duration::from_secs_f32(0.05));
                        }
                    }

                    last_buffer_fill_time = Some(Instant::now());

                    if reached_end_of_stream {
                        log::info!(
                            "Reached end of stream for file: {:?}",
                            audio_file_streamer.file_path(),
                        );
                        break;
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    crate::thread::sleep(Duration::from_secs_f32(max_chunk_size_length_seconds));
                }
                Err(err) => {
                    log::error!(
                        "Error loading audio asset {:?}: {}\n{}",
                        audio_file_streamer.file_path(),
                        err,
                        err.backtrace()
                    );
                }
            }
        });
    }

    pub fn load_skybox(&self, paths: SkyboxPaths) -> AssetId {
        let asset_id = self.next_asset_id();

        let mut pending_skyboxes_guard = self.pending_skyboxes.lock().unwrap();
        pending_skyboxes_guard.push((asset_id, paths));

        if pending_skyboxes_guard.len() == 1 {
            let pending_skyboxes = self.pending_skyboxes.clone();
            let bindable_skyboxes = self.bindable_skyboxes.clone();

            crate::thread::spawn(move || {
                profiling::register_thread!("Skybox loader");
                crate::block_on(async move {
                    while pending_skyboxes.lock().unwrap().len() > 0 {
                        let (next_skybox_id, next_skybox_paths) =
                            pending_skyboxes.lock().unwrap().remove(0);

                        let _get_skybox_paths_str = || {
                            let mut result = next_skybox_paths
                                .to_flattened_file_paths()
                                .iter()
                                .map(|path| path.relative_path.to_string_lossy().to_string())
                                .collect::<Vec<_>>()
                                .join(", ");
                            result.truncate(100);
                            result
                        };
                        profiling::scope!("Load skybox", &_get_skybox_paths_str());

                        match make_bindable_skybox(&next_skybox_paths).await {
                            Ok(result) => {
                                let _replaced_ignored = bindable_skyboxes
                                    .lock()
                                    .unwrap()
                                    .insert(next_skybox_id, result);
                            }
                            Err(err) => {
                                log::error!(
                                    "Error loading skybox asset {:?}: {}\n{}",
                                    next_skybox_paths,
                                    err,
                                    err.backtrace()
                                );
                            }
                        }
                    }
                });
            });
        }

        asset_id
    }
}

impl AssetBinder {
    pub fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let scene_binder = Box::new(ThreadedSceneBinder::new(WasmNotArc::new(
            WasmNotMutex::new(HashMap::new()),
        )));

        #[cfg(target_arch = "wasm32")]
        let scene_binder = Box::new(TimeSlicedSceneBinder::new(WasmNotArc::new(
            WasmNotMutex::new(HashMap::new()),
        )));

        let skybox_binder: Box<dyn BindSkybox> = Box::new(TimeSlicedSkyboxBinder {
            loaded_skyboxes: WasmNotArc::new(WasmNotMutex::new(HashMap::new())),
        });

        Self {
            scene_binder: WasmNotArc::new(scene_binder),
            skybox_binder: WasmNotArc::new(skybox_binder),
        }
    }

    pub fn update(
        &self,
        base_renderer: WasmNotArc<BaseRenderer>,
        renderer_constant_data: WasmNotArc<RendererConstantData>,
        asset_loader: Arc<AssetLoader>,
    ) {
        self.scene_binder.update(
            base_renderer.clone(),
            renderer_constant_data.clone(),
            asset_loader.clone(),
        );
        self.skybox_binder.update(
            base_renderer.clone(),
            renderer_constant_data.clone(),
            asset_loader.clone(),
        );
    }

    pub fn loaded_scenes(&self) -> WasmNotArc<WasmNotMutex<HashMap<AssetId, BindedScene>>> {
        self.scene_binder.loaded_scenes().clone()
    }

    pub fn loaded_skyboxes(&self) -> WasmNotArc<WasmNotMutex<HashMap<AssetId, BindedSkybox>>> {
        self.skybox_binder.loaded_skyboxes().clone()
    }
}

impl Default for AssetBinder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn load_skybox_image_raw_compressed(
    face_image_paths: &[GameFilePath; 6],
) -> Result<BindableSkyboxBackground> {
    let mut joiner = RawImageDepthJoiner::with_capacity(
        TextureCompressor.transcode_image(
            &FileManager::read(
                &crate::texture_compression::texture_path_to_compressed_path(&face_image_paths[0]),
            )
            .await?,
            true,
            false,
        )?,
        6,
    );

    for face_image_path in &face_image_paths[1..] {
        joiner.append_image(
            TextureCompressor.transcode_image(
                &FileManager::read(
                    &crate::texture_compression::texture_path_to_compressed_path(face_image_path),
                )
                .await?,
                true,
                false,
            )?,
        );
    }

    Ok(BindableSkyboxBackground::Cube(joiner.complete()))
}

async fn load_skybox_image_raw(
    face_image_paths: &[GameFilePath; 6],
) -> Result<BindableSkyboxBackground> {
    let mut joiner = RawImageDepthJoiner::with_capacity(
        RawImage::from_dynamic_image(
            image::load_from_memory(&FileManager::read(&face_image_paths[0]).await?)?,
            true,
        ),
        6,
    );

    for face_image_path in &face_image_paths[1..] {
        joiner.append_image(RawImage::from_dynamic_image(
            image::load_from_memory(&FileManager::read(face_image_path).await?)?,
            true,
        ));
    }

    Ok(BindableSkyboxBackground::Cube(joiner.complete()))
}

pub async fn make_bindable_skybox(paths: &SkyboxPaths) -> Result<BindableSkybox> {
    let background = match &paths.background {
        SkyboxBackgroundPath::Equirectangular(image_path) => {
            BindableSkyboxBackground::Equirectangular(RawImage::from_dynamic_image(
                image::load_from_memory(&FileManager::read(image_path).await?)?,
                true,
            ))
        }
        SkyboxBackgroundPath::Cube(face_image_paths) => {
            let mut joiner = RawImageDepthJoiner::with_capacity(
                RawImage::from_dynamic_image(
                    image::load_from_memory(&FileManager::read(&face_image_paths[0]).await?)?,
                    true,
                ),
                6,
            );

            for face_image_path in &face_image_paths[1..] {
                joiner.append_image(RawImage::from_dynamic_image(
                    image::load_from_memory(&FileManager::read(face_image_path).await?)?,
                    true,
                ));
            }

            BindableSkyboxBackground::Cube(joiner.complete())
        }
        #[cfg(not(target_arch = "wasm32"))]
        SkyboxBackgroundPath::ProcessedCube(face_image_paths) => {
            match load_skybox_image_raw_compressed(face_image_paths).await {
                Ok(compressed_cube) => compressed_cube,
                Err(error) => {
                    log::error!("Failed to load compressed version of some face of skybox {face_image_paths:?}. Will fallback to uncompressed version.\n{error:?}");
                    load_skybox_image_raw(face_image_paths).await?
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        SkyboxBackgroundPath::ProcessedCube(face_image_paths) => {
            load_skybox_image_raw(face_image_paths).await?
        }
    };

    let mut bindable_environment_hdr = None;
    match &paths.environment_hdr {
        Some(SkyboxEnvironmentHDRPath::Equirectangular(image_path)) => {
            let image_bytes = FileManager::read(image_path).await?;
            let skybox_rad_texture_decoder =
                image::codecs::hdr::HdrDecoder::new(image_bytes.as_slice())?;
            let (width, height) = {
                let metadata = skybox_rad_texture_decoder.metadata();
                (metadata.width, metadata.height)
            };
            let skybox_rad_texture_decoded: Vec<Float16> = {
                let rgb_values = skybox_rad_texture_decoder.read_image_hdr()?;
                rgb_values
                    .iter()
                    .copied()
                    .flat_map(|rbg| {
                        rbg.to_rgba()
                            .0
                            .into_iter()
                            .map(|c| Float16(half::f16::from_f32(c)))
                    })
                    .collect()
            };

            bindable_environment_hdr =
                Some(BindableSkyboxHDREnvironment::Equirectangular(RawImage {
                    width,
                    height,
                    depth: 1,
                    mip_count: 1,
                    format: wgpu::TextureFormat::Rgba16Float,
                    bytes: bytemuck::cast_slice(&skybox_rad_texture_decoded).to_vec(),
                }));
        }
        Some(SkyboxEnvironmentHDRPath::ProcessedCube { diffuse, specular }) => {
            bindable_environment_hdr = Some(BindableSkyboxHDREnvironment::Cube {
                diffuse: TextureCompressor
                    .transcode_float_image(&FileManager::read(diffuse).await?)?,
                specular: TextureCompressor
                    .transcode_float_image(&FileManager::read(specular).await?)?,
            });
        }
        None => {}
    };

    Ok(BindableSkybox {
        paths: paths.clone(),
        background,
        environment_hdr: bindable_environment_hdr,
    })
}

#[cfg(not(target_arch = "wasm32"))]
struct ThreadedSceneBinder {
    loaded_scenes: Arc<Mutex<HashMap<AssetId, BindedScene>>>,
    bind_group_caches: Arc<Mutex<HashMap<AssetId, BindGroupCache>>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl ThreadedSceneBinder {
    pub fn new(loaded_scenes: Arc<Mutex<HashMap<AssetId, BindedScene>>>) -> Self {
        Self {
            loaded_scenes,
            bind_group_caches: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn bind_scene(
        base_renderer: &BaseRenderer,
        renderer_constant_data: &RendererConstantData,
        bind_group_cache: &mut BindGroupCache,
        bindable_scene: BindableScene,
    ) -> Result<BindedScene> {
        profiling::scope!(
            "bind_scene",
            &bindable_scene.path.relative_path.to_string_lossy()
        );

        let start_time = crate::time::Instant::now();

        let mut textures: Vec<Texture> = Vec::with_capacity(bindable_scene.textures.len());
        for bindable_texture in bindable_scene.textures.iter() {
            textures.push(bind_texture(base_renderer, bindable_texture)?);
        }

        let mut binded_meshes: Vec<BindedGeometryBuffers> =
            Vec::with_capacity(bindable_scene.bindable_meshes.len());
        let mut binded_pbr_materials: Vec<BindedPbrMaterial> =
            Vec::with_capacity(bindable_scene.bindable_pbr_materials.len());
        for bindable_mesh in bindable_scene.bindable_meshes.iter() {
            binded_meshes.push(bind_mesh(base_renderer, bindable_mesh)?);
        }

        for bindable_pbr_material in bindable_scene.bindable_pbr_materials.iter() {
            binded_pbr_materials.push(bind_pbr_material(
                base_renderer,
                renderer_constant_data,
                &textures,
                bind_group_cache,
                bindable_pbr_material,
            )?);
        }

        bind_group_cache.clear();

        let mut binded_wireframe_meshes: Vec<BindedWireframeMesh> =
            Vec::with_capacity(bindable_scene.bindable_wireframe_meshes.len());
        for wireframe_mesh in bindable_scene.bindable_wireframe_meshes.iter() {
            binded_wireframe_meshes.push(bind_wireframe_mesh(base_renderer, wireframe_mesh)?);
        }

        log::debug!(
            "Scene {:?} binded in {:?}:",
            bindable_scene.path.relative_path,
            start_time.elapsed()
        );

        Ok(BindedScene {
            path: bindable_scene.path,
            scene: bindable_scene.scene,
            binded_meshes,
            binded_wireframe_meshes,
            binded_pbr_materials,
            textures,
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl BindScene for ThreadedSceneBinder {
    fn update(
        &self,
        base_renderer: Arc<BaseRenderer>,
        renderer_constant_data: Arc<RendererConstantData>,
        asset_loader: Arc<AssetLoader>,
    ) {
        let loaded_scenes_clone = self.loaded_scenes.clone();
        let bind_group_caches_clone = self.bind_group_caches.clone();

        let mut bindable_scenes: Vec<_> = vec![];
        {
            let mut bindable_scenes_guard = asset_loader.bindable_scenes.lock().unwrap();
            for item in bindable_scenes_guard.drain() {
                bindable_scenes.push(item);
            }
        }

        if bindable_scenes.is_empty() {
            return;
        }

        crate::thread::spawn(move || {
            for (scene_id, bindable_scene) in bindable_scenes {
                let mut bind_group_caches_guard = bind_group_caches_clone.lock().unwrap();
                let bind_group_cache = bind_group_caches_guard.entry(scene_id).or_default();

                match Self::bind_scene(
                    &base_renderer.clone(),
                    &renderer_constant_data.clone(),
                    bind_group_cache,
                    bindable_scene,
                ) {
                    Ok(binded_scene) => {
                        let _replaced_ignored = loaded_scenes_clone
                            .lock()
                            .unwrap()
                            .insert(scene_id, binded_scene);
                    }
                    Err(err) => {
                        log::error!(
                            "Error loading scene asset {:?}: {}\n{}",
                            scene_id,
                            err,
                            err.backtrace()
                        );
                    }
                }
            }
        });
    }

    fn loaded_scenes(&self) -> Arc<Mutex<HashMap<AssetId, BindedScene>>> {
        self.loaded_scenes.clone()
    }
}

impl TimeSlicedSceneBinder {
    #[allow(dead_code)]
    pub fn new(loaded_scenes: WasmNotArc<WasmNotMutex<HashMap<AssetId, BindedScene>>>) -> Self {
        Self {
            loaded_scenes,
            staged_scenes: WasmNotMutex::new(HashMap::new()),
            bind_group_caches: WasmNotArc::new(WasmNotMutex::new(HashMap::new())),
        }
    }

    fn bind_scene_slice(
        base_renderer: &BaseRenderer,
        renderer_constant_data: &RendererConstantData,
        staged_scenes: &mut HashMap<AssetId, BindedScene>,
        bind_group_cache: &mut BindGroupCache,
        scene_id: AssetId,
        bindable_scene: &BindableScene,
    ) -> Result<Option<BindedScene>> {
        {
            let staged_scene = staged_scenes
                .entry(scene_id)
                .or_insert_with(|| BindedScene {
                    path: bindable_scene.path.clone(),
                    scene: bindable_scene.scene.clone(),
                    binded_meshes: Vec::with_capacity(bindable_scene.bindable_meshes.len()),
                    binded_wireframe_meshes: Vec::with_capacity(
                        bindable_scene.bindable_wireframe_meshes.len(),
                    ),
                    binded_pbr_materials: Vec::with_capacity(
                        bindable_scene.bindable_pbr_materials.len(),
                    ),
                    textures: Vec::with_capacity(bindable_scene.textures.len()),
                });

            let staged_texture_count = staged_scene.textures.len();
            if staged_texture_count < bindable_scene.textures.len() {
                staged_scene.textures.push(bind_texture(
                    base_renderer,
                    &bindable_scene.textures[staged_texture_count],
                )?);

                return Ok(None);
            }

            let staged_mesh_count = staged_scene.binded_meshes.len();
            if staged_mesh_count < bindable_scene.bindable_meshes.len() {
                staged_scene.binded_meshes.push(bind_mesh(
                    base_renderer,
                    &bindable_scene.bindable_meshes[staged_mesh_count],
                )?);

                return Ok(None);
            }

            let staged_wf_mesh_count = staged_scene.binded_wireframe_meshes.len();
            if staged_wf_mesh_count < bindable_scene.bindable_wireframe_meshes.len() {
                staged_scene
                    .binded_wireframe_meshes
                    .push(bind_wireframe_mesh(
                        base_renderer,
                        &bindable_scene.bindable_wireframe_meshes[staged_wf_mesh_count],
                    )?);

                if staged_scene.binded_wireframe_meshes.len()
                    < bindable_scene.bindable_wireframe_meshes.len()
                {
                    return Ok(None);
                }
            }

            let staged_pbr_mat_count = staged_scene.binded_pbr_materials.len();
            if staged_pbr_mat_count < bindable_scene.bindable_pbr_materials.len() {
                staged_scene.binded_pbr_materials.push(bind_pbr_material(
                    base_renderer,
                    renderer_constant_data,
                    &staged_scene.textures,
                    bind_group_cache,
                    &bindable_scene.bindable_pbr_materials[staged_pbr_mat_count],
                )?);

                if staged_scene.binded_pbr_materials.len()
                    < bindable_scene.bindable_pbr_materials.len()
                {
                    return Ok(None);
                }
            }

            bind_group_cache.clear();
        }

        let staged_scene = staged_scenes.remove(&scene_id).unwrap();
        Ok(Some(staged_scene))
    }
}

impl BindScene for TimeSlicedSceneBinder {
    fn update(
        &self,
        base_renderer: WasmNotArc<BaseRenderer>,
        renderer_constant_data: WasmNotArc<RendererConstantData>,
        asset_loader: Arc<AssetLoader>,
    ) {
        const SLICE_BUDGET_SECONDS: f32 = 0.001;

        let start_time = crate::time::Instant::now();
        let mut bindable_scenes_guard = asset_loader.bindable_scenes.lock().unwrap();
        let loaded_scenes_clone = self.loaded_scenes.clone();

        let scene_ids: Vec<_> = bindable_scenes_guard.keys().cloned().collect();
        for scene_id in scene_ids {
            while start_time.elapsed().as_secs_f32() < SLICE_BUDGET_SECONDS {
                let scene_id = scene_id;

                let bindable_scene = bindable_scenes_guard.remove(&scene_id).unwrap();

                let mut bind_group_caches_guard = self.bind_group_caches.lock().unwrap();
                let bind_group_cache = bind_group_caches_guard.entry(scene_id).or_default();

                match Self::bind_scene_slice(
                    &base_renderer,
                    &renderer_constant_data,
                    &mut self.staged_scenes.lock().unwrap(),
                    bind_group_cache,
                    scene_id,
                    &bindable_scene,
                ) {
                    Ok(Some(binded_scene)) => {
                        let _replaced_ignored = loaded_scenes_clone
                            .lock()
                            .unwrap()
                            .insert(scene_id, binded_scene);
                        break;
                    }
                    Ok(None) => {
                        // not done binding this scene yet, add it back to the map to be processed next frame
                        bindable_scenes_guard.insert(scene_id, bindable_scene);
                    }
                    Err(err) => {
                        log::error!(
                            "Error loading asset {:?}: {}\n{}",
                            scene_id,
                            err,
                            err.backtrace()
                        );
                        break;
                    }
                }
            }

            // wait until the next frame to continue uploading to gpu
            if start_time.elapsed().as_secs_f32() > SLICE_BUDGET_SECONDS {
                break;
            }
        }
    }

    fn loaded_scenes(&self) -> WasmNotArc<WasmNotMutex<HashMap<AssetId, BindedScene>>> {
        self.loaded_scenes.clone()
    }
}

pub fn bind_skybox(
    base_renderer: &BaseRenderer,
    renderer_constant_data: &RendererConstantData,
    bindable_skybox: BindableSkybox,
) -> Result<BindedSkybox> {
    let start = crate::time::Instant::now();

    let background = match bindable_skybox.background {
        BindableSkyboxBackground::Equirectangular(image) => {
            let er_background_texture = Texture::from_raw_image(
                base_renderer,
                &image,
                Some("er_skybox_texture"),
                false,
                &SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                },
            )?;

            Texture::create_cubemap_from_equirectangular(
                base_renderer,
                renderer_constant_data,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                Some("cubemap_skybox_texture"),
                &er_background_texture,
            )?
        }
        BindableSkyboxBackground::Cube(image) => {
            Texture::create_cubemap(base_renderer, &image, Some("cubemap_skybox_texture"))
        }
    };

    let generate_diffuse_and_specular_maps = |hdr_env_texture: &Texture| {
        (
            Texture::create_diffuse_env_map(
                base_renderer,
                renderer_constant_data,
                Some("diffuse env map"),
                hdr_env_texture,
            ),
            Texture::create_specular_env_map(
                base_renderer,
                renderer_constant_data,
                Some("specular env map"),
                hdr_env_texture,
            ),
        )
    };

    let (diffuse_environment_map, specular_environment_map) = match bindable_skybox.environment_hdr
    {
        Some(BindableSkyboxHDREnvironment::Equirectangular(image)) => {
            let er_hdr_env_texture = Texture::from_raw_image(
                base_renderer,
                &image,
                None,
                false,
                &SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                },
            )?;

            let hdr_env_texture = Texture::create_cubemap_from_equirectangular(
                base_renderer,
                renderer_constant_data,
                wgpu::TextureFormat::Rgba16Float,
                None,
                &er_hdr_env_texture,
            )?;

            generate_diffuse_and_specular_maps(&hdr_env_texture)
        }
        Some(BindableSkyboxHDREnvironment::Cube { diffuse, specular }) => (
            Texture::create_cubemap(base_renderer, &diffuse, Some("diffuse env map")),
            Texture::create_cubemap(base_renderer, &specular, Some("specular env map")),
        ),
        None => generate_diffuse_and_specular_maps(&background),
    };

    log::debug!("skybox bind time: {:?}", start.elapsed());

    Ok(BindedSkybox {
        background,
        diffuse_environment_map,
        specular_environment_map,
    })
}

impl BindSkybox for TimeSlicedSkyboxBinder {
    fn update(
        &self,
        base_renderer: WasmNotArc<BaseRenderer>,
        renderer_constant_data: WasmNotArc<RendererConstantData>,
        asset_loader: Arc<AssetLoader>,
    ) {
        let next_skybox_id = {
            let guard = asset_loader.bindable_skyboxes.lock().unwrap();
            guard.keys().next().cloned()
        };
        if let Some(next_skybox_id) = next_skybox_id {
            if let Some(next_bindable_skybox) = asset_loader
                .bindable_skyboxes
                .lock()
                .unwrap()
                .remove(&next_skybox_id)
            {
                match bind_skybox(
                    &base_renderer.clone(),
                    &renderer_constant_data,
                    next_bindable_skybox,
                ) {
                    Ok(result) => {
                        let _replaced_ignored = self
                            .loaded_skyboxes
                            .lock()
                            .unwrap()
                            .insert(next_skybox_id, result);
                    }
                    Err(err) => {
                        log::error!(
                            "Error loading skybox asset {:?}: {}\n{}",
                            next_skybox_id,
                            err,
                            err.backtrace()
                        );
                    }
                }
            }
        }
    }

    fn loaded_skyboxes(&self) -> WasmNotArc<WasmNotMutex<HashMap<AssetId, BindedSkybox>>> {
        self.loaded_skyboxes.clone()
    }
}

#[profiling::function]
fn bind_texture(
    base_renderer: &BaseRenderer,
    bindable_texture: &BindableTexture,
) -> Result<Texture> {
    let BindableTexture {
        raw_image,
        name,
        sampler_descriptor,
    } = bindable_texture;
    Texture::from_raw_image(
        base_renderer,
        raw_image,
        name.as_deref(),
        raw_image.mip_count <= 1,
        sampler_descriptor,
    )
}

#[profiling::function]
fn bind_mesh(
    base_renderer: &BaseRenderer,
    mesh: &BindableGeometryBuffers,
) -> Result<BindedGeometryBuffers> {
    let vertex_buffer_bytes = bytemuck::cast_slice(&mesh.vertices);

    if vertex_buffer_bytes.len() as u64 > base_renderer.limits.max_buffer_size {
        bail!(
            "Tried to upload a vertex buffer of size {:?} which is larger than the max buffer size of {:?}", 
            vertex_buffer_bytes.len(), base_renderer.limits.max_buffer_size
        );
    }

    let vertex_buffer = GpuBuffer::from_bytes(
        &base_renderer.device,
        vertex_buffer_bytes,
        std::mem::size_of::<Vertex>(),
        wgpu::BufferUsages::VERTEX,
    );

    let geometry_buffers = BindedGeometryBuffers {
        vertex_buffer,
        index_buffer: bind_index_buffer(base_renderer, &mesh.indices)?,
        bounding_box: mesh.bounding_box,
    };

    Ok(geometry_buffers)
}

#[profiling::function]
fn bind_pbr_material(
    base_renderer: &BaseRenderer,
    renderer_constant_data: &RendererConstantData,
    textures: &[Texture],
    bind_group_cache: &mut BindGroupCache,
    material: &BindablePbrMaterial,
) -> Result<BindedPbrMaterial> {
    let textures_bind_group = match bind_group_cache.entry(material.textures.clone()) {
        Entry::Occupied(entry) => entry.get().clone(),
        Entry::Vacant(vacant_entry) => {
            let get_texture = |texture_index: Option<usize>| {
                texture_index.map(|texture_index| &textures[texture_index])
            };

            let pbr_textures = PbrTextures {
                base_color: get_texture(material.textures.base_color),
                normal: get_texture(material.textures.normal),
                emissive: get_texture(material.textures.emissive),
                ambient_occlusion: get_texture(material.textures.ambient_occlusion),
                metallic_roughness: get_texture(material.textures.metallic_roughness),
            };
            let textures_bind_group = WasmNotArc::new(Renderer::make_pbr_textures_bind_group(
                base_renderer,
                renderer_constant_data,
                &pbr_textures,
                true,
            )?);
            vacant_entry.insert(textures_bind_group.clone());
            textures_bind_group
        }
    };

    Ok(BindedPbrMaterial {
        textures_bind_group,
        dynamic_pbr_params: material.dynamic_pbr_params,
    })
}

#[profiling::function]
fn bind_wireframe_mesh(
    base_renderer: &BaseRenderer,
    wireframe_mesh: &BindableWireframeMesh,
) -> Result<BindedWireframeMesh> {
    Ok(BindedWireframeMesh {
        source_mesh_index: wireframe_mesh.source_mesh_index,
        index_buffer: bind_index_buffer(base_renderer, &wireframe_mesh.indices)?,
    })
}

#[profiling::function]
fn bind_index_buffer(
    base_renderer: &BaseRenderer,
    indices: &BindableIndices,
) -> Result<BindedIndexBuffer> {
    let (index_buffer_bytes, index_buffer_format) = match indices {
        BindableIndices::U16(indices_u16) => (
            bytemuck::cast_slice::<u16, u8>(indices_u16),
            wgpu::IndexFormat::Uint16,
        ),
        BindableIndices::U32(indices_u32) => (
            bytemuck::cast_slice::<u32, u8>(indices_u32),
            wgpu::IndexFormat::Uint32,
        ),
    };

    if index_buffer_bytes.len() as u64 > base_renderer.limits.max_buffer_size {
        bail!("Tried to upload an index buffer of size {:?} which is larger than the max buffer size of {:?}", index_buffer_bytes.len(), base_renderer.limits.max_buffer_size);
    }

    Ok(BindedIndexBuffer {
        buffer: GpuBuffer::from_bytes(
            &base_renderer.device,
            index_buffer_bytes,
            match index_buffer_format {
                wgpu::IndexFormat::Uint16 => std::mem::size_of::<u16>(),
                wgpu::IndexFormat::Uint32 => std::mem::size_of::<u32>(),
            },
            wgpu::BufferUsages::INDEX,
        ),
        format: index_buffer_format,
    })
}
