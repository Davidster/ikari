use crate::audio::*;
use crate::buffer::*;
use crate::file_manager::FileManager;
use crate::file_manager::GameFilePath;
use crate::gltf_loader::*;
use crate::mesh::*;
use crate::renderer::*;
use crate::sampler_cache::*;
use crate::scene::*;
use crate::texture::*;
use crate::texture_compression::TextureCompressor;
use crate::time::*;
use crate::wasm_not_sync::WasmNotArc;
use crate::wasm_not_sync::WasmNotMutex;
use crate::wasm_not_sync::{WasmNotSend, WasmNotSync};

use anyhow::bail;
use anyhow::Result;
use image::Pixel;
use std::collections::{hash_map::Entry, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// TODO: replace with log::debug and use RUST_LOG module filter to view logs?
const DEBUG_AUDIO_STREAMING: bool = false;

type PendingSkybox = (
    String,
    SkyboxBackgroundPath,
    Option<SkyboxHDREnvironmentPath>,
);

pub struct AssetLoader {
    pending_audio: Arc<Mutex<Vec<(GameFilePath, AudioFileFormat, SoundParams)>>>,
    pub loaded_audio: Arc<Mutex<HashMap<PathBuf, usize>>>,

    audio_manager: Arc<Mutex<AudioManager>>,
    pending_scenes: Arc<Mutex<Vec<GameFilePath>>>,
    bindable_scenes: Arc<Mutex<HashMap<PathBuf, (Scene, BindableSceneData)>>>,

    pending_skyboxes: Arc<Mutex<Vec<PendingSkybox>>>,
    bindable_skyboxes: Arc<Mutex<HashMap<String, BindableSkybox>>>,
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
/// TODO: instead of using a PathBuf of the relative path of the scene, make user pass an ID in load_scene function and use that instead.
trait BindScene: WasmNotSend + WasmNotSync {
    fn update(&self, base_renderer: WasmNotArc<BaseRenderer>, asset_loader: Arc<AssetLoader>);
    fn loaded_scenes(&self)
        -> WasmNotArc<WasmNotMutex<HashMap<PathBuf, (Scene, BindedSceneData)>>>;
}

struct TimeSlicedSceneBinder {
    #[allow(clippy::type_complexity)]
    staged_scenes: WasmNotMutex<
        HashMap<
            PathBuf,
            (
                BindedSceneData,
                // texture bind group cache
                HashMap<IndexedPbrMaterial, WasmNotArc<wgpu::BindGroup>>,
            ),
        >,
    >,
    loaded_scenes: WasmNotArc<WasmNotMutex<HashMap<PathBuf, (Scene, BindedSceneData)>>>,
}

trait BindSkybox: WasmNotSend + WasmNotSync {
    fn update(
        &self,
        base_renderer: WasmNotArc<BaseRenderer>,
        renderer_constant_data: WasmNotArc<RendererConstantData>,
        asset_loader: Arc<AssetLoader>,
    );
    fn loaded_skyboxes(&self) -> WasmNotArc<WasmNotMutex<HashMap<String, BindedSkybox>>>;
}

struct TimeSlicedSkyboxBinder {
    loaded_skyboxes: WasmNotArc<WasmNotMutex<HashMap<String, BindedSkybox>>>,
}

impl AssetLoader {
    pub fn new(audio_manager: Arc<Mutex<AudioManager>>) -> Self {
        Self {
            pending_scenes: Arc::new(Mutex::new(Vec::new())),
            bindable_scenes: Arc::new(Mutex::new(HashMap::new())),
            audio_manager,
            pending_audio: Arc::new(Mutex::new(Vec::new())),
            loaded_audio: Arc::new(Mutex::new(HashMap::new())),

            pending_skyboxes: Arc::new(Mutex::new(Vec::new())),
            bindable_skyboxes: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn load_gltf_scene(&self, path: GameFilePath) {
        let pending_scene_count = {
            let mut pending_scenes = self.pending_scenes.lock().unwrap();
            pending_scenes.push(path);
            pending_scenes.len()
        };

        if pending_scene_count == 1 {
            let pending_scenes = self.pending_scenes.clone();
            let bindable_scenes = self.bindable_scenes.clone();

            crate::thread::spawn(move || {
                profiling::register_thread!("GLTF loader");
                crate::block_on(async move {
                    while pending_scenes.lock().unwrap().len() > 0 {
                        let next_scene_path = pending_scenes.lock().unwrap().remove(0);

                        let do_load = || async {
                            profiling::scope!(
                                "Load scene",
                                &next_scene_path.relative_path.to_string_lossy()
                            );
                            let gltf_slice;
                            {
                                profiling::scope!("Read root file");
                                gltf_slice = FileManager::read(&next_scene_path).await?;
                            }

                            let (document, buffers, images);
                            {
                                profiling::scope!("Parse root & children");
                                (document, buffers, images) = gltf::import_slice(&gltf_slice)?;
                            }
                            let (other_scene, other_scene_bindable_data) =
                                build_scene((&document, &buffers, &images), &next_scene_path)
                                    .await?;

                            if !other_scene_bindable_data.bindable_unlit_meshes.is_empty() {
                                log::warn!("Warning: loading unlit meshes is not yet supported in the asset loader");
                            }

                            if !other_scene_bindable_data
                                .bindable_transparent_meshes
                                .is_empty()
                            {
                                log::warn!(
                                    "Warning: loading transparent meshes is not yet supported in the asset loader",
                                );
                            }

                            anyhow::Ok((other_scene, other_scene_bindable_data))
                        };
                        match do_load().await {
                            Ok(result) => {
                                let _replaced_ignored = bindable_scenes
                                    .lock()
                                    .unwrap()
                                    .insert(next_scene_path.relative_path, result);
                            }
                            Err(err) => {
                                log::error!(
                                    "Error loading scene asset {:?}: {}\n{}",
                                    next_scene_path,
                                    err,
                                    err.backtrace()
                                );
                            }
                        }
                    }
                });
            });
        }
    }

    pub fn load_audio(&self, path: GameFilePath, format: AudioFileFormat, params: SoundParams) {
        let pending_audio_clone = self.pending_audio.clone();
        let mut pending_audio_clone_guard = pending_audio_clone.lock().unwrap();
        pending_audio_clone_guard.push((path, format, params));

        if pending_audio_clone_guard.len() == 1 {
            let pending_audio = self.pending_audio.clone();
            let loaded_audio = self.loaded_audio.clone();
            let audio_manager = self.audio_manager.clone();

            crate::thread::spawn(move || {
                profiling::register_thread!("Audio loader");
                crate::block_on(async move {
                    while pending_audio.lock().unwrap().len() > 0 {
                        let (next_audio_path, next_audio_format, next_audio_params) =
                            pending_audio.lock().unwrap().remove(0);

                        let do_load = || async {
                            let device_sample_rate =
                                audio_manager.lock().unwrap().device_sample_rate();
                            let mut audio_file_streamer = AudioFileStreamer::new(
                                device_sample_rate,
                                next_audio_path.clone(),
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
                                next_audio_path.clone(),
                                sound_data,
                                audio_file_streamer.track_length_seconds(),
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
                                let _replaced_ignored = loaded_audio
                                    .lock()
                                    .unwrap()
                                    .insert(next_audio_path.relative_path, result);
                            }
                            Err(err) => {
                                log::error!(
                                    "Error loading audio asset {:?}: {}\n{}",
                                    next_audio_path,
                                    err,
                                    err.backtrace()
                                );
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
                if DEBUG_AUDIO_STREAMING {
                    log::info!(
                        "buffered_amount_seconds={buffered_amount_seconds:?}, deficit_seconds={deficit_seconds:?}",
                    );
                }
                (max_chunk_size_length_seconds + deficit_seconds).max(0.0)
            };
            if DEBUG_AUDIO_STREAMING {
                log::info!("requested_chunk_size_seconds={requested_chunk_size_seconds:?}");
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
                    buffered_amount_seconds += added_buffer_seconds - removed_buffer_seconds;

                    if DEBUG_AUDIO_STREAMING {
                        log::info!(
                            "Streamed in {:?} samples ({:?} seconds) from file: {:?}",
                            sample_count,
                            sample_count as f32 / device_sample_rate as f32,
                            audio_file_streamer.file_path(),
                        );
                    }

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

    pub fn load_skybox(
        &self,
        id: String,
        background: SkyboxBackgroundPath,
        environment_hdr: Option<SkyboxHDREnvironmentPath>,
    ) {
        let pending_skybox_count = {
            let mut pending_skyboxes = self.pending_skyboxes.lock().unwrap();
            pending_skyboxes.push((id, background, environment_hdr));
            pending_skyboxes.len()
        };

        if pending_skybox_count == 1 {
            let pending_skyboxes = self.pending_skyboxes.clone();
            let bindable_skyboxes = self.bindable_skyboxes.clone();

            crate::thread::spawn(move || {
                profiling::register_thread!("Skybox loader");
                crate::block_on(async move {
                    while pending_skyboxes.lock().unwrap().len() > 0 {
                        let (next_skybox_id, next_skybox_background, next_skybox_env_hdr) =
                            pending_skyboxes.lock().unwrap().remove(0);

                        profiling::scope!("Load skybox", &next_skybox_id);

                        match make_bindable_skybox(
                            &next_skybox_background,
                            next_skybox_env_hdr.as_ref(),
                        )
                        .await
                        {
                            Ok(result) => {
                                let _replaced_ignored = bindable_skyboxes
                                    .lock()
                                    .unwrap()
                                    .insert(next_skybox_id, result);
                            }
                            Err(err) => {
                                log::error!(
                                    "Error loading skybox asset {}: {}\n{}",
                                    next_skybox_id,
                                    err,
                                    err.backtrace()
                                );
                            }
                        }
                    }
                });
            });
        }
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
        self.scene_binder
            .update(base_renderer.clone(), asset_loader.clone());
        self.skybox_binder.update(
            base_renderer.clone(),
            renderer_constant_data,
            asset_loader.clone(),
        );
    }

    pub fn loaded_scenes(
        &self,
    ) -> WasmNotArc<WasmNotMutex<HashMap<PathBuf, (Scene, BindedSceneData)>>> {
        self.scene_binder.loaded_scenes().clone()
    }

    pub fn loaded_skyboxes(&self) -> WasmNotArc<WasmNotMutex<HashMap<String, BindedSkybox>>> {
        self.skybox_binder.loaded_skyboxes().clone()
    }
}

impl Default for AssetBinder {
    fn default() -> Self {
        Self::new()
    }
}

pub async fn make_bindable_skybox(
    background: &SkyboxBackgroundPath,
    environment_hdr: Option<&SkyboxHDREnvironmentPath>,
) -> Result<BindableSkybox> {
    let background = match background {
        SkyboxBackgroundPath::Equirectangular(image_path) => {
            BindableSkyboxBackground::Equirectangular(
                image::load_from_memory(&FileManager::read(image_path).await?)?
                    .to_rgba8()
                    .into(),
            )
        }
        SkyboxBackgroundPath::Cube(face_image_paths) => {
            async fn to_img(img_path: &GameFilePath) -> Result<image::DynamicImage> {
                Ok(
                    image::load_from_memory(&FileManager::read(img_path).await?)?
                        .to_rgba8()
                        .into(),
                )
            }

            let first_img = to_img(&face_image_paths[0]).await?;
            let mut raw =
                Vec::with_capacity((first_img.width() * first_img.height() * 4 * 6) as usize);
            raw.extend_from_slice(first_img.as_bytes());
            for path in &face_image_paths[1..] {
                raw.extend_from_slice(to_img(path).await?.as_bytes());
            }

            BindableSkyboxBackground::Cube(RawImage {
                width: first_img.width(),
                height: first_img.height(),
                depth: 6,
                mip_count: 1,
                raw,
            })
        }
        #[cfg(not(target_arch = "wasm32"))]
        SkyboxBackgroundPath::ProcessedCube(face_image_paths) => {
            async fn to_img(
                img_path: &GameFilePath,
            ) -> Result<crate::texture_compression::CompressedTexture> {
                TextureCompressor.transcode_image(
                    &FileManager::read(
                        &crate::texture_compression::texture_path_to_compressed_path(img_path),
                    )
                    .await?,
                    false,
                )
            }

            let first_img = to_img(&face_image_paths[0]).await?;
            let mut raw = vec![];
            raw.extend_from_slice(&first_img.raw);
            for path in &face_image_paths[1..] {
                raw.extend_from_slice(&to_img(path).await?.raw);
            }

            BindableSkyboxBackground::CompressedCube(RawImage {
                width: first_img.width,
                height: first_img.height,
                depth: 6,
                mip_count: 1,
                raw,
            })
        }
        #[cfg(target_arch = "wasm32")]
        SkyboxBackgroundPath::ProcessedCube(face_image_paths) => {
            async fn to_img(img_path: &GameFilePath) -> Result<image::RgbaImage> {
                Ok(image::load_from_memory(&FileManager::read(img_path).await?)?.to_rgba8())
            }

            let first_img = to_img(&face_image_paths[0]).await?;
            let mut raw = vec![];
            raw.extend_from_slice(first_img.as_raw());
            for path in &face_image_paths[1..] {
                raw.extend_from_slice(to_img(path).await?.as_raw());
            }

            BindableSkyboxBackground::Cube(RawImage {
                width: first_img.width(),
                height: first_img.height(),
                depth: 6,
                mip_count: 1,
                raw,
            })
        }
    };

    let mut bindable_environment_hdr = None;
    match environment_hdr {
        Some(SkyboxHDREnvironmentPath::Equirectangular(image_path)) => {
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
                    raw: bytemuck::cast_slice(&skybox_rad_texture_decoded).to_vec(),
                }));
        }
        Some(SkyboxHDREnvironmentPath::ProcessedCube { diffuse, specular }) => {
            bindable_environment_hdr = Some(BindableSkyboxHDREnvironment::ProcessedCube {
                diffuse: TextureCompressor
                    .transcode_float_image(&FileManager::read(diffuse).await?)?,
                specular: TextureCompressor
                    .transcode_float_image(&FileManager::read(specular).await?)?,
            });
        }
        None => {}
    };

    Ok(BindableSkybox {
        background,
        environment_hdr: bindable_environment_hdr,
    })
}

#[cfg(not(target_arch = "wasm32"))]
struct ThreadedSceneBinder {
    loaded_scenes: Arc<Mutex<HashMap<PathBuf, (Scene, BindedSceneData)>>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl ThreadedSceneBinder {
    pub fn new(loaded_scenes: Arc<Mutex<HashMap<PathBuf, (Scene, BindedSceneData)>>>) -> Self {
        Self { loaded_scenes }
    }

    fn bind_scene(
        base_renderer: &BaseRenderer,
        bindable_scene: BindableSceneData,
    ) -> Result<BindedSceneData> {
        let mut textures: Vec<Texture> = Vec::with_capacity(bindable_scene.textures.len());
        for bindable_texture in bindable_scene.textures.iter() {
            textures.push(bind_texture(base_renderer, bindable_texture)?);
        }

        let mut binded_pbr_meshes: Vec<BindedPbrMesh> =
            Vec::with_capacity(bindable_scene.bindable_pbr_meshes.len());
        let mut textures_bind_group_cache: HashMap<IndexedPbrMaterial, Arc<wgpu::BindGroup>> =
            HashMap::new();
        for pbr_mesh in bindable_scene.bindable_pbr_meshes.iter() {
            binded_pbr_meshes.push(bind_pbr_mesh(
                base_renderer,
                pbr_mesh,
                &textures,
                &mut textures_bind_group_cache,
            )?);
        }

        let mut binded_wireframe_meshes: Vec<BindedWireframeMesh> =
            Vec::with_capacity(bindable_scene.bindable_wireframe_meshes.len());
        for wireframe_mesh in bindable_scene.bindable_wireframe_meshes.iter() {
            binded_wireframe_meshes.push(bind_wireframe_mesh(base_renderer, wireframe_mesh)?);
        }

        Ok(BindedSceneData {
            binded_pbr_meshes,
            binded_unlit_meshes: vec![],
            binded_transparent_meshes: vec![],
            binded_wireframe_meshes,
            textures,
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl BindScene for ThreadedSceneBinder {
    fn update(&self, base_renderer: Arc<BaseRenderer>, asset_loader: Arc<AssetLoader>) {
        let loaded_scenes_clone = self.loaded_scenes.clone();

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
            for (scene_id, (scene, bindable_scene)) in bindable_scenes {
                let binded_scene_result = Self::bind_scene(&base_renderer.clone(), bindable_scene);
                match binded_scene_result {
                    Ok(result) => {
                        let _replaced_ignored = loaded_scenes_clone
                            .lock()
                            .unwrap()
                            .insert(scene_id, (scene, result));
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

    fn loaded_scenes(&self) -> Arc<Mutex<HashMap<PathBuf, (Scene, BindedSceneData)>>> {
        self.loaded_scenes.clone()
    }
}

impl TimeSlicedSceneBinder {
    #[allow(dead_code)]
    pub fn new(
        loaded_scenes: WasmNotArc<WasmNotMutex<HashMap<PathBuf, (Scene, BindedSceneData)>>>,
    ) -> Self {
        Self {
            loaded_scenes,
            staged_scenes: WasmNotMutex::new(HashMap::new()),
        }
    }

    fn bind_scene_slice(
        base_renderer: &BaseRenderer,
        staged_scenes: &mut HashMap<
            PathBuf,
            (
                BindedSceneData,
                // texture bind group cache
                HashMap<IndexedPbrMaterial, WasmNotArc<wgpu::BindGroup>>,
            ),
        >,
        scene_id: PathBuf,
        bindable_scene: &BindableSceneData,
    ) -> Result<Option<BindedSceneData>> {
        {
            let (staged_scene, textures_bind_group_cache) =
                staged_scenes.entry(scene_id.clone()).or_insert_with(|| {
                    (
                        BindedSceneData {
                            binded_pbr_meshes: Vec::with_capacity(
                                bindable_scene.bindable_pbr_meshes.len(),
                            ),
                            binded_unlit_meshes: vec![],
                            binded_transparent_meshes: vec![],
                            binded_wireframe_meshes: Vec::with_capacity(
                                bindable_scene.bindable_wireframe_meshes.len(),
                            ),
                            textures: Vec::with_capacity(bindable_scene.textures.len()),
                        },
                        HashMap::new(),
                    )
                });

            let staged_texture_count = staged_scene.textures.len();
            if staged_texture_count < bindable_scene.textures.len() {
                staged_scene.textures.push(bind_texture(
                    base_renderer,
                    &bindable_scene.textures[staged_texture_count],
                )?);

                return Ok(None);
            }

            let staged_pbr_mesh_count = staged_scene.binded_pbr_meshes.len();
            if staged_pbr_mesh_count < bindable_scene.bindable_pbr_meshes.len() {
                staged_scene.binded_pbr_meshes.push(bind_pbr_mesh(
                    base_renderer,
                    &bindable_scene.bindable_pbr_meshes[staged_pbr_mesh_count],
                    &staged_scene.textures,
                    textures_bind_group_cache,
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
        }

        let (staged_scene, _) = staged_scenes.remove(&scene_id).unwrap();
        Ok(Some(staged_scene))
    }
}

impl BindScene for TimeSlicedSceneBinder {
    fn update(&self, base_renderer: WasmNotArc<BaseRenderer>, asset_loader: Arc<AssetLoader>) {
        const SLICE_BUDGET_SECONDS: f32 = 0.001;

        let start_time = crate::time::Instant::now();
        let mut bindable_scenes_guard = asset_loader.bindable_scenes.lock().unwrap();
        let loaded_scenes_clone = self.loaded_scenes.clone();

        let scene_ids: Vec<_> = bindable_scenes_guard.keys().cloned().collect();
        for scene_id in scene_ids {
            while start_time.elapsed().as_secs_f32() < SLICE_BUDGET_SECONDS {
                let scene_id = scene_id.clone();

                let (scene, bindable_scene) = bindable_scenes_guard.remove(&scene_id).unwrap();

                let binded_scene_result = Self::bind_scene_slice(
                    &base_renderer,
                    &mut self.staged_scenes.lock().unwrap(),
                    scene_id.clone(),
                    &bindable_scene,
                );
                match binded_scene_result {
                    Ok(Some(result)) => {
                        let _replaced_ignored = loaded_scenes_clone
                            .lock()
                            .unwrap()
                            .insert(scene_id, (scene, result));
                        break;
                    }
                    Ok(None) => {
                        // not done binding this scene yet, add it back to the map to be processed next frame
                        bindable_scenes_guard.insert(scene_id, (scene, bindable_scene));
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

    fn loaded_scenes(
        &self,
    ) -> WasmNotArc<WasmNotMutex<HashMap<PathBuf, (Scene, BindedSceneData)>>> {
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
            let er_background_texture = Texture::from_decoded_image(
                base_renderer,
                &image.raw,
                (image.width, image.height),
                image.mip_count,
                Some("er_skybox_texture"),
                Some(wgpu::TextureFormat::Rgba8UnormSrgb),
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
                false, // an artifact occurs between the edges of the texture with mipmaps enabled
            )?
        }
        BindableSkyboxBackground::Cube(image) => Texture::create_cubemap(
            base_renderer,
            image.slice(),
            Some("cubemap_skybox_texture"),
            wgpu::TextureFormat::Rgba8UnormSrgb,
        ),
        BindableSkyboxBackground::CompressedCube(image) => Texture::create_cubemap(
            base_renderer,
            image.slice(),
            Some("cubemap_skybox_texture"),
            wgpu::TextureFormat::Bc7RgbaUnormSrgb,
        ),
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
            let er_hdr_env_texture = Texture::from_decoded_image(
                base_renderer,
                &image.raw,
                (image.width, image.height),
                image.mip_count,
                None,
                Some(wgpu::TextureFormat::Rgba16Float),
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
                false,
            )?;

            generate_diffuse_and_specular_maps(&hdr_env_texture)
        }
        Some(BindableSkyboxHDREnvironment::ProcessedCube { diffuse, specular }) => (
            Texture::create_cubemap(
                base_renderer,
                diffuse.slice(),
                Some("diffuse env map"),
                wgpu::TextureFormat::Rgba16Float,
            ),
            Texture::create_cubemap(
                base_renderer,
                specular.slice(),
                Some("specular env map"),
                wgpu::TextureFormat::Rgba16Float,
            ),
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
                            .insert(next_skybox_id.clone(), result);
                    }
                    Err(err) => {
                        log::error!(
                            "Error loading skybox asset {}: {}\n{}",
                            next_skybox_id,
                            err,
                            err.backtrace()
                        );
                    }
                }
            }
        }
    }

    fn loaded_skyboxes(&self) -> WasmNotArc<WasmNotMutex<HashMap<String, BindedSkybox>>> {
        self.loaded_skyboxes.clone()
    }
}

fn bind_texture(
    base_renderer: &BaseRenderer,
    bindable_texture: &BindableTexture,
) -> Result<Texture> {
    let BindableTexture {
        image_pixels,
        image_dimensions,
        baked_mip_levels,
        name,
        format,
        generate_mipmaps,
        sampler_descriptor,
    } = bindable_texture;
    Texture::from_decoded_image(
        base_renderer,
        image_pixels,
        *image_dimensions,
        *baked_mip_levels,
        name.as_deref(),
        *format,
        *generate_mipmaps,
        sampler_descriptor,
    )
}

fn bind_pbr_mesh(
    base_renderer: &BaseRenderer,
    mesh: &BindablePbrMesh,
    textures: &[Texture],
    textures_bind_group_cache: &mut HashMap<IndexedPbrMaterial, WasmNotArc<wgpu::BindGroup>>,
) -> Result<BindedPbrMesh> {
    let material = &mesh.material;
    let textures_bind_group = match textures_bind_group_cache.entry(material.clone()) {
        Entry::Occupied(entry) => entry.get().clone(),
        Entry::Vacant(vacant_entry) => {
            let get_texture = |texture_index: Option<usize>| {
                texture_index.map(|texture_index| &textures[texture_index])
            };

            let pbr_material = PbrMaterial {
                base_color: get_texture(material.base_color),
                normal: get_texture(material.normal),
                emissive: get_texture(material.emissive),
                ambient_occlusion: get_texture(material.ambient_occlusion),
                metallic_roughness: get_texture(material.metallic_roughness),
            };
            let textures_bind_group =
                WasmNotArc::new(base_renderer.make_pbr_textures_bind_group(&pbr_material, true)?);
            vacant_entry.insert(textures_bind_group.clone());
            textures_bind_group
        }
    };

    let vertex_buffer_bytes = bytemuck::cast_slice(&mesh.geometry.vertices);

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
        index_buffer: bind_index_buffer(base_renderer, &mesh.geometry.indices)?,
        bounding_box: mesh.geometry.bounding_box,
    };

    Ok(BindedPbrMesh {
        geometry_buffers,
        textures_bind_group,
        dynamic_pbr_params: mesh.dynamic_pbr_params,
        alpha_mode: mesh.alpha_mode,
        primitive_mode: mesh.primitive_mode,
    })
}

fn bind_wireframe_mesh(
    base_renderer: &BaseRenderer,
    wireframe_mesh: &BindableWireframeMesh,
) -> Result<BindedWireframeMesh> {
    Ok(BindedWireframeMesh {
        source_mesh_type: wireframe_mesh.source_mesh_type,
        source_mesh_index: wireframe_mesh.source_mesh_index,
        index_buffer: bind_index_buffer(base_renderer, &wireframe_mesh.indices)?,
    })
}

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
