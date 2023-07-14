use crate::audio::*;
use crate::buffer::*;
use crate::gltf_loader::*;
use crate::logger::*;
use crate::mesh::*;
use crate::renderer::*;
use crate::scene::*;
use crate::texture::*;
use crate::time::*;

use anyhow::bail;
use anyhow::Result;
use std::collections::{hash_map::Entry, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};

const DEBUG_AUDIO_STREAMING: bool = false;

pub struct AssetLoader {
    pending_audio: Arc<Mutex<Vec<(String, AudioFileFormat, SoundParams)>>>,
    pub loaded_audio: Arc<Mutex<HashMap<String, usize>>>,

    audio_manager: Arc<Mutex<AudioManager>>,
    pending_scenes: Arc<Mutex<Vec<String>>>,
    scene_binder: Arc<Mutex<Box<dyn BindScene + Send + Sync>>>,
}

/// loading must be split into two phases:
/// 1) Do all CPU work of loading the asset into memory and preprare it for upload to GPU (fill bindable_scenes)
/// 2) Upload to GPU (fill loaded_scenes)
///
/// This is because step 2) can only be done on the main thread on the web due to the fact that the webgpu device
/// can't be used on a thread other than the one where it was created
trait BindScene {
    fn update(&mut self, base_renderer: Arc<BaseRenderer>);
    fn bindable_scenes(&self) -> Arc<Mutex<HashMap<String, (Scene, BindableSceneData)>>>;
    fn loaded_scenes(&self) -> Arc<Mutex<HashMap<String, (Scene, BindedSceneData)>>>;
}

struct ThreadedSceneBinder {
    bindable_scenes: Arc<Mutex<HashMap<String, (Scene, BindableSceneData)>>>,
    loaded_scenes: Arc<Mutex<HashMap<String, (Scene, BindedSceneData)>>>,
}

struct TimeSlicedSceneBinder {
    bindable_scenes: Arc<Mutex<HashMap<String, (Scene, BindableSceneData)>>>,
    staged_scenes: HashMap<
        String,
        (
            BindedSceneData,
            // texture bind group cache
            HashMap<IndexedPbrMaterial, Arc<wgpu::BindGroup>>,
        ),
    >,
    loaded_scenes: Arc<Mutex<HashMap<String, (Scene, BindedSceneData)>>>,
}

impl AssetLoader {
    pub fn new(audio_manager: Arc<Mutex<AudioManager>>) -> Self {
        let bindable_scenes = Arc::new(Mutex::new(HashMap::new()));
        let loaded_scenes = Arc::new(Mutex::new(HashMap::new()));
        let scene_binder: Box<dyn BindScene + Send + Sync> = if cfg!(target_arch = "wasm32") {
            Box::new(TimeSlicedSceneBinder::new(bindable_scenes, loaded_scenes))
        } else {
            Box::new(ThreadedSceneBinder::new(bindable_scenes, loaded_scenes))
        };
        Self {
            scene_binder: Arc::new(Mutex::new(scene_binder)),
            pending_scenes: Arc::new(Mutex::new(Vec::new())),
            audio_manager,
            pending_audio: Arc::new(Mutex::new(Vec::new())),
            loaded_audio: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn load_gltf_scene(&self, path: &str) {
        let pending_scene_count = {
            let mut pending_scenes = self.pending_scenes.lock().unwrap();
            pending_scenes.push(path.to_string());
            pending_scenes.len()
        };

        if pending_scene_count == 1 {
            let pending_scenes = self.pending_scenes.clone();
            let bindable_scenes = self.scene_binder.lock().unwrap().bindable_scenes();

            crate::thread::spawn(move || {
                profiling::register_thread!("GLTF loader");
                crate::block_on(async move {
                    while pending_scenes.lock().unwrap().len() > 0 {
                        let next_scene_path = pending_scenes.lock().unwrap().remove(0);

                        let do_load = || async {
                            profiling::scope!("Load scene", &next_scene_path);
                            let gltf_slice;
                            {
                                profiling::scope!("Read root file");
                                gltf_slice = crate::file_loader::read(&next_scene_path).await?;
                            }

                            let (document, buffers, images);
                            {
                                profiling::scope!("Parse root & children");
                                (document, buffers, images) = gltf::import_slice(&gltf_slice)?;
                            }
                            let (other_scene, other_scene_bindable_data) = build_scene(
                                (&document, &buffers, &images),
                                Path::new(&next_scene_path),
                            )
                            .await?;

                            if !other_scene_bindable_data.bindable_unlit_meshes.is_empty() {
                                logger_log("Warning: loading unlit meshes is not yet supported in the asset loader");
                            }

                            if !other_scene_bindable_data
                                .bindable_transparent_meshes
                                .is_empty()
                            {
                                logger_log(
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
                                    .insert(next_scene_path, result);
                            }
                            Err(err) => {
                                logger_log(&format!(
                                    "Error loading scene asset {}: {}\n{}",
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

    pub fn loaded_scenes(&self) -> Arc<Mutex<HashMap<String, (Scene, BindedSceneData)>>> {
        self.scene_binder.lock().unwrap().loaded_scenes()
    }

    pub fn update(&self, base_renderer: Arc<BaseRenderer>) {
        self.scene_binder.lock().unwrap().update(base_renderer);
    }

    pub fn load_audio(&self, path: &str, format: AudioFileFormat, params: SoundParams) {
        let pending_audio_clone = self.pending_audio.clone();
        let mut pending_audio_clone_guard = pending_audio_clone.lock().unwrap();
        pending_audio_clone_guard.push((path.to_string(), format, params));

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
                                &next_audio_path,
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
                    logger_log(&format!(
                        "buffered_amount_seconds={buffered_amount_seconds:?}, deficit_seconds={deficit_seconds:?}",
                    ));
                }
                (max_chunk_size_length_seconds + deficit_seconds).max(0.0)
            };
            if DEBUG_AUDIO_STREAMING {
                logger_log(&format!(
                    "requested_chunk_size_seconds={requested_chunk_size_seconds:?}"
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
                        logger_log(&format!(
                            "Reached end of stream for file: {}",
                            audio_file_streamer.file_path(),
                        ));
                        break;
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    crate::thread::sleep(Duration::from_secs_f32(max_chunk_size_length_seconds));
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

impl ThreadedSceneBinder {
    pub fn new(
        bindable_scenes: Arc<Mutex<HashMap<String, (Scene, BindableSceneData)>>>,
        loaded_scenes: Arc<Mutex<HashMap<String, (Scene, BindedSceneData)>>>,
    ) -> Self {
        Self {
            bindable_scenes,
            loaded_scenes,
        }
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

impl BindScene for ThreadedSceneBinder {
    fn update(&mut self, base_renderer: Arc<BaseRenderer>) {
        let loaded_scenes_clone = self.loaded_scenes.clone();

        let mut bindable_scenes: Vec<_> = vec![];
        {
            let mut bindable_scenes_guard = self.bindable_scenes.lock().unwrap();
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
                        logger_log(&format!(
                            "Error loading scene asset {}: {}\n{}",
                            scene_id,
                            err,
                            err.backtrace()
                        ));
                    }
                }
            }
        });
    }

    fn bindable_scenes(&self) -> Arc<Mutex<HashMap<String, (Scene, BindableSceneData)>>> {
        self.bindable_scenes.clone()
    }

    fn loaded_scenes(&self) -> Arc<Mutex<HashMap<String, (Scene, BindedSceneData)>>> {
        self.loaded_scenes.clone()
    }
}

impl TimeSlicedSceneBinder {
    pub fn new(
        bindable_scenes: Arc<Mutex<HashMap<String, (Scene, BindableSceneData)>>>,
        loaded_scenes: Arc<Mutex<HashMap<String, (Scene, BindedSceneData)>>>,
    ) -> Self {
        Self {
            bindable_scenes,
            loaded_scenes,

            staged_scenes: HashMap::new(),
        }
    }

    fn bind_scene_slice(
        base_renderer: &BaseRenderer,
        staged_scenes: &mut HashMap<
            String,
            (
                BindedSceneData,
                // texture bind group cache
                HashMap<IndexedPbrMaterial, Arc<wgpu::BindGroup>>,
            ),
        >,
        scene_id: String,
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
    fn update(&mut self, base_renderer: Arc<BaseRenderer>) {
        const SLICE_BUDGET_SECONDS: f32 = 0.001;

        let start_time = crate::time::Instant::now();
        let mut bindable_scenes_guard = self.bindable_scenes.lock().unwrap();
        let loaded_scenes_clone = self.loaded_scenes.clone();

        let scene_ids: Vec<_> = bindable_scenes_guard.keys().cloned().collect();
        for scene_id in scene_ids {
            while start_time.elapsed().as_secs_f32() < SLICE_BUDGET_SECONDS {
                let scene_id = scene_id.clone();

                let (scene, bindable_scene) = bindable_scenes_guard.remove(&scene_id).unwrap();

                let binded_scene_result = Self::bind_scene_slice(
                    &base_renderer,
                    &mut self.staged_scenes,
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
                        logger_log(&format!(
                            "Error loading asset {}: {}\n{}",
                            scene_id,
                            err,
                            err.backtrace()
                        ));
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

    fn bindable_scenes(&self) -> Arc<Mutex<HashMap<String, (Scene, BindableSceneData)>>> {
        self.bindable_scenes.clone()
    }

    fn loaded_scenes(&self) -> Arc<Mutex<HashMap<String, (Scene, BindedSceneData)>>> {
        self.loaded_scenes.clone()
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
    textures_bind_group_cache: &mut HashMap<IndexedPbrMaterial, Arc<wgpu::BindGroup>>,
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
                Arc::new(base_renderer.make_pbr_textures_bind_group(&pbr_material, true)?);
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
