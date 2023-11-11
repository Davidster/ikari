use crate::buffer::*;
use crate::camera::*;
use crate::collisions::*;
use crate::engine_state::EngineState;
use crate::file_manager::GameFilePath;
use crate::math::*;
use crate::mesh::*;
use crate::sampler_cache::*;
use crate::scene::*;
use crate::skinning::*;
use crate::texture::*;
use crate::transform::*;
use crate::ui::*;
use crate::wasm_not_sync::WasmNotArc;

use std::collections::{hash_map::Entry, HashMap};
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Result;
use glam::f32::{Mat4, Vec3};
use glam::Vec4;

use wgpu::util::DeviceExt;
use wgpu::InstanceDescriptor;

use wgpu_profiler::wgpu_profiler;
use wgpu_profiler::GpuProfiler;

pub(crate) const USE_LABELS: bool = true;
pub(crate) const USE_ORTHOGRAPHIC_CAMERA: bool = false;
pub(crate) const USE_EXTRA_SHADOW_MAP_CULLING: bool = true;

pub const MAX_LIGHT_COUNT: usize = 32;
pub const NEAR_PLANE_DISTANCE: f32 = 0.001;
pub const FAR_PLANE_DISTANCE: f32 = 100000.0;
pub const FOV_Y_DEG: f32 = 45.0;
pub const DEFAULT_WIREFRAME_COLOR: [f32; 4] = [0.0, 1.0, 1.0, 1.0];
pub const POINT_LIGHT_SHADOW_MAP_FRUSTUM_NEAR_PLANE: f32 = 0.1;
pub const POINT_LIGHT_SHADOW_MAP_FRUSTUM_FAR_PLANE: f32 = 1000.0;
pub const POINT_LIGHT_SHADOW_MAP_RESOLUTION: u32 = 1024;
pub const DIRECTIONAL_LIGHT_SHADOW_MAP_RESOLUTION: u32 = 2048;
pub const POINT_LIGHT_SHOW_MAP_COUNT: u32 = 2;
pub const DIRECTIONAL_LIGHT_SHOW_MAP_COUNT: u32 = 2;
pub const DIRECTIONAL_LIGHT_PROJ_BOX_RADIUS: f32 = 50.0;
pub const DIRECTIONAL_LIGHT_PROJ_BOX_LENGTH: f32 = 1000.0;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Float16(pub half::f16);

unsafe impl bytemuck::Pod for Float16 {}
unsafe impl bytemuck::Zeroable for Float16 {}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PointLightUniform {
    position: [f32; 4],
    color: [f32; 4],
}

impl Default for PointLightUniform {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0, 1.0],
            color: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

fn make_point_light_uniform_buffer(engine_state: &EngineState) -> Vec<PointLightUniform> {
    let mut light_uniforms = Vec::new();

    let active_light_count = engine_state.scene.point_lights.len();
    let mut active_lights = engine_state
        .scene
        .point_lights
        .iter()
        .flat_map(|point_light| {
            engine_state
                .scene
                .get_node(point_light.node_id)
                .map(|light_node| {
                    let position = light_node.transform.position();
                    PointLightUniform {
                        position: [position.x, position.y, position.z, 1.0],
                        color: [
                            point_light.color.x,
                            point_light.color.y,
                            point_light.color.z,
                            point_light.intensity,
                        ],
                    }
                })
        })
        .collect::<Vec<_>>();
    light_uniforms.append(&mut active_lights);

    let mut inactive_lights = (0..(MAX_LIGHT_COUNT - active_light_count))
        .map(|_| PointLightUniform::default())
        .collect::<Vec<_>>();
    light_uniforms.append(&mut inactive_lights);

    light_uniforms
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DirectionalLightUniform {
    world_space_to_light_space: [[f32; 4]; 4],
    direction: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct PbrShaderOptionsUniform {
    options_1: [f32; 4],
    options_2: [f32; 4],
    options_3: [f32; 4],
    options_4: [f32; 4],
}

impl From<&DirectionalLight> for DirectionalLightUniform {
    fn from(light: &DirectionalLight) -> Self {
        let DirectionalLight {
            direction,
            color,
            intensity,
        } = light;
        let shader_camera_data = ShaderCameraData::orthographic(
            look_in_dir(Vec3::new(0.0, 0.0, 0.0), light.direction),
            DIRECTIONAL_LIGHT_PROJ_BOX_RADIUS * 2.0,
            DIRECTIONAL_LIGHT_PROJ_BOX_RADIUS * 2.0,
            -DIRECTIONAL_LIGHT_PROJ_BOX_LENGTH / 2.0,
            DIRECTIONAL_LIGHT_PROJ_BOX_LENGTH / 2.0,
            false,
        );
        Self {
            world_space_to_light_space: (shader_camera_data.proj * shader_camera_data.view)
                .to_cols_array_2d(),
            direction: [direction.x, direction.y, direction.z, 1.0],
            color: [color.x, color.y, color.z, *intensity],
        }
    }
}

impl Default for DirectionalLightUniform {
    fn default() -> Self {
        Self {
            world_space_to_light_space: Mat4::IDENTITY.to_cols_array_2d(),
            direction: [0.0, -1.0, 0.0, 1.0],
            color: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

fn make_directional_light_uniform_buffer(
    lights: &[DirectionalLight],
) -> Vec<DirectionalLightUniform> {
    let mut light_uniforms = Vec::new();

    let active_light_count = lights.len();
    let mut active_lights = lights
        .iter()
        .map(DirectionalLightUniform::from)
        .collect::<Vec<_>>();
    light_uniforms.append(&mut active_lights);

    let mut inactive_lights = (0..(MAX_LIGHT_COUNT - active_light_count))
        .map(|_| DirectionalLightUniform::default())
        .collect::<Vec<_>>();
    light_uniforms.append(&mut inactive_lights);

    light_uniforms
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum CullingFrustumLockMode {
    Full,
    FocalPoint,
    #[default]
    None,
}

impl CullingFrustumLockMode {
    pub const ALL: [CullingFrustumLockMode; 3] = [
        CullingFrustumLockMode::None,
        CullingFrustumLockMode::Full,
        CullingFrustumLockMode::FocalPoint,
    ];
}

impl From<CullingFrustumLock> for CullingFrustumLockMode {
    fn from(value: CullingFrustumLock) -> Self {
        match value {
            CullingFrustumLock::Full(_) => CullingFrustumLockMode::Full,
            CullingFrustumLock::FocalPoint(_) => CullingFrustumLockMode::FocalPoint,
            CullingFrustumLock::None => CullingFrustumLockMode::None,
        }
    }
}

impl std::fmt::Display for CullingFrustumLockMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CullingFrustumLockMode::Full => "On",
                CullingFrustumLockMode::FocalPoint => "Only FocalPoint",
                CullingFrustumLockMode::None => "Off",
            }
        )
    }
}

pub struct GpuTimerScopeResultWrapper(pub wgpu_profiler::GpuTimerScopeResult);

impl std::ops::Deref for GpuTimerScopeResultWrapper {
    type Target = wgpu_profiler::GpuTimerScopeResult;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Debug for GpuTimerScopeResultWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "(TODO implement better formatter for type) {:?} -> {:?}",
            self.label, self.time
        )
    }
}

impl Clone for GpuTimerScopeResultWrapper {
    fn clone(&self) -> Self {
        Self(wgpu_profiler::GpuTimerScopeResult {
            label: self.label.clone(),
            time: self.time.clone(),
            nested_scopes: clone_nested_scopes(&self.nested_scopes),
            pid: self.pid,
            tid: self.tid,
        })
    }
}

fn clone_nested_scopes(
    nested_scopes: &[wgpu_profiler::GpuTimerScopeResult],
) -> Vec<wgpu_profiler::GpuTimerScopeResult> {
    nested_scopes
        .iter()
        .map(|nested_scope| wgpu_profiler::GpuTimerScopeResult {
            label: nested_scope.label.clone(),
            time: nested_scope.time.clone(),
            nested_scopes: clone_nested_scopes(&nested_scope.nested_scopes),
            pid: nested_scope.pid,
            tid: nested_scope.tid,
        })
        .collect()
}

fn make_pbr_shader_options_uniform_buffer(
    enable_soft_shadows: bool,
    shadow_bias: f32,
    soft_shadow_factor: f32,
    enable_shadow_debug: bool,
    soft_shadow_grid_dims: u32,
) -> PbrShaderOptionsUniform {
    let options_1 = [
        if enable_soft_shadows { 1.0 } else { 0.0 },
        soft_shadow_factor,
        if enable_shadow_debug { 1.0 } else { 0.0 },
        soft_shadow_grid_dims as f32,
    ];

    let options_2 = [shadow_bias, 0.0, 0.0, 0.0];

    PbrShaderOptionsUniform {
        options_1,
        options_2,
        ..Default::default()
    }
}

#[derive(Debug)]
pub struct BindableTexture {
    pub image_pixels: Vec<u8>,
    pub image_dimensions: (u32, u32),
    pub baked_mip_levels: u32,
    pub name: Option<String>,
    pub format: Option<wgpu::TextureFormat>,
    pub generate_mipmaps: bool,
    pub sampler_descriptor: crate::sampler_cache::SamplerDescriptor,
}

#[derive(Debug)]
pub struct BindablePbrMaterial {
    pub textures: IndexedPbrTextures,
    pub dynamic_pbr_params: DynamicPbrParams,
}

#[derive(Debug)]
pub struct BindedPbrMaterial {
    pub textures_bind_group: wgpu::BindGroup,
    pub dynamic_pbr_params: DynamicPbrParams,
}

#[derive(Debug, Clone)]
pub enum BindableIndices {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl BindableIndices {
    pub fn format(&self) -> wgpu::IndexFormat {
        match self {
            BindableIndices::U16(_) => wgpu::IndexFormat::Uint16,
            BindableIndices::U32(_) => wgpu::IndexFormat::Uint32,
        }
    }
}

#[derive(Debug)]
pub struct BindableGeometryBuffers {
    pub vertices: Vec<Vertex>,
    pub indices: BindableIndices,
    pub bounding_box: crate::collisions::Aabb,
}

#[derive(Debug)]
pub struct BindedIndexBuffer {
    pub buffer: GpuBuffer,
    pub format: wgpu::IndexFormat,
}

#[derive(Debug)]
pub struct BindedGeometryBuffers {
    pub vertex_buffer: GpuBuffer,
    pub index_buffer: BindedIndexBuffer,
    pub bounding_box: crate::collisions::Aabb,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum MaterialType {
    Pbr,
    Unlit,
    Transparent,
}

impl From<Material> for MaterialType {
    fn from(material: Material) -> Self {
        match material {
            Material::Pbr { .. } => MaterialType::Pbr,
            Material::Unlit { .. } => MaterialType::Unlit,
            Material::Transparent { .. } => MaterialType::Transparent,
        }
    }
}

#[derive(Debug)]
pub struct BindableWireframeMesh {
    pub source_mesh_index: usize,
    pub indices: BindableIndices,
}

#[derive(Debug)]
pub struct BindedWireframeMesh {
    pub source_mesh_index: usize,
    pub index_buffer: BindedIndexBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefaultTextureType {
    BaseColor,
    Normal,
    MetallicRoughness,
    MetallicRoughnessGLTF,
    Emissive,
    EmissiveGLTF,
    AmbientOcclusion,
}

#[derive(Clone, Debug)]
pub struct PointLight {
    pub node_id: GameNodeId,
    pub color: Vec3,
    pub intensity: f32,
}

#[derive(Clone, Debug)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

pub struct BaseRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
    pub limits: wgpu::Limits,
    default_texture_cache: Mutex<HashMap<DefaultTextureType, WasmNotArc<Texture>>>,
    pub sampler_cache: Mutex<SamplerCache>,
}

pub struct SurfaceData {
    pub surface: wgpu::Surface,
    pub surface_config: wgpu::SurfaceConfiguration,
}

impl BaseRenderer {
    pub async fn offscreen(backends: wgpu::Backends, dxc_path: Option<PathBuf>) -> Result<Self> {
        let instance = Self::make_instance(backends, dxc_path);
        Self::new(instance, None).await
    }

    pub async fn with_window(
        backends: wgpu::Backends,
        dxc_path: Option<PathBuf>,
        window: &winit::window::Window,
    ) -> Result<(Self, SurfaceData)> {
        let instance = Self::make_instance(backends, dxc_path);
        let surface = unsafe { instance.create_surface(&window).unwrap() };
        let base = Self::new(instance, Some(&surface)).await?;

        let window_size = window.inner_size();

        let mut surface_config = surface
            .get_default_config(&base.adapter, window_size.width, window_size.height)
            .ok_or_else(|| {
                anyhow::anyhow!("Window surface is incompatible with the graphics adapter")
            })?;
        surface_config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        // surface_config.format = wgpu::TextureFormat::Bgra8UnormSrgb;
        surface_config.alpha_mode = wgpu::CompositeAlphaMode::Auto;
        surface_config.present_mode = wgpu::PresentMode::AutoNoVsync;
        surface.configure(&base.device, &surface_config);

        let capabilities = surface.get_capabilities(&base.adapter);

        log::info!(
            "WGPU surface initialized with:\nPossible formats: {:?}\nChosen format: {:?}\nPossible present modes: {:?}\nAlpha modes: {:?}",
            capabilities.formats,
            surface_config.format,
            capabilities.present_modes,
            capabilities.alpha_modes,
        );

        let surface_data = SurfaceData {
            surface,
            surface_config,
        };

        Ok((base, surface_data))
    }

    fn make_instance(backends: wgpu::Backends, dxc_path: Option<PathBuf>) -> wgpu::Instance {
        wgpu::Instance::new(InstanceDescriptor {
            backends,
            dx12_shader_compiler: wgpu::Dx12Compiler::Dxc {
                dxil_path: dxc_path.clone(),
                dxc_path,
            },
        })
    }

    async fn new(instance: wgpu::Instance, surface: Option<&wgpu::Surface>) -> Result<Self> {
        let request_adapter_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
        };
        let adapter = instance
            .request_adapter(&request_adapter_options)
            .await
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Failed to find a wgpu adapter with options: {request_adapter_options:?}"
                )
            })?;

        let mut features = adapter.features();

        // use time features if they're available on the adapter
        features &= wgpu_profiler::GpuProfiler::ALL_WGPU_TIMER_FEATURES;

        // panic if these features are missing
        features |= wgpu::Features::TEXTURE_COMPRESSION_BC;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features,
                    limits: Default::default(),
                },
                None,
            )
            .await
            .map_err(|err| anyhow::anyhow!("Failed to create wgpu device: {err}"))?;

        log::info!(
            "WGPU device initialized with:\nAdapter: {:?}\nFeatures: {:?}",
            adapter.get_info(),
            device.features()
        );

        let limits = device.limits();

        Ok(Self {
            device,
            adapter,
            queue,
            limits,

            default_texture_cache: Mutex::new(HashMap::new()),
            sampler_cache: Mutex::new(SamplerCache::default()),
        })
    }

    pub fn get_default_texture(
        &self,
        default_texture_type: DefaultTextureType,
    ) -> anyhow::Result<WasmNotArc<Texture>> {
        let mut default_texture_cache_guard = self.default_texture_cache.lock().unwrap();
        let default_texture = match default_texture_cache_guard.entry(default_texture_type) {
            Entry::Occupied(texture) => texture.get().clone(),
            Entry::Vacant(_) => {
                let color: [u8; 4] = match default_texture_type {
                    DefaultTextureType::BaseColor => [255, 255, 255, 255],
                    DefaultTextureType::Normal => [127, 127, 255, 255],
                    DefaultTextureType::MetallicRoughness => [255, 255, 255, 255],
                    DefaultTextureType::MetallicRoughnessGLTF => [255, 127, 0, 255],
                    DefaultTextureType::Emissive => [0, 0, 0, 255],
                    DefaultTextureType::EmissiveGLTF => [255, 255, 255, 255],
                    DefaultTextureType::AmbientOcclusion => [255, 255, 255, 255],
                };
                WasmNotArc::new(Texture::from_color(self, color)?)
            }
        };
        if let Entry::Vacant(entry) = default_texture_cache_guard.entry(default_texture_type) {
            entry.insert(default_texture.clone());
        }
        Ok(default_texture)
    }
}

#[derive(Clone, Debug)]
pub enum CullingFrustumLock {
    Full(CameraFrustumDescriptor),
    FocalPoint(Vec3),
    None,
}

pub struct RendererPrivateData {
    // cpu
    all_bone_transforms: AllBoneTransforms,
    all_pbr_instances: ChunkedBuffer<GpuPbrMeshInstance, (usize, usize)>,
    all_pbr_instances_culling_masks: Vec<u32>,
    all_unlit_instances: ChunkedBuffer<GpuUnlitMeshInstance, usize>,
    all_transparent_instances: ChunkedBuffer<GpuUnlitMeshInstance, usize>,
    all_wireframe_instances: ChunkedBuffer<GpuWireframeMeshInstance, usize>,
    debug_node_bounding_spheres_nodes: Vec<GameNodeId>,
    debug_culling_frustum_nodes: Vec<GameNodeId>,
    debug_culling_frustum_mesh_index: Option<usize>,

    bloom_threshold_cleared: bool,
    frustum_culling_lock: CullingFrustumLock, // for debug
    skybox_weights: [f32; 2],

    // gpu
    camera_lights_and_pbr_shader_options_bind_group_layout: wgpu::BindGroupLayout,

    camera_lights_and_pbr_shader_options_bind_groups: Vec<wgpu::BindGroup>,
    bones_and_pbr_instances_bind_group: wgpu::BindGroup,
    bones_and_unlit_instances_bind_group: wgpu::BindGroup,
    bones_and_transparent_instances_bind_group: wgpu::BindGroup,
    bones_and_wireframe_instances_bind_group: wgpu::BindGroup,
    bloom_config_bind_groups: [wgpu::BindGroup; 2],
    tone_mapping_config_bind_group: wgpu::BindGroup,
    environment_textures_bind_group: wgpu::BindGroup,
    shading_and_bloom_textures_bind_group: wgpu::BindGroup,
    tone_mapping_texture_bind_group: wgpu::BindGroup,
    pre_gamma_fb_bind_group: Option<wgpu::BindGroup>,
    shading_texture_bind_group: wgpu::BindGroup,
    bloom_pingpong_texture_bind_groups: [wgpu::BindGroup; 2],

    camera_buffers: Vec<wgpu::Buffer>,
    point_lights_buffer: wgpu::Buffer,
    directional_lights_buffer: wgpu::Buffer,
    pbr_shader_options_buffer: wgpu::Buffer,
    bloom_config_buffers: [wgpu::Buffer; 2],
    tone_mapping_config_buffer: wgpu::Buffer,
    bones_buffer: GpuBuffer,
    pbr_instances_buffer: GpuBuffer,
    unlit_instances_buffer: GpuBuffer,
    transparent_instances_buffer: GpuBuffer,
    wireframe_instances_buffer: GpuBuffer,

    skyboxes: [BindedSkybox; 2],
    skybox_weights_buffer: wgpu::Buffer,

    point_shadow_map_textures: Texture,
    directional_shadow_map_textures: Texture,
    // only needed if the surface is not srgb, to be able to render the UI in the same resolution as
    // the surface and before the gamma correction that will happen in the surface blit pipeline
    pre_gamma_fb: Option<Texture>,
    shading_texture: Texture,
    tone_mapping_texture: Texture,
    depth_texture: Texture,
    bloom_pingpong_textures: [Texture; 2],
    brdf_lut: Texture,
}

#[derive(Debug)]
pub struct BindableSceneData {
    pub bindable_meshes: Vec<BindableGeometryBuffers>,
    pub bindable_wireframe_meshes: Vec<BindableWireframeMesh>,
    pub bindable_pbr_materials: Vec<BindablePbrMaterial>,
    pub textures: Vec<BindableTexture>,
}

#[derive(Debug, Default)]
pub struct BindedSceneData {
    pub binded_meshes: Vec<BindedGeometryBuffers>,
    pub binded_wireframe_meshes: Vec<BindedWireframeMesh>,
    pub binded_pbr_materials: Vec<BindedPbrMaterial>,
    pub textures: Vec<Texture>,
}

#[derive(Debug, Clone)]
pub enum SkyboxBackgroundPath {
    Cube([GameFilePath; 6]),
    ProcessedCube([GameFilePath; 6]),
    Equirectangular(GameFilePath),
}

#[derive(Debug, Clone)]
pub enum SkyboxEnvironmentHDRPath {
    Equirectangular(GameFilePath),
    ProcessedCube {
        diffuse: GameFilePath,
        specular: GameFilePath,
    },
}

#[derive(Debug, Clone)]
pub struct SkyboxPaths {
    pub background: SkyboxBackgroundPath,
    pub environment_hdr: Option<SkyboxEnvironmentHDRPath>,
}

impl SkyboxPaths {
    pub fn to_flattened_file_paths(&self) -> Vec<GameFilePath> {
        let backgrounds = match &self.background {
            SkyboxBackgroundPath::Cube(paths) => paths.to_vec(),
            SkyboxBackgroundPath::ProcessedCube(paths) => paths.to_vec(),
            SkyboxBackgroundPath::Equirectangular(path) => vec![path.clone()],
        };

        let environment_hdrs = match &self.environment_hdr {
            Some(SkyboxEnvironmentHDRPath::Equirectangular(path)) => vec![path.clone()],
            Some(SkyboxEnvironmentHDRPath::ProcessedCube { diffuse, specular }) => {
                vec![diffuse.clone(), specular.clone()]
            }
            None => vec![],
        };

        backgrounds
            .iter()
            .chain(environment_hdrs.iter())
            .cloned()
            .collect()
    }
}

#[derive(Debug)]
pub enum BindableSkyboxBackground {
    Cube(RawImage),
    CompressedCube(RawImage),
    Equirectangular(RawImage),
}

#[derive(Debug)]
pub enum BindableSkyboxHDREnvironment {
    Equirectangular(RawImage),
    ProcessedCube {
        diffuse: RawImage,
        specular: RawImage,
    },
}

#[derive(Debug)]
pub struct BindableSkybox {
    pub paths: SkyboxPaths,
    pub background: BindableSkyboxBackground,
    pub environment_hdr: Option<BindableSkyboxHDREnvironment>,
}

#[derive(Debug)]
pub struct BindedSkybox {
    pub background: Texture,
    pub diffuse_environment_map: Texture,
    pub specular_environment_map: Texture,
}

#[derive(Debug)]
pub enum SkyboxSlot {
    One,
    Two,
}

impl SkyboxSlot {
    pub fn as_index(&self) -> usize {
        match self {
            SkyboxSlot::One => 0,
            SkyboxSlot::Two => 1,
        }
    }
}

pub struct RendererData {
    pub binded_meshes: Vec<BindedGeometryBuffers>,
    pub binded_wireframe_meshes: Vec<BindedWireframeMesh>,
    pub binded_pbr_materials: Vec<BindedPbrMaterial>,
    pub textures: Vec<Texture>,

    pub tone_mapping_exposure: f32,
    pub bloom_threshold: f32,
    pub bloom_ramp_size: f32,
    pub render_scale: f32,
    pub enable_bloom: bool,
    pub enable_shadows: bool,
    pub enable_wireframe_mode: bool,
    pub draw_node_bounding_spheres: bool,
    pub draw_culling_frustum: bool,
    pub draw_point_light_culling_frusta: bool,
    pub draw_directional_light_culling_frusta: bool,
    pub enable_soft_shadows: bool,
    pub shadow_bias: f32,
    pub soft_shadow_factor: f32,
    pub enable_shadow_debug: bool,
    pub soft_shadow_grid_dims: u32,
    pub camera_node_id: Option<GameNodeId>,
}

pub struct RendererConstantData {
    pub skybox_mesh: BindedGeometryBuffers,

    pub single_texture_bind_group_layout: wgpu::BindGroupLayout,
    pub two_texture_bind_group_layout: wgpu::BindGroupLayout,
    pub single_cube_texture_bind_group_layout: wgpu::BindGroupLayout,
    pub single_uniform_bind_group_layout: wgpu::BindGroupLayout,
    pub two_uniform_bind_group_layout: wgpu::BindGroupLayout,
    pub bones_and_instances_bind_group_layout: wgpu::BindGroupLayout,
    pub pbr_textures_bind_group_layout: wgpu::BindGroupLayout,
    pub environment_textures_bind_group_layout: wgpu::BindGroupLayout,

    pub mesh_pipeline: wgpu::RenderPipeline,
    pub unlit_mesh_pipeline: wgpu::RenderPipeline,
    pub transparent_mesh_pipeline: wgpu::RenderPipeline,
    pub wireframe_pipeline: wgpu::RenderPipeline,
    pub skybox_pipeline: wgpu::RenderPipeline,
    pub tone_mapping_pipeline: wgpu::RenderPipeline,
    pub surface_blit_pipeline: Option<wgpu::RenderPipeline>,
    pub pre_gamma_surface_blit_pipeline: Option<wgpu::RenderPipeline>,
    pub point_shadow_map_pipeline: wgpu::RenderPipeline,
    pub directional_shadow_map_pipeline: wgpu::RenderPipeline,
    pub bloom_threshold_pipeline: wgpu::RenderPipeline,
    pub bloom_blur_pipeline: wgpu::RenderPipeline,
    pub equirectangular_to_cubemap_pipeline: wgpu::RenderPipeline,
    pub equirectangular_to_cubemap_hdr_pipeline: wgpu::RenderPipeline,
    pub diffuse_env_map_gen_pipeline: wgpu::RenderPipeline,
    pub specular_env_map_gen_pipeline: wgpu::RenderPipeline,

    pub cube_mesh_index: usize,
    pub sphere_mesh_index: usize,
    pub plane_mesh_index: usize,
}

pub struct Renderer {
    pub base: WasmNotArc<BaseRenderer>,
    pub data: WasmNotArc<Mutex<RendererData>>,
    pub constant_data: WasmNotArc<RendererConstantData>,
    private_data: Mutex<RendererPrivateData>,
    profiler: Mutex<wgpu_profiler::GpuProfiler>,
}

type PointLightFrustaWithCullingInfo = Vec<Option<(Vec<(Frustum, bool)>, bool)>>;

impl Renderer {
    pub async fn new(
        base: BaseRenderer,
        framebuffer_format: wgpu::TextureFormat,
        framebuffer_size: (u32, u32),
    ) -> Result<Self> {
        let unlit_mesh_shader = base
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: USE_LABELS.then_some("Unlit Mesh Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/unlit_mesh.wgsl").into()),
            });

        let blit_shader = base
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: USE_LABELS.then_some("Blit Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit.wgsl").into()),
            });

        let textured_mesh_shader = base
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: USE_LABELS.then_some("Textured Mesh Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/textured_mesh.wgsl").into()),
            });

        let skybox_shader = base
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: USE_LABELS.then_some("Skybox Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/skybox.wgsl").into()),
            });

        let single_texture_bind_group_layout =
            base.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                    label: USE_LABELS.then_some("single_texture_bind_group_layout"),
                });

        let two_texture_bind_group_layout =
            base.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                    label: USE_LABELS.then_some("two_texture_bind_group_layout"),
                });

        let single_cube_texture_bind_group_layout =
            base.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::Cube,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                    label: USE_LABELS.then_some("single_cube_texture_bind_group_layout"),
                });

        let single_uniform_bind_group_layout =
            base.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                    label: USE_LABELS.then_some("single_uniform_bind_group_layout"),
                });

        let two_uniform_bind_group_layout =
            base.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                    label: USE_LABELS.then_some("two_uniform_bind_group_layout"),
                });

        let pbr_textures_bind_group_layout =
            base.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 8,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 9,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                    label: USE_LABELS.then_some("pbr_textures_bind_group_layout"),
                });

        let environment_textures_bind_group_layout =
            base.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        // skybox_texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::Cube,
                            },
                            count: None,
                        },
                        // skybox_sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // skybox_texture_2
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::Cube,
                            },
                            count: None,
                        },
                        // skybox_weights
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // diffuse_env_map_texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::Cube,
                            },
                            count: None,
                        },
                        // diffuse_env_map_texture_2
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::Cube,
                            },
                            count: None,
                        },
                        // specular_env_map_texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::Cube,
                            },
                            count: None,
                        },
                        // specular_env_map_texture_2
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::Cube,
                            },
                            count: None,
                        },
                        // brdf_lut_texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 8,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        // brdf_lut_sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 9,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // point_shadow_map_textures
                        wgpu::BindGroupLayoutEntry {
                            binding: 10,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            },
                            count: None,
                        },
                        // directional_shadow_map_textures
                        wgpu::BindGroupLayoutEntry {
                            binding: 11,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            },
                            count: None,
                        },
                        // shadow_map_sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 12,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                    ],
                    label: USE_LABELS.then_some("environment_textures_bind_group_layout"),
                });

        let bones_and_instances_bind_group_layout =
            base.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: true,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: true,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                    label: USE_LABELS.then_some("bones_and_instances_bind_group_layout"),
                });

        let camera_lights_and_pbr_shader_options_bind_group_layout = base
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: USE_LABELS
                    .then_some("camera_lights_and_pbr_shader_options_bind_group_layout"),
            });

        let fragment_shader_color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let mesh_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: USE_LABELS.then_some("Mesh Pipeline Layout"),
                    bind_group_layouts: &[
                        &camera_lights_and_pbr_shader_options_bind_group_layout,
                        &environment_textures_bind_group_layout,
                        &bones_and_instances_bind_group_layout,
                        &pbr_textures_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        let mesh_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: USE_LABELS.then_some("Mesh Pipeline"),
            layout: Some(&mesh_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &textured_mesh_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &textured_mesh_shader,
                entry_point: "fs_main",
                targets: fragment_shader_color_targets,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        };

        let mesh_pipeline = base
            .device
            .create_render_pipeline(&mesh_pipeline_descriptor);

        let unlit_mesh_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: USE_LABELS.then_some("Unlit Mesh Pipeline Layout"),
                    bind_group_layouts: &[
                        &camera_lights_and_pbr_shader_options_bind_group_layout,
                        &bones_and_instances_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let mut unlit_mesh_pipeline_descriptor = mesh_pipeline_descriptor.clone();
        unlit_mesh_pipeline_descriptor.label = Some("Unlit Mesh Render Pipeline");
        unlit_mesh_pipeline_descriptor.layout = Some(&unlit_mesh_pipeline_layout);
        let unlit_mesh_pipeline_v_buffers = &[Vertex::desc()];
        unlit_mesh_pipeline_descriptor.vertex = wgpu::VertexState {
            module: &unlit_mesh_shader,
            entry_point: "vs_main",
            buffers: unlit_mesh_pipeline_v_buffers,
        };
        unlit_mesh_pipeline_descriptor.fragment = Some(wgpu::FragmentState {
            module: &unlit_mesh_shader,
            entry_point: "fs_main",
            targets: fragment_shader_color_targets,
        });
        let unlit_mesh_pipeline = base
            .device
            .create_render_pipeline(&unlit_mesh_pipeline_descriptor);

        let transparent_fragment_shader_color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let mut transparent_mesh_pipeline_descriptor = unlit_mesh_pipeline_descriptor.clone();
        transparent_mesh_pipeline_descriptor.fragment = Some(wgpu::FragmentState {
            module: &unlit_mesh_shader,
            entry_point: "fs_main",
            targets: transparent_fragment_shader_color_targets,
        });
        if let Some(depth_stencil) = &mut transparent_mesh_pipeline_descriptor.depth_stencil {
            depth_stencil.depth_write_enabled = false;
        }
        let transparent_mesh_pipeline = base
            .device
            .create_render_pipeline(&transparent_mesh_pipeline_descriptor);

        let mut wireframe_pipeline_descriptor = unlit_mesh_pipeline_descriptor.clone();
        wireframe_pipeline_descriptor.label = Some("Wireframe Render Pipeline");
        let wireframe_mesh_pipeline_v_buffers = &[Vertex::desc()];
        wireframe_pipeline_descriptor.vertex.buffers = wireframe_mesh_pipeline_v_buffers;
        wireframe_pipeline_descriptor.primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            ..Default::default()
        };
        let wireframe_pipeline = base
            .device
            .create_render_pipeline(&wireframe_pipeline_descriptor);

        let bloom_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        &single_texture_bind_group_layout,
                        &single_uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let bloom_threshold_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: USE_LABELS.then_some("Bloom Threshold Pipeline"),
            layout: Some(&bloom_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: "bloom_threshold_fs_main",
                targets: fragment_shader_color_targets,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        };
        let bloom_threshold_pipeline = base
            .device
            .create_render_pipeline(&bloom_threshold_pipeline_descriptor);

        let bloom_blur_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: USE_LABELS.then_some("Bloom Blur Pipeline"),
            layout: Some(&bloom_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: "bloom_blur_fs_main",
                targets: fragment_shader_color_targets,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        };
        let bloom_blur_pipeline = base
            .device
            .create_render_pipeline(&bloom_blur_pipeline_descriptor);

        let surface_pipelines = {
            let surface_blit_color_targets = &[Some(wgpu::ColorTargetState {
                format: framebuffer_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })];
            let surface_blit_pipeline_layout =
                base.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[
                            &single_texture_bind_group_layout,
                            &single_uniform_bind_group_layout,
                        ],
                        push_constant_ranges: &[],
                    });
            let surface_blit_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
                label: USE_LABELS.then_some("Surface Blit Render Pipeline"),
                layout: Some(&surface_blit_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &blit_shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blit_shader,
                    entry_point: "surface_blit_fs_main",
                    targets: surface_blit_color_targets,
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            };
            let surface_blit_pipeline = base
                .device
                .create_render_pipeline(&surface_blit_pipeline_descriptor);

            let mut pre_gamma_surface_blit_pipeline_descriptor =
                surface_blit_pipeline_descriptor.clone();
            let pre_gamma_surface_blit_pipeline_layout =
                base.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&single_texture_bind_group_layout],
                        push_constant_ranges: &[],
                    });
            pre_gamma_surface_blit_pipeline_descriptor.layout =
                Some(&pre_gamma_surface_blit_pipeline_layout);
            pre_gamma_surface_blit_pipeline_descriptor.fragment = Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: "fs_main",
                targets: fragment_shader_color_targets,
            });
            let pre_gamma_surface_blit_pipeline = base
                .device
                .create_render_pipeline(&pre_gamma_surface_blit_pipeline_descriptor);

            Some((surface_blit_pipeline, pre_gamma_surface_blit_pipeline))
        };

        let (surface_blit_pipeline, pre_gamma_surface_blit_pipeline) = match surface_pipelines {
            Some((surface_blit_pipeline, pre_gamma_surface_blit_pipeline)) => (
                Some(surface_blit_pipeline),
                Some(pre_gamma_surface_blit_pipeline),
            ),
            None => (None, None),
        };

        let tone_mapping_colors_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::REPLACE,
            }),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let tone_mapping_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        &two_texture_bind_group_layout,
                        &single_uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let tone_mapping_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: USE_LABELS.then_some("Tone Mapping Render Pipeline"),
            layout: Some(&tone_mapping_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: "tone_mapping_fs_main",
                targets: tone_mapping_colors_targets,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        };
        let tone_mapping_pipeline = base
            .device
            .create_render_pipeline(&tone_mapping_pipeline_descriptor);

        let skybox_pipeline_primitive_state = wgpu::PrimitiveState {
            front_face: wgpu::FrontFace::Cw,
            ..Default::default()
        };
        let skybox_depth_stencil_state = Some(wgpu::DepthStencilState {
            format: Texture::DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::GreaterEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });
        let skybox_render_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: USE_LABELS.then_some("Skybox Render Pipeline Layout"),
                    bind_group_layouts: &[
                        &environment_textures_bind_group_layout,
                        &camera_lights_and_pbr_shader_options_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        let skybox_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: USE_LABELS.then_some("Skybox Render Pipeline"),
            layout: Some(&skybox_render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &skybox_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &skybox_shader,
                entry_point: "background_fs_main",
                targets: fragment_shader_color_targets,
            }),
            primitive: skybox_pipeline_primitive_state,
            depth_stencil: skybox_depth_stencil_state,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        };
        let skybox_pipeline = base
            .device
            .create_render_pipeline(&skybox_pipeline_descriptor);

        let equirectangular_to_cubemap_color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let equirectangular_to_cubemap_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: USE_LABELS
                        .then_some("Equirectangular To Cubemap Render Pipeline Layout"),
                    bind_group_layouts: &[
                        &single_texture_bind_group_layout,
                        &single_uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        let equirectangular_to_cubemap_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: USE_LABELS.then_some("Equirectangular To Cubemap Render Pipeline"),
            layout: Some(&equirectangular_to_cubemap_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &skybox_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &skybox_shader,
                entry_point: "equirectangular_to_cubemap_fs_main",
                targets: equirectangular_to_cubemap_color_targets,
            }),
            primitive: skybox_pipeline_primitive_state,
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        };
        let equirectangular_to_cubemap_pipeline = base
            .device
            .create_render_pipeline(&equirectangular_to_cubemap_pipeline_descriptor);

        let mut equirectangular_to_cubemap_hdr_pipeline_descriptor =
            equirectangular_to_cubemap_pipeline_descriptor.clone();
        equirectangular_to_cubemap_hdr_pipeline_descriptor
            .fragment
            .as_mut()
            .unwrap()
            .targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let equirectangular_to_cubemap_hdr_pipeline = base
            .device
            .create_render_pipeline(&equirectangular_to_cubemap_hdr_pipeline_descriptor);

        let diffuse_env_map_color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let diffuse_env_map_gen_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: USE_LABELS.then_some("diffuse env map Gen Pipeline Layout"),
                    bind_group_layouts: &[
                        &single_cube_texture_bind_group_layout,
                        &single_uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let diffuse_env_map_gen_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: USE_LABELS.then_some("diffuse env map Gen Pipeline"),
            layout: Some(&diffuse_env_map_gen_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &skybox_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &skybox_shader,
                entry_point: "diffuse_env_map_gen_fs_main",
                targets: diffuse_env_map_color_targets,
            }),
            primitive: skybox_pipeline_primitive_state,
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        };
        let diffuse_env_map_gen_pipeline = base
            .device
            .create_render_pipeline(&diffuse_env_map_gen_pipeline_descriptor);

        let specular_env_map_color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let specular_env_map_gen_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: USE_LABELS.then_some("specular env map Gen Pipeline Layout"),
                    bind_group_layouts: &[
                        &single_cube_texture_bind_group_layout,
                        &two_uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        let specular_env_map_gen_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: USE_LABELS.then_some("specular env map Gen Pipeline"),
            layout: Some(&specular_env_map_gen_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &skybox_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &skybox_shader,
                entry_point: "specular_env_map_gen_fs_main",
                targets: specular_env_map_color_targets,
            }),
            primitive: skybox_pipeline_primitive_state,
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        };
        let specular_env_map_gen_pipeline = base
            .device
            .create_render_pipeline(&specular_env_map_gen_pipeline_descriptor);

        let brdf_lut_gen_color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rg16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let brdf_lut_gen_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: USE_LABELS.then_some("Brdf Lut Gen Pipeline Layout"),
                    bind_group_layouts: &[],
                    push_constant_ranges: &[],
                });

        let brdf_lut_gen_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: USE_LABELS.then_some("Brdf Lut Gen Pipeline"),
            layout: Some(&brdf_lut_gen_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: "brdf_lut_gen_fs_main",
                targets: brdf_lut_gen_color_targets,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        };
        let brdf_lut_gen_pipeline = base
            .device
            .create_render_pipeline(&brdf_lut_gen_pipeline_descriptor);

        let shadow_map_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: USE_LABELS.then_some("Shadow Map Pipeline Layout"),
                    bind_group_layouts: &[
                        &camera_lights_and_pbr_shader_options_bind_group_layout,
                        &bones_and_instances_bind_group_layout,
                        &pbr_textures_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let point_shadow_map_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: USE_LABELS.then_some("Point Shadow Map Pipeline"),
            layout: Some(&shadow_map_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &textured_mesh_shader,
                entry_point: "shadow_map_vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &textured_mesh_shader,
                entry_point: "point_shadow_map_fs_main",
                targets: &[],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        };
        let point_shadow_map_pipeline = base
            .device
            .create_render_pipeline(&point_shadow_map_pipeline_descriptor);

        let directional_shadow_map_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: USE_LABELS.then_some("Directional Shadow Map Pipeline"),
            layout: Some(&shadow_map_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &textured_mesh_shader,
                entry_point: "shadow_map_vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        };
        let directional_shadow_map_pipeline = base
            .device
            .create_render_pipeline(&directional_shadow_map_pipeline_descriptor);

        let initial_render_scale = 1.0;

        let cube_mesh = BasicMesh::new(include_bytes!("models/cube.obj"))?;

        let skybox_mesh = Self::bind_geometry_buffers_for_basic_mesh_impl(&base.device, &cube_mesh);

        let mut constant_data = RendererConstantData {
            skybox_mesh,

            single_texture_bind_group_layout,
            two_texture_bind_group_layout,
            single_cube_texture_bind_group_layout,
            single_uniform_bind_group_layout,
            two_uniform_bind_group_layout,
            bones_and_instances_bind_group_layout,
            pbr_textures_bind_group_layout,
            environment_textures_bind_group_layout,

            mesh_pipeline,
            unlit_mesh_pipeline,
            transparent_mesh_pipeline,
            wireframe_pipeline,
            skybox_pipeline,
            tone_mapping_pipeline,
            surface_blit_pipeline,
            pre_gamma_surface_blit_pipeline,
            point_shadow_map_pipeline,
            directional_shadow_map_pipeline,
            bloom_threshold_pipeline,
            bloom_blur_pipeline,
            equirectangular_to_cubemap_pipeline,
            equirectangular_to_cubemap_hdr_pipeline,
            diffuse_env_map_gen_pipeline,
            specular_env_map_gen_pipeline,

            cube_mesh_index: 0,
            sphere_mesh_index: 0,
            plane_mesh_index: 0,
        };

        let pre_gamma_fb = (!framebuffer_format.is_srgb()).then(|| {
            Texture::create_scaled_surface_texture(&base, framebuffer_size, 1.0, "pre_gamma_fb")
        });

        let shading_texture = Texture::create_scaled_surface_texture(
            &base,
            framebuffer_size,
            initial_render_scale,
            "shading_texture",
        );
        let bloom_pingpong_textures = [
            Texture::create_scaled_surface_texture(
                &base,
                framebuffer_size,
                initial_render_scale,
                "bloom_texture_1",
            ),
            Texture::create_scaled_surface_texture(
                &base,
                framebuffer_size,
                initial_render_scale,
                "bloom_texture_2",
            ),
        ];
        let tone_mapping_texture = Texture::create_scaled_surface_texture(
            &base,
            framebuffer_size,
            initial_render_scale,
            "tone_mapping_texture",
        );

        let pre_gamma_fb_bind_group;
        let shading_texture_bind_group;
        let tone_mapping_texture_bind_group;
        let shading_and_bloom_textures_bind_group;
        let bloom_pingpong_texture_bind_groups;
        {
            let sampler_cache_guard = base.sampler_cache.lock().unwrap();

            pre_gamma_fb_bind_group = pre_gamma_fb.as_ref().map(|pre_gamma_fb| {
                base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &constant_data.single_texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&pre_gamma_fb.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                sampler_cache_guard
                                    .get_sampler_by_index(pre_gamma_fb.sampler_index),
                            ),
                        },
                    ],
                    label: USE_LABELS.then_some("pre_gamma_fb_bind_group"),
                })
            });
            shading_texture_bind_group =
                base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &constant_data.single_texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&shading_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                sampler_cache_guard
                                    .get_sampler_by_index(shading_texture.sampler_index),
                            ),
                        },
                    ],
                    label: USE_LABELS.then_some("shading_texture_bind_group"),
                });
            tone_mapping_texture_bind_group =
                base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &constant_data.single_texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &tone_mapping_texture.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                sampler_cache_guard
                                    .get_sampler_by_index(tone_mapping_texture.sampler_index),
                            ),
                        },
                    ],
                    label: USE_LABELS.then_some("tone_mapping_texture_bind_group"),
                });
            shading_and_bloom_textures_bind_group =
                base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &constant_data.two_texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&shading_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                sampler_cache_guard
                                    .get_sampler_by_index(shading_texture.sampler_index),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                &bloom_pingpong_textures[0].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(
                                sampler_cache_guard
                                    .get_sampler_by_index(bloom_pingpong_textures[0].sampler_index),
                            ),
                        },
                    ],
                    label: USE_LABELS.then_some("surface_blit_textures_bind_group"),
                });

            bloom_pingpong_texture_bind_groups = [
                base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &constant_data.single_texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &bloom_pingpong_textures[0].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                sampler_cache_guard
                                    .get_sampler_by_index(bloom_pingpong_textures[0].sampler_index),
                            ),
                        },
                    ],
                    label: USE_LABELS.then_some("bloom_texture_bind_group_1"),
                }),
                base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &constant_data.single_texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &bloom_pingpong_textures[1].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                sampler_cache_guard
                                    .get_sampler_by_index(bloom_pingpong_textures[1].sampler_index),
                            ),
                        },
                    ],
                    label: USE_LABELS.then_some("bloom_texture_bind_group_2"),
                }),
            ];
        }

        let bloom_config_buffers = [
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: USE_LABELS.then_some("Bloom Config Buffer 0"),
                    contents: bytemuck::cast_slice(&[0f32, 0f32, 0f32, 0.0f32]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                }),
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: USE_LABELS.then_some("Bloom Config Buffer 1"),
                    contents: bytemuck::cast_slice(&[0f32, 0f32, 0f32, 0.0f32]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                }),
        ];

        let bloom_config_bind_groups = [
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &constant_data.single_uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bloom_config_buffers[0].as_entire_binding(),
                }],
                label: USE_LABELS.then_some("bloom_config_bind_group_0"),
            }),
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &constant_data.single_uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bloom_config_buffers[1].as_entire_binding(),
                }],
                label: USE_LABELS.then_some("bloom_config_bind_group_1"),
            }),
        ];

        let tone_mapping_config_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: USE_LABELS.then_some("Tone Mapping Config Buffer"),
                    contents: bytemuck::cast_slice(&[0f32, 0f32, 0f32, 0f32]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let tone_mapping_config_bind_group =
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &constant_data.single_uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tone_mapping_config_buffer.as_entire_binding(),
                }],
                label: USE_LABELS.then_some("tone_mapping_config_bind_group"),
            });

        let depth_texture = Texture::create_depth_texture(
            &base,
            framebuffer_size,
            initial_render_scale,
            "depth_texture",
        );

        let start = crate::time::Instant::now();

        let skybox_dim = 32;
        let sky_color = [8, 113, 184, 255];
        let sun_color = [253, 251, 211, 255];
        let background_texture_er = {
            let image = image::RgbaImage::from_pixel(skybox_dim, skybox_dim, sky_color.into());
            Texture::from_decoded_image(
                &base,
                &image,
                image.dimensions(),
                1,
                Some("skybox_image texture"),
                wgpu::TextureFormat::Rgba8UnormSrgb.into(),
                false,
                &SamplerDescriptor {
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                },
            )?
        };

        let background_texture_rad = {
            let pixel_count = skybox_dim * skybox_dim;
            let color_hdr =
                sun_color.map(|val| Float16(half::f16::from_f32(0.2 * val as f32 / 255.0)));
            let mut image_raw: Vec<[Float16; 4]> = Vec::with_capacity(pixel_count as usize);
            image_raw.resize(pixel_count as usize, color_hdr);
            let texture_er = Texture::from_decoded_image(
                &base,
                bytemuck::cast_slice(&image_raw),
                (skybox_dim, skybox_dim),
                1,
                None,
                wgpu::TextureFormat::Rgba16Float.into(),
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
                &base,
                &constant_data,
                wgpu::TextureFormat::Rgba16Float,
                None,
                &texture_er,
                false,
            )?
        };

        fn make_skybox(
            base: &BaseRenderer,
            constant_data: &RendererConstantData,
            background_texture_er: &Texture,
            background_texture_rad: &Texture,
        ) -> Result<BindedSkybox> {
            Ok(BindedSkybox {
                background: Texture::create_cubemap_from_equirectangular(
                    base,
                    constant_data,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    None,
                    background_texture_er,
                    false, // an artifact occurs between the edges of the texture with mipmaps enabled
                )?,
                diffuse_environment_map: Texture::create_diffuse_env_map(
                    base,
                    constant_data,
                    Some("diffuse env map"),
                    background_texture_rad,
                ),
                specular_environment_map: Texture::create_specular_env_map(
                    base,
                    constant_data,
                    Some("specular env map 2"),
                    background_texture_rad,
                ),
            })
        }

        let skyboxes = [
            make_skybox(
                &base,
                &constant_data,
                &background_texture_er,
                &background_texture_rad,
            )?,
            make_skybox(
                &base,
                &constant_data,
                &background_texture_er,
                &background_texture_rad,
            )?,
        ];

        let skybox_weights = [1.0, 0.0];

        let skybox_weights_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: USE_LABELS.then_some("Skybox Weights Buffer"),
                    contents: bytemuck::cast_slice(&[
                        skybox_weights[0],
                        skybox_weights[1],
                        0.0,
                        0.0,
                    ]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let brdf_lut = Texture::create_brdf_lut(&base, &brdf_lut_gen_pipeline);

        let skybox_gen_time = start.elapsed();
        log::debug!("skybox_gen_time={skybox_gen_time:?}");

        let initial_point_lights_buffer: Vec<u8> = (0..(MAX_LIGHT_COUNT
            * std::mem::size_of::<PointLightUniform>()))
            .map(|_| 0u8)
            .collect();
        let point_lights_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: USE_LABELS.then_some("Point Lights Buffer"),
                    contents: &initial_point_lights_buffer,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let initial_directional_lights_buffer: Vec<u8> = (0..(MAX_LIGHT_COUNT
            * std::mem::size_of::<DirectionalLightUniform>()))
            .map(|_| 0u8)
            .collect();
        let directional_lights_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: USE_LABELS.then_some("Directional Lights Buffer"),
                    contents: &initial_directional_lights_buffer,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let enable_soft_shadows = Default::default();
        let shadow_bias = Default::default();
        let soft_shadow_factor = Default::default();
        let enable_shadow_debug = Default::default();
        let soft_shadow_grid_dims = Default::default();
        let initial_pbr_shader_options_buffer = make_pbr_shader_options_uniform_buffer(
            enable_soft_shadows,
            shadow_bias,
            soft_shadow_factor,
            enable_shadow_debug,
            soft_shadow_grid_dims,
        );
        let pbr_shader_options_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: USE_LABELS.then_some("PBR Shader Options Buffer"),
                    contents: bytemuck::cast_slice(&[initial_pbr_shader_options_buffer]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let bones_buffer = GpuBuffer::empty(
            &base.device,
            std::mem::size_of::<Mat4>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let pbr_instances_buffer = GpuBuffer::empty(
            &base.device,
            std::mem::size_of::<GpuPbrMeshInstance>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let unlit_instances_buffer = GpuBuffer::empty(
            &base.device,
            std::mem::size_of::<GpuUnlitMeshInstance>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let transparent_instances_buffer = GpuBuffer::empty(
            &base.device,
            std::mem::size_of::<GpuUnlitMeshInstance>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let wireframe_instances_buffer = GpuBuffer::empty(
            &base.device,
            std::mem::size_of::<GpuWireframeMeshInstance>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let bones_and_pbr_instances_bind_group =
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &constant_data.bones_and_instances_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: bones_buffer.src(),
                            offset: 0,
                            size: NonZeroU64::new(bones_buffer.length_bytes().try_into().unwrap()),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: pbr_instances_buffer.src(),
                            offset: 0,
                            size: NonZeroU64::new(
                                pbr_instances_buffer.length_bytes().try_into().unwrap(),
                            ),
                        }),
                    },
                ],
                label: USE_LABELS.then_some("bones_and_pbr_instances_bind_group"),
            });

        let bones_and_unlit_instances_bind_group =
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &constant_data.bones_and_instances_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: bones_buffer.src(),
                            offset: 0,
                            size: NonZeroU64::new(bones_buffer.length_bytes().try_into().unwrap()),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: unlit_instances_buffer.src(),
                            offset: 0,
                            size: NonZeroU64::new(
                                unlit_instances_buffer.length_bytes().try_into().unwrap(),
                            ),
                        }),
                    },
                ],
                label: USE_LABELS.then_some("bones_and_unlit_instances_bind_group"),
            });

        let bones_and_transparent_instances_bind_group =
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &constant_data.bones_and_instances_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: bones_buffer.src(),
                            offset: 0,
                            size: NonZeroU64::new(bones_buffer.length_bytes().try_into().unwrap()),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: transparent_instances_buffer.src(),
                            offset: 0,
                            size: NonZeroU64::new(
                                transparent_instances_buffer
                                    .length_bytes()
                                    .try_into()
                                    .unwrap(),
                            ),
                        }),
                    },
                ],
                label: USE_LABELS.then_some("bones_and_transparent_instances_bind_group"),
            });

        let bones_and_wireframe_instances_bind_group =
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &constant_data.bones_and_instances_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: bones_buffer.src(),
                            offset: 0,
                            size: NonZeroU64::new(bones_buffer.length_bytes().try_into().unwrap()),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: wireframe_instances_buffer.src(),
                            offset: 0,
                            size: NonZeroU64::new(
                                wireframe_instances_buffer
                                    .length_bytes()
                                    .try_into()
                                    .unwrap(),
                            ),
                        }),
                    },
                ],
                label: USE_LABELS.then_some("bones_and_wireframe_instances_bind_group"),
            });

        let point_shadow_map_textures = Texture::create_depth_texture_array(
            &base,
            (
                6 * POINT_LIGHT_SHADOW_MAP_RESOLUTION,
                POINT_LIGHT_SHADOW_MAP_RESOLUTION,
            ),
            Some("point_shadow_map_texture"),
            POINT_LIGHT_SHOW_MAP_COUNT,
        );

        let directional_shadow_map_textures = Texture::create_depth_texture_array(
            &base,
            (
                DIRECTIONAL_LIGHT_SHADOW_MAP_RESOLUTION,
                DIRECTIONAL_LIGHT_SHADOW_MAP_RESOLUTION,
            ),
            Some("directional_shadow_map_texture"),
            DIRECTIONAL_LIGHT_SHOW_MAP_COUNT,
        );

        let environment_textures_bind_group = Self::get_environment_textures_bind_group(
            &base,
            &constant_data,
            &skyboxes,
            &skybox_weights_buffer,
            &brdf_lut,
            &point_shadow_map_textures,
            &directional_shadow_map_textures,
        );

        let mut data = RendererData {
            binded_meshes: vec![],
            binded_wireframe_meshes: vec![],
            binded_pbr_materials: vec![],
            textures: vec![],

            tone_mapping_exposure: 1.0,
            bloom_threshold: 0.8,
            bloom_ramp_size: 0.2,
            render_scale: initial_render_scale,
            enable_bloom: true,
            enable_shadows: true,
            enable_wireframe_mode: false,
            draw_node_bounding_spheres: false,
            draw_culling_frustum: false,
            draw_point_light_culling_frusta: false,
            draw_directional_light_culling_frusta: false,
            enable_soft_shadows,
            shadow_bias,
            soft_shadow_factor,
            enable_shadow_debug,
            soft_shadow_grid_dims,
            camera_node_id: None,
        };

        constant_data.cube_mesh_index = Self::bind_basic_mesh(&base, &mut data, &cube_mesh, true);

        let sphere_mesh = BasicMesh::new(include_bytes!("models/sphere.obj"))?;
        constant_data.sphere_mesh_index =
            Self::bind_basic_mesh(&base, &mut data, &sphere_mesh, true);

        let plane_mesh = BasicMesh::new(include_bytes!("models/plane.obj"))?;
        constant_data.plane_mesh_index = Self::bind_basic_mesh(&base, &mut data, &plane_mesh, true);

        // buffer up to 4 frames
        let profiler = wgpu_profiler::GpuProfiler::new(&base.adapter, &base.device, &base.queue, 4);

        let renderer = Self {
            base: WasmNotArc::new(base),
            data: WasmNotArc::new(Mutex::new(data)),
            constant_data: WasmNotArc::new(constant_data),

            private_data: Mutex::new(RendererPrivateData {
                all_bone_transforms: AllBoneTransforms {
                    buffer: vec![],
                    animated_bone_transforms: vec![],
                    identity_slice: (0, 0),
                },
                all_pbr_instances: ChunkedBuffer::new(),
                all_pbr_instances_culling_masks: vec![],
                all_unlit_instances: ChunkedBuffer::new(),
                all_transparent_instances: ChunkedBuffer::new(),
                all_wireframe_instances: ChunkedBuffer::new(),
                debug_node_bounding_spheres_nodes: vec![],
                debug_culling_frustum_nodes: vec![],
                debug_culling_frustum_mesh_index: None,

                bloom_threshold_cleared: true,
                frustum_culling_lock: CullingFrustumLock::None,
                skybox_weights,

                camera_lights_and_pbr_shader_options_bind_group_layout,

                camera_lights_and_pbr_shader_options_bind_groups: vec![],
                bones_and_pbr_instances_bind_group,
                bones_and_unlit_instances_bind_group,
                bones_and_transparent_instances_bind_group,
                bones_and_wireframe_instances_bind_group,
                bloom_config_bind_groups,
                tone_mapping_config_bind_group,
                environment_textures_bind_group,
                shading_and_bloom_textures_bind_group,
                tone_mapping_texture_bind_group,
                pre_gamma_fb_bind_group,
                shading_texture_bind_group,
                bloom_pingpong_texture_bind_groups,

                camera_buffers: vec![],
                point_lights_buffer,
                directional_lights_buffer,
                bloom_config_buffers,
                tone_mapping_config_buffer,
                pbr_shader_options_buffer,
                bones_buffer,
                pbr_instances_buffer,
                unlit_instances_buffer,
                transparent_instances_buffer,
                wireframe_instances_buffer,

                skyboxes,
                skybox_weights_buffer,

                point_shadow_map_textures,
                directional_shadow_map_textures,

                pre_gamma_fb,
                shading_texture,
                tone_mapping_texture,
                depth_texture,
                bloom_pingpong_textures,
                brdf_lut,
            }),

            profiler: Mutex::new(profiler),
        };

        Ok(renderer)
    }

    #[profiling::function]
    pub fn make_pbr_textures_bind_group(
        base: &BaseRenderer,
        constant_data: &RendererConstantData,
        pbr_textures: &PbrTextures,
        use_gltf_defaults: bool,
    ) -> Result<wgpu::BindGroup> {
        let auto_generated_diffuse_texture;
        let diffuse_texture = match pbr_textures.base_color {
            Some(diffuse_texture) => diffuse_texture,
            None => {
                auto_generated_diffuse_texture =
                    base.get_default_texture(DefaultTextureType::BaseColor)?;
                &auto_generated_diffuse_texture
            }
        };
        let auto_generated_normal_map;
        let normal_map = match pbr_textures.normal {
            Some(normal_map) => normal_map,
            None => {
                auto_generated_normal_map = base.get_default_texture(DefaultTextureType::Normal)?;
                &auto_generated_normal_map
            }
        };
        let auto_generated_metallic_roughness_map;
        let metallic_roughness_map = match pbr_textures.metallic_roughness {
            Some(metallic_roughness_map) => metallic_roughness_map,
            None => {
                auto_generated_metallic_roughness_map =
                    base.get_default_texture(if use_gltf_defaults {
                        DefaultTextureType::MetallicRoughnessGLTF
                    } else {
                        DefaultTextureType::MetallicRoughness
                    })?;
                &auto_generated_metallic_roughness_map
            }
        };
        let auto_generated_emissive_map;
        let emissive_map = match pbr_textures.emissive {
            Some(emissive_map) => emissive_map,
            None => {
                auto_generated_emissive_map = base.get_default_texture(if use_gltf_defaults {
                    DefaultTextureType::EmissiveGLTF
                } else {
                    DefaultTextureType::Emissive
                })?;
                &auto_generated_emissive_map
            }
        };
        let auto_generated_ambient_occlusion_map;
        let ambient_occlusion_map = match pbr_textures.ambient_occlusion {
            Some(ambient_occlusion_map) => ambient_occlusion_map,
            None => {
                auto_generated_ambient_occlusion_map =
                    base.get_default_texture(DefaultTextureType::AmbientOcclusion)?;
                &auto_generated_ambient_occlusion_map
            }
        };

        let sampler_cache_guard = base.sampler_cache.lock().unwrap();

        let textures_bind_group = base.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &constant_data.pbr_textures_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(
                        sampler_cache_guard.get_sampler_by_index(diffuse_texture.sampler_index),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&normal_map.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(
                        sampler_cache_guard.get_sampler_by_index(normal_map.sampler_index),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&metallic_roughness_map.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(
                        sampler_cache_guard
                            .get_sampler_by_index(metallic_roughness_map.sampler_index),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&emissive_map.view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(
                        sampler_cache_guard.get_sampler_by_index(emissive_map.sampler_index),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&ambient_occlusion_map.view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(
                        sampler_cache_guard
                            .get_sampler_by_index(ambient_occlusion_map.sampler_index),
                    ),
                },
            ],
            label: USE_LABELS.then_some("InstancedMeshComponent textures_bind_group"),
        });

        Ok(textures_bind_group)
    }

    /// generate_wireframe_mesh: generates wireframe counterpart, making the mesh renderable in wireframe mode
    pub fn bind_basic_mesh(
        base: &BaseRenderer,
        data: &mut RendererData,
        mesh: &BasicMesh,
        generate_wireframe_mesh: bool,
    ) -> usize {
        let geometry_buffers = Self::bind_geometry_buffers_for_basic_mesh(base, mesh);

        data.binded_meshes.push(geometry_buffers);
        let mesh_index = data.binded_meshes.len() - 1;

        if generate_wireframe_mesh {
            let wireframe_index_buffer =
                Self::make_wireframe_index_buffer_for_basic_mesh(base, mesh);
            data.binded_wireframe_meshes.push(BindedWireframeMesh {
                source_mesh_index: mesh_index,
                index_buffer: BindedIndexBuffer {
                    buffer: wireframe_index_buffer,
                    format: wgpu::IndexFormat::Uint16,
                },
            });
        }

        mesh_index
    }

    pub fn unbind_mesh(data: &RendererData, mesh_index: usize) {
        let geometry_buffers = &data.binded_meshes[mesh_index];
        let wireframe_mesh = data
            .binded_wireframe_meshes
            .iter()
            .find(|wireframe_mesh| wireframe_mesh.source_mesh_index == mesh_index)
            .unwrap();

        geometry_buffers.vertex_buffer.destroy();
        geometry_buffers.index_buffer.buffer.destroy();
        wireframe_mesh.index_buffer.buffer.destroy();
    }

    // returns index of mesh in the RenderScene::binded_pbr_meshes list
    pub fn bind_pbr_material(
        base: &BaseRenderer,
        constant_data: &RendererConstantData,
        data: &mut RendererData,
        pbr_textures: &PbrTextures,
        dynamic_pbr_params: DynamicPbrParams,
    ) -> Result<usize> {
        let textures_bind_group =
            Self::make_pbr_textures_bind_group(base, constant_data, pbr_textures, false)?;

        data.binded_pbr_materials.push(BindedPbrMaterial {
            dynamic_pbr_params,
            textures_bind_group,
        });
        let material_index = data.binded_pbr_materials.len() - 1;

        Ok(material_index)
    }

    fn bind_geometry_buffers_for_basic_mesh(
        base: &BaseRenderer,
        mesh: &BasicMesh,
    ) -> BindedGeometryBuffers {
        Self::bind_geometry_buffers_for_basic_mesh_impl(&base.device, mesh)
    }

    fn bind_geometry_buffers_for_basic_mesh_impl(
        device: &wgpu::Device,
        mesh: &BasicMesh,
    ) -> BindedGeometryBuffers {
        let vertex_buffer = GpuBuffer::from_bytes(
            device,
            bytemuck::cast_slice(&mesh.vertices),
            std::mem::size_of::<Vertex>(),
            wgpu::BufferUsages::VERTEX,
        );

        let index_buffer = GpuBuffer::from_bytes(
            device,
            bytemuck::cast_slice(&mesh.indices),
            std::mem::size_of::<u16>(),
            wgpu::BufferUsages::INDEX,
        );

        let bounding_box = {
            let mut min_point = Vec3::new(
                mesh.vertices[0].position[0],
                mesh.vertices[0].position[1],
                mesh.vertices[0].position[2],
            );
            let mut max_point = min_point;
            for vertex in &mesh.vertices {
                min_point.x = min_point.x.min(vertex.position[0]);
                min_point.y = min_point.y.min(vertex.position[1]);
                min_point.z = min_point.z.min(vertex.position[2]);
                max_point.x = max_point.x.max(vertex.position[0]);
                max_point.y = max_point.y.max(vertex.position[1]);
                max_point.z = max_point.z.max(vertex.position[2]);
            }
            crate::collisions::Aabb {
                min: min_point,
                max: max_point,
            }
        };

        BindedGeometryBuffers {
            vertex_buffer,
            index_buffer: BindedIndexBuffer {
                buffer: index_buffer,
                format: wgpu::IndexFormat::Uint16,
            },
            bounding_box,
        }
    }

    fn make_wireframe_index_buffer_for_basic_mesh(
        base: &BaseRenderer,
        mesh: &BasicMesh,
    ) -> GpuBuffer {
        Self::make_wireframe_index_buffer_for_basic_mesh_impl(&base.device, mesh)
    }

    fn make_wireframe_index_buffer_for_basic_mesh_impl(
        device: &wgpu::Device,
        mesh: &BasicMesh,
    ) -> GpuBuffer {
        let index_buffer = GpuBuffer::from_bytes(
            device,
            bytemuck::cast_slice(
                &mesh
                    .indices
                    .chunks(3)
                    .flat_map(|triangle| {
                        vec![
                            triangle[0],
                            triangle[1],
                            triangle[1],
                            triangle[2],
                            triangle[2],
                            triangle[0],
                        ]
                    })
                    .collect::<Vec<_>>(),
            ),
            std::mem::size_of::<u16>(),
            wgpu::BufferUsages::INDEX,
        );

        index_buffer
    }

    pub fn set_vsync(&self, vsync: bool, surface_data: &mut SurfaceData) {
        let new_present_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };

        if surface_data.surface_config.present_mode == new_present_mode {
            return;
        }

        surface_data.surface_config.present_mode = new_present_mode;

        surface_data
            .surface
            .configure(&self.base.device, &surface_data.surface_config);
    }

    pub fn resize_surface(
        &mut self,
        surface_data: &mut SurfaceData,
        new_unscaled_framebuffer_size: winit::dpi::PhysicalSize<u32>,
    ) {
        let new_unscaled_framebuffer_size = (
            new_unscaled_framebuffer_size.width,
            new_unscaled_framebuffer_size.height,
        );
        let (new_width, new_height) = new_unscaled_framebuffer_size;

        surface_data.surface_config.width = new_width;
        surface_data.surface_config.height = new_height;

        surface_data
            .surface
            .configure(&self.base.device, &surface_data.surface_config);

        let data_guard = self.data.lock().unwrap();
        let mut private_data_guard = self.private_data.lock().unwrap();
        let render_scale = data_guard.render_scale;

        private_data_guard.pre_gamma_fb = private_data_guard.pre_gamma_fb.is_some().then(|| {
            Texture::create_scaled_surface_texture(
                &self.base,
                new_unscaled_framebuffer_size,
                1.0,
                "pre_gamma_fb",
            )
        });
        private_data_guard.shading_texture = Texture::create_scaled_surface_texture(
            &self.base,
            new_unscaled_framebuffer_size,
            render_scale,
            "shading_texture",
        );
        private_data_guard.bloom_pingpong_textures = [
            Texture::create_scaled_surface_texture(
                &self.base,
                new_unscaled_framebuffer_size,
                render_scale,
                "bloom_texture_1",
            ),
            Texture::create_scaled_surface_texture(
                &self.base,
                new_unscaled_framebuffer_size,
                render_scale,
                "bloom_texture_2",
            ),
        ];
        private_data_guard.tone_mapping_texture = Texture::create_scaled_surface_texture(
            &self.base,
            new_unscaled_framebuffer_size,
            render_scale,
            "tone_mapping_texture",
        );
        private_data_guard.depth_texture = Texture::create_depth_texture(
            &self.base,
            new_unscaled_framebuffer_size,
            render_scale,
            "depth_texture",
        );

        let device = &self.base.device;
        let single_texture_bind_group_layout = &self.constant_data.single_texture_bind_group_layout;
        let two_texture_bind_group_layout = &self.constant_data.two_texture_bind_group_layout;

        let sampler_cache_guard = self.base.sampler_cache.lock().unwrap();
        private_data_guard.pre_gamma_fb_bind_group = private_data_guard
            .pre_gamma_fb_bind_group
            .is_some()
            .then(|| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: single_texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &private_data_guard.pre_gamma_fb.as_ref().unwrap().view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                sampler_cache_guard.get_sampler_by_index(
                                    private_data_guard
                                        .pre_gamma_fb
                                        .as_ref()
                                        .unwrap()
                                        .sampler_index,
                                ),
                            ),
                        },
                    ],
                    label: USE_LABELS.then_some("pre_gamma_fb_bind_group"),
                })
            });

        private_data_guard.shading_texture_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: single_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &private_data_guard.shading_texture.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            sampler_cache_guard.get_sampler_by_index(
                                private_data_guard.shading_texture.sampler_index,
                            ),
                        ),
                    },
                ],
                label: USE_LABELS.then_some("shading_texture_bind_group"),
            });

        private_data_guard.tone_mapping_texture_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: single_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &private_data_guard.tone_mapping_texture.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            sampler_cache_guard.get_sampler_by_index(
                                private_data_guard.tone_mapping_texture.sampler_index,
                            ),
                        ),
                    },
                ],
                label: USE_LABELS.then_some("tone_mapping_texture_bind_group"),
            });
        private_data_guard.shading_and_bloom_textures_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: two_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &private_data_guard.shading_texture.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            sampler_cache_guard.get_sampler_by_index(
                                private_data_guard.shading_texture.sampler_index,
                            ),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &private_data_guard.bloom_pingpong_textures[0].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(
                            sampler_cache_guard.get_sampler_by_index(
                                private_data_guard.bloom_pingpong_textures[0].sampler_index,
                            ),
                        ),
                    },
                ],
                label: USE_LABELS.then_some("shading_and_bloom_textures_bind_group"),
            });
        private_data_guard.bloom_pingpong_texture_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: single_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &private_data_guard.bloom_pingpong_textures[0].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            sampler_cache_guard.get_sampler_by_index(
                                private_data_guard.bloom_pingpong_textures[0].sampler_index,
                            ),
                        ),
                    },
                ],
                label: USE_LABELS.then_some("bloom_texture_bind_group_1"),
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: single_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &private_data_guard.bloom_pingpong_textures[1].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            sampler_cache_guard.get_sampler_by_index(
                                private_data_guard.bloom_pingpong_textures[1].sampler_index,
                            ),
                        ),
                    },
                ],
                label: USE_LABELS.then_some("bloom_texture_bind_group_2"),
            }),
        ];
    }

    #[profiling::function]
    pub fn add_debug_nodes(
        &self,
        data: &mut RendererData,
        private_data: &mut RendererPrivateData,
        engine_state: &mut EngineState,
        main_culling_frustum: &Frustum,
        main_culling_frustum_desc: &CameraFrustumDescriptor,
    ) {
        let scene = &mut engine_state.scene;

        for node_id in private_data
            .debug_node_bounding_spheres_nodes
            .iter()
            .copied()
        {
            scene.remove_node(node_id);
        }
        private_data.debug_node_bounding_spheres_nodes.clear();

        for node_id in private_data.debug_culling_frustum_nodes.iter().copied() {
            scene.remove_node(node_id);
        }
        private_data.debug_culling_frustum_nodes.clear();

        if let Some(mesh_index) = private_data.debug_culling_frustum_mesh_index.take() {
            Self::unbind_mesh(data, mesh_index);
        }

        if data.draw_node_bounding_spheres {
            let node_ids: Vec<_> = scene.nodes().map(|node| node.id()).collect();
            for node_id in node_ids {
                if let Some(bounding_sphere) = scene.get_node_bounding_sphere(node_id, data) {
                    let culling_frustum_intersection_result =
                        Self::get_node_cam_intersection_result(
                            scene.get_node(node_id).unwrap(),
                            data,
                            scene,
                            main_culling_frustum,
                        );

                    let debug_sphere_color = match culling_frustum_intersection_result {
                        Some(IntersectionResult::FullyContained) => [0.0, 1.0, 0.0, 0.1],
                        Some(IntersectionResult::PartiallyIntersecting) => [1.0, 1.0, 0.0, 0.15],
                        Some(IntersectionResult::NotIntersecting) => [1.0, 0.0, 0.0, 0.1],
                        None => [1.0, 1.0, 1.0, 0.1],
                    };

                    private_data.debug_node_bounding_spheres_nodes.push(
                        scene
                            .add_node(
                                GameNodeDescBuilder::new()
                                    .transform(
                                        TransformBuilder::new()
                                            .scale(
                                                bounding_sphere.radius * Vec3::new(1.0, 1.0, 1.0),
                                            )
                                            .position(bounding_sphere.center)
                                            .build(),
                                    )
                                    .visual(Some(GameNodeVisual {
                                        material: Material::Transparent {
                                            color: debug_sphere_color.into(),
                                            premultiplied_alpha: false,
                                        },
                                        mesh_index: self.constant_data.sphere_mesh_index,
                                        wireframe: false,
                                        cullable: false,
                                    }))
                                    .build(),
                            )
                            .id(),
                    );
                }
            }
        }

        let debug_main_camera_frustum_descriptor = CameraFrustumDescriptor {
            // shrink the frustum along the view direction for the debug view
            near_plane_distance: 1.0,
            far_plane_distance: 500.0,
            ..(*main_culling_frustum_desc)
        };

        if data.draw_culling_frustum {
            let debug_main_camera_frustum_mesh =
                debug_main_camera_frustum_descriptor.to_basic_mesh();

            private_data.debug_culling_frustum_mesh_index = Some(Self::bind_basic_mesh(
                &self.base,
                data,
                &debug_main_camera_frustum_mesh,
                true,
            ));

            let culling_frustum_mesh = GameNodeVisual {
                material: Material::Transparent {
                    color: Vec4::new(1.0, 0.0, 0.0, 0.1),
                    premultiplied_alpha: false,
                },
                mesh_index: private_data
                    .debug_culling_frustum_mesh_index
                    .expect("Mesh index should have just been set above"),
                wireframe: false,
                cullable: false,
            };

            let culling_frustum_mesh_wf = GameNodeVisual {
                wireframe: true,
                ..culling_frustum_mesh.clone()
            };

            private_data.debug_culling_frustum_nodes.push(
                scene
                    .add_node(
                        GameNodeDescBuilder::new()
                            .visual(Some(culling_frustum_mesh))
                            .build(),
                    )
                    .id(),
            );
            private_data.debug_culling_frustum_nodes.push(
                scene
                    .add_node(
                        GameNodeDescBuilder::new()
                            .visual(Some(culling_frustum_mesh_wf))
                            .build(),
                    )
                    .id(),
            );
        }

        if data.draw_point_light_culling_frusta {
            let mut new_node_descs = vec![];
            for point_light in &scene.point_lights {
                for controlled_direction in build_cubemap_face_camera_view_directions() {
                    let frustum_descriptor = CameraFrustumDescriptor {
                        focal_point: scene
                            .get_global_transform_for_node(point_light.node_id)
                            .position(),
                        forward_vector: controlled_direction.to_vector(),
                        near_plane_distance: POINT_LIGHT_SHADOW_MAP_FRUSTUM_NEAR_PLANE,
                        far_plane_distance: POINT_LIGHT_SHADOW_MAP_FRUSTUM_FAR_PLANE,
                        fov_y_rad: deg_to_rad(90.0),
                        aspect_ratio: 1.0,
                    };

                    let debug_frustum_descriptor = CameraFrustumDescriptor {
                        // shrink the frustum along the view direction for the debug view
                        near_plane_distance: 0.5,
                        far_plane_distance: 5.0,
                        ..frustum_descriptor
                    };
                    let debug_culling_frustum_mesh = debug_frustum_descriptor.to_basic_mesh();

                    let collision_based_color = if frustum_descriptor
                        .frustum_intersection_test(main_culling_frustum_desc)
                    {
                        Vec4::new(0.0, 1.0, 0.0, 0.1)
                    } else {
                        Vec4::new(1.0, 0.0, 0.0, 0.1)
                    };

                    private_data.debug_culling_frustum_mesh_index = Some(Self::bind_basic_mesh(
                        &self.base,
                        data,
                        &debug_culling_frustum_mesh,
                        true,
                    ));

                    let culling_frustum_mesh = GameNodeVisual {
                        material: Material::Transparent {
                            color: collision_based_color,
                            premultiplied_alpha: false,
                        },
                        mesh_index: private_data
                            .debug_culling_frustum_mesh_index
                            .expect("Mesh index should have just been set above"),
                        wireframe: false,
                        cullable: false,
                    };

                    let culling_frustum_mesh_wf = GameNodeVisual {
                        wireframe: true,
                        ..culling_frustum_mesh.clone()
                    };

                    new_node_descs.push(
                        GameNodeDescBuilder::new()
                            .visual(Some(culling_frustum_mesh))
                            .build(),
                    );
                    new_node_descs.push(
                        GameNodeDescBuilder::new()
                            .visual(Some(culling_frustum_mesh_wf))
                            .build(),
                    );
                }
            }

            for new_node_desc in new_node_descs {
                private_data
                    .debug_culling_frustum_nodes
                    .push(scene.add_node(new_node_desc).id());
            }
        }

        if data.draw_directional_light_culling_frusta {
            // let mut new_node_descs = vec![];
            for directional_light in &scene.directional_lights {
                // -directional_light.direction,
                // DIRECTIONAL_LIGHT_PROJ_BOX_RADIUS * 2.0,
                // DIRECTIONAL_LIGHT_PROJ_BOX_RADIUS * 2.0,
                // DIRECTIONAL_LIGHT_PROJ_BOX_LENGTH,
            }
        }
    }

    pub fn set_culling_frustum_lock(
        &self,
        engine_state: &EngineState,
        surface_config: &wgpu::SurfaceConfiguration,
        lock_mode: CullingFrustumLockMode,
    ) {
        let aspect_ratio = surface_config.width as f32 / surface_config.height as f32;

        let mut private_data_guard = self.private_data.lock().unwrap();

        if CullingFrustumLockMode::from(private_data_guard.frustum_culling_lock.clone())
            == lock_mode
        {
            return;
        }

        let camera_transform = self
            .data
            .lock()
            .unwrap()
            .camera_node_id
            .and_then(|camera_node_id| engine_state.scene.get_node(camera_node_id))
            .map(|camera_node| camera_node.transform);

        if camera_transform.is_none() {
            log::error!("Couldn't set the frustum culling lock as there is currently no camera");
            return;
        }

        let camera_transform = camera_transform.unwrap();

        let position = match private_data_guard.frustum_culling_lock {
            CullingFrustumLock::Full(desc) => desc.focal_point,
            CullingFrustumLock::FocalPoint(locked_focal_point) => locked_focal_point,
            CullingFrustumLock::None => camera_transform.position(),
        };

        private_data_guard.frustum_culling_lock = match lock_mode {
            CullingFrustumLockMode::Full => CullingFrustumLock::Full(CameraFrustumDescriptor {
                focal_point: camera_transform.position(),
                forward_vector: (-camera_transform.z_axis).into(),
                aspect_ratio,
                near_plane_distance: NEAR_PLANE_DISTANCE,
                far_plane_distance: FAR_PLANE_DISTANCE,
                fov_y_rad: deg_to_rad(FOV_Y_DEG),
            }),
            CullingFrustumLockMode::FocalPoint => CullingFrustumLock::FocalPoint(position),
            CullingFrustumLockMode::None => CullingFrustumLock::None,
        };
    }

    pub fn render<UiOverlay>(
        &mut self,
        engine_state: &mut EngineState,
        surface_data: &SurfaceData,
        ui_overlay: &mut IkariUiContainer<UiOverlay>,
    ) -> anyhow::Result<()>
    where
        UiOverlay: iced_winit::runtime::Program<Renderer = iced::Renderer> + 'static,
    {
        self.update_internal(engine_state, &surface_data.surface_config);
        self.render_internal(
            engine_state,
            surface_data.surface.get_current_texture()?,
            ui_overlay,
        )
    }

    fn get_node_cam_intersection_result(
        node: &GameNode,
        data: &RendererData,
        scene: &Scene,
        camera_culling_frustum: &Frustum,
    ) -> Option<IntersectionResult> {
        node.visual.as_ref()?;

        /* bounding boxes will be wrong for skinned meshes so we currently can't cull them */
        if node.skin_index.is_some() || !node.visual.as_ref().unwrap().cullable {
            return None;
        }

        let node_bounding_sphere = scene.get_node_bounding_sphere_opt(node.id(), data);

        node_bounding_sphere?;

        let node_bounding_sphere = node_bounding_sphere.unwrap();

        Some(camera_culling_frustum.sphere_intersection_test(node_bounding_sphere))
    }

    // culling mask is a bitmask where each bit corresponds to a frustum
    // and the value of the bit represents whether or not the object
    // is touching that frustum or not. the first bit represnts the main
    // camera frustum, the subsequent bits represent the directional shadow
    // mapping frusta and the rest of the bits represent the point light shadow
    // mapping frusta, of which there are 6 per point light so 6 bits are used
    // per point light.
    fn get_node_culling_mask(
        node: &GameNode,
        data: &RendererData,
        engine_state: &EngineState,
        camera_culling_frustum: &Frustum,
        point_lights_frusta: &PointLightFrustaWithCullingInfo,
    ) -> u32 {
        if USE_ORTHOGRAPHIC_CAMERA {
            return u32::MAX;
        }

        assert!(1 + engine_state.scene.directional_lights.len() + engine_state.scene.point_lights.len() * 6 <= 32,
            "u32 can only store a max of 5 point lights, might be worth using a larger bitvec or a Vec<bool> or something"
        );

        if node.visual.is_none() {
            return 0;
        }

        /* bounding boxes will be wrong for skinned meshes so we currently can't cull them */
        if node.skin_index.is_some() || !node.visual.as_ref().unwrap().cullable {
            return u32::MAX;
        }

        let node_bounding_sphere = engine_state
            .scene
            .get_node_bounding_sphere_opt(node.id(), data);

        if node_bounding_sphere.is_none() {
            return 0;
        }

        let node_bounding_sphere = node_bounding_sphere.unwrap();

        let is_touching_frustum = |frustum: &Frustum| {
            matches!(
                frustum.sphere_intersection_test(node_bounding_sphere),
                IntersectionResult::FullyContained | IntersectionResult::PartiallyIntersecting
            )
        };

        let mut culling_mask = 0u32;
        let mut mask_pos = 0;

        let is_node_on_screen = is_touching_frustum(camera_culling_frustum);

        if is_node_on_screen {
            culling_mask |= 2u32.pow(mask_pos);
        }

        mask_pos += 1;

        for _ in &engine_state.scene.directional_lights {
            // TODO: add support for frustum culling directional lights shadow map gen?
            culling_mask |= 2u32.pow(mask_pos);

            mask_pos += 1;
        }

        for frusta in point_lights_frusta {
            match frusta {
                Some((frusta, can_cull_offscreen_objects)) => {
                    for (frustum, is_light_view_culled) in frusta {
                        let is_culled = if USE_EXTRA_SHADOW_MAP_CULLING {
                            let is_offscreen_culled =
                                *can_cull_offscreen_objects && !is_node_on_screen;
                            is_offscreen_culled
                                || *is_light_view_culled
                                || !is_touching_frustum(frustum)
                        } else {
                            !is_touching_frustum(frustum)
                        };

                        if !is_culled {
                            culling_mask |= 2u32.pow(mask_pos);
                        }

                        mask_pos += 1;
                    }
                }
                None => {
                    mask_pos += 6;
                }
            }
        }

        culling_mask
    }

    fn get_environment_textures_bind_group(
        base: &BaseRenderer,
        constant_data: &RendererConstantData,
        skyboxes: &[BindedSkybox; 2],
        skybox_weights_buffer: &wgpu::Buffer,
        brdf_lut: &Texture,
        point_shadow_map_textures: &Texture,
        directional_shadow_map_textures: &Texture,
    ) -> wgpu::BindGroup {
        let sampler_cache_guard = base.sampler_cache.lock().unwrap();

        base.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &constant_data.environment_textures_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&skyboxes[0].background.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(
                        sampler_cache_guard
                            .get_sampler_by_index(skyboxes[0].background.sampler_index),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&skyboxes[1].background.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: skybox_weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        &skyboxes[0].diffuse_environment_map.view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(
                        &skyboxes[1].diffuse_environment_map.view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(
                        &skyboxes[0].specular_environment_map.view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(
                        &skyboxes[1].specular_environment_map.view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&brdf_lut.view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(
                        sampler_cache_guard.get_sampler_by_index(brdf_lut.sampler_index),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&point_shadow_map_textures.view),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(
                        &directional_shadow_map_textures.view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Sampler(
                        sampler_cache_guard
                            .get_sampler_by_index(point_shadow_map_textures.sampler_index),
                    ),
                },
            ],
            label: USE_LABELS.then_some("environment_textures_bind_group"),
        })
    }

    pub fn set_skybox(&self, slot: SkyboxSlot, skybox: BindedSkybox) {
        let mut private_data_guard = self.private_data.lock().unwrap();

        private_data_guard.skyboxes[slot.as_index()] = skybox;

        private_data_guard.environment_textures_bind_group =
            Self::get_environment_textures_bind_group(
                &self.base,
                &self.constant_data,
                &private_data_guard.skyboxes,
                &private_data_guard.skybox_weights_buffer,
                &private_data_guard.brdf_lut,
                &private_data_guard.point_shadow_map_textures,
                &private_data_guard.directional_shadow_map_textures,
            );
    }

    pub fn set_skybox_weights(&self, weights: [f32; 2]) {
        let normalized = {
            let total = weights[0] + weights[1];
            [weights[0] / total, weights[1] / total]
        };
        self.private_data.lock().unwrap().skybox_weights = normalized;
    }

    pub fn get_skybox_weights(&self) -> [f32; 2] {
        self.private_data.lock().unwrap().skybox_weights
    }

    /// Prepare and send all data to gpu so it's ready to render
    #[profiling::function]
    fn update_internal(
        &mut self,
        engine_state: &mut EngineState,
        surface_config: &wgpu::SurfaceConfiguration,
    ) {
        let mut data_guard = self.data.lock().unwrap();
        let data: &mut RendererData = &mut data_guard;

        let mut private_data_guard = self.private_data.lock().unwrap();
        let private_data: &mut RendererPrivateData = &mut private_data_guard;

        let aspect_ratio = surface_config.width as f32 / surface_config.height as f32;

        let camera_transform = data
            .camera_node_id
            .and_then(|camera_node_id| engine_state.scene.get_node(camera_node_id))
            .map(|camera_node| camera_node.transform)
            .unwrap_or_default();

        let camera_position = camera_transform.position();

        let camera_frustum_desc = CameraFrustumDescriptor {
            focal_point: camera_position,
            forward_vector: (-camera_transform.z_axis).into(),
            aspect_ratio,
            near_plane_distance: NEAR_PLANE_DISTANCE,
            far_plane_distance: FAR_PLANE_DISTANCE,
            fov_y_rad: deg_to_rad(FOV_Y_DEG),
        };

        let culling_frustum_desc = match private_data.frustum_culling_lock {
            CullingFrustumLock::Full(locked) => locked,
            CullingFrustumLock::FocalPoint(locked_position) => CameraFrustumDescriptor {
                focal_point: locked_position,
                ..camera_frustum_desc
            },
            CullingFrustumLock::None => camera_frustum_desc,
        };

        let culling_frustum = Frustum::from(culling_frustum_desc);

        self.add_debug_nodes(
            data,
            private_data,
            engine_state,
            &culling_frustum,
            &culling_frustum_desc,
        );

        // TODO: compute node bounding spheres here too?
        engine_state.scene.recompute_global_node_transforms();

        let limits = &self.base.limits;
        let queue = &self.base.queue;
        let device = &self.base.device;
        let bones_and_instances_bind_group_layout =
            &self.constant_data.bones_and_instances_bind_group_layout;

        private_data.all_bone_transforms = get_all_bone_data(
            &engine_state.scene,
            limits.min_storage_buffer_offset_alignment,
        );
        let previous_bones_buffer_capacity_bytes = private_data.bones_buffer.capacity_bytes();

        // composite index (mesh_index, material_index)
        let mut pbr_mesh_index_to_gpu_instances: HashMap<(usize, usize), Vec<GpuPbrMeshInstance>> =
            HashMap::new();
        let mut unlit_mesh_index_to_gpu_instances: HashMap<usize, Vec<GpuUnlitMeshInstance>> =
            HashMap::new();
        let mut wireframe_mesh_index_to_gpu_instances: HashMap<
            usize,
            Vec<GpuWireframeMeshInstance>,
        > = HashMap::new();
        // no instancing for transparent meshes to allow for sorting
        let mut transparent_meshes: Vec<(usize, GpuTransparentMeshInstance, f32)> = Vec::new();

        // list of 6 frusta for each point light including culling information
        let point_lights_frusta: PointLightFrustaWithCullingInfo = engine_state
            .scene
            .point_lights
            .iter()
            .map(|point_light| {
                engine_state
                    .scene
                    .get_node(point_light.node_id)
                    .map(|point_light_node| {
                        let frustum_descriptors = build_cubemap_face_frusta(
                            point_light_node.transform.position(),
                            POINT_LIGHT_SHADOW_MAP_FRUSTUM_NEAR_PLANE,
                            POINT_LIGHT_SHADOW_MAP_FRUSTUM_FAR_PLANE,
                        );

                        // if the point light is inside the main camera view this means that
                        // no objects outside the main view can cast shadows on objects that
                        // are inside the main view so we cull any such objects from the
                        // shadow map render pass
                        let can_cull_offscreen_objects = if USE_EXTRA_SHADOW_MAP_CULLING {
                            culling_frustum.contains_point(point_light_node.transform.position())
                        } else {
                            false
                        };

                        (
                            frustum_descriptors
                                .iter()
                                .map(|desc| {
                                    // if the light view doesn't intersect with the main camera view at all
                                    // then none of the objects inside of it can cast shadows on objects
                                    // that are inside the main view so we can completely skip the shadow map
                                    // render pass for this light view
                                    let is_light_view_culled = if USE_EXTRA_SHADOW_MAP_CULLING {
                                        !desc.frustum_intersection_test(&culling_frustum_desc)
                                    } else {
                                        false
                                    };
                                    ((*desc).into(), is_light_view_culled)
                                })
                                .collect(),
                            can_cull_offscreen_objects,
                        )
                    })
            })
            .collect();

        {
            profiling::scope!("Prepare/cull instance lists");

            for node in engine_state.scene.nodes() {
                let transform = Mat4::from(
                    engine_state
                        .scene
                        .get_global_transform_for_node_opt(node.id()),
                );
                if let Some(GameNodeVisual {
                    mesh_index,
                    material,
                    wireframe,
                    ..
                }) = node.visual.clone()
                {
                    let (scale, _, translation) = transform.to_scale_rotation_translation();
                    let aabb_world_space = data.binded_meshes[mesh_index]
                        .bounding_box
                        .scale_translate(scale, translation);
                    let closest_point_to_player =
                        aabb_world_space.find_closest_surface_point(camera_position);
                    let distance_from_player = closest_point_to_player.distance(camera_position);

                    match (material, data.enable_wireframe_mode, wireframe) {
                        (
                            Material::Pbr {
                                binded_material_index,
                                dynamic_pbr_params,
                            },
                            false,
                            false,
                        ) => {
                            let culling_mask = Self::get_node_culling_mask(
                                node,
                                data,
                                engine_state,
                                &culling_frustum,
                                &point_lights_frusta,
                            );
                            let gpu_instance = GpuPbrMeshInstance::new(
                                transform,
                                dynamic_pbr_params.unwrap_or_else(|| {
                                    data.binded_pbr_materials[binded_material_index]
                                        .dynamic_pbr_params
                                }),
                                culling_mask,
                            );
                            match pbr_mesh_index_to_gpu_instances
                                .entry((mesh_index, binded_material_index))
                            {
                                Entry::Occupied(mut entry) => {
                                    entry.get_mut().push(gpu_instance);
                                }
                                Entry::Vacant(entry) => {
                                    entry.insert(vec![gpu_instance]);
                                }
                            }
                        }
                        (material, enable_wireframe_mode, is_node_wireframe) => {
                            let (color, is_transparent) = match material {
                                Material::Unlit { color } => {
                                    ([color.x, color.y, color.z, 1.0], false)
                                }
                                Material::Transparent {
                                    color,
                                    premultiplied_alpha,
                                } => {
                                    (
                                        if premultiplied_alpha {
                                            [color.x, color.y, color.z, color.w]
                                        } else {
                                            // transparent pipeline requires alpha to be premultiplied.
                                            [
                                                color.w * color.x,
                                                color.w * color.y,
                                                color.w * color.z,
                                                color.w,
                                            ]
                                        },
                                        true,
                                    )
                                }
                                Material::Pbr {
                                    binded_material_index,
                                    dynamic_pbr_params,
                                } => {
                                    // fancy logic for picking what the wireframe lines
                                    // color will be by checking the pbr material
                                    let DynamicPbrParams {
                                        base_color_factor,
                                        emissive_factor,
                                        ..
                                    } = dynamic_pbr_params.unwrap_or_else(|| {
                                        data.binded_pbr_materials[binded_material_index]
                                            .dynamic_pbr_params
                                    });
                                    let should_take_color = |as_slice: &[f32]| {
                                        let is_all_zero = as_slice.iter().all(|&x| x == 0.0);
                                        let is_all_one = as_slice.iter().all(|&x| x == 1.0);
                                        !is_all_zero && !is_all_one
                                    };
                                    let base_color_factor_arr: [f32; 4] = base_color_factor.into();
                                    let emissive_factor_arr: [f32; 3] = emissive_factor.into();
                                    (
                                        if should_take_color(&base_color_factor_arr[0..3]) {
                                            [
                                                base_color_factor.x,
                                                base_color_factor.y,
                                                base_color_factor.z,
                                                base_color_factor.w,
                                            ]
                                        } else if should_take_color(&emissive_factor_arr) {
                                            [
                                                emissive_factor.x,
                                                emissive_factor.y,
                                                emissive_factor.z,
                                                1.0,
                                            ]
                                        } else {
                                            DEFAULT_WIREFRAME_COLOR
                                        },
                                        false,
                                    )
                                }
                            };

                            if enable_wireframe_mode || is_node_wireframe {
                                let wireframe_mesh_index = data
                                    .binded_wireframe_meshes
                                    .iter()
                                    .enumerate()
                                    .find(|(_, wireframe_mesh)| {
                                        wireframe_mesh.source_mesh_index == mesh_index
                                    })
                                    .map(|(index, _)| index);

                                if wireframe_mesh_index.is_none() {
                                    if is_node_wireframe {
                                        log::error!("Attempted to draw mesh in wireframe mode without binding a corresponding wireframe mesh. Mesh will not be draw. mesh_index={mesh_index:?} node={node:?}");
                                    }
                                    continue;
                                }

                                let wireframe_mesh_index = wireframe_mesh_index
                                    .expect("Should have checked that it isn't None");

                                let gpu_instance = GpuWireframeMeshInstance {
                                    color,
                                    model_transform: transform,
                                };
                                match wireframe_mesh_index_to_gpu_instances
                                    .entry(wireframe_mesh_index)
                                {
                                    Entry::Occupied(mut entry) => {
                                        entry.get_mut().push(gpu_instance);
                                    }
                                    Entry::Vacant(entry) => {
                                        entry.insert(vec![gpu_instance]);
                                    }
                                }
                            } else if is_transparent {
                                let gpu_instance = GpuTransparentMeshInstance {
                                    color,
                                    model_transform: transform,
                                };
                                transparent_meshes.push((
                                    mesh_index,
                                    gpu_instance,
                                    distance_from_player,
                                ));
                            } else {
                                let gpu_instance = GpuUnlitMeshInstance {
                                    color,
                                    model_transform: transform,
                                };
                                match unlit_mesh_index_to_gpu_instances.entry(mesh_index) {
                                    Entry::Occupied(mut entry) => {
                                        entry.get_mut().push(gpu_instance);
                                    }
                                    Entry::Vacant(entry) => {
                                        entry.insert(vec![gpu_instance]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let min_storage_buffer_offset_alignment =
            self.base.limits.min_storage_buffer_offset_alignment;

        let mut pbr_mesh_instances: Vec<_> = pbr_mesh_index_to_gpu_instances.into_iter().collect();

        // TODO: sort opaque instances front to back to maximize z-buffer early outs
        pbr_mesh_instances.sort_unstable_by_key(|((_, material_index), _)| *material_index);
        pbr_mesh_instances.sort_by_key(|((mesh_index, _), _)| *mesh_index);

        private_data.all_pbr_instances_culling_masks.clear();

        {
            profiling::scope!("Combine culling masks");

            for (_, instances) in &pbr_mesh_instances {
                // we only have one culling mask per chunk of instances,
                // meaning that instances can't be culled individually
                let mut combined_culling_mask = 0u32;
                for instance in instances {
                    combined_culling_mask |= instance.culling_mask;
                }
                private_data
                    .all_pbr_instances_culling_masks
                    .push(combined_culling_mask);
            }
        }

        private_data.all_pbr_instances.replace(
            pbr_mesh_instances.into_iter(),
            min_storage_buffer_offset_alignment as usize,
        );

        let bones_buffer_changed_capacity = private_data.bones_buffer.write(
            device,
            queue,
            &private_data.all_bone_transforms.buffer,
        );
        if bones_buffer_changed_capacity {
            log::debug!(
                "Resized bones instances buffer capacity from {:?} bytes to {:?}, length={:?}, buffer_length={:?}",
                previous_bones_buffer_capacity_bytes,
                private_data.bones_buffer.capacity_bytes(),
                private_data.bones_buffer.length_bytes(),
                private_data.all_bone_transforms.buffer.len(),
            );
        }

        let previous_pbr_instances_buffer_capacity_bytes =
            private_data.pbr_instances_buffer.capacity_bytes();
        let pbr_instances_buffer_changed_capacity = private_data.pbr_instances_buffer.write(
            device,
            queue,
            private_data.all_pbr_instances.buffer(),
        );

        if pbr_instances_buffer_changed_capacity {
            log::debug!(
                "Resized pbr instances buffer capacity from {:?} bytes to {:?}, length={:?}, buffer_length={:?}",
                previous_pbr_instances_buffer_capacity_bytes,
                private_data.pbr_instances_buffer.capacity_bytes(),
                private_data.pbr_instances_buffer.length_bytes(),
                private_data.all_pbr_instances.buffer().len(),
            );
        }

        private_data.all_unlit_instances.replace(
            unlit_mesh_index_to_gpu_instances.into_iter(),
            min_storage_buffer_offset_alignment as usize,
        );

        let previous_unlit_instances_buffer_capacity_bytes =
            private_data.unlit_instances_buffer.capacity_bytes();
        let unlit_instances_buffer_changed_capacity = private_data.unlit_instances_buffer.write(
            device,
            queue,
            private_data.all_unlit_instances.buffer(),
        );

        if unlit_instances_buffer_changed_capacity {
            log::debug!(
                "Resized unlit instances buffer capacity from {:?} bytes to {:?}, length={:?}, buffer_length={:?}",
                previous_unlit_instances_buffer_capacity_bytes,
                private_data.unlit_instances_buffer.capacity_bytes(),
                private_data.unlit_instances_buffer.length_bytes(),
                private_data.all_unlit_instances.buffer().len(),
            );
        }

        // draw furthest transparent meshes first
        transparent_meshes.sort_by(
            |(_, _, distance_from_player_a), (_, _, distance_from_player_b)| {
                distance_from_player_b
                    .partial_cmp(distance_from_player_a)
                    .unwrap()
            },
        );

        private_data.all_transparent_instances.replace(
            transparent_meshes
                .into_iter()
                .map(|(mesh_index, instance, _)| (mesh_index, vec![instance])),
            min_storage_buffer_offset_alignment as usize,
        );

        let previous_transparent_instances_buffer_capacity_bytes =
            private_data.transparent_instances_buffer.capacity_bytes();
        let transparent_instances_buffer_changed_capacity =
            private_data.transparent_instances_buffer.write(
                device,
                queue,
                private_data.all_transparent_instances.buffer(),
            );

        if transparent_instances_buffer_changed_capacity {
            log::debug!(
                "Resized transparent instances buffer capacity from {:?} bytes to {:?}, length={:?}, buffer_length={:?}",
                previous_transparent_instances_buffer_capacity_bytes,
                private_data.transparent_instances_buffer.capacity_bytes(),
                private_data.transparent_instances_buffer.length_bytes(),
                private_data.all_transparent_instances.buffer().len(),
            );
        }

        private_data.all_wireframe_instances.replace(
            wireframe_mesh_index_to_gpu_instances.into_iter(),
            min_storage_buffer_offset_alignment as usize,
        );

        let previous_wireframe_instances_buffer_capacity_bytes =
            private_data.wireframe_instances_buffer.capacity_bytes();
        let wireframe_instances_buffer_changed_capacity = private_data
            .wireframe_instances_buffer
            .write(device, queue, private_data.all_wireframe_instances.buffer());

        if wireframe_instances_buffer_changed_capacity {
            log::debug!(
                "Resized wireframe instances buffer capacity from {:?} bytes to {:?}, length={:?}, buffer_length={:?}",
                previous_wireframe_instances_buffer_capacity_bytes,
                private_data.wireframe_instances_buffer.capacity_bytes(),
                private_data.wireframe_instances_buffer.length_bytes(),
                private_data.all_wireframe_instances.buffer().len(),
            );
        }

        {
            profiling::scope!("Recreate bind groups");

            private_data.bones_and_pbr_instances_bind_group =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: bones_and_instances_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: private_data.bones_buffer.src(),
                                offset: 0,
                                size: NonZeroU64::new(
                                    private_data.bones_buffer.length_bytes().try_into().unwrap(),
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: private_data.pbr_instances_buffer.src(),
                                offset: 0,
                                size: NonZeroU64::new(
                                    (private_data.all_pbr_instances.biggest_chunk_length()
                                        * private_data.pbr_instances_buffer.stride())
                                    .try_into()
                                    .unwrap(),
                                ),
                            }),
                        },
                    ],
                    label: USE_LABELS.then_some("bones_and_pbr_instances_bind_group"),
                });

            private_data.bones_and_unlit_instances_bind_group =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: bones_and_instances_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: private_data.bones_buffer.src(),
                                offset: 0,
                                size: NonZeroU64::new(
                                    private_data.bones_buffer.length_bytes().try_into().unwrap(),
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: private_data.unlit_instances_buffer.src(),
                                offset: 0,
                                size: NonZeroU64::new(
                                    (private_data.all_unlit_instances.biggest_chunk_length()
                                        * private_data.unlit_instances_buffer.stride())
                                    .try_into()
                                    .unwrap(),
                                ),
                            }),
                        },
                    ],
                    label: USE_LABELS.then_some("bones_and_unlit_instances_bind_group"),
                });

            private_data.bones_and_transparent_instances_bind_group =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: bones_and_instances_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: private_data.bones_buffer.src(),
                                offset: 0,
                                size: NonZeroU64::new(
                                    private_data.bones_buffer.length_bytes().try_into().unwrap(),
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: private_data.transparent_instances_buffer.src(),
                                offset: 0,
                                size: NonZeroU64::new(
                                    (private_data
                                        .all_transparent_instances
                                        .biggest_chunk_length()
                                        * private_data.transparent_instances_buffer.stride())
                                    .try_into()
                                    .unwrap(),
                                ),
                            }),
                        },
                    ],
                    label: USE_LABELS.then_some("bones_and_transparent_instances_bind_group"),
                });

            private_data.bones_and_wireframe_instances_bind_group =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: bones_and_instances_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: private_data.bones_buffer.src(),
                                offset: 0,
                                size: NonZeroU64::new(
                                    private_data.bones_buffer.length_bytes().try_into().unwrap(),
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: private_data.wireframe_instances_buffer.src(),
                                offset: 0,
                                size: NonZeroU64::new(
                                    (private_data.all_wireframe_instances.biggest_chunk_length()
                                        * private_data.wireframe_instances_buffer.stride())
                                    .try_into()
                                    .unwrap(),
                                ),
                            }),
                        },
                    ],
                    label: USE_LABELS.then_some("bones_and_wireframe_instances_bind_group"),
                });
        }

        let mut all_camera_data: Vec<ShaderCameraData> = vec![];

        // collect all camera data

        // main camera
        let main_camera_shader_data = if USE_ORTHOGRAPHIC_CAMERA {
            ShaderCameraData::orthographic(
                camera_transform.into(),
                20.0 * aspect_ratio,
                20.0,
                -1000.0,
                1000.0,
                false,
            )
        } else {
            ShaderCameraData::perspective(
                camera_transform.into(),
                aspect_ratio,
                NEAR_PLANE_DISTANCE,
                FAR_PLANE_DISTANCE,
                deg_to_rad(FOV_Y_DEG),
                true,
            )
        };
        all_camera_data.push(main_camera_shader_data);

        // directional lights
        for directional_light in &engine_state.scene.directional_lights {
            all_camera_data.push(ShaderCameraData::orthographic(
                look_in_dir(Vec3::new(0.0, 0.0, 0.0), directional_light.direction),
                DIRECTIONAL_LIGHT_PROJ_BOX_RADIUS * 2.0,
                DIRECTIONAL_LIGHT_PROJ_BOX_RADIUS * 2.0,
                -DIRECTIONAL_LIGHT_PROJ_BOX_LENGTH / 2.0,
                DIRECTIONAL_LIGHT_PROJ_BOX_LENGTH / 2.0,
                false,
            ));
        }

        // point lights
        for point_light in &engine_state.scene.point_lights {
            let light_position = engine_state
                .scene
                .get_node(point_light.node_id)
                .map(|node| node.transform.position())
                .unwrap_or_default();
            all_camera_data.append(&mut build_cubemap_face_camera_views(
                light_position,
                POINT_LIGHT_SHADOW_MAP_FRUSTUM_NEAR_PLANE,
                POINT_LIGHT_SHADOW_MAP_FRUSTUM_FAR_PLANE,
                false,
            ));
        }

        // main camera but only rotation, for skybox
        all_camera_data.push(all_camera_data[0]);

        // write all camera data, adding new buffers if necessary
        for (i, camera_data) in all_camera_data.iter().enumerate() {
            let contents = if i == all_camera_data.len() - 1 {
                bytemuck::cast_slice(&[SkyboxShaderCameraRaw::from(*camera_data)]).to_vec()
            } else {
                bytemuck::cast_slice(&[MeshShaderCameraRaw::from(*camera_data)]).to_vec()
            };
            if private_data.camera_buffers.len() == i {
                private_data
                    .camera_buffers
                    .push(
                        self.base
                            .device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: USE_LABELS.then_some("Camera Buffer"),
                                contents: &contents,
                                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            }),
                    );
                private_data
                    .camera_lights_and_pbr_shader_options_bind_groups
                    .push(
                        self.base
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                layout: &private_data
                                    .camera_lights_and_pbr_shader_options_bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: private_data.camera_buffers[i]
                                            .as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: private_data
                                            .point_lights_buffer
                                            .as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: private_data
                                            .directional_lights_buffer
                                            .as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: private_data
                                            .pbr_shader_options_buffer
                                            .as_entire_binding(),
                                    },
                                ],
                                label: USE_LABELS
                                    .then_some("camera_lights_and_pbr_shader_options_bind_group"),
                            }),
                    );
            } else {
                queue.write_buffer(&private_data.camera_buffers[i], 0, &contents)
            }
        }

        let fmt_bytes = |bytes: usize| {
            byte_unit::Byte::from_bytes(bytes.try_into().unwrap())
                .get_appropriate_unit(false)
                .to_string()
        };
        log::debug!(
            "Memory usage:\n  Instance buffers: {}\n  Index buffers: {}\n  Vertex buffers: {}",
            fmt_bytes(
                private_data.pbr_instances_buffer.length_bytes()
                    + private_data.unlit_instances_buffer.length_bytes()
                    + private_data.transparent_instances_buffer.length_bytes()
                    + private_data.wireframe_instances_buffer.length_bytes()
            ),
            fmt_bytes(
                data.binded_meshes
                    .iter()
                    .map(|mesh| mesh.index_buffer.buffer.length_bytes())
                    .chain(
                        data.binded_wireframe_meshes
                            .iter()
                            .map(|mesh| mesh.index_buffer.buffer.length_bytes()),
                    )
                    .reduce(|acc, val| acc + val)
                    .unwrap_or(0)
            ),
            fmt_bytes(
                data.binded_meshes
                    .iter()
                    .map(|mesh| mesh.vertex_buffer.length_bytes())
                    .reduce(|acc, val| acc + val)
                    .unwrap_or(0)
            ),
        );

        queue.write_buffer(
            &private_data.point_lights_buffer,
            0,
            bytemuck::cast_slice(&make_point_light_uniform_buffer(engine_state)),
        );
        queue.write_buffer(
            &private_data.directional_lights_buffer,
            0,
            bytemuck::cast_slice(&make_directional_light_uniform_buffer(
                &engine_state.scene.directional_lights,
            )),
        );
        queue.write_buffer(
            &private_data.tone_mapping_config_buffer,
            0,
            bytemuck::cast_slice(&[
                data.tone_mapping_exposure,
                if surface_config.format.is_srgb() {
                    0f32
                } else {
                    1f32
                },
                0f32,
                0f32,
            ]),
        );
        queue.write_buffer(
            &private_data.bloom_config_buffers[0],
            0,
            bytemuck::cast_slice(&[0.0f32, data.bloom_threshold, data.bloom_ramp_size, 0.0f32]),
        );
        queue.write_buffer(
            &private_data.bloom_config_buffers[1],
            0,
            bytemuck::cast_slice(&[1.0f32, data.bloom_threshold, data.bloom_ramp_size, 0.0f32]),
        );
        queue.write_buffer(
            &private_data.pbr_shader_options_buffer,
            0,
            bytemuck::cast_slice(&[make_pbr_shader_options_uniform_buffer(
                data.enable_soft_shadows,
                data.shadow_bias,
                data.soft_shadow_factor,
                data.enable_shadow_debug,
                data.soft_shadow_grid_dims,
            )]),
        );
        queue.write_buffer(
            &private_data.skybox_weights_buffer,
            0,
            bytemuck::cast_slice(&[
                private_data.skybox_weights[0],
                private_data.skybox_weights[1],
                0.0,
                0.0,
            ]),
        );
    }

    #[profiling::function]
    pub fn render_internal<UiOverlay>(
        &mut self,
        engine_state: &mut EngineState,
        surface_texture: wgpu::SurfaceTexture,
        ui_overlay: &mut IkariUiContainer<UiOverlay>,
    ) -> anyhow::Result<()>
    where
        UiOverlay: iced_winit::runtime::Program<Renderer = iced::Renderer> + 'static,
    {
        let mut data_guard = self.data.lock().unwrap();
        let data: &mut RendererData = &mut data_guard;

        let mut private_data_guard = self.private_data.lock().unwrap();
        let private_data: &mut RendererPrivateData = &mut private_data_guard;

        let mut profiler_guard = self.profiler.lock().unwrap();
        let profiler: &mut GpuProfiler = &mut profiler_guard;

        let surface_texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .base
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        if data.enable_shadows {
            for (light_index, _light) in engine_state.scene.directional_lights.iter().enumerate() {
                if light_index >= DIRECTIONAL_LIGHT_SHOW_MAP_COUNT as usize {
                    continue;
                }
                let texture_view = private_data
                    .directional_shadow_map_textures
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: light_index.try_into().unwrap(),
                        array_layer_count: Some(1),
                        ..Default::default()
                    });
                let shadow_render_pass_desc = wgpu::RenderPassDescriptor {
                    label: USE_LABELS.then_some("Directional light shadow map"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &texture_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                };
                Self::render_pbr_meshes(
                    &self.base,
                    data,
                    private_data,
                    profiler,
                    &mut encoder,
                    &shadow_render_pass_desc,
                    &self.constant_data.directional_shadow_map_pipeline,
                    &private_data.camera_lights_and_pbr_shader_options_bind_groups[1 + light_index],
                    true,
                    2u32.pow((1 + light_index).try_into().unwrap()),
                    None,
                );
            }
            for light_index in 0..engine_state.scene.point_lights.len() {
                if light_index >= POINT_LIGHT_SHOW_MAP_COUNT as usize {
                    continue;
                }
                if let Some(light_node) = engine_state
                    .scene
                    .get_node(engine_state.scene.point_lights[light_index].node_id)
                {
                    build_cubemap_face_camera_views(
                        light_node.transform.position(),
                        POINT_LIGHT_SHADOW_MAP_FRUSTUM_NEAR_PLANE,
                        POINT_LIGHT_SHADOW_MAP_FRUSTUM_FAR_PLANE,
                        false,
                    )
                    .iter()
                    .copied()
                    .enumerate()
                    .for_each(|(face_index, _face_view_proj_matrices)| {
                        let culling_mask = 2u32.pow(
                            (1 + engine_state.scene.directional_lights.len()
                                + light_index * 6
                                + face_index)
                                .try_into()
                                .unwrap(),
                        );
                        let face_texture_view = private_data
                            .point_shadow_map_textures
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor {
                                dimension: Some(wgpu::TextureViewDimension::D2),
                                base_array_layer: light_index.try_into().unwrap(),
                                array_layer_count: Some(1),
                                ..Default::default()
                            });
                        let shadow_render_pass_desc = wgpu::RenderPassDescriptor {
                            label: USE_LABELS.then_some("Point light shadow map"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &face_texture_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: if face_index == 0 {
                                            wgpu::LoadOp::Clear(1.0)
                                        } else {
                                            wgpu::LoadOp::Load
                                        },
                                        store: true,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                        };

                        Self::render_pbr_meshes(
                            &self.base,
                            data,
                            private_data,
                            profiler,
                            &mut encoder,
                            &shadow_render_pass_desc,
                            &self.constant_data.point_shadow_map_pipeline,
                            &private_data.camera_lights_and_pbr_shader_options_bind_groups[1
                                + engine_state.scene.directional_lights.len()
                                + light_index * 6
                                + face_index],
                            true,
                            culling_mask,
                            Some(face_index.try_into().unwrap()),
                        );
                    });
                }
            }
        }

        let black = wgpu::Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };

        let shading_render_pass_desc = wgpu::RenderPassDescriptor {
            label: USE_LABELS.then_some("Pbr meshes"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &private_data.shading_texture.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(black),
                    store: true,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &private_data.depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        };

        Self::render_pbr_meshes(
            &self.base,
            data,
            private_data,
            profiler,
            &mut encoder,
            &shading_render_pass_desc,
            &self.constant_data.mesh_pipeline,
            &private_data.camera_lights_and_pbr_shader_options_bind_groups[0],
            false,
            1, // use main camera culling mask
            None,
        );

        {
            let label = "Unlit and wireframe";
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: USE_LABELS.then_some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &private_data.shading_texture.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &private_data.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            wgpu_profiler!(label, profiler, &mut render_pass, &self.base.device, {
                render_pass.set_pipeline(&self.constant_data.unlit_mesh_pipeline);

                render_pass.set_bind_group(
                    0,
                    &private_data.camera_lights_and_pbr_shader_options_bind_groups[0],
                    &[],
                );
                for unlit_instance_chunk in private_data.all_unlit_instances.chunks() {
                    let binded_unlit_mesh_index = unlit_instance_chunk.id;
                    let instances_buffer_start_index = unlit_instance_chunk.start_index as u32;
                    let instance_count = (unlit_instance_chunk.end_index
                        - unlit_instance_chunk.start_index)
                        / private_data.all_unlit_instances.stride();

                    let geometry_buffers = &data.binded_meshes[binded_unlit_mesh_index];

                    render_pass.set_bind_group(
                        1,
                        &private_data.bones_and_unlit_instances_bind_group,
                        &[0, instances_buffer_start_index],
                    );
                    render_pass
                        .set_vertex_buffer(0, geometry_buffers.vertex_buffer.src().slice(..));
                    render_pass.set_index_buffer(
                        geometry_buffers.index_buffer.buffer.src().slice(..),
                        geometry_buffers.index_buffer.format,
                    );
                    render_pass.draw_indexed(
                        0..geometry_buffers.index_buffer.buffer.length() as u32,
                        0,
                        0..instance_count as u32,
                    );
                }

                render_pass.set_pipeline(&self.constant_data.wireframe_pipeline);

                for wireframe_instance_chunk in private_data.all_wireframe_instances.chunks() {
                    let binded_wireframe_mesh_index = wireframe_instance_chunk.id;
                    let instances_buffer_start_index = wireframe_instance_chunk.start_index as u32;
                    let instance_count = (wireframe_instance_chunk.end_index
                        - wireframe_instance_chunk.start_index)
                        / private_data.all_wireframe_instances.stride();

                    let BindedWireframeMesh {
                        source_mesh_index,
                        index_buffer,
                        ..
                    } = &data.binded_wireframe_meshes[binded_wireframe_mesh_index];

                    let bone_transforms_buffer_start_index = private_data
                        .all_bone_transforms
                        .animated_bone_transforms
                        .iter()
                        .find(|bone_slice| bone_slice.mesh_index == *source_mesh_index)
                        .map(|bone_slice| bone_slice.start_index.try_into().unwrap())
                        .unwrap_or(0);

                    render_pass.set_bind_group(
                        1,
                        &private_data.bones_and_wireframe_instances_bind_group,
                        &[
                            bone_transforms_buffer_start_index,
                            instances_buffer_start_index,
                        ],
                    );
                    render_pass.set_vertex_buffer(
                        0,
                        data.binded_meshes[*source_mesh_index]
                            .vertex_buffer
                            .src()
                            .slice(..),
                    );
                    render_pass
                        .set_index_buffer(index_buffer.buffer.src().slice(..), index_buffer.format);
                    render_pass.draw_indexed(
                        0..index_buffer.buffer.length() as u32,
                        0,
                        0..instance_count as u32,
                    );
                }
            });
        }

        if data.enable_bloom {
            private_data.bloom_threshold_cleared = false;

            {
                let label = "Bloom threshold";
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: USE_LABELS.then_some(label),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &private_data.bloom_pingpong_textures[0].view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(black),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: None,
                });

                wgpu_profiler!(label, profiler, &mut render_pass, &self.base.device, {
                    render_pass.set_pipeline(&self.constant_data.bloom_threshold_pipeline);
                    render_pass.set_bind_group(0, &private_data.shading_texture_bind_group, &[]);
                    render_pass.set_bind_group(1, &private_data.bloom_config_bind_groups[0], &[]);
                    render_pass.draw(0..3, 0..1);
                });
            }

            let mut do_bloom_blur_pass =
                |encoder: &mut wgpu::CommandEncoder,
                 src_texture: &wgpu::BindGroup,
                 dst_texture: &wgpu::TextureView,
                 horizontal: bool| {
                    let label = "Bloom blur";
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: USE_LABELS.then_some(label),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: dst_texture,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(black),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });

                    wgpu_profiler!(label, profiler, &mut render_pass, &self.base.device, {
                        render_pass.set_pipeline(&self.constant_data.bloom_blur_pipeline);
                        render_pass.set_bind_group(0, src_texture, &[]);
                        render_pass.set_bind_group(
                            1,
                            &private_data.bloom_config_bind_groups[if horizontal { 0 } else { 1 }],
                            &[],
                        );
                        render_pass.draw(0..3, 0..1);
                    });
                };

            // do 10 gaussian blur passes, switching between horizontal and vertical and ping ponging between
            // the two textures, effectively doing 5 full blurs
            let blur_passes = 10;
            (0..blur_passes).for_each(|i| {
                do_bloom_blur_pass(
                    &mut encoder,
                    &private_data.bloom_pingpong_texture_bind_groups[i % 2],
                    &private_data.bloom_pingpong_textures[(i + 1) % 2].view,
                    i % 2 == 0,
                );
            });
        } else if !private_data.bloom_threshold_cleared {
            // clear bloom texture
            let label = "Bloom clear";
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &private_data.bloom_pingpong_textures[0].view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(black),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            wgpu_profiler!(label, profiler, &mut render_pass, &self.base.device, {});
            private_data.bloom_threshold_cleared = true;
        }

        {
            let label = "Skybox";
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: USE_LABELS.then_some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &private_data.tone_mapping_texture.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(black),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &private_data.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            wgpu_profiler!(label, profiler, &mut render_pass, &self.base.device, {
                render_pass.set_pipeline(&self.constant_data.skybox_pipeline);
                render_pass.set_bind_group(0, &private_data.environment_textures_bind_group, &[]);
                render_pass.set_bind_group(
                    1,
                    &private_data.camera_lights_and_pbr_shader_options_bind_groups[private_data
                        .camera_lights_and_pbr_shader_options_bind_groups
                        .len()
                        - 1],
                    &[],
                );
                render_pass.set_vertex_buffer(
                    0,
                    self.constant_data.skybox_mesh.vertex_buffer.src().slice(..),
                );
                render_pass.set_index_buffer(
                    self.constant_data
                        .skybox_mesh
                        .index_buffer
                        .buffer
                        .src()
                        .slice(..),
                    self.constant_data.skybox_mesh.index_buffer.format,
                );
                render_pass.draw_indexed(
                    0..(self.constant_data.skybox_mesh.index_buffer.buffer.length() as u32),
                    0,
                    0..1,
                );
            });
        }
        {
            let label = "Tone mapping";
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: USE_LABELS.then_some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &private_data.tone_mapping_texture.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            wgpu_profiler!(label, profiler, &mut render_pass, &self.base.device, {
                render_pass.set_pipeline(&self.constant_data.tone_mapping_pipeline);
                render_pass.set_bind_group(
                    0,
                    &private_data.shading_and_bloom_textures_bind_group,
                    &[],
                );
                render_pass.set_bind_group(1, &private_data.tone_mapping_config_bind_group, &[]);
                render_pass.draw(0..3, 0..1);
            });
        }

        {
            let label = "Transparent";
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: USE_LABELS.then_some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &private_data.tone_mapping_texture.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &private_data.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            wgpu_profiler!(label, profiler, &mut render_pass, &self.base.device, {
                render_pass.set_pipeline(&self.constant_data.transparent_mesh_pipeline);

                render_pass.set_bind_group(
                    0,
                    &private_data.camera_lights_and_pbr_shader_options_bind_groups[0],
                    &[],
                );

                for transparent_instance_chunk in private_data.all_transparent_instances.chunks() {
                    let binded_transparent_mesh_index = transparent_instance_chunk.id;
                    let instances_buffer_start_index =
                        transparent_instance_chunk.start_index as u32;
                    let instance_count = (transparent_instance_chunk.end_index
                        - transparent_instance_chunk.start_index)
                        / private_data.all_transparent_instances.stride();

                    let geometry_buffers = &data.binded_meshes[binded_transparent_mesh_index];

                    render_pass.set_bind_group(
                        1,
                        &private_data.bones_and_transparent_instances_bind_group,
                        &[0, instances_buffer_start_index],
                    );
                    render_pass
                        .set_vertex_buffer(0, geometry_buffers.vertex_buffer.src().slice(..));
                    render_pass.set_index_buffer(
                        geometry_buffers.index_buffer.buffer.src().slice(..),
                        geometry_buffers.index_buffer.format,
                    );
                    render_pass.draw_indexed(
                        0..geometry_buffers.index_buffer.buffer.length() as u32,
                        0,
                        0..instance_count as u32,
                    );
                }
            });
        }

        if let (Some(pre_gamma_fb), Some(pre_gamma_surface_blit_pipeline)) = (
            &private_data.pre_gamma_fb,
            &self.constant_data.pre_gamma_surface_blit_pipeline,
        ) {
            {
                let label = "Pre-gamma Surface blit";
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: USE_LABELS.then_some(label),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &pre_gamma_fb.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(black),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: None,
                });

                wgpu_profiler!(label, profiler, &mut render_pass, &self.base.device, {
                    {
                        render_pass.set_pipeline(pre_gamma_surface_blit_pipeline);
                        render_pass.set_bind_group(
                            0,
                            &private_data.tone_mapping_texture_bind_group,
                            &[],
                        );
                        render_pass.draw(0..3, 0..1);
                    }
                });
            }

            // TODO: pass a separate encoder to the ui overlay so it can be profiled
            ui_overlay.render(
                &self.base.device,
                &self.base.queue,
                &mut encoder,
                &pre_gamma_fb.view,
            );
        }

        if let Some(surface_blit_pipeline) = self.constant_data.surface_blit_pipeline.as_ref() {
            let label = "Surface blit";
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: USE_LABELS.then_some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(black),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            wgpu_profiler!(label, profiler, &mut render_pass, &self.base.device, {
                {
                    render_pass.set_pipeline(surface_blit_pipeline);
                    render_pass.set_bind_group(
                        0,
                        private_data
                            .pre_gamma_fb_bind_group
                            .as_ref()
                            .unwrap_or(&private_data.tone_mapping_texture_bind_group),
                        &[],
                    );
                    render_pass.set_bind_group(
                        1,
                        &private_data.tone_mapping_config_bind_group,
                        &[],
                    );
                    render_pass.draw(0..3, 0..1);
                }
            });
        }

        if private_data.pre_gamma_fb.is_none() {
            // TODO: pass a separate encoder to the ui overlay so it can be profiled
            ui_overlay.render(
                &self.base.device,
                &self.base.queue,
                &mut encoder,
                &surface_texture_view,
            )
        }

        profiler.resolve_queries(&mut encoder);

        self.base.queue.submit(std::iter::once(encoder.finish()));

        surface_texture.present();

        profiler.end_frame().map_err(|_| anyhow::anyhow!(
            "Something went wrong with wgpu_profiler. Does the crate still not report error details?"
        ))?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn render_pbr_meshes<'a>(
        base: &BaseRenderer,
        data: &RendererData,
        private_data: &RendererPrivateData,
        profiler: &mut wgpu_profiler::GpuProfiler,
        encoder: &mut wgpu::CommandEncoder,
        render_pass_descriptor: &wgpu::RenderPassDescriptor<'a, 'a>,
        pipeline: &'a wgpu::RenderPipeline,
        camera_lights_shader_options_bind_group: &'a wgpu::BindGroup,
        is_shadow: bool,
        culling_mask: u32,
        cubemap_face_index: Option<u32>,
    ) {
        let device = &base.device;

        let mut render_pass = encoder.begin_render_pass(render_pass_descriptor);

        if let Some(cubemap_face_index) = cubemap_face_index {
            render_pass.set_viewport(
                (cubemap_face_index * POINT_LIGHT_SHADOW_MAP_RESOLUTION) as f32,
                0.0,
                POINT_LIGHT_SHADOW_MAP_RESOLUTION as f32,
                POINT_LIGHT_SHADOW_MAP_RESOLUTION as f32,
                0.0,
                1.0,
            )
        }

        wgpu_profiler!(
            render_pass_descriptor
                .label
                .unwrap_or("render_pbr_meshes unlabelled"),
            profiler,
            &mut render_pass,
            device,
            {
                render_pass.set_pipeline(pipeline);

                render_pass.set_bind_group(0, camera_lights_shader_options_bind_group, &[]);
                if !is_shadow {
                    render_pass.set_bind_group(
                        1,
                        &private_data.environment_textures_bind_group,
                        &[],
                    );
                }
                for (pbr_instance_chunk_index, pbr_instance_chunk) in
                    private_data.all_pbr_instances.chunks().iter().enumerate()
                {
                    // TODO: if none of the instances pass the culling test, we should
                    //       early out at the beginning of this function to avoid creating
                    //       the render pass / clearing the texture at all.
                    if private_data.all_pbr_instances_culling_masks[pbr_instance_chunk_index]
                        & culling_mask
                        == 0
                    {
                        continue;
                    }

                    let (mesh_index, pbr_material_index) = pbr_instance_chunk.id;
                    let bone_transforms_buffer_start_index = private_data
                        .all_bone_transforms
                        .animated_bone_transforms
                        .iter()
                        .find(|bone_slice| bone_slice.mesh_index == mesh_index)
                        .map(|bone_slice| bone_slice.start_index.try_into().unwrap())
                        .unwrap_or(0);
                    let instances_buffer_start_index = pbr_instance_chunk.start_index as u32;
                    let instance_count = (pbr_instance_chunk.end_index
                        - pbr_instance_chunk.start_index)
                        / private_data.all_pbr_instances.stride();

                    let geometry_buffers = &data.binded_meshes[mesh_index];

                    let material = &data.binded_pbr_materials[pbr_material_index];

                    render_pass.set_bind_group(
                        if is_shadow { 1 } else { 2 },
                        &private_data.bones_and_pbr_instances_bind_group,
                        &[
                            bone_transforms_buffer_start_index,
                            instances_buffer_start_index,
                        ],
                    );
                    render_pass.set_bind_group(
                        if is_shadow { 2 } else { 3 },
                        &material.textures_bind_group,
                        &[],
                    );
                    render_pass
                        .set_vertex_buffer(0, geometry_buffers.vertex_buffer.src().slice(..));
                    render_pass.set_index_buffer(
                        geometry_buffers.index_buffer.buffer.src().slice(..),
                        geometry_buffers.index_buffer.format,
                    );
                    render_pass.draw_indexed(
                        0..geometry_buffers.index_buffer.buffer.length() as u32,
                        0,
                        0..instance_count as u32,
                    );
                }
            }
        );
    }

    pub fn process_profiler_frame(&self) -> Option<Vec<GpuTimerScopeResultWrapper>> {
        self.profiler
            .lock()
            .unwrap()
            .process_finished_frame()
            .map(|frames| {
                let mut result: Vec<_> = vec![];
                for frame in frames {
                    result.push(GpuTimerScopeResultWrapper(frame));
                }
                result
            })
    }
}
