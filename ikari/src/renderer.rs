use crate::buffer::{ChunkedBuffer, GpuBuffer};
use crate::camera::{
    build_cubemap_face_camera_view_directions, build_cubemap_face_camera_views,
    build_cubemap_face_frusta, MeshShaderCameraRaw, ShaderCameraData, SkyboxShaderCameraRaw,
};
use crate::collisions::{CameraFrustumDescriptor, Frustum, IntersectionResult, Sphere};
use crate::engine_state::EngineState;
use crate::file_manager::GameFilePath;
use crate::mesh::{
    BasicMesh, DynamicPbrParams, GpuPbrMeshInstance, GpuTransparentMeshInstance,
    GpuUnlitMeshInstance, GpuWireframeMeshInstance, IndexedPbrTextures, PbrTextures, ShaderVertex,
};
use crate::mutex::Mutex;
use crate::physics::rapier3d_f64::prelude::*;
use crate::raw_image::RawImage;
use crate::sampler_cache::{SamplerCache, SamplerDescriptor};
use crate::scene::{GameNode, GameNodeDescBuilder, GameNodeId, GameNodeVisual, Material, Scene};
use crate::skinning::{get_all_bone_data, AllBoneTransforms};
use crate::texture::Texture;
use crate::time::Duration;
use crate::transform::{look_in_dir, Transform, TransformBuilder};
use crate::ui::{IkariUiContainer, UiProgramEvents};
use crate::wasm_not_sync::WasmNotArc;

use std::cmp::Ordering;
use std::collections::{hash_map::Entry, HashMap};
use std::num::NonZeroU64;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use bitvec::prelude::*;

use anyhow::Result;
use glam::f32::{Mat4, Vec3};
use glam::Vec4;

use crate::physics::rapier3d_f64::na::Vector3;
use crate::physics::rapier3d_f64::parry::query::PointQuery;
use serde::Deserialize;
use serde::Serialize;
use smallvec::{smallvec, SmallVec};
use wgpu::util::DeviceExt;

pub(crate) const USE_LABELS: bool = true;
pub(crate) const USE_ORTHOGRAPHIC_CAMERA: bool = false;
pub(crate) const DISABLE_EXPENSIVE_CULLING: bool = false;
pub(crate) const USE_EXTRA_SHADOW_MAP_CULLING: bool = true;
pub(crate) const DRAW_FRUSTUM_BOUNDING_SPHERE_FOR_SHADOW_MAPS: bool = false;
pub(crate) const ENABLE_GRAPHICS_API_VALIDATION: bool = false;
pub(crate) const PRESORT_INSTANCES_BY_MESH_MATERIAL: bool = false;
pub(crate) const USE_ADAPTER_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT: bool = true;

pub const MAX_LIGHT_COUNT: usize = 32;
pub const MAX_SHADOW_CASCADES: usize = 4;
pub const DEFAULT_WIREFRAME_COLOR: [f32; 4] = [0.0, 1.0, 1.0, 1.0];
pub const POINT_LIGHT_SHADOW_MAP_FRUSTUM_NEAR_PLANE: f32 = 0.1;
pub const POINT_LIGHT_SHADOW_MAP_FRUSTUM_FAR_PLANE: f32 = 1000.0;
pub const POINT_LIGHT_SHADOW_MAP_RESOLUTION: u32 = 1024;
pub const DIRECTIONAL_LIGHT_SHADOW_MAP_RESOLUTION: u32 = 2048;
pub const POINT_LIGHT_SHOW_MAP_COUNT: u32 = 2;
pub const DIRECTIONAL_LIGHT_SHOW_MAP_COUNT: u32 = 2;
pub const DIRECTIONAL_LIGHT_PROJ_BOX_LENGTH: f32 = 50.0;
pub const MIN_SHADOW_MAP_BIAS: f32 = 0.00005;
pub const BLOOM_TARGET_MIP_COUNT: u32 = 5;

// see last comment here for why we don't have 'Hash': https://internals.rust-lang.org/t/f32-f64-should-implement-hash/5436/33
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct F16(pub half::f16);

unsafe impl bytemuck::Pod for F16 {}
unsafe impl bytemuck::Zeroable for F16 {}

impl Deref for F16 {
    type Target = half::f16;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<f32> for F16 {
    fn from(value: f32) -> Self {
        Self(half::f16::from_f32(value))
    }
}

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
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct PbrShaderOptionsUniform {
    options_1: [f32; 4],
    options_2: [f32; 4],
    options_3: [f32; 4],
    options_4: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DirectionalLightUniform {
    direction: [f32; 4],
    color: [f32; 4],
}

impl DirectionalLightUniform {
    pub fn new(light: &DirectionalLight) -> Self {
        let DirectionalLight {
            direction,
            color,
            intensity,
            ..
        } = light;

        Self {
            direction: [direction.x, direction.y, direction.z, 1.0],
            color: [color.x, color.y, color.z, *intensity],
        }
    }
}

impl Default for DirectionalLightUniform {
    fn default() -> Self {
        Self {
            direction: [0.0, -1.0, 0.0, 1.0],
            color: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

/// is a square prism
#[derive(Debug, Copy, Clone)]
pub struct CascadeProjectionVolume {
    pub half_thickness: f32,
    pub half_depth: f32,
    pub center: Vec3,
    pub direction: Vec3,
    /// side length of a pixel of the shadow map in world units
    pub pixel_size: f32,
    pub aabb: rapier3d_f64::prelude::Aabb,
}

impl CascadeProjectionVolume {
    /// creates a transform that transforms a 2x2 cube centered at the origin to become this volume
    pub fn box_transform(&self) -> Transform {
        let mut transform = look_in_dir(self.center, self.direction);

        transform.set_scale(Vec3::new(
            self.half_thickness,
            self.half_thickness,
            self.half_depth,
        ));

        transform
    }

    pub fn shader_orthographic_projection(&self) -> ShaderCameraData {
        ShaderCameraData::orthographic(
            look_in_dir(self.center, self.direction),
            self.half_thickness * 2.0,
            self.half_thickness * 2.0,
            -self.half_depth,
            self.half_depth,
            false,
        )
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ResolvedDirectionalLightCascade {
    // distance from minimum_cascade_distance to the far place of the slice of the
    // frustum that was used for computing the bounding volume for the slice
    frustum_slice_far_distance: f32,
    projection_volume: CascadeProjectionVolume,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct DirectionalLightCascadeUniform {
    world_space_to_light_space: [[f32; 4]; 4],
    frustum_slice_far_distance: f32,
    pixel_size: f32,
    _padding: [f32; 2],
}

impl DirectionalLightCascadeUniform {
    pub fn new(resolved_cascade: ResolvedDirectionalLightCascade, light_direction: Vec3) -> Self {
        let projection_volume = resolved_cascade.projection_volume;

        let shader_camera_data = ShaderCameraData::orthographic(
            look_in_dir(projection_volume.center, light_direction),
            projection_volume.half_thickness * 2.0,
            projection_volume.half_thickness * 2.0,
            -projection_volume.half_depth,
            projection_volume.half_depth,
            false,
        );

        Self {
            world_space_to_light_space: (shader_camera_data.proj * shader_camera_data.view)
                .to_cols_array_2d(),
            frustum_slice_far_distance: resolved_cascade.frustum_slice_far_distance,
            pixel_size: projection_volume.pixel_size,
            _padding: Default::default(),
        }
    }
}

fn make_directional_light_uniform_buffer(
    lights: &[DirectionalLight],
    all_resolved_cascades: &[Vec<ResolvedDirectionalLightCascade>],
) -> Vec<u8> {
    let mut light_uniforms = vec![];
    light_uniforms.reserve_exact(MAX_LIGHT_COUNT);

    let active_light_count = lights.len();
    let mut active_lights = lights
        .iter()
        .map(DirectionalLightUniform::new)
        .collect::<Vec<_>>();
    light_uniforms.append(&mut active_lights);
    light_uniforms.resize(MAX_LIGHT_COUNT, DirectionalLightUniform::default());

    let mut cascade_uniforms = vec![];
    cascade_uniforms.reserve_exact(MAX_LIGHT_COUNT * MAX_SHADOW_CASCADES);
    let mut tmp_cascade_distances = vec![];
    tmp_cascade_distances.reserve_exact(MAX_SHADOW_CASCADES);

    for light_index in 0..active_light_count {
        let light_direction = lights[light_index].direction;

        for i in 0..all_resolved_cascades[light_index].len() {
            tmp_cascade_distances.push(DirectionalLightCascadeUniform::new(
                all_resolved_cascades[light_index][i],
                light_direction,
            ));
        }

        tmp_cascade_distances.resize(
            MAX_SHADOW_CASCADES,
            DirectionalLightCascadeUniform::default(),
        );

        cascade_uniforms.append(&mut tmp_cascade_distances);
    }

    cascade_uniforms.resize(
        MAX_LIGHT_COUNT * MAX_SHADOW_CASCADES,
        DirectionalLightCascadeUniform::default(),
    );

    let mut result_bytes = bytemuck::cast_slice(&light_uniforms).to_vec();
    result_bytes.extend_from_slice(bytemuck::cast_slice(&cascade_uniforms));

    result_bytes
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

pub fn get_point_light_frustum_collider() -> &'static ConvexPolyhedron {
    static INSTANCE: OnceLock<ConvexPolyhedron> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        CameraFrustumDescriptor {
            focal_point: Vec3::ZERO,
            forward_vector: Vec3::Z,
            aspect_ratio: 1.0,
            near_plane_distance: POINT_LIGHT_SHADOW_MAP_FRUSTUM_NEAR_PLANE,
            far_plane_distance: POINT_LIGHT_SHADOW_MAP_FRUSTUM_FAR_PLANE,
            fov_x: 90.0_f32.to_radians(),
        }
        .as_convex_polyhedron()
    })
}

struct CachedCameraFrustumCollider {
    collider: ConvexPolyhedron,
    isometry: Isometry<f64>,
    descriptor: CameraFrustumDescriptor,
}

impl CachedCameraFrustumCollider {
    pub fn new(descriptor: CameraFrustumDescriptor) -> Self {
        Self {
            collider: CameraFrustumDescriptor {
                focal_point: Vec3::ZERO,
                forward_vector: Vec3::Z,
                ..descriptor
            }
            .as_convex_polyhedron(),
            isometry: descriptor.get_isometry(),
            descriptor,
        }
    }

    pub fn collider(&self) -> &ConvexPolyhedron {
        &self.collider
    }

    pub fn isometry(&self) -> &Isometry<f64> {
        &self.isometry
    }

    pub fn update(&mut self, new_descriptor: &CameraFrustumDescriptor) {
        if self.descriptor.aspect_ratio != new_descriptor.aspect_ratio
            || self.descriptor.fov_x != new_descriptor.fov_x
            || self.descriptor.near_plane_distance != new_descriptor.near_plane_distance
            || self.descriptor.far_plane_distance != new_descriptor.far_plane_distance
        {
            self.collider = CameraFrustumDescriptor {
                focal_point: Vec3::ZERO,
                forward_vector: Vec3::Z,
                ..*new_descriptor
            }
            .as_convex_polyhedron();
        }

        if self.descriptor.focal_point != new_descriptor.focal_point
            || self.descriptor.forward_vector != new_descriptor.forward_vector
        {
            self.isometry = new_descriptor.get_isometry();
        }

        self.descriptor = *new_descriptor;
    }
}

fn make_pbr_shader_options_uniform_buffer(
    shadow_settings: ShadowSettings,
    enable_shadow_debug: bool,
    enable_cascade_debug: bool,
) -> PbrShaderOptionsUniform {
    let ShadowSettings {
        enable_soft_shadows,
        shadow_bias,
        soft_shadow_factor,
        soft_shadows_max_distance,
        soft_shadow_grid_dims,
        ..
    } = shadow_settings;

    let options_1 = [
        if enable_soft_shadows { 1.0 } else { 0.0 },
        soft_shadow_factor,
        if enable_shadow_debug { 1.0 } else { 0.0 },
        soft_shadow_grid_dims as f32,
    ];

    let options_2 = [
        shadow_bias,
        if enable_cascade_debug { 1.0 } else { 0.0 },
        soft_shadows_max_distance,
        0.0,
    ];

    PbrShaderOptionsUniform {
        options_1,
        options_2,
        ..Default::default()
    }
}

#[derive(Debug)]
pub struct BindableTexture {
    pub raw_image: RawImage,
    pub name: Option<String>,
    pub sampler_descriptor: crate::sampler_cache::SamplerDescriptor,
}

#[derive(Debug)]
pub struct BindablePbrMaterial {
    pub textures: IndexedPbrTextures,
    pub dynamic_pbr_params: DynamicPbrParams,
}

#[derive(Debug)]
pub struct BindedPbrMaterial {
    pub textures_bind_group: WasmNotArc<wgpu::BindGroup>,
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
    pub vertices: Vec<ShaderVertex>,
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

#[derive(Copy, Clone, Debug)]
pub struct DirectionalLightShadowMappingConfig {
    pub num_cascades: u32,
    pub maximum_distance: f32,
    pub first_cascade_far_bound: f32,
}

impl Default for DirectionalLightShadowMappingConfig {
    fn default() -> Self {
        Self {
            num_cascades: 4,
            maximum_distance: 250.0,
            first_cascade_far_bound: 5.0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub shadow_mapping_config: DirectionalLightShadowMappingConfig,
}

#[derive(Clone, Debug, Default)]
pub struct CullingStats {
    pub time_to_cull: Duration,
    pub total_count: u64,
    pub completely_culled_count: u64,
    pub main_camera_culled_count: u64,
    // one per cascade per light
    pub directional_lights_culled_counts: Vec<Vec<u64>>,
    // one per frustum per light
    pub point_light_culled_counts: Vec<Vec<u64>>,
}

pub struct BaseRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
    pub limits: wgpu::Limits,
    pub mip_pipeline_cache: Mutex<HashMap<wgpu::TextureFormat, WasmNotArc<wgpu::RenderPipeline>>>,
    default_texture_cache: Mutex<HashMap<DefaultTextureType, WasmNotArc<Texture>>>,
    pub sampler_cache: Mutex<SamplerCache>,
}

pub struct SurfaceData {
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
}

impl BaseRenderer {
    #[profiling::function]
    pub async fn offscreen(backends: wgpu::Backends, dxc_path: Option<PathBuf>) -> Result<Self> {
        let instance = Self::make_instance(backends, dxc_path);
        Self::new(instance, None).await
    }

    #[profiling::function]
    pub async fn with_window(
        backends: wgpu::Backends,
        dxc_path: Option<PathBuf>,
        window: Arc<winit::window::Window>,
    ) -> Result<(Self, SurfaceData)> {
        let window_size = window.inner_size();

        let instance = Self::make_instance(backends, dxc_path);
        let surface = instance.create_surface(window)?;

        let base = Self::new(instance, Some(&surface)).await?;

        let mut surface_config = surface
            .get_default_config(&base.adapter, window_size.width, window_size.height)
            .ok_or_else(|| {
                anyhow::anyhow!("Window surface is incompatible with the graphics adapter")
            })?;
        surface_config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        // surface_config.format = wgpu::TextureFormat::Bgra8UnormSrgb;
        surface_config.alpha_mode = wgpu::CompositeAlphaMode::Auto;
        surface_config.present_mode = wgpu::PresentMode::AutoVsync;
        surface_config.view_formats = vec![surface_config.format.add_srgb_suffix()];
        surface_config.desired_maximum_frame_latency = 2;
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

    #[profiling::function]
    fn make_instance(backends: wgpu::Backends, dxc_path: Option<PathBuf>) -> wgpu::Instance {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            dx12_shader_compiler: wgpu::Dx12Compiler::Dxc {
                dxil_path: dxc_path.clone(),
                dxc_path,
            },
            flags: if ENABLE_GRAPHICS_API_VALIDATION {
                wgpu::InstanceFlags::debugging()
            } else {
                wgpu::InstanceFlags::empty()
            },
            ..Default::default()
        })
    }

    #[profiling::function]
    async fn new(instance: wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<Self> {
        let request_adapter_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
        };
        let adapter = instance
            .request_adapter(&request_adapter_options)
            .await
            .ok_or_else(|| {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    anyhow::anyhow!(
                        "Failed to find a wgpu adapter with options: {request_adapter_options:?}"
                    )
                }
                #[cfg(target_arch = "wasm32")]
                {
                    anyhow::anyhow!(
                        "Failed to request a WebGPU adapter. Is WebGPU supported by your browser?"
                    )
                }
            })?;

        let mut optional_features = wgpu::Features::empty();

        // used by wgpu_profiler
        optional_features |= wgpu_profiler::GpuProfiler::ALL_WGPU_TIMER_FEATURES;

        // uses half of the memory of a rgba16f texture, so it saves a nice chunk of VRAM for the bloom effect
        // without any difference in visual quality that I could detect
        // the feature should be available "everywhere we would care about". see https://github.com/gpuweb/gpuweb/issues/3566
        optional_features |= wgpu::Features::RG11B10UFLOAT_RENDERABLE;

        // panic if these features are missing
        let mut required_features = wgpu::Features::empty();

        required_features |= wgpu::Features::TEXTURE_COMPRESSION_BC;

        let features = (adapter.features() & optional_features) | required_features;

        let mut required_limits = wgpu::Limits::default();

        if USE_ADAPTER_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT {
            required_limits.min_storage_buffer_offset_alignment =
                adapter.limits().min_storage_buffer_offset_alignment;
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: features,
                    required_limits,
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
            mip_pipeline_cache: Mutex::new(HashMap::new()),
            default_texture_cache: Mutex::new(HashMap::new()),
            sampler_cache: Mutex::new(SamplerCache::default()),
        })
    }

    pub fn get_default_texture(
        &self,
        default_texture_type: DefaultTextureType,
    ) -> anyhow::Result<WasmNotArc<Texture>> {
        let mut default_texture_cache_guard = self.default_texture_cache.lock();
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

type MeshMaterialIndexPair = (usize, usize);

pub struct RendererPrivateData {
    // cpu
    all_bone_transforms: AllBoneTransforms,
    all_pbr_instances: ChunkedBuffer<GpuPbrMeshInstance, MeshMaterialIndexPair>,
    all_pbr_instances_culling_masks: Vec<BitVec>,
    pbr_mesh_index_to_gpu_instances:
        HashMap<MeshMaterialIndexPair, (SmallVec<[GpuPbrMeshInstance; 1]>, BitVec, f32)>,
    all_unlit_instances: ChunkedBuffer<GpuUnlitMeshInstance, usize>,
    all_transparent_instances: ChunkedBuffer<GpuUnlitMeshInstance, usize>,
    all_wireframe_instances: ChunkedBuffer<GpuWireframeMeshInstance, usize>,
    debug_node_bounding_spheres_nodes: Vec<GameNodeId>,
    debug_culling_frustum_nodes: Vec<GameNodeId>,
    debug_culling_frustum_mesh_index: Option<usize>,

    culling_frustum_collider: Option<CachedCameraFrustumCollider>,
    frustum_culling_lock: CullingFrustumLock,
    skybox_weights: [f32; 2],
    bloom_cleared: bool,

    new_pending_surface_config: Option<wgpu::SurfaceConfiguration>,
    current_render_scale: f32,

    // gpu
    camera_lights_and_pbr_shader_options_bind_group_layout: wgpu::BindGroupLayout,

    camera_lights_and_pbr_shader_options_bind_groups: Vec<wgpu::BindGroup>,
    bones_and_pbr_instances_bind_group: wgpu::BindGroup,
    bones_and_unlit_instances_bind_group: wgpu::BindGroup,
    bones_and_transparent_instances_bind_group: wgpu::BindGroup,
    bones_and_wireframe_instances_bind_group: wgpu::BindGroup,
    bloom_downscale_config_bind_groups: Vec<wgpu::BindGroup>,
    bloom_upscale_config_bind_group: wgpu::BindGroup,
    tone_mapping_config_bind_group: wgpu::BindGroup,
    environment_textures_bind_group: wgpu::BindGroup,
    shading_and_bloom_texture_bind_group: wgpu::BindGroup,
    tone_mapping_texture_bind_group: wgpu::BindGroup,
    shading_texture_bind_group: wgpu::BindGroup,
    bloom_texture_bind_group: wgpu::BindGroup,
    bloom_texture_mip_bind_groups: Vec<wgpu::BindGroup>,

    camera_buffers: Vec<wgpu::Buffer>,
    point_lights_buffer: wgpu::Buffer,
    directional_lights_buffer: wgpu::Buffer,
    pbr_shader_options_buffer: wgpu::Buffer,
    bloom_downscale_config_buffers: Vec<wgpu::Buffer>,
    bloom_upscale_config_buffer: wgpu::Buffer,
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

    shading_texture: Texture,
    tone_mapping_texture: Texture,
    depth_texture: Texture,
    bloom_texture: Texture,
    bloom_texture_mip_views: Vec<wgpu::TextureView>,
    brdf_lut: Texture,

    bloom_mip_count: u32,
}

#[derive(Debug)]
pub struct BindableScene {
    pub path: GameFilePath,
    pub scene: Scene,
    pub bindable_meshes: Vec<BindableGeometryBuffers>,
    pub bindable_wireframe_meshes: Vec<BindableWireframeMesh>,
    pub bindable_pbr_materials: Vec<BindablePbrMaterial>,
    pub textures: Vec<BindableTexture>,
}

#[derive(Debug)]
pub struct BindedScene {
    pub path: GameFilePath,
    pub scene: Scene,
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
    Equirectangular(RawImage),
}

#[derive(Debug)]
pub enum BindableSkyboxHDREnvironment {
    Equirectangular(RawImage),
    Cube {
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

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct GeneralSettings {
    pub render_scale: f32,
    pub enable_depth_prepass: bool,
}

impl Default for GeneralSettings {
    fn default() -> Self {
        Self {
            render_scale: 1.0,
            enable_depth_prepass: false,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CameraSettings {
    pub fov_x: f32,
    pub near_plane_distance: f32,
    pub far_plane_distance: f32,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            fov_x: 103.0f32.to_radians(),
            near_plane_distance: 0.001,
            far_plane_distance: 100000.0,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PostEffectSettings {
    pub tone_mapping_exposure: f32,

    pub enable_bloom: bool,
    pub bloom_radius: f32,
    pub bloom_intensity: f32,
}

impl Default for PostEffectSettings {
    fn default() -> Self {
        Self {
            tone_mapping_exposure: 1.0,

            enable_bloom: true,
            bloom_radius: 0.005,
            bloom_intensity: 0.04,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ShadowSettings {
    pub enable_shadows: bool,
    pub enable_soft_shadows: bool,
    pub shadow_bias: f32,
    pub soft_shadow_factor: f32,
    pub soft_shadows_max_distance: f32,
    pub soft_shadow_grid_dims: u32,
    pub shadow_small_object_culling_size_pixels: f32,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            enable_shadows: true,
            enable_soft_shadows: true,
            shadow_bias: 0.001,
            soft_shadow_factor: 0.00003,
            soft_shadows_max_distance: 100.0,
            soft_shadow_grid_dims: 4,
            shadow_small_object_culling_size_pixels: 16.0,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub struct DebugSettings {
    pub enable_wireframe: bool,
    pub draw_node_bounding_spheres: bool,
    pub draw_culling_frustum: bool,
    pub draw_point_light_culling_frusta: bool,
    pub draw_directional_light_culling_frusta: bool,
    pub enable_shadow_debug: bool,
    pub enable_cascade_debug: bool,
    pub record_culling_stats: bool,
}

pub struct RendererData {
    pub camera_node_id: Option<GameNodeId>,
    pub last_frame_culling_stats: Option<CullingStats>,

    pub general_settings: GeneralSettings,
    pub camera_settings: CameraSettings,
    pub post_effect_settings: PostEffectSettings,
    pub shadow_settings: ShadowSettings,
    pub debug_settings: DebugSettings,

    // binded (uploaded to gpu) data
    pub binded_meshes: Vec<BindedGeometryBuffers>,
    pub binded_wireframe_meshes: Vec<BindedWireframeMesh>,
    pub binded_pbr_materials: Vec<BindedPbrMaterial>,
    pub textures: Vec<Texture>,
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
    pub depth_prepass_pipeline: wgpu::RenderPipeline,
    pub unlit_mesh_pipeline: wgpu::RenderPipeline,
    pub transparent_mesh_pipeline: wgpu::RenderPipeline,
    pub wireframe_pipeline: wgpu::RenderPipeline,
    pub skybox_pipeline: wgpu::RenderPipeline,
    pub tone_mapping_pipeline: wgpu::RenderPipeline,
    pub surface_blit_pipeline: wgpu::RenderPipeline,
    pub point_shadow_map_pipeline: wgpu::RenderPipeline,
    pub directional_shadow_map_pipeline: wgpu::RenderPipeline,
    pub bloom_downscale_pipeline: wgpu::RenderPipeline,
    pub bloom_upscale_pipeline: wgpu::RenderPipeline,
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
    #[profiling::function]
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
                buffers: &[ShaderVertex::desc()],
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

        // depth_prepass_fs_main
        let mut depth_prepass_pipeline_descriptor = mesh_pipeline_descriptor.clone();
        depth_prepass_pipeline_descriptor.fragment = Some(wgpu::FragmentState {
            module: &textured_mesh_shader,
            entry_point: "depth_prepass_fs_main",
            targets: &[],
        });
        let depth_prepass_pipeline = base
            .device
            .create_render_pipeline(&depth_prepass_pipeline_descriptor);

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
        let unlit_mesh_pipeline_v_buffers = &[ShaderVertex::desc()];
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
        let wireframe_mesh_pipeline_v_buffers = &[ShaderVertex::desc()];
        wireframe_pipeline_descriptor.vertex.buffers = wireframe_mesh_pipeline_v_buffers;
        wireframe_pipeline_descriptor.primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            ..Default::default()
        };
        let wireframe_pipeline = base
            .device
            .create_render_pipeline(&wireframe_pipeline_descriptor);

        let bloom_texture_format = if base
            .device
            .features()
            .contains(wgpu::Features::RG11B10UFLOAT_RENDERABLE)
        {
            wgpu::TextureFormat::Rg11b10Float
        } else {
            log::warn!(
                "{:?} is missing. bloom quality will be slightly lower",
                wgpu::Features::RG11B10UFLOAT_RENDERABLE
            );
            wgpu::TextureFormat::Rgba16Float
        };

        let bloom_color_targets = &[Some(wgpu::ColorTargetState {
            format: bloom_texture_format,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::REPLACE,
            }),
            write_mask: wgpu::ColorWrites::COLOR,
        })];
        let bloom_downscale_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        &single_texture_bind_group_layout,
                        &single_uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let bloom_downscale_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: USE_LABELS.then_some("Bloom Downscale Pipeline"),
            layout: Some(&bloom_downscale_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: "bloom_downscale_fs_main",
                targets: bloom_color_targets,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        };
        let bloom_downscale_pipeline = base
            .device
            .create_render_pipeline(&bloom_downscale_pipeline_descriptor);

        let mut bloom_upscale_pipeline_descriptor = bloom_downscale_pipeline_descriptor.clone();
        bloom_upscale_pipeline_descriptor.fragment = Some(wgpu::FragmentState {
            module: &blit_shader,
            entry_point: "bloom_upscale_fs_main",
            targets: bloom_color_targets,
        });
        let bloom_upscale_pipeline = base
            .device
            .create_render_pipeline(&bloom_upscale_pipeline_descriptor);

        let surface_blit_color_targets = &[Some(wgpu::ColorTargetState {
            format: framebuffer_format.add_srgb_suffix(),
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
                buffers: &[ShaderVertex::desc()],
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
                buffers: &[ShaderVertex::desc()],
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
            .expect("Expected non-hdr counterpart to have a fragmet stage")
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
                buffers: &[ShaderVertex::desc()],
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
                buffers: &[ShaderVertex::desc()],
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
                buffers: &[ShaderVertex::desc()],
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
                // back face culling can be more performant here but it increases peter panning
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
                buffers: &[ShaderVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &textured_mesh_shader,
                entry_point: "directional_shadow_map_fs_main",
                targets: &[],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // back face culling can be more performant here but it increases peter panning
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

        let general_settings = GeneralSettings::default();

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
            depth_prepass_pipeline,
            unlit_mesh_pipeline,
            transparent_mesh_pipeline,
            wireframe_pipeline,
            skybox_pipeline,
            tone_mapping_pipeline,
            surface_blit_pipeline,
            point_shadow_map_pipeline,
            directional_shadow_map_pipeline,
            bloom_downscale_pipeline,
            bloom_upscale_pipeline,
            equirectangular_to_cubemap_pipeline,
            equirectangular_to_cubemap_hdr_pipeline,
            diffuse_env_map_gen_pipeline,
            specular_env_map_gen_pipeline,

            cube_mesh_index: 0,
            sphere_mesh_index: 0,
            plane_mesh_index: 0,
        };

        let shading_texture = Texture::create_scaled_surface_texture(
            &base,
            framebuffer_size,
            general_settings.render_scale,
            "shading_texture",
        );

        let bloom_texture = Texture::create_bloom_texture(
            &base,
            shading_texture.size,
            bloom_texture_format,
            BLOOM_TARGET_MIP_COUNT,
            "bloom_texture",
        );

        let bloom_mip_count = bloom_texture.texture.mip_level_count();

        let bloom_texture_mip_views = (0..bloom_mip_count)
            .map(|mip_index| {
                bloom_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        label: USE_LABELS
                            .then_some(&format!("Bloom Downscale Texture Mip View {}", mip_index)),
                        base_mip_level: mip_index,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
            })
            .collect::<Vec<_>>();

        let tone_mapping_texture = Texture::create_scaled_surface_texture(
            &base,
            framebuffer_size,
            general_settings.render_scale,
            "tone_mapping_texture",
        );

        let shading_texture_bind_group;
        let tone_mapping_texture_bind_group;
        let shading_and_bloom_texture_bind_group;
        let bloom_texture_bind_group;
        let bloom_texture_mip_bind_groups;
        {
            let sampler_cache_guard = base.sampler_cache.lock();

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
            shading_and_bloom_texture_bind_group =
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
                            resource: wgpu::BindingResource::TextureView(&bloom_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(
                                sampler_cache_guard
                                    .get_sampler_by_index(bloom_texture.sampler_index),
                            ),
                        },
                    ],
                    label: USE_LABELS.then_some("shading_and_bloom_texture_bind_group"),
                });

            bloom_texture_bind_group = base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &constant_data.single_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&bloom_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            sampler_cache_guard.get_sampler_by_index(bloom_texture.sampler_index),
                        ),
                    },
                ],
                label: USE_LABELS.then_some("bloom_texture_bind_group"),
            });

            bloom_texture_mip_bind_groups = (0..bloom_mip_count)
                .map(|mip_index| {
                    base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &constant_data.single_texture_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &bloom_texture_mip_views[mip_index as usize],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    sampler_cache_guard
                                        .get_sampler_by_index(bloom_texture.sampler_index),
                                ),
                            },
                        ],
                        label: USE_LABELS
                            .then_some(&format!("bloom_texture_mip_bind_group {}", mip_index)),
                    })
                })
                .collect::<Vec<_>>();
        }

        let bloom_downscale_config_buffers = (0..BLOOM_TARGET_MIP_COUNT)
            .map(|mip_index| {
                base.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: USE_LABELS
                            .then_some(&format!("Bloom Downscale Config Buffer {}", mip_index)),
                        contents: bytemuck::cast_slice(&[0f32, 0f32, 0f32, 0.0f32]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    })
            })
            .collect::<Vec<_>>();

        let bloom_upscale_config_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: USE_LABELS.then_some("Bloom Upscale Config Buffer"),
                    contents: bytemuck::cast_slice(&[0f32, 0f32, 0f32, 0.0f32]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let bloom_downscale_config_bind_groups = (0..bloom_downscale_config_buffers.len())
            .map(|mip_index| {
                base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &constant_data.single_uniform_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bloom_downscale_config_buffers[mip_index].as_entire_binding(),
                    }],
                    label: USE_LABELS
                        .then_some(&format!("bloom_downscale_config_bind_groups {}", mip_index,)),
                })
            })
            .collect::<Vec<_>>();

        let bloom_upscale_config_bind_group =
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &constant_data.single_uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bloom_upscale_config_buffer.as_entire_binding(),
                }],
                label: USE_LABELS.then_some("bloom_upscale_config_bind_group"),
            });

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
            general_settings.render_scale,
            "depth_texture",
        );

        let start = crate::time::Instant::now();

        let skybox_dim = 32;
        let sky_color = [8, 113, 184, 255];
        let sun_color = [253, 251, 211, 255];
        let skybox_background_texture = {
            let image = RawImage::from_dynamic_image(
                image::RgbaImage::from_pixel(skybox_dim, skybox_dim, sky_color.into()).into(),
                true,
            );
            Texture::from_raw_image(
                &base,
                &image,
                Some("skybox_image texture"),
                false,
                &SamplerDescriptor {
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                },
            )?
        };

        let skybox_environment_hdr = {
            let pixel_count = skybox_dim * skybox_dim;
            let color_hdr = sun_color.map(|val| F16::from(0.2 * val as f32 / 255.0));
            let mut image_raw: Vec<[F16; 4]> = Vec::with_capacity(pixel_count as usize);
            image_raw.resize(pixel_count as usize, color_hdr);
            let texture_er = Texture::from_raw_image(
                &base,
                &RawImage {
                    width: skybox_dim,
                    height: skybox_dim,
                    depth: 1,
                    mip_count: 1,
                    format: wgpu::TextureFormat::Rgba16Float,
                    bytes: bytemuck::cast_slice(&image_raw).to_vec(),
                },
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

            Texture::create_cubemap_from_equirectangular(
                &base,
                &constant_data,
                wgpu::TextureFormat::Rgba16Float,
                None,
                &texture_er,
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
                &skybox_background_texture,
                &skybox_environment_hdr,
            )?,
            make_skybox(
                &base,
                &constant_data,
                &skybox_background_texture,
                &skybox_environment_hdr,
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

        let directional_lights_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: USE_LABELS.then_some("Directional Lights Buffer"),
                    contents: &make_directional_light_uniform_buffer(&[], &[]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let shadow_settings = ShadowSettings::default();
        let debug_settings = DebugSettings::default();
        let initial_pbr_shader_options_buffer = make_pbr_shader_options_uniform_buffer(
            shadow_settings,
            debug_settings.enable_shadow_debug,
            debug_settings.enable_cascade_debug,
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
                            size: NonZeroU64::new(bones_buffer.length_bytes() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: pbr_instances_buffer.src(),
                            offset: 0,
                            size: NonZeroU64::new(pbr_instances_buffer.length_bytes() as u64),
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
                            size: NonZeroU64::new(bones_buffer.length_bytes() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: unlit_instances_buffer.src(),
                            offset: 0,
                            size: NonZeroU64::new(unlit_instances_buffer.length_bytes() as u64),
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
                            size: NonZeroU64::new(bones_buffer.length_bytes() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: transparent_instances_buffer.src(),
                            offset: 0,
                            size: NonZeroU64::new(
                                transparent_instances_buffer.length_bytes() as u64
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
                            size: NonZeroU64::new(bones_buffer.length_bytes() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: wireframe_instances_buffer.src(),
                            offset: 0,
                            size: NonZeroU64::new(wireframe_instances_buffer.length_bytes() as u64),
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
            DIRECTIONAL_LIGHT_SHOW_MAP_COUNT * MAX_SHADOW_CASCADES as u32,
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
            camera_node_id: None,
            last_frame_culling_stats: None,

            general_settings,
            camera_settings: CameraSettings::default(),
            post_effect_settings: PostEffectSettings::default(),
            shadow_settings,
            debug_settings,

            binded_meshes: vec![],
            binded_wireframe_meshes: vec![],
            binded_pbr_materials: vec![],
            textures: vec![],
        };

        constant_data.cube_mesh_index = Self::bind_basic_mesh(&base, &mut data, &cube_mesh, true);

        let sphere_mesh = BasicMesh::new(include_bytes!("models/sphere.obj"))?;
        constant_data.sphere_mesh_index =
            Self::bind_basic_mesh(&base, &mut data, &sphere_mesh, true);

        let plane_mesh = BasicMesh::new(include_bytes!("models/plane.obj"))?;
        constant_data.plane_mesh_index = Self::bind_basic_mesh(&base, &mut data, &plane_mesh, true);

        let wgpu_profiler_settings = wgpu_profiler::GpuProfilerSettings {
            enable_timer_queries: !cfg!(target_arch = "wasm32") && cfg!(feature = "gpu-profiling"),
            enable_debug_groups: true,
            // buffer up to 4 frames
            max_num_pending_frames: 4,
        };
        #[cfg(all(
            feature = "gpu-profiling",
            feature = "tracy-profiling",
            not(target_arch = "wasm32")
        ))]
        let profiler = wgpu_profiler::GpuProfiler::new_with_tracy_client(
            wgpu_profiler_settings,
            base.adapter.get_info().backend,
            &base.device,
            &base.queue,
        )?;

        #[cfg(not(all(
            feature = "gpu-profiling",
            feature = "tracy-profiling",
            not(target_arch = "wasm32")
        )))]
        let profiler = wgpu_profiler::GpuProfiler::new(wgpu_profiler_settings)?;

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
                pbr_mesh_index_to_gpu_instances: HashMap::new(),
                all_pbr_instances: ChunkedBuffer::new(),
                all_pbr_instances_culling_masks: vec![],
                all_unlit_instances: ChunkedBuffer::new(),
                all_transparent_instances: ChunkedBuffer::new(),
                all_wireframe_instances: ChunkedBuffer::new(),
                debug_node_bounding_spheres_nodes: vec![],
                debug_culling_frustum_nodes: vec![],
                debug_culling_frustum_mesh_index: None,

                culling_frustum_collider: None,
                frustum_culling_lock: CullingFrustumLock::None,
                skybox_weights,
                // TODO: when bloom is not enabled, we should really free up the graphics memory used for it.
                //       same goes for shadow maps
                bloom_cleared: true,

                new_pending_surface_config: None,
                current_render_scale: general_settings.render_scale,

                camera_lights_and_pbr_shader_options_bind_group_layout,

                camera_lights_and_pbr_shader_options_bind_groups: vec![],
                bones_and_pbr_instances_bind_group,
                bones_and_unlit_instances_bind_group,
                bones_and_transparent_instances_bind_group,
                bones_and_wireframe_instances_bind_group,
                bloom_downscale_config_bind_groups,
                bloom_upscale_config_bind_group,
                tone_mapping_config_bind_group,
                environment_textures_bind_group,
                shading_and_bloom_texture_bind_group,
                tone_mapping_texture_bind_group,
                shading_texture_bind_group,
                bloom_texture_bind_group,
                bloom_texture_mip_bind_groups,

                camera_buffers: vec![],
                point_lights_buffer,
                directional_lights_buffer,
                bloom_downscale_config_buffers,
                bloom_upscale_config_buffer,
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

                shading_texture,
                tone_mapping_texture,
                depth_texture,
                bloom_texture,
                bloom_texture_mip_views,
                brdf_lut,

                bloom_mip_count,
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

        let sampler_cache_guard = base.sampler_cache.lock();

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
        geometry_buffers.vertex_buffer.destroy();
        geometry_buffers.index_buffer.buffer.destroy();

        if let Some(wireframe_mesh) = data
            .binded_wireframe_meshes
            .iter()
            .find(|wireframe_mesh| wireframe_mesh.source_mesh_index == mesh_index)
        {
            wireframe_mesh.index_buffer.buffer.destroy();
        }
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
            textures_bind_group: WasmNotArc::new(textures_bind_group),
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
            std::mem::size_of::<ShaderVertex>(),
            wgpu::BufferUsages::VERTEX,
        );

        let index_buffer = GpuBuffer::from_bytes(
            device,
            bytemuck::cast_slice(&mesh.indices),
            std::mem::size_of::<u16>(),
            wgpu::BufferUsages::INDEX,
        );

        let bounding_box = crate::collisions::Aabb::make_from_points(
            mesh.vertices.iter().map(|vertex| vertex.position.into()),
        )
        .expect("Expected model to have at least two vertex positions");

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

    pub fn set_present_mode(&self, surface_data: &SurfaceData, present_mode: wgpu::PresentMode) {
        self.private_data
            .lock()
            .new_pending_surface_config
            .get_or_insert_with(|| surface_data.surface_config.clone())
            .present_mode = present_mode
    }

    pub fn resize_surface(
        &mut self,
        surface_data: &SurfaceData,
        new_unscaled_framebuffer_size: winit::dpi::PhysicalSize<u32>,
    ) {
        if new_unscaled_framebuffer_size.width == 0 || new_unscaled_framebuffer_size.height == 0 {
            log::warn!("Tried resizing the surface to an invalid resolution: {new_unscaled_framebuffer_size:?}. This request will be ignored");
            return;
        }

        let mut private_data = self.private_data.lock();
        let new_pending_surface_config = private_data
            .new_pending_surface_config
            .get_or_insert_with(|| surface_data.surface_config.clone());

        new_pending_surface_config.width = new_unscaled_framebuffer_size.width;
        new_pending_surface_config.height = new_unscaled_framebuffer_size.height;
    }

    pub fn reconfigure_surface_if_needed(
        &self,
        surface_data: &mut SurfaceData,
        force_reconfigure: bool,
    ) -> bool {
        let mut private_data_guard = self.private_data.lock();

        let new_surface_config = private_data_guard
            .new_pending_surface_config
            .take()
            .unwrap_or_else(|| surface_data.surface_config.clone());

        if new_surface_config.width == 0 || new_surface_config.height == 0 {
            return false;
        }

        let surface_config_changed = surface_data.surface_config != new_surface_config;

        let surface_resized = (
            surface_data.surface_config.width,
            surface_data.surface_config.height,
        ) != (new_surface_config.width, new_surface_config.height);

        if surface_resized {
            log::debug!(
                "Resizing surface to {:?}",
                (new_surface_config.width, new_surface_config.height)
            );
        }

        let data_guard = self.data.lock();

        let new_render_scale = data_guard.general_settings.render_scale;
        let current_render_scale = private_data_guard.current_render_scale;
        let render_scale_changed = current_render_scale != new_render_scale;

        if !surface_config_changed && !render_scale_changed && !force_reconfigure {
            return false;
        }

        let new_unscaled_framebuffer_size = (new_surface_config.width, new_surface_config.height);

        surface_data.surface_config = new_surface_config;
        private_data_guard.current_render_scale = new_render_scale;

        surface_data
            .surface
            .configure(&self.base.device, &surface_data.surface_config);

        if !surface_resized && !render_scale_changed {
            return false;
        }

        private_data_guard.shading_texture = Texture::create_scaled_surface_texture(
            &self.base,
            new_unscaled_framebuffer_size,
            new_render_scale,
            "shading_texture",
        );

        private_data_guard.bloom_texture = Texture::create_bloom_texture(
            &self.base,
            private_data_guard.shading_texture.size,
            private_data_guard.bloom_texture.texture.format(),
            BLOOM_TARGET_MIP_COUNT,
            "bloom_texture",
        );

        private_data_guard.bloom_mip_count =
            private_data_guard.bloom_texture.texture.mip_level_count();

        private_data_guard.bloom_texture_mip_views = (0..private_data_guard.bloom_mip_count)
            .map(|mip_index| {
                private_data_guard
                    .bloom_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        label: USE_LABELS
                            .then_some(&format!("Bloom Downscale Texture Mip View {}", mip_index)),
                        base_mip_level: mip_index,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
            })
            .collect::<Vec<_>>();

        private_data_guard.tone_mapping_texture = Texture::create_scaled_surface_texture(
            &self.base,
            new_unscaled_framebuffer_size,
            new_render_scale,
            "tone_mapping_texture",
        );
        private_data_guard.depth_texture = Texture::create_depth_texture(
            &self.base,
            new_unscaled_framebuffer_size,
            new_render_scale,
            "depth_texture",
        );

        let device = &self.base.device;
        let single_texture_bind_group_layout = &self.constant_data.single_texture_bind_group_layout;

        let sampler_cache_guard = self.base.sampler_cache.lock();

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

        private_data_guard.shading_and_bloom_texture_bind_group = self
            .base
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.constant_data.two_texture_bind_group_layout,
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
                            &private_data_guard.bloom_texture.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(
                            sampler_cache_guard.get_sampler_by_index(
                                private_data_guard.bloom_texture.sampler_index,
                            ),
                        ),
                    },
                ],
                label: USE_LABELS.then_some("shading_and_bloom_texture_bind_group"),
            });

        private_data_guard.bloom_texture_bind_group =
            self.base
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.constant_data.single_texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &private_data_guard.bloom_texture.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                sampler_cache_guard.get_sampler_by_index(
                                    private_data_guard.bloom_texture.sampler_index,
                                ),
                            ),
                        },
                    ],
                    label: USE_LABELS.then_some("bloom_texture_bind_group"),
                });

        private_data_guard.bloom_texture_mip_bind_groups = (0..private_data_guard.bloom_mip_count)
            .map(|mip_index| {
                self.base
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &self.constant_data.single_texture_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &private_data_guard.bloom_texture_mip_views[mip_index as usize],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    sampler_cache_guard.get_sampler_by_index(
                                        private_data_guard.bloom_texture.sampler_index,
                                    ),
                                ),
                            },
                        ],
                        label: USE_LABELS
                            .then_some(&format!("bloom_texture_mip_bind_group {}", mip_index)),
                    })
            })
            .collect::<Vec<_>>();

        surface_resized
    }

    #[profiling::function]
    pub fn add_debug_nodes(
        &self,
        data: &mut RendererData,
        private_data: &mut RendererPrivateData,
        engine_state: &mut EngineState,
        main_culling_frustum: &Frustum,
        main_culling_frustum_desc: &CameraFrustumDescriptor,
        resolved_directional_light_cascades: &[Vec<ResolvedDirectionalLightCascade>],
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

        let DebugSettings {
            draw_node_bounding_spheres,
            draw_culling_frustum,
            draw_point_light_culling_frusta,
            draw_directional_light_culling_frusta,
            ..
        } = data.debug_settings;

        if draw_node_bounding_spheres {
            let node_ids: Vec<_> = scene.nodes().map(|node| node.id()).collect();
            for node_id in node_ids {
                if let Some(bounding_sphere) = scene.get_node_bounding_sphere(node_id, data) {
                    let culling_frustum_intersection_result =
                        Self::get_node_cam_intersection_result(
                            scene
                                .get_node(node_id)
                                .expect("Node id should still be valid at this point"),
                            bounding_sphere,
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

        if draw_culling_frustum {
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

        if draw_point_light_culling_frusta {
            let mut new_node_descs = vec![];
            for point_light in &scene.point_lights {
                for controlled_direction in build_cubemap_face_camera_view_directions() {
                    let frustum_descriptor = CameraFrustumDescriptor {
                        focal_point: scene
                            .get_global_transform_for_node(point_light.node_id)
                            .unwrap_or_default()
                            .position(),
                        forward_vector: controlled_direction.to_vector(),
                        near_plane_distance: POINT_LIGHT_SHADOW_MAP_FRUSTUM_NEAR_PLANE,
                        far_plane_distance: POINT_LIGHT_SHADOW_MAP_FRUSTUM_FAR_PLANE,
                        fov_x: 90.0_f32.to_radians(),
                        aspect_ratio: 1.0,
                    };

                    let debug_frustum_descriptor = CameraFrustumDescriptor {
                        // shrink the frustum along the view direction for the debug view
                        near_plane_distance: 0.5,
                        far_plane_distance: 5.0,
                        ..frustum_descriptor
                    };
                    let debug_culling_frustum_mesh = debug_frustum_descriptor.to_basic_mesh();

                    let culling_frustum_collider = private_data
                        .culling_frustum_collider
                        .as_ref()
                        .expect("Should have have checked for None above");

                    let collision_based_color = if rapier3d_f64::parry::query::intersection_test(
                        &frustum_descriptor.get_isometry(),
                        get_point_light_frustum_collider(),
                        culling_frustum_collider.isometry(),
                        culling_frustum_collider.collider(),
                    )
                    .expect("Frustum-Frustum query should be supported")
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

        if draw_directional_light_culling_frusta {
            let mut new_node_descs = vec![];
            for (light_index, _) in scene.directional_lights.iter().enumerate() {
                for (cascade_index, resolved_cascade) in resolved_directional_light_cascades
                    [light_index]
                    .iter()
                    .enumerate()
                {
                    let projection_volume = resolved_cascade.projection_volume;

                    let transform = projection_volume.box_transform();

                    let culling_box_mesh = GameNodeVisual {
                        material: Material::Transparent {
                            color: match cascade_index {
                                0 => Vec4::new(0.0, 1.0, 0.0, 0.1),
                                1 => Vec4::new(0.0, 0.0, 1.0, 0.1),
                                2 => Vec4::new(1.0, 0.0, 1.0, 0.1),
                                _ => Vec4::new(1.0, 0.0, 0.0, 0.1),
                            },
                            premultiplied_alpha: false,
                        },
                        mesh_index: self.constant_data.cube_mesh_index,
                        wireframe: false,
                        cullable: false,
                    };

                    let culling_box_mesh_wf = GameNodeVisual {
                        wireframe: true,
                        ..culling_box_mesh.clone()
                    };

                    new_node_descs.push(
                        GameNodeDescBuilder::new()
                            .transform(transform)
                            .visual(Some(culling_box_mesh))
                            .build(),
                    );
                    new_node_descs.push(
                        GameNodeDescBuilder::new()
                            .transform(transform)
                            .visual(Some(culling_box_mesh_wf))
                            .build(),
                    );

                    if DRAW_FRUSTUM_BOUNDING_SPHERE_FOR_SHADOW_MAPS {
                        let frustum_bounding_sphere_mesh = GameNodeVisual {
                            material: Material::Transparent {
                                color: Vec4::new(0.0, 0.0, 1.0, 0.1),
                                premultiplied_alpha: false,
                            },
                            mesh_index: self.constant_data.sphere_mesh_index,
                            wireframe: false,
                            cullable: false,
                        };

                        new_node_descs.push(
                            GameNodeDescBuilder::new()
                                .transform(
                                    TransformBuilder::new()
                                        .position(projection_volume.center)
                                        .scale(Vec3::splat(projection_volume.half_thickness))
                                        .build(),
                                )
                                .visual(Some(frustum_bounding_sphere_mesh))
                                .build(),
                        );
                    }
                }
            }

            for new_node_desc in new_node_descs {
                private_data
                    .debug_culling_frustum_nodes
                    .push(scene.add_node(new_node_desc).id());
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

        let data_guard = self.data.lock();
        let mut private_data_guard = self.private_data.lock();

        if CullingFrustumLockMode::from(private_data_guard.frustum_culling_lock.clone())
            == lock_mode
        {
            return;
        }

        let Some(camera_transform) = data_guard
            .camera_node_id
            .and_then(|camera_node_id| engine_state.scene.get_node(camera_node_id))
            .map(|camera_node| camera_node.transform)
        else {
            log::error!("Couldn't set the frustum culling lock as there is currently no camera");
            return;
        };

        let position = match private_data_guard.frustum_culling_lock {
            CullingFrustumLock::Full(desc) => desc.focal_point,
            CullingFrustumLock::FocalPoint(locked_focal_point) => locked_focal_point,
            CullingFrustumLock::None => camera_transform.position(),
        };

        let CameraSettings {
            fov_x,
            near_plane_distance,
            far_plane_distance,
        } = data_guard.camera_settings;

        private_data_guard.frustum_culling_lock = match lock_mode {
            CullingFrustumLockMode::Full => CullingFrustumLock::Full(CameraFrustumDescriptor {
                focal_point: camera_transform.position(),
                forward_vector: (-camera_transform.z_axis).into(),
                aspect_ratio,
                near_plane_distance,
                far_plane_distance,
                fov_x,
            }),
            CullingFrustumLockMode::FocalPoint => CullingFrustumLock::FocalPoint(position),
            CullingFrustumLockMode::None => CullingFrustumLock::None,
        };
    }

    pub fn render<UiOverlay>(
        &mut self,
        engine_state: &mut EngineState,
        surface_data: &SurfaceData,
        surface_texture: wgpu::SurfaceTexture,
        ui_overlay: &mut IkariUiContainer<UiOverlay>,
    ) -> anyhow::Result<()>
    where
        UiOverlay:
            iced_winit::runtime::Program<Renderer = iced::Renderer> + UiProgramEvents + 'static,
    {
        if surface_data.surface_config.width == 0 || surface_data.surface_config.height == 0 {
            return Ok(());
        }

        self.update_internal(engine_state, &surface_data.surface_config);
        self.render_internal(engine_state, surface_texture, ui_overlay)
    }

    fn get_node_cam_intersection_result(
        node: &GameNode,
        node_bounding_sphere: Sphere,
        _data: &RendererData,
        _scene: &Scene,
        camera_culling_frustum: &Frustum,
    ) -> Option<IntersectionResult> {
        let visual = node.visual.as_ref()?;

        /* bounding boxes will be wrong for skinned meshes so we currently can't cull them */
        if node.skin_index.is_some() || !visual.cullable {
            return None;
        }

        Some(camera_culling_frustum.sphere_intersection_test(node_bounding_sphere))
    }

    // culling mask is a bitvec where each bit corresponds to a frustum
    // and the value of the bit represents whether the object
    // is touching that frustum or not. the first bit represnts the main
    // camera frustum, the subsequent bits represent the directional shadow
    // mapping boxes and the rest of the bits represent the point light shadow
    // mapping frusta, of which there are 6 per point light so 6 bits are used
    // per point light.
    fn get_node_culling_mask(
        node: &GameNode,
        data: &RendererData,
        engine_state: &EngineState,
        camera_culling_frustum: &Frustum,
        point_lights_frusta: &PointLightFrustaWithCullingInfo,
        resolved_directional_light_cascades: &[Vec<ResolvedDirectionalLightCascade>],
        culling_mask: &mut BitVec,
    ) {
        culling_mask.set_elements(0);

        let Some(visual) = node.visual.as_ref() else {
            return;
        };

        // bounding boxes will be wrong for skinned meshes so we currently can't cull them
        if DISABLE_EXPENSIVE_CULLING
            || USE_ORTHOGRAPHIC_CAMERA
            || !visual.cullable
            || node.skin_index.is_some()
        {
            if data.shadow_settings.enable_shadows {
                culling_mask.set_elements(usize::MAX);
            } else {
                culling_mask.set(0, true);
            }

            return;
        }

        let node_bounding_sphere = engine_state.scene.get_node_bounding_sphere_opt(node.id());
        let node_bounding_sphere_point = rapier3d_f64::na::Point3::new(
            node_bounding_sphere.center.x as f64,
            node_bounding_sphere.center.y as f64,
            node_bounding_sphere.center.z as f64,
        );

        let is_touching_frustum = |frustum: &Frustum| {
            matches!(
                frustum.sphere_intersection_test(node_bounding_sphere),
                IntersectionResult::FullyContained | IntersectionResult::PartiallyIntersecting
            )
        };

        let mut mask_pos = 0;

        let is_node_on_screen = is_touching_frustum(camera_culling_frustum);

        if is_node_on_screen {
            culling_mask.set(mask_pos, true);
        }

        if !data.shadow_settings.enable_shadows {
            return;
        }

        mask_pos += 1;

        for (light_index, _) in engine_state.scene.directional_lights.iter().enumerate() {
            let light_cascades = &resolved_directional_light_cascades[light_index];

            for (cascade_index, resolved_cascade) in light_cascades.iter().enumerate() {
                let projection_volume = resolved_cascade.projection_volume;

                // Cull objects that cover less than one pixel in the shadow map
                if projection_volume.pixel_size
                    * data.shadow_settings.shadow_small_object_culling_size_pixels
                    > node_bounding_sphere.radius * 2.0
                {
                    mask_pos += light_cascades.len() - cascade_index;
                    break;
                }

                if projection_volume
                    .aabb
                    .distance_to_local_point(&node_bounding_sphere_point, true)
                    < node_bounding_sphere.radius.into()
                {
                    culling_mask.set(mask_pos, true);
                }

                mask_pos += 1;
            }
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
                            culling_mask.set(mask_pos, true);
                        }

                        mask_pos += 1;
                    }
                }
                None => {
                    mask_pos += 6;
                }
            }
        }
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
        let sampler_cache_guard = base.sampler_cache.lock();

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
        let mut private_data_guard = self.private_data.lock();

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
        self.private_data.lock().skybox_weights = normalized;
    }

    pub fn get_skybox_weights(&self) -> [f32; 2] {
        self.private_data.lock().skybox_weights
    }

    /// Prepare and send all data to gpu so it's ready to render
    #[profiling::function]
    fn update_internal(
        &mut self,
        engine_state: &mut EngineState,
        surface_config: &wgpu::SurfaceConfiguration,
    ) {
        let mut data_guard = self.data.lock();
        let data: &mut RendererData = &mut data_guard;

        let mut private_data_guard = self.private_data.lock();
        let private_data: &mut RendererPrivateData = &mut private_data_guard;

        let aspect_ratio = surface_config.width as f32 / surface_config.height as f32;

        let CameraSettings {
            fov_x,
            near_plane_distance,
            far_plane_distance,
        } = data.camera_settings;

        let DebugSettings {
            enable_shadow_debug,
            enable_cascade_debug,
            ..
        } = data.debug_settings;

        let PostEffectSettings {
            tone_mapping_exposure,

            enable_bloom,
            bloom_radius,
            bloom_intensity,
        } = data.post_effect_settings;

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
            near_plane_distance,
            far_plane_distance,
            fov_x,
        };

        let culling_frustum_desc = match private_data.frustum_culling_lock {
            CullingFrustumLock::Full(locked) => locked,
            CullingFrustumLock::FocalPoint(locked_position) => CameraFrustumDescriptor {
                focal_point: locked_position,
                ..camera_frustum_desc
            },
            CullingFrustumLock::None => camera_frustum_desc,
        };

        match private_data.culling_frustum_collider.as_mut() {
            Some(collider) => {
                collider.update(&culling_frustum_desc);
            }
            None => {
                private_data.culling_frustum_collider =
                    Some(CachedCameraFrustumCollider::new(culling_frustum_desc));
            }
        };

        let culling_frustum = Frustum::from(culling_frustum_desc);

        let mut resolved_directional_light_cascades = vec![];
        for directional_light in &engine_state.scene.directional_lights {
            let from_light_space =
                look_in_dir(Vec3::new(0.0, 0.0, 0.0), directional_light.direction);
            let to_light_space = from_light_space.inverse();

            let DirectionalLightShadowMappingConfig {
                num_cascades,
                maximum_distance,
                first_cascade_far_bound,
                ..
            } = directional_light.shadow_mapping_config;

            // https://github.com/bevyengine/bevy/blob/951c9bb1a25caddc97ac69bbe8a2937f97227e90/crates/bevy_pbr/src/light.rs#L258
            let cascade_distances = if num_cascades == 1 {
                vec![maximum_distance]
            } else {
                let base = (maximum_distance / first_cascade_far_bound)
                    .powf(1.0 / (num_cascades - 1) as f32);
                (0..num_cascades)
                    .map(|i| first_cascade_far_bound * base.powf(i as f32))
                    .collect()
            };

            let mut resolved_cascades = vec![];

            let minimum_cascade_distance = 0.5;
            let overlap_proportion = 0.0;

            for cascade_index in 0..cascade_distances.len() {
                let light_space_frustum_slice = CameraFrustumDescriptor {
                    focal_point: to_light_space.transform_point3(culling_frustum_desc.focal_point),
                    forward_vector: to_light_space
                        .transform_vector3(culling_frustum_desc.forward_vector),
                    near_plane_distance: if cascade_index > 0 {
                        (1.0 - overlap_proportion) * cascade_distances[cascade_index - 1]
                    } else {
                        minimum_cascade_distance
                    },
                    far_plane_distance: cascade_distances[cascade_index],
                    ..culling_frustum_desc
                };

                // we use a bounding sphere to make sure the shadow map camera box's thickness remains consistent regardless
                // of the rotation of the main camera's view frustum.
                //
                // we round the position to the nearest pixel_size to make sure the shadow map camera box
                // moves in pixel-sized increments.
                // combined with the above bounding sphere logic, this ensures that the shadow map is always sampled at the same
                // location regardless of the camera transform. Otherwise, there would be visible shimmering/aliasing as the user
                // moves the camera around that is quite distracting.
                //
                // See https://www.gamedev.net/forums/topic/591684-xna-40---shimmering-shadow-maps/
                //     https://www.youtube.com/watch?v=u0pk1LyLKYQ
                let bounding_sphere =
                    light_space_frustum_slice.make_rotation_independent_bounding_sphere();

                let pixel_size =
                    2.0 * bounding_sphere.radius / DIRECTIONAL_LIGHT_SHADOW_MAP_RESOLUTION as f32;

                let rounded_bounding_sphere_center = Vec3::new(
                    bounding_sphere.center.x - bounding_sphere.center.x % pixel_size,
                    bounding_sphere.center.y - bounding_sphere.center.y % pixel_size,
                    bounding_sphere.center.z - bounding_sphere.center.z % pixel_size,
                );

                let projection_volume_half_thickness = bounding_sphere.radius;

                let projection_half_depth = (DIRECTIONAL_LIGHT_PROJ_BOX_LENGTH
                    * projection_volume_half_thickness.sqrt()
                    + projection_volume_half_thickness)
                    / 2.0;

                // make sure the box's "far plane" is roughly at the edge of the frustum slice
                let projection_center = from_light_space
                    .transform_point3(rounded_bounding_sphere_center)
                    - directional_light.direction
                        * (projection_half_depth - projection_volume_half_thickness);

                let transform = {
                    let mut transform = look_in_dir(projection_center, directional_light.direction);

                    transform.set_scale(Vec3::new(
                        projection_volume_half_thickness,
                        projection_volume_half_thickness,
                        projection_half_depth,
                    ));
                    transform
                };

                let projection_collider = Cuboid::new(Vector3::new(
                    transform.scale().x as f64,
                    transform.scale().y as f64,
                    transform.scale().z as f64,
                ));
                let projection_isometry = Isometry::from_parts(
                    nalgebra::Translation3::new(
                        transform.position().x as f64,
                        transform.position().y as f64,
                        transform.position().z as f64,
                    ),
                    nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                        transform.rotation().w as f64,
                        transform.rotation().x as f64,
                        transform.rotation().y as f64,
                        transform.rotation().z as f64,
                    )),
                );

                let projection_volume = CascadeProjectionVolume {
                    half_thickness: projection_volume_half_thickness,
                    half_depth: projection_half_depth,
                    center: projection_center,
                    direction: directional_light.direction,
                    pixel_size,
                    aabb: projection_collider.aabb(&projection_isometry),
                };

                resolved_cascades.push(ResolvedDirectionalLightCascade {
                    frustum_slice_far_distance: cascade_distances[cascade_index],
                    projection_volume,
                });
            }

            resolved_directional_light_cascades.push(resolved_cascades);
        }

        self.add_debug_nodes(
            data,
            private_data,
            engine_state,
            &culling_frustum,
            &culling_frustum_desc,
            &resolved_directional_light_cascades,
        );

        let culling_frustum_collider = private_data
            .culling_frustum_collider
            .as_ref()
            .expect("Should have have checked for None above");

        engine_state.scene.recompute_global_node_transforms(data);

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
        //
        // Since instances must be packed in continguous lists for the instanced draw calls,
        // we would need to create a separate instance buffer for each camera to be able to
        // draw a different set of instances per camera.
        //
        // Instead, we conservatively combine the culling mask of all instances into one
        // and use it for the whole group, meaning frustum culling can be much less effective
        // for large groups of instances because if only one of them is visible by the camera
        // then all the rest of them need to be drawn
        private_data.pbr_mesh_index_to_gpu_instances.clear();
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
                                .map(|desc| {
                                    // if the light view doesn't intersect with the main camera view at all
                                    // then none of the objects inside of it can cast shadows on objects
                                    // that are inside the main view so we can completely skip the shadow map
                                    // render pass for this light view
                                    let is_light_view_culled = if USE_EXTRA_SHADOW_MAP_CULLING {
                                        !rapier3d_f64::parry::query::intersection_test(
                                            culling_frustum_collider.isometry(),
                                            culling_frustum_collider.collider(),
                                            &desc.get_isometry(),
                                            get_point_light_frustum_collider(),
                                        )
                                        .expect("Frustum-Frustum query should be supported")
                                    } else {
                                        false
                                    };
                                    (desc.into(), is_light_view_culled)
                                })
                                .collect(),
                            can_cull_offscreen_objects,
                        )
                    })
            })
            .collect();

        Self::prepare_and_cull_instances(
            engine_state,
            data,
            private_data,
            &culling_frustum,
            &point_lights_frusta,
            &resolved_directional_light_cascades,
            &mut wireframe_mesh_index_to_gpu_instances,
            &mut unlit_mesh_index_to_gpu_instances,
            &mut transparent_meshes,
            camera_position,
        );

        let min_storage_buffer_offset_alignment =
            self.base.limits.min_storage_buffer_offset_alignment;

        let mut pbr_mesh_instances: Vec<_> = private_data
            .pbr_mesh_index_to_gpu_instances
            .drain()
            .collect();

        if PRESORT_INSTANCES_BY_MESH_MATERIAL {
            pbr_mesh_instances.sort_by_key(|((mesh_index, material_index), _)| {
                ((*mesh_index as u128) << 64) + *material_index as u128
            });
        }

        pbr_mesh_instances.sort_by(
            |(_, (_, _, dist_sq_from_player_a)), (_, (_, _, dist_sq_from_player_b))| {
                dist_sq_from_player_a
                    .partial_cmp(dist_sq_from_player_b)
                    .unwrap_or(Ordering::Equal)
            },
        );

        private_data.all_pbr_instances_culling_masks = pbr_mesh_instances
            .iter()
            .map(|(_, (_, culling_mask, _))| culling_mask.clone())
            .collect();

        private_data.all_pbr_instances.replace(
            pbr_mesh_instances
                .into_iter()
                .map(|(key, (instances, _, _))| (key, instances.into_boxed_slice())),
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
            unlit_mesh_index_to_gpu_instances
                .into_iter()
                .map(|(key, instances)| (key, instances.into_boxed_slice())),
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
            |(_, _, dist_sq_from_player_a), (_, _, dist_sq_from_player_b)| {
                dist_sq_from_player_b
                    .partial_cmp(dist_sq_from_player_a)
                    .unwrap_or(Ordering::Equal)
            },
        );

        private_data.all_transparent_instances.replace(
            transparent_meshes
                .into_iter()
                .map(|(mesh_index, instance, _)| (mesh_index, Box::from([instance]))),
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
            wireframe_mesh_index_to_gpu_instances
                .into_iter()
                .map(|(key, instances)| (key, instances.into_boxed_slice())),
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
                                    private_data.bones_buffer.length_bytes() as u64
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
                                        as u64,
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
                                    private_data.bones_buffer.length_bytes() as u64
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
                                        as u64,
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
                                    private_data.bones_buffer.length_bytes() as u64
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
                                        as u64,
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
                                    private_data.bones_buffer.length_bytes() as u64
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
                                        as u64,
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
                camera_transform,
                20.0 * aspect_ratio,
                20.0,
                -1000.0,
                1000.0,
                false,
            )
        } else {
            ShaderCameraData::perspective(
                camera_transform,
                aspect_ratio,
                near_plane_distance,
                far_plane_distance,
                fov_x,
                true,
            )
        };
        all_camera_data.push(main_camera_shader_data);

        // directional lights
        for (light_index, _light) in engine_state.scene.directional_lights.iter().enumerate() {
            for resolved_cascade in &resolved_directional_light_cascades[light_index] {
                all_camera_data.push(
                    resolved_cascade
                        .projection_volume
                        .shader_orthographic_projection(),
                );
            }
        }

        // point lights
        for point_light in &engine_state.scene.point_lights {
            let light_position = engine_state
                .scene
                .get_node(point_light.node_id)
                .map(|node| node.transform.position())
                .unwrap_or_default();
            all_camera_data.reserve(6);
            all_camera_data.extend(&mut build_cubemap_face_camera_views(
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
            byte_unit::Byte::from_bytes(bytes as u128)
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
            &make_directional_light_uniform_buffer(
                &engine_state.scene.directional_lights,
                &resolved_directional_light_cascades,
            ),
        );
        queue.write_buffer(
            &private_data.tone_mapping_config_buffer,
            0,
            bytemuck::cast_slice(&[
                tone_mapping_exposure,
                if enable_bloom {
                    bloom_intensity
                } else {
                    -1.0f32
                },
                0f32,
                0f32,
            ]),
        );
        for mip_index in 0..private_data.bloom_mip_count {
            queue.write_buffer(
                &private_data.bloom_downscale_config_buffers[mip_index as usize],
                0,
                bytemuck::cast_slice(&[
                    (private_data.bloom_texture.size.width / 2u32.pow(mip_index)) as f32,
                    (private_data.bloom_texture.size.height / 2u32.pow(mip_index)) as f32,
                    0.0f32,
                    0.0f32,
                ]),
            );
        }
        queue.write_buffer(
            &private_data.bloom_upscale_config_buffer,
            0,
            bytemuck::cast_slice(&[bloom_radius, 0.0f32, 0.0f32, 0.0f32]),
        );
        queue.write_buffer(
            &private_data.pbr_shader_options_buffer,
            0,
            bytemuck::cast_slice(&[make_pbr_shader_options_uniform_buffer(
                data.shadow_settings,
                enable_shadow_debug,
                enable_cascade_debug,
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
        UiOverlay:
            iced_winit::runtime::Program<Renderer = iced::Renderer> + UiProgramEvents + 'static,
    {
        let mut data_guard = self.data.lock();
        let data: &mut RendererData = &mut data_guard;

        let mut private_data_guard = self.private_data.lock();
        let private_data: &mut RendererPrivateData = &mut private_data_guard;

        let mut profiler_guard = self.profiler.lock();
        let profiler: &mut wgpu_profiler::GpuProfiler = &mut profiler_guard;

        let GeneralSettings {
            enable_depth_prepass,
            ..
        } = data.general_settings;

        let PostEffectSettings { enable_bloom, .. } = data.post_effect_settings;

        let ShadowSettings { enable_shadows, .. } = data.shadow_settings;

        let surface_texture_view =
            surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    format: Some(surface_texture.texture.format().add_srgb_suffix()),
                    ..Default::default()
                });

        let mut encoder = self
            .base
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        if enable_shadows {
            let mut culling_mask_camera_index = 1; // start at one to skip main camera

            for (light_index, light) in engine_state.scene.directional_lights.iter().enumerate() {
                if light_index >= DIRECTIONAL_LIGHT_SHOW_MAP_COUNT as usize {
                    continue;
                }

                for cascade_index in 0..light.shadow_mapping_config.num_cascades {
                    let texture_view = private_data
                        .directional_shadow_map_textures
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            base_array_layer: cascade_index
                                + MAX_SHADOW_CASCADES as u32 * light_index as u32,
                            array_layer_count: Some(1),
                            ..Default::default()
                        });

                    let pass_label = "Directional light shadow map";

                    let shadow_render_pass_desc = wgpu::RenderPassDescriptor {
                        label: USE_LABELS.then_some(pass_label),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &texture_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        occlusion_query_set: None,
                        timestamp_writes: None, // overwritten by wgpu_profiler
                    };

                    let mut profiler_scope =
                        profiler.scope(pass_label, &mut encoder, &self.base.device);

                    let mut render_pass = profiler_scope.scoped_render_pass(
                        pass_label,
                        &self.base.device,
                        shadow_render_pass_desc,
                    );

                    Self::render_pbr_meshes(
                        data,
                        private_data,
                        &mut render_pass,
                        &self.constant_data.directional_shadow_map_pipeline,
                        &private_data.camera_lights_and_pbr_shader_options_bind_groups
                            [culling_mask_camera_index],
                        true,
                        culling_mask_camera_index,
                    );

                    culling_mask_camera_index += 1;
                }
            }

            for light_index in 0..engine_state.scene.point_lights.len() {
                if light_index >= POINT_LIGHT_SHOW_MAP_COUNT as usize {
                    continue;
                }
                if let Some(light_node) = engine_state
                    .scene
                    .get_node(engine_state.scene.point_lights[light_index].node_id)
                {
                    let pass_label = "Point light shadow map";

                    let texture_view = private_data.point_shadow_map_textures.texture.create_view(
                        &wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            base_array_layer: light_index.try_into().unwrap(),
                            array_layer_count: Some(1),
                            ..Default::default()
                        },
                    );

                    let shadow_render_pass_desc = wgpu::RenderPassDescriptor {
                        label: USE_LABELS.then_some(pass_label),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &texture_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        occlusion_query_set: None,
                        timestamp_writes: None, // overwritten by wgpu_profiler
                    };

                    let mut profiler_scope =
                        profiler.scope(pass_label, &mut encoder, &self.base.device);

                    let mut render_pass = profiler_scope.scoped_render_pass(
                        pass_label,
                        &self.base.device,
                        shadow_render_pass_desc,
                    );

                    build_cubemap_face_camera_views(
                        light_node.transform.position(),
                        POINT_LIGHT_SHADOW_MAP_FRUSTUM_NEAR_PLANE,
                        POINT_LIGHT_SHADOW_MAP_FRUSTUM_FAR_PLANE,
                        false,
                    )
                    .enumerate()
                    .for_each(|(face_index, _face_view_proj_matrices)| {
                        render_pass.set_viewport(
                            (face_index * POINT_LIGHT_SHADOW_MAP_RESOLUTION as usize) as f32,
                            0.0,
                            POINT_LIGHT_SHADOW_MAP_RESOLUTION as f32,
                            POINT_LIGHT_SHADOW_MAP_RESOLUTION as f32,
                            0.0,
                            1.0,
                        );

                        Self::render_pbr_meshes(
                            data,
                            private_data,
                            &mut render_pass,
                            &self.constant_data.point_shadow_map_pipeline,
                            &private_data.camera_lights_and_pbr_shader_options_bind_groups
                                [culling_mask_camera_index],
                            true,
                            culling_mask_camera_index,
                        );

                        culling_mask_camera_index += 1;
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

        if enable_depth_prepass {
            let depth_prepass_pass_label = "Depth pre-pass";

            let depth_prepass_render_pass_desc = wgpu::RenderPassDescriptor {
                label: USE_LABELS.then_some(depth_prepass_pass_label),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &private_data.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None, // overwritten by wgpu_profiler
            };

            let mut profiler_scope =
                profiler.scope(depth_prepass_pass_label, &mut encoder, &self.base.device);

            let mut render_pass = profiler_scope.scoped_render_pass(
                depth_prepass_pass_label,
                &self.base.device,
                depth_prepass_render_pass_desc,
            );

            Self::render_pbr_meshes(
                data,
                private_data,
                &mut render_pass,
                &self.constant_data.depth_prepass_pipeline,
                &private_data.camera_lights_and_pbr_shader_options_bind_groups[0],
                false,
                0, // use main camera culling mask
            );
        }

        {
            let pbr_meshes_pass_label = "Pbr meshes";

            let shading_render_pass_desc = wgpu::RenderPassDescriptor {
                label: USE_LABELS.then_some(pbr_meshes_pass_label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &private_data.shading_texture.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(black),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &private_data.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: if enable_depth_prepass {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(0.0)
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None, // overwritten by wgpu_profiler
            };

            let mut profiler_scope =
                profiler.scope(pbr_meshes_pass_label, &mut encoder, &self.base.device);

            let mut render_pass = profiler_scope.scoped_render_pass(
                pbr_meshes_pass_label,
                &self.base.device,
                shading_render_pass_desc,
            );

            Self::render_pbr_meshes(
                data,
                private_data,
                &mut render_pass,
                &self.constant_data.mesh_pipeline,
                &private_data.camera_lights_and_pbr_shader_options_bind_groups[0],
                false,
                0, // use main camera culling mask
            );
        }

        {
            let pass_label = "Unlit and wireframe";

            let mut profiler_scope = profiler.scope(pass_label, &mut encoder, &self.base.device);

            let mut render_pass = profiler_scope.scoped_render_pass(
                pass_label,
                &self.base.device,
                wgpu::RenderPassDescriptor {
                    label: USE_LABELS.then_some(pass_label),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &private_data.shading_texture.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &private_data.depth_texture.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None, // overwritten by wgpu_profiler
                },
            );

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
                render_pass.set_vertex_buffer(0, geometry_buffers.vertex_buffer.src().slice(..));
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
        }

        if enable_bloom {
            private_data.bloom_cleared = false;

            for mip_index in 0..private_data.bloom_mip_count as usize {
                let pass_label = "Bloom Downscale";

                let mut profiler_scope =
                    profiler.scope(pass_label, &mut encoder, &self.base.device);

                let src_texture_bind_group = if mip_index == 0 {
                    &private_data.shading_texture_bind_group
                } else {
                    &private_data.bloom_texture_mip_bind_groups[mip_index - 1]
                };
                let dst_texture_view = &private_data.bloom_texture_mip_views[mip_index];

                let mut render_pass = profiler_scope.scoped_render_pass(
                    pass_label,
                    &self.base.device,
                    wgpu::RenderPassDescriptor {
                        label: USE_LABELS.then_some(pass_label),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: dst_texture_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(black),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None, // overwritten by wgpu_profiler
                    },
                );

                render_pass.set_pipeline(&self.constant_data.bloom_downscale_pipeline);
                render_pass.set_bind_group(0, src_texture_bind_group, &[]);
                render_pass.set_bind_group(
                    1,
                    &private_data.bloom_downscale_config_bind_groups[mip_index],
                    &[],
                );
                render_pass.draw(0..3, 0..1);
            }

            for mip_index in (1..private_data.bloom_mip_count as usize).rev() {
                let pass_label = "Bloom Upscale";

                let mut profiler_scope =
                    profiler.scope(pass_label, &mut encoder, &self.base.device);

                let src_texture_bind_group = &private_data.bloom_texture_mip_bind_groups[mip_index];
                let dst_texture_view = &private_data.bloom_texture_mip_views[mip_index - 1];

                let mut render_pass = profiler_scope.scoped_render_pass(
                    pass_label,
                    &self.base.device,
                    wgpu::RenderPassDescriptor {
                        label: USE_LABELS.then_some(pass_label),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: dst_texture_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None, // overwritten by wgpu_profiler
                    },
                );

                render_pass.set_pipeline(&self.constant_data.bloom_upscale_pipeline);
                render_pass.set_bind_group(0, src_texture_bind_group, &[]);
                render_pass.set_bind_group(1, &private_data.bloom_upscale_config_bind_group, &[]);
                render_pass.draw(0..3, 0..1);
            }
        }

        if !enable_bloom && !private_data.bloom_cleared {
            let pass_label = "Bloom clear";

            let mut profiler_scope = profiler.scope(pass_label, &mut encoder, &self.base.device);

            profiler_scope.scoped_render_pass(
                pass_label,
                &self.base.device,
                wgpu::RenderPassDescriptor {
                    label: USE_LABELS.then_some(pass_label),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &private_data.bloom_texture_mip_views[0],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(black),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None, // overwritten by wgpu_profiler
                },
            );
            private_data.bloom_cleared = true;
        }

        {
            let pass_label = "Skybox";

            let mut profiler_scope = profiler.scope(pass_label, &mut encoder, &self.base.device);

            let mut render_pass = profiler_scope.scoped_render_pass(
                pass_label,
                &self.base.device,
                wgpu::RenderPassDescriptor {
                    label: USE_LABELS.then_some(pass_label),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &private_data.tone_mapping_texture.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(black),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &private_data.depth_texture.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None, // overwritten by wgpu_profiler
                },
            );

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
        }
        {
            let pass_label = "Tone mapping";

            let mut profiler_scope = profiler.scope(pass_label, &mut encoder, &self.base.device);

            let mut render_pass = profiler_scope.scoped_render_pass(
                pass_label,
                &self.base.device,
                wgpu::RenderPassDescriptor {
                    label: USE_LABELS.then_some(pass_label),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &private_data.tone_mapping_texture.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None, // overwritten by wgpu_profiler
                },
            );
            render_pass.set_pipeline(&self.constant_data.tone_mapping_pipeline);
            render_pass.set_bind_group(0, &private_data.shading_and_bloom_texture_bind_group, &[]);
            render_pass.set_bind_group(1, &private_data.tone_mapping_config_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        {
            let pass_label = "Transparent";

            let mut profiler_scope = profiler.scope(pass_label, &mut encoder, &self.base.device);

            let mut render_pass = profiler_scope.scoped_render_pass(
                pass_label,
                &self.base.device,
                wgpu::RenderPassDescriptor {
                    label: USE_LABELS.then_some(pass_label),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &private_data.tone_mapping_texture.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &private_data.depth_texture.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None, // overwritten by wgpu_profiler
                },
            );

            render_pass.set_pipeline(&self.constant_data.transparent_mesh_pipeline);

            render_pass.set_bind_group(
                0,
                &private_data.camera_lights_and_pbr_shader_options_bind_groups[0],
                &[],
            );

            for transparent_instance_chunk in private_data.all_transparent_instances.chunks() {
                let binded_transparent_mesh_index = transparent_instance_chunk.id;
                let instances_buffer_start_index = transparent_instance_chunk.start_index as u32;
                let instance_count = (transparent_instance_chunk.end_index
                    - transparent_instance_chunk.start_index)
                    / private_data.all_transparent_instances.stride();

                let geometry_buffers = &data.binded_meshes[binded_transparent_mesh_index];

                render_pass.set_bind_group(
                    1,
                    &private_data.bones_and_transparent_instances_bind_group,
                    &[0, instances_buffer_start_index],
                );
                render_pass.set_vertex_buffer(0, geometry_buffers.vertex_buffer.src().slice(..));
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

        {
            let pass_label = "Surface blit";

            let mut profiler_scope = profiler.scope(pass_label, &mut encoder, &self.base.device);

            let mut render_pass = profiler_scope.scoped_render_pass(
                pass_label,
                &self.base.device,
                wgpu::RenderPassDescriptor {
                    label: USE_LABELS.then_some(pass_label),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &surface_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(black),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None, // overwritten by wgpu_profiler
                },
            );

            render_pass.set_pipeline(&self.constant_data.surface_blit_pipeline);
            render_pass.set_bind_group(0, &private_data.tone_mapping_texture_bind_group, &[]);
            render_pass.set_bind_group(1, &private_data.tone_mapping_config_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        {
            let profiler_scope = profiler.scope("UI overlay", &mut encoder, &self.base.device);

            ui_overlay.render(
                &self.base.device,
                &self.base.queue,
                profiler_scope.recorder,
                &surface_texture_view,
            );
        }

        profiler.resolve_queries(&mut encoder);

        self.base.queue.submit(std::iter::once(encoder.finish()));

        surface_texture.present();

        profiler.end_frame()?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn render_pbr_meshes<'a>(
        data: &'a RendererData,
        private_data: &'a RendererPrivateData,
        render_pass: &mut wgpu::RenderPass<'a>,
        pipeline: &'a wgpu::RenderPipeline,
        camera_lights_shader_options_bind_group: &'a wgpu::BindGroup,
        is_shadow: bool,
        culling_mask_camera_index: usize,
    ) {
        // early out if all objects are culled from current pass
        if (0..private_data.all_pbr_instances.chunks().len()).all(|pbr_instance_chunk_index| {
            !private_data.all_pbr_instances_culling_masks[pbr_instance_chunk_index]
                [culling_mask_camera_index]
        }) {
            return;
        }

        render_pass.set_pipeline(pipeline);

        render_pass.set_bind_group(0, camera_lights_shader_options_bind_group, &[]);
        if !is_shadow {
            render_pass.set_bind_group(1, &private_data.environment_textures_bind_group, &[]);
        }

        for (pbr_instance_chunk_index, pbr_instance_chunk) in
            private_data.all_pbr_instances.chunks().iter().enumerate()
        {
            if !private_data.all_pbr_instances_culling_masks[pbr_instance_chunk_index]
                [culling_mask_camera_index]
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
            let instance_count = (pbr_instance_chunk.end_index - pbr_instance_chunk.start_index)
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
            render_pass.set_vertex_buffer(0, geometry_buffers.vertex_buffer.src().slice(..));
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

    pub fn process_profiler_frame(&self) -> Option<Vec<wgpu_profiler::GpuTimerQueryResult>> {
        self.profiler
            .lock()
            .process_finished_frame(self.base.queue.get_timestamp_period())
    }

    #[profiling::function]
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_and_cull_instances(
        engine_state: &EngineState,
        data: &mut RendererData,
        private_data: &mut RendererPrivateData,
        culling_frustum: &Frustum,
        point_lights_frusta: &PointLightFrustaWithCullingInfo,
        resolved_directional_light_cascades: &[Vec<ResolvedDirectionalLightCascade>],
        wireframe_mesh_index_to_gpu_instances: &mut HashMap<usize, Vec<GpuWireframeMeshInstance>>,
        unlit_mesh_index_to_gpu_instances: &mut HashMap<usize, Vec<GpuUnlitMeshInstance>>,
        transparent_meshes: &mut Vec<(usize, GpuTransparentMeshInstance, f32)>,
        camera_position: Vec3,
    ) {
        let mut stats = CullingStats::default();
        let start = crate::time::Instant::now();

        let directional_light_camera_count: usize = engine_state
            .scene
            .directional_lights
            .iter()
            .map(|light| light.shadow_mapping_config.num_cascades as usize)
            .sum();
        let point_light_camera_count = engine_state.scene.point_lights.len() * 6;
        let camera_count = 1 + directional_light_camera_count + point_light_camera_count;

        let mut tmp_node_culling_mask = BitVec::repeat(false, camera_count);
        let mut culled_object_counts_per_camera: Vec<u64> = vec![0; camera_count];

        let DebugSettings {
            enable_wireframe,
            record_culling_stats,
            ..
        } = data.debug_settings;

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
                let node_bounding_sphere =
                    engine_state.scene.get_node_bounding_sphere_opt(node.id());
                let dist_sq_from_player = node_bounding_sphere
                    .center
                    .distance_squared(camera_position)
                    - node_bounding_sphere.radius;

                match (material, enable_wireframe, wireframe) {
                    (
                        Material::Pbr {
                            binded_material_index,
                            dynamic_pbr_params,
                        },
                        false,
                        false,
                    ) => {
                        stats.total_count += 1;

                        Self::get_node_culling_mask(
                            node,
                            data,
                            engine_state,
                            culling_frustum,
                            point_lights_frusta,
                            resolved_directional_light_cascades,
                            &mut tmp_node_culling_mask,
                        );

                        let mut completely_culled = true;

                        for element in tmp_node_culling_mask.iter().by_vals() {
                            if element {
                                completely_culled = false;
                                break;
                            }
                        }

                        if record_culling_stats {
                            for (camera_index, element) in
                                tmp_node_culling_mask.iter().by_vals().enumerate()
                            {
                                if element {
                                    culled_object_counts_per_camera[camera_index] += 1;
                                }
                            }

                            if completely_culled {
                                stats.completely_culled_count += 1;
                            }
                        }

                        if completely_culled {
                            continue;
                        }

                        let gpu_instance = GpuPbrMeshInstance::new(
                            transform,
                            dynamic_pbr_params.unwrap_or_else(|| {
                                data.binded_pbr_materials[binded_material_index].dynamic_pbr_params
                            }),
                        );

                        match private_data
                            .pbr_mesh_index_to_gpu_instances
                            .entry((mesh_index, binded_material_index))
                        {
                            Entry::Occupied(mut entry) => {
                                entry.get_mut().0.push(gpu_instance);
                                // combine instance culling masks
                                *entry.get_mut().1 |= &tmp_node_culling_mask;
                            }
                            Entry::Vacant(entry) => {
                                entry.insert((
                                    smallvec![gpu_instance],
                                    tmp_node_culling_mask.clone(),
                                    dist_sq_from_player,
                                ));
                            }
                        }
                    }
                    (material, enable_wireframe, is_node_wireframe) => {
                        let (color, is_transparent) = match material {
                            Material::Unlit { color } => ([color.x, color.y, color.z, 1.0], false),
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

                        if enable_wireframe || is_node_wireframe {
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
                            match wireframe_mesh_index_to_gpu_instances.entry(wireframe_mesh_index)
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
                                dist_sq_from_player,
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

        if record_culling_stats {
            stats.main_camera_culled_count = culled_object_counts_per_camera[0];

            let mut directional_light_index_acc = 1;

            for (light_index, cascades) in resolved_directional_light_cascades.iter().enumerate() {
                let mut cascade_counts = Vec::with_capacity(cascades.len());

                for cascade_index in 0..cascades.len() {
                    let cull_index = 1 + light_index + cascade_index;
                    cascade_counts.push(culled_object_counts_per_camera[cull_index]);

                    directional_light_index_acc += 1;
                }
                stats.directional_lights_culled_counts.push(cascade_counts);
            }

            for frusta in point_lights_frusta.iter() {
                let mut frusta_counts = vec![];
                if let Some(frusta) = &frusta {
                    for frustum_index in 0..frusta.0.len() {
                        let cull_index = directional_light_index_acc + frustum_index;
                        frusta_counts.push(culled_object_counts_per_camera[cull_index]);
                    }
                }

                stats.point_light_culled_counts.push(frusta_counts);
            }

            stats.time_to_cull = start.elapsed();

            data.last_frame_culling_stats = Some(stats);
        } else {
            data.last_frame_culling_stats = None;
        }
    }
}
