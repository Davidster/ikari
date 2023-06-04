use crate::buffer::*;
use crate::camera::*;
use crate::collisions::*;
use crate::game::*;
use crate::game_state::*;
use crate::light::*;
use crate::logger::*;
use crate::math::*;
use crate::mesh::*;
use crate::sampler_cache::*;
use crate::scene::*;
use crate::skinning::*;
use crate::texture::*;
use crate::transform::*;
use crate::ui_overlay::*;

use std::collections::{hash_map::Entry, HashMap};
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use glam::f32::{Mat4, Vec3};
use glam::Vec4;
use image::Pixel;
use wgpu::util::DeviceExt;
use wgpu::InstanceDescriptor;
use wgpu_profiler::wgpu_profiler;
use winit::window::Window;

pub const MAX_LIGHT_COUNT: usize = 32;
pub const NEAR_PLANE_DISTANCE: f32 = 0.001;
pub const FAR_PLANE_DISTANCE: f32 = 100000.0;
pub const FOV_Y_DEG: f32 = 45.0;
pub const DEFAULT_WIREFRAME_COLOR: [f32; 4] = [0.0, 1.0, 1.0, 1.0];
pub const POINT_LIGHT_SHADOW_MAP_FRUSTUM_NEAR_PLANE: f32 = 0.1;
pub const POINT_LIGHT_SHADOW_MAP_FRUSTUM_FAR_PLANE: f32 = 1000.0;
pub const POINT_LIGHT_SHADOW_MAP_RESOLUTION: u32 = 1024;
pub const DIRECTIONAL_LIGHT_SHADOW_MAP_RESOLUTION: u32 = 2048;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Float16(half::f16);

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

fn make_point_light_uniform_buffer(game_state: &GameState) -> Vec<PointLightUniform> {
    let mut light_uniforms = Vec::new();

    let active_light_count = game_state.point_lights.len();
    let mut active_lights = game_state
        .point_lights
        .iter()
        .flat_map(|point_light| {
            game_state
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
    position: [f32; 4],
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

impl From<&DirectionalLightComponent> for DirectionalLightUniform {
    fn from(light: &DirectionalLightComponent) -> Self {
        let DirectionalLightComponent {
            position,
            direction,
            color,
            intensity,
        } = light;
        let view_proj_matrices =
            build_directional_light_camera_view(-light.direction, 100.0, 100.0, 1000.0);
        Self {
            world_space_to_light_space: (view_proj_matrices.proj * view_proj_matrices.view)
                .to_cols_array_2d(),
            position: [position.x, position.y, position.z, 1.0],
            direction: [direction.x, direction.y, direction.z, 1.0],
            color: [color.x, color.y, color.z, *intensity],
        }
    }
}

impl Default for DirectionalLightUniform {
    fn default() -> Self {
        Self {
            world_space_to_light_space: Mat4::IDENTITY.to_cols_array_2d(),
            position: [0.0, 0.0, 0.0, 1.0],
            direction: [0.0, -1.0, 0.0, 1.0],
            color: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

fn make_directional_light_uniform_buffer(
    lights: &[DirectionalLightComponent],
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

pub enum SkyboxBackground<'a> {
    Cube { face_image_paths: [&'a str; 6] },
    Equirectangular { image_path: &'a str },
}

pub enum SkyboxHDREnvironment<'a> {
    Equirectangular { image_path: &'a str },
}

// TODO: store a global list of GeometryBuffers, and store indices into them in BindedPbrMesh and BindedUnlitMesh.
//       this would not work if the Vertex attributes / format every becomes different between these two shaders
#[derive(Debug)]
pub struct BindedPbrMesh {
    pub geometry_buffers: GeometryBuffers,
    pub textures_bind_group: Arc<wgpu::BindGroup>,
    pub dynamic_pbr_params: DynamicPbrParams,

    pub alpha_mode: AlphaMode,
    pub primitive_mode: PrimitiveMode,
}

#[derive(Debug)]
pub struct GeometryBuffers {
    pub vertex_buffer: GpuBuffer,
    pub index_buffer: GpuBuffer,
    pub index_buffer_format: wgpu::IndexFormat,
    pub bounding_box: crate::collisions::Aabb,
}

pub type BindedUnlitMesh = GeometryBuffers;

pub type BindedTransparentMesh = GeometryBuffers;

#[derive(Debug, PartialEq, Eq)]
pub enum MeshType {
    Pbr,
    Unlit,
    Transparent,
}

impl From<GameNodeMeshType> for MeshType {
    fn from(game_node_mesh_type: GameNodeMeshType) -> Self {
        match game_node_mesh_type {
            GameNodeMeshType::Pbr { .. } => MeshType::Pbr,
            GameNodeMeshType::Unlit { .. } => MeshType::Unlit,
            GameNodeMeshType::Transparent { .. } => MeshType::Transparent,
        }
    }
}

#[derive(Debug)]
pub struct BindedWireframeMesh {
    pub source_mesh_type: MeshType,
    pub source_mesh_index: usize,
    pub index_buffer: GpuBuffer,
    pub index_buffer_format: wgpu::IndexFormat,
}

#[derive(Debug)]
pub enum AlphaMode {
    Opaque,
    Mask,
}

#[derive(Debug)]
pub enum PrimitiveMode {
    Triangles,
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

pub struct BaseRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
    pub surface: wgpu::Surface,
    pub surface_config: Mutex<wgpu::SurfaceConfiguration>,
    pub limits: wgpu::Limits,
    pub window_size: Mutex<winit::dpi::PhysicalSize<u32>>,
    pub single_texture_bind_group_layout: wgpu::BindGroupLayout,
    pub two_texture_bind_group_layout: wgpu::BindGroupLayout,
    pub bones_and_instances_bind_group_layout: wgpu::BindGroupLayout,
    pub pbr_textures_bind_group_layout: wgpu::BindGroupLayout,
    default_texture_cache: Mutex<HashMap<DefaultTextureType, Arc<Texture>>>,
    pub sampler_cache: Mutex<SamplerCache>,
}

impl BaseRenderer {
    pub async fn new(
        window: &winit::window::Window,
        backends: wgpu::Backends,
        present_mode: wgpu::PresentMode,
    ) -> Self {
        let window_size = window.inner_size();

        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends,
            dx12_shader_compiler: wgpu::Dx12Compiler::Dxc {
                dxil_path: Some(PathBuf::from("dxc/")),
                dxc_path: Some(PathBuf::from("dxc/")),
            },
        });
        let surface = unsafe { instance.create_surface(&window).unwrap() };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let adapter_info = adapter.get_info();
        log::info!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

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
            .expect("Failed to create device");

        let mut surface_config = surface
            .get_default_config(&adapter, window_size.width, window_size.height)
            .expect("Window surface is incompatible with the graphics adapter");
        surface_config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        // surface_config.format = wgpu::TextureFormat::Bgra8UnormSrgb;
        surface_config.alpha_mode = wgpu::CompositeAlphaMode::Auto;
        surface_config.present_mode = present_mode;

        println!("swapchain_format={:?}", surface_config.format);

        surface.configure(&device, &surface_config);

        let single_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some("single_texture_bind_group_layout"),
            });
        let two_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some("two_texture_bind_group_layout"),
            });
        let pbr_textures_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some("pbr_textures_bind_group_layout"),
            });

        let bones_and_instances_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some("bones_and_instances_bind_group_layout"),
            });

        let limits = device.limits();

        Self {
            device,
            adapter,
            queue,
            surface,
            surface_config: Mutex::new(surface_config),
            limits,
            window_size: Mutex::new(window_size),
            single_texture_bind_group_layout,
            two_texture_bind_group_layout,
            bones_and_instances_bind_group_layout,
            pbr_textures_bind_group_layout,
            default_texture_cache: Mutex::new(HashMap::new()),
            sampler_cache: Mutex::new(SamplerCache::new()),
        }
    }

    #[profiling::function]
    pub fn make_pbr_textures_bind_group(
        &self,
        material: &PbrMaterial,
        use_gltf_defaults: bool,
    ) -> Result<wgpu::BindGroup> {
        let auto_generated_diffuse_texture;
        let diffuse_texture = match material.base_color {
            Some(diffuse_texture) => diffuse_texture,
            None => {
                auto_generated_diffuse_texture =
                    self.get_default_texture(DefaultTextureType::BaseColor)?;
                &auto_generated_diffuse_texture
            }
        };
        let auto_generated_normal_map;
        let normal_map = match material.normal {
            Some(normal_map) => normal_map,
            None => {
                auto_generated_normal_map = self.get_default_texture(DefaultTextureType::Normal)?;
                &auto_generated_normal_map
            }
        };
        let auto_generated_metallic_roughness_map;
        let metallic_roughness_map = match material.metallic_roughness {
            Some(metallic_roughness_map) => metallic_roughness_map,
            None => {
                auto_generated_metallic_roughness_map =
                    self.get_default_texture(if use_gltf_defaults {
                        DefaultTextureType::MetallicRoughnessGLTF
                    } else {
                        DefaultTextureType::MetallicRoughness
                    })?;
                &auto_generated_metallic_roughness_map
            }
        };
        let auto_generated_emissive_map;
        let emissive_map = match material.emissive {
            Some(emissive_map) => emissive_map,
            None => {
                auto_generated_emissive_map = self.get_default_texture(if use_gltf_defaults {
                    DefaultTextureType::EmissiveGLTF
                } else {
                    DefaultTextureType::Emissive
                })?;
                &auto_generated_emissive_map
            }
        };
        let auto_generated_ambient_occlusion_map;
        let ambient_occlusion_map = match material.ambient_occlusion {
            Some(ambient_occlusion_map) => ambient_occlusion_map,
            None => {
                auto_generated_ambient_occlusion_map =
                    self.get_default_texture(DefaultTextureType::AmbientOcclusion)?;
                &auto_generated_ambient_occlusion_map
            }
        };

        let sampler_cache_guard = self.sampler_cache.lock().unwrap();

        let textures_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.pbr_textures_bind_group_layout,
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
            label: Some("InstancedMeshComponent textures_bind_group"),
        });

        Ok(textures_bind_group)
    }

    pub fn get_default_texture(
        &self,
        default_texture_type: DefaultTextureType,
    ) -> anyhow::Result<Arc<Texture>> {
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
                Arc::new(Texture::from_color(self, color)?)
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
    Full((Frustum, Vec3, Vec3)),
    FocalPoint(Vec3),
    None,
}

pub struct RendererPrivateData {
    // cpu
    all_bone_transforms: AllBoneTransforms,
    all_pbr_instances: ChunkedBuffer<GpuPbrMeshInstance>,
    all_pbr_instances_culling_masks: Vec<u32>,
    all_unlit_instances: ChunkedBuffer<GpuUnlitMeshInstance>,
    all_transparent_instances: ChunkedBuffer<GpuUnlitMeshInstance>,
    all_wireframe_instances: ChunkedBuffer<GpuWireframeMeshInstance>,
    debug_node_bounding_spheres_nodes: Vec<GameNodeId>,
    debug_culling_frustum_nodes: Vec<GameNodeId>,
    debug_culling_frustum_mesh_index: Option<usize>,

    bloom_threshold_cleared: bool,
    frustum_culling_lock: CullingFrustumLock, // for debug

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

    point_shadow_map_textures: Texture,
    directional_shadow_map_textures: Texture,
    shading_texture: Texture,
    tone_mapping_texture: Texture,
    depth_texture: Texture,
    bloom_pingpong_textures: [Texture; 2],
}

#[derive(Debug)]
pub struct RenderBuffers {
    pub binded_pbr_meshes: Vec<BindedPbrMesh>,
    pub binded_unlit_meshes: Vec<BindedUnlitMesh>,
    pub binded_transparent_meshes: Vec<BindedTransparentMesh>,
    pub binded_wireframe_meshes: Vec<BindedWireframeMesh>,
    pub textures: Vec<Texture>,
}

pub struct RendererPublicData {
    pub binded_pbr_meshes: Vec<BindedPbrMesh>,
    pub binded_unlit_meshes: Vec<BindedUnlitMesh>,
    pub binded_transparent_meshes: Vec<BindedTransparentMesh>,
    pub binded_wireframe_meshes: Vec<BindedWireframeMesh>,
    pub textures: Vec<Texture>,

    pub skybox_mesh: GeometryBuffers,

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
    pub enable_soft_shadows: bool,
    pub shadow_bias: f32,
    pub soft_shadow_factor: f32,
    pub enable_shadow_debug: bool,
    pub soft_shadow_grid_dims: u32,

    pub ui_overlay: IkariUiOverlay,
}

pub struct Renderer {
    pub base: Arc<BaseRenderer>,
    pub data: Arc<Mutex<RendererPublicData>>,

    private_data: Mutex<RendererPrivateData>,

    profiler: Mutex<wgpu_profiler::GpuProfiler>,

    mesh_pipeline: wgpu::RenderPipeline,
    unlit_mesh_pipeline: wgpu::RenderPipeline,
    transparent_mesh_pipeline: wgpu::RenderPipeline,
    wireframe_pipeline: wgpu::RenderPipeline,
    skybox_pipeline: wgpu::RenderPipeline,
    tone_mapping_pipeline: wgpu::RenderPipeline,
    surface_blit_pipeline: wgpu::RenderPipeline,
    point_shadow_map_pipeline: wgpu::RenderPipeline,
    directional_shadow_map_pipeline: wgpu::RenderPipeline,
    bloom_threshold_pipeline: wgpu::RenderPipeline,
    bloom_blur_pipeline: wgpu::RenderPipeline,

    #[allow(dead_code)]
    box_mesh_index: i32,
    #[allow(dead_code)]
    transparent_box_mesh_index: i32,
    // TODO: since unlit and transparent share the same shader, these don't reaaaally need to be stored in different lists.
    #[allow(dead_code)]
    sphere_mesh_index: i32,
    #[allow(dead_code)]
    transparent_sphere_mesh_index: i32,
    #[allow(dead_code)]
    plane_mesh_index: i32,
}

impl Renderer {
    pub async fn new(base: BaseRenderer, window: &Window) -> Result<Self> {
        logger_log("Controls:");
        vec![
            "Look Around:             Mouse",
            "Move Around:             WASD, Space Bar, Ctrl",
            "Adjust Speed:            Scroll",
            "Adjust Render Scale:     Z / X",
            "Adjust Exposure:         E / R",
            "Adjust Bloom Threshold:  T / Y",
            "Pause/Resume Animations: P",
            "Toggle Bloom Effect:     B",
            "Toggle Shadows:          M",
            "Toggle Wireframe:        F",
            "Toggle Collision Boxes:  C",
            "Draw Bounding Spheres:   J",
            "Open Options Menu:       Escape",
        ]
        .iter()
        .for_each(|line| {
            logger_log(&format!("  {line}"));
        });

        let unlit_mesh_shader = base
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Unlit Mesh Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    crate::file_loader::read_to_string("src/shaders/unlit_mesh.wgsl")
                        .await?
                        .into(),
                ),
            });

        let blit_shader = base
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Blit Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    crate::file_loader::read_to_string("src/shaders/blit.wgsl")
                        .await?
                        .into(),
                ),
            });

        let textured_mesh_shader = base
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Textured Mesh Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    crate::file_loader::read_to_string("src/shaders/textured_mesh.wgsl")
                        .await?
                        .into(),
                ),
            });

        let skybox_shader = base
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Skybox Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    crate::file_loader::read_to_string("src/shaders/skybox.wgsl")
                        .await?
                        .into(),
                ),
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
                    label: Some("single_cube_texture_bind_group_layout"),
                });

        let environment_textures_bind_group_layout =
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
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::Cube,
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
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 9,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 11,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                    ],
                    label: Some("environment_textures_bind_group_layout"),
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
                    label: Some("single_uniform_bind_group_layout"),
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
                    label: Some("two_uniform_bind_group_layout"),
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
                label: Some("camera_lights_and_pbr_shader_options_bind_group_layout"),
            });

        let fragment_shader_color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let mesh_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Mesh Pipeline Layout"),
                    bind_group_layouts: &[
                        &camera_lights_and_pbr_shader_options_bind_group_layout,
                        &environment_textures_bind_group_layout,
                        &base.bones_and_instances_bind_group_layout,
                        &base.pbr_textures_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        let mesh_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Pipeline"),
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
                    label: Some("Unlit Mesh Pipeline Layout"),
                    bind_group_layouts: &[
                        &camera_lights_and_pbr_shader_options_bind_group_layout,
                        &base.bones_and_instances_bind_group_layout,
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
                        &base.single_texture_bind_group_layout,
                        &single_uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let bloom_threshold_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Threshold Pipeline"),
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
            label: Some("Bloom Blur Pipeline"),
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

        let surface_blit_color_targets = &[Some(wgpu::ColorTargetState {
            format: base.surface_config.lock().unwrap().format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let surface_blit_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        &base.single_texture_bind_group_layout,
                        &single_uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let surface_blit_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Surface Blit Render Pipeline"),
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
                        &base.two_texture_bind_group_layout,
                        &single_uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let tone_mapping_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Tone Mapping Render Pipeline"),
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
                    label: Some("Skybox Render Pipeline Layout"),
                    bind_group_layouts: &[
                        &environment_textures_bind_group_layout,
                        &camera_lights_and_pbr_shader_options_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        let skybox_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Skybox Render Pipeline"),
            layout: Some(&skybox_render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &skybox_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &skybox_shader,
                entry_point: "cubemap_fs_main",
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
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let equirectangular_to_cubemap_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Equirectangular To Cubemap Render Pipeline Layout"),
                    bind_group_layouts: &[
                        &base.single_texture_bind_group_layout,
                        &single_uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        let equirectangular_to_cubemap_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Equirectangular To Cubemap Render Pipeline"),
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

        let diffuse_env_map_color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let diffuse_env_map_gen_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("diffuse env map Gen Pipeline Layout"),
                    bind_group_layouts: &[
                        &single_cube_texture_bind_group_layout,
                        &single_uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let diffuse_env_map_gen_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("diffuse env map Gen Pipeline"),
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
                    label: Some("specular env map Gen Pipeline Layout"),
                    bind_group_layouts: &[
                        &single_cube_texture_bind_group_layout,
                        &two_uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        let specular_env_map_gen_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("specular env map Gen Pipeline"),
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
                    label: Some("Brdf Lut Gen Pipeline Layout"),
                    bind_group_layouts: &[],
                    push_constant_ranges: &[],
                });

        let brdf_lut_gen_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Brdf Lut Gen Pipeline"),
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
                    label: Some("Shadow Map Pipeline Layout"),
                    bind_group_layouts: &[
                        &camera_lights_and_pbr_shader_options_bind_group_layout,
                        &base.bones_and_instances_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let point_shadow_map_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Point Shadow Map Pipeline"),
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
            label: Some("Directional Shadow Map Pipeline"),
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

        let initial_render_scale = INITIAL_RENDER_SCALE;

        let cube_mesh = BasicMesh::new("src/models/cube.obj").await?;

        let skybox_mesh = Self::bind_geometry_buffers_for_basic_mesh_impl(&base.device, &cube_mesh);

        let shading_texture =
            Texture::create_scaled_surface_texture(&base, initial_render_scale, "shading_texture");
        let bloom_pingpong_textures = [
            Texture::create_scaled_surface_texture(&base, initial_render_scale, "bloom_texture_1"),
            Texture::create_scaled_surface_texture(&base, initial_render_scale, "bloom_texture_2"),
        ];
        let tone_mapping_texture = Texture::create_scaled_surface_texture(
            &base,
            initial_render_scale,
            "tone_mapping_texture",
        );
        let shading_texture_bind_group;
        let tone_mapping_texture_bind_group;
        let shading_and_bloom_textures_bind_group;
        let bloom_pingpong_texture_bind_groups;
        {
            let sampler_cache_guard = base.sampler_cache.lock().unwrap();
            shading_texture_bind_group =
                base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &base.single_texture_bind_group_layout,
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
                    label: Some("shading_texture_bind_group"),
                });
            tone_mapping_texture_bind_group =
                base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &base.single_texture_bind_group_layout,
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
                    label: Some("tone_mapping_texture_bind_group"),
                });
            shading_and_bloom_textures_bind_group =
                base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &base.two_texture_bind_group_layout,
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
                    label: Some("surface_blit_textures_bind_group"),
                });

            bloom_pingpong_texture_bind_groups = [
                base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &base.single_texture_bind_group_layout,
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
                    label: Some("bloom_texture_bind_group_1"),
                }),
                base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &base.single_texture_bind_group_layout,
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
                    label: Some("bloom_texture_bind_group_2"),
                }),
            ];
        }

        let bloom_config_buffers = [
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Bloom Config Buffer 0"),
                    contents: bytemuck::cast_slice(&[0f32, 0f32, 0f32]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                }),
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Bloom Config Buffer 1"),
                    contents: bytemuck::cast_slice(&[0f32, 0f32, 0f32]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                }),
        ];

        let bloom_config_bind_groups = [
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &single_uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bloom_config_buffers[0].as_entire_binding(),
                }],
                label: Some("bloom_config_bind_group_0"),
            }),
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &single_uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bloom_config_buffers[1].as_entire_binding(),
                }],
                label: Some("bloom_config_bind_group_1"),
            }),
        ];

        let tone_mapping_config_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Tone Mapping Config Buffer"),
                    contents: bytemuck::cast_slice(&[0f32, 0f32, 0f32, 0f32]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let tone_mapping_config_bind_group =
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &single_uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tone_mapping_config_buffer.as_entire_binding(),
                }],
                label: Some("tone_mapping_config_bind_group"),
            });

        let depth_texture =
            Texture::create_depth_texture(&base, initial_render_scale, "depth_texture");

        let (skybox_background, skybox_hdr_environment) = get_skybox_path();

        let skybox_texture = match skybox_background {
            SkyboxBackground::Equirectangular { image_path } => {
                let er_skybox_texture_bytes = crate::file_loader::read(image_path).await?;
                let er_skybox_texture = Texture::from_encoded_image(
                    &base,
                    &er_skybox_texture_bytes,
                    image_path,
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
                    Some(image_path),
                    &skybox_mesh,
                    &equirectangular_to_cubemap_pipeline,
                    &er_skybox_texture,
                    false, // an artifact occurs between the edges of the texture with mipmaps enabled
                )
            }
            SkyboxBackground::Cube { face_image_paths } => {
                let mut cubemap_skybox_images: Vec<image::DynamicImage> = vec![];
                for path in face_image_paths {
                    cubemap_skybox_images.push(image::load_from_memory(
                        &crate::file_loader::read(path).await?,
                    )?);
                }

                Texture::create_cubemap(
                    &base,
                    CreateCubeMapImagesParam {
                        pos_x: &cubemap_skybox_images[0],
                        neg_x: &cubemap_skybox_images[1],
                        pos_y: &cubemap_skybox_images[2],
                        neg_y: &cubemap_skybox_images[3],
                        pos_z: &cubemap_skybox_images[4],
                        neg_z: &cubemap_skybox_images[5],
                    },
                    Some("cubemap_skybox_texture"),
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    false,
                )
            }
        };

        let er_to_cube_texture;
        let skybox_rad_texture = match skybox_hdr_environment {
            Some(SkyboxHDREnvironment::Equirectangular { image_path }) => {
                let image_bytes = crate::file_loader::read(image_path).await?;
                let skybox_rad_texture_decoder =
                    image::codecs::hdr::HdrDecoder::new(image_bytes.as_slice())?;
                let skybox_rad_texture_dimensions = {
                    let md = skybox_rad_texture_decoder.metadata();
                    (md.width, md.height)
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

                let skybox_rad_texture_er = Texture::from_decoded_image(
                    &base,
                    bytemuck::cast_slice(&skybox_rad_texture_decoded),
                    skybox_rad_texture_dimensions,
                    1,
                    image_path.into(),
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

                er_to_cube_texture = Texture::create_cubemap_from_equirectangular(
                    &base,
                    Some(image_path),
                    &skybox_mesh,
                    &equirectangular_to_cubemap_pipeline,
                    &skybox_rad_texture_er,
                    false,
                );

                &er_to_cube_texture
            }
            None => &skybox_texture,
        };

        let diffuse_env_map = Texture::create_diffuse_env_map(
            &base,
            Some("diffuse env map"),
            &skybox_mesh,
            &diffuse_env_map_gen_pipeline,
            skybox_rad_texture,
            false,
        );

        let specular_env_map = Texture::create_specular_env_map(
            &base,
            Some("specular env map"),
            &skybox_mesh,
            &specular_env_map_gen_pipeline,
            skybox_rad_texture,
        );

        let brdf_lut = Texture::create_brdf_lut(&base, &brdf_lut_gen_pipeline);

        let initial_point_lights_buffer: Vec<u8> = (0..(MAX_LIGHT_COUNT
            * std::mem::size_of::<PointLightUniform>()))
            .map(|_| 0u8)
            .collect();
        let point_lights_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Point Lights Buffer"),
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
                    label: Some("Directional Lights Buffer"),
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
                    label: Some("PBR Shader Options Buffer"),
                    contents: bytemuck::cast_slice(&[initial_pbr_shader_options_buffer]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        // let camera_lights_and_pbr_shader_options_bind_group =
        //     base.device.create_bind_group(&wgpu::BindGroupDescriptor {
        //         layout: &camera_lights_and_pbr_shader_options_bind_group_layout,
        //         entries: &[
        //             wgpu::BindGroupEntry {
        //                 binding: 0,
        //                 resource: point_lights_buffer.as_entire_binding(),
        //             },
        //             wgpu::BindGroupEntry {
        //                 binding: 1,
        //                 resource: directional_lights_buffer.as_entire_binding(),
        //             },
        //             wgpu::BindGroupEntry {
        //                 binding: 2,
        //                 resource: pbr_shader_options_buffer.as_entire_binding(),
        //             },
        //         ],
        //         label: Some("lights_and_pbr_shader_options_bind_group"),
        //     });

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
                layout: &base.bones_and_instances_bind_group_layout,
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
                label: Some("bones_and_pbr_instances_bind_group"),
            });

        let bones_and_unlit_instances_bind_group =
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &base.bones_and_instances_bind_group_layout,
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
                label: Some("bones_and_unlit_instances_bind_group"),
            });

        let bones_and_transparent_instances_bind_group =
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &base.bones_and_instances_bind_group_layout,
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
                label: Some("bones_and_transparent_instances_bind_group"),
            });

        let bones_and_wireframe_instances_bind_group =
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &base.bones_and_instances_bind_group_layout,
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
                label: Some("bones_and_wireframe_instances_bind_group"),
            });

        let point_shadow_map_textures = Texture::create_depth_texture_array(
            &base,
            (
                6 * POINT_LIGHT_SHADOW_MAP_RESOLUTION,
                POINT_LIGHT_SHADOW_MAP_RESOLUTION,
            ),
            Some("point_shadow_map_texture"),
            2, // TODO: this currently puts on hard limit on number of point lights at a time
        );

        let directional_shadow_map_textures = Texture::create_depth_texture_array(
            &base,
            (
                DIRECTIONAL_LIGHT_SHADOW_MAP_RESOLUTION,
                DIRECTIONAL_LIGHT_SHADOW_MAP_RESOLUTION,
            ),
            Some("directional_shadow_map_texture"),
            2, // TODO: this currently puts on hard limit on number of directional lights at a time
        );

        let environment_textures_bind_group = {
            let sampler_cache_guard = base.sampler_cache.lock().unwrap();
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &environment_textures_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&skybox_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            sampler_cache_guard.get_sampler_by_index(skybox_texture.sampler_index),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&diffuse_env_map.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(
                            sampler_cache_guard.get_sampler_by_index(diffuse_env_map.sampler_index),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&specular_env_map.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(
                            sampler_cache_guard
                                .get_sampler_by_index(specular_env_map.sampler_index),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&brdf_lut.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(
                            sampler_cache_guard.get_sampler_by_index(brdf_lut.sampler_index),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(
                            &point_shadow_map_textures.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Sampler(
                            sampler_cache_guard
                                .get_sampler_by_index(point_shadow_map_textures.sampler_index),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(
                            &directional_shadow_map_textures.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::Sampler(
                            sampler_cache_guard.get_sampler_by_index(
                                directional_shadow_map_textures.sampler_index,
                            ),
                        ),
                    },
                ],
                label: Some("skybox_texture_bind_group"),
            })
        };

        let ui_overlay = IkariUiOverlay::new(window, &base.device);

        let mut data = RendererPublicData {
            binded_pbr_meshes: vec![],
            binded_unlit_meshes: vec![],
            binded_transparent_meshes: vec![],
            binded_wireframe_meshes: vec![],
            textures: vec![],

            skybox_mesh,

            tone_mapping_exposure: INITIAL_TONE_MAPPING_EXPOSURE,
            bloom_threshold: INITIAL_BLOOM_THRESHOLD,
            bloom_ramp_size: INITIAL_BLOOM_RAMP_SIZE,
            render_scale: initial_render_scale,
            enable_bloom: false,
            enable_shadows: true,
            enable_wireframe_mode: false,
            draw_node_bounding_spheres: false,
            draw_culling_frustum: false,
            draw_point_light_culling_frusta: false,

            enable_soft_shadows,
            shadow_bias,
            soft_shadow_factor,
            enable_shadow_debug,
            soft_shadow_grid_dims,

            ui_overlay,
        };

        let box_mesh_index = Self::bind_basic_unlit_mesh(&base, &mut data, &cube_mesh)
            .try_into()
            .unwrap();
        let transparent_box_mesh_index =
            Self::bind_basic_transparent_mesh(&base, &mut data, &cube_mesh)
                .try_into()
                .unwrap();

        let sphere_mesh = BasicMesh::new("src/models/sphere.obj").await?;
        let sphere_mesh_index = Self::bind_basic_unlit_mesh(&base, &mut data, &sphere_mesh)
            .try_into()
            .unwrap();
        let transparent_sphere_mesh_index =
            Self::bind_basic_transparent_mesh(&base, &mut data, &sphere_mesh)
                .try_into()
                .unwrap();

        let plane_mesh = BasicMesh::new("src/models/plane.obj").await?;
        let plane_mesh_index = Self::bind_basic_unlit_mesh(&base, &mut data, &plane_mesh)
            .try_into()
            .unwrap();

        // buffer up to 4 frames
        let profiler = wgpu_profiler::GpuProfiler::new(
            4,
            base.queue.get_timestamp_period(),
            base.device.features(),
        );

        let renderer = Self {
            base: Arc::new(base),
            data: Arc::new(Mutex::new(data)),

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

                point_shadow_map_textures,
                directional_shadow_map_textures,
                shading_texture,
                tone_mapping_texture,
                depth_texture,
                bloom_pingpong_textures,
            }),

            profiler: Mutex::new(profiler),

            mesh_pipeline,
            unlit_mesh_pipeline,
            transparent_mesh_pipeline,
            wireframe_pipeline,
            skybox_pipeline,
            tone_mapping_pipeline,
            surface_blit_pipeline,
            point_shadow_map_pipeline,
            directional_shadow_map_pipeline,
            bloom_threshold_pipeline,
            bloom_blur_pipeline,

            box_mesh_index,
            transparent_box_mesh_index,
            sphere_mesh_index,
            transparent_sphere_mesh_index,
            plane_mesh_index,
        };

        Ok(renderer)
    }

    pub fn bind_basic_unlit_mesh(
        base: &BaseRenderer,
        data: &mut RendererPublicData,
        mesh: &BasicMesh,
    ) -> usize {
        let geometry_buffers = Self::bind_geometry_buffers_for_basic_mesh(base, mesh);

        data.binded_unlit_meshes.push(geometry_buffers);
        let unlit_mesh_index = data.binded_unlit_meshes.len() - 1;

        let wireframe_index_buffer = Self::make_wireframe_index_buffer_for_basic_mesh(base, mesh);
        data.binded_wireframe_meshes.push(BindedWireframeMesh {
            source_mesh_type: MeshType::Unlit,
            source_mesh_index: unlit_mesh_index,
            index_buffer: wireframe_index_buffer,
            index_buffer_format: wgpu::IndexFormat::Uint16,
        });

        unlit_mesh_index
    }

    pub fn bind_basic_transparent_mesh(
        base: &BaseRenderer,
        data: &mut RendererPublicData,
        mesh: &BasicMesh,
    ) -> usize {
        let geometry_buffers = Self::bind_geometry_buffers_for_basic_mesh(base, mesh);

        data.binded_transparent_meshes.push(geometry_buffers);
        let transparent_mesh_index = data.binded_transparent_meshes.len() - 1;

        let wireframe_index_buffer = Self::make_wireframe_index_buffer_for_basic_mesh(base, mesh);
        data.binded_wireframe_meshes.push(BindedWireframeMesh {
            source_mesh_type: MeshType::Transparent,
            source_mesh_index: transparent_mesh_index,
            index_buffer: wireframe_index_buffer,
            index_buffer_format: wgpu::IndexFormat::Uint16,
        });

        transparent_mesh_index
    }

    pub fn unbind_transparent_mesh(data: &mut RendererPublicData, mesh_index: usize) {
        let geometry_buffers = &data.binded_transparent_meshes[mesh_index];
        let wireframe_mesh = data
            .binded_wireframe_meshes
            .iter()
            .find(
                |BindedWireframeMesh {
                     source_mesh_type,
                     source_mesh_index,
                     ..
                 }| {
                    *source_mesh_type == MeshType::Transparent && *source_mesh_index == mesh_index
                },
            )
            .unwrap();

        geometry_buffers.vertex_buffer.destroy();
        geometry_buffers.index_buffer.destroy();
        wireframe_mesh.index_buffer.destroy();
    }

    // returns index of mesh in the RenderScene::binded_pbr_meshes list
    pub fn bind_basic_pbr_mesh(
        base: &BaseRenderer,
        data: &mut RendererPublicData,
        mesh: &BasicMesh,
        material: &PbrMaterial,
        dynamic_pbr_params: DynamicPbrParams,
    ) -> Result<usize> {
        let geometry_buffers = Self::bind_geometry_buffers_for_basic_mesh(base, mesh);

        let textures_bind_group = Arc::new(base.make_pbr_textures_bind_group(material, false)?);

        data.binded_pbr_meshes.push(BindedPbrMesh {
            geometry_buffers,
            dynamic_pbr_params,
            textures_bind_group,
            alpha_mode: AlphaMode::Opaque,
            primitive_mode: PrimitiveMode::Triangles,
        });
        let pbr_mesh_index = data.binded_pbr_meshes.len() - 1;

        let wireframe_index_buffer = Self::make_wireframe_index_buffer_for_basic_mesh(base, mesh);
        data.binded_wireframe_meshes.push(BindedWireframeMesh {
            source_mesh_type: MeshType::Pbr,
            source_mesh_index: pbr_mesh_index,
            index_buffer: wireframe_index_buffer,
            index_buffer_format: wgpu::IndexFormat::Uint16,
        });

        Ok(pbr_mesh_index)
    }

    fn bind_geometry_buffers_for_basic_mesh(
        base: &BaseRenderer,
        mesh: &BasicMesh,
    ) -> GeometryBuffers {
        Self::bind_geometry_buffers_for_basic_mesh_impl(&base.device, mesh)
    }

    fn bind_geometry_buffers_for_basic_mesh_impl(
        device: &wgpu::Device,
        mesh: &BasicMesh,
    ) -> GeometryBuffers {
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

        GeometryBuffers {
            vertex_buffer,
            index_buffer,
            index_buffer_format: wgpu::IndexFormat::Uint16,
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

    pub fn set_vsync(&self, vsync: bool) {
        let surface_config = {
            let mut surface_config_guard = self.base.surface_config.lock().unwrap();
            let new_present_mode = if vsync {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            };
            if surface_config_guard.present_mode == new_present_mode {
                return;
            }
            surface_config_guard.present_mode = new_present_mode;
            surface_config_guard.clone()
        };

        self.base
            .surface
            .configure(&self.base.device, &surface_config);
    }

    pub fn resize(&self, new_window_size: winit::dpi::PhysicalSize<u32>, scale_factor: f64) {
        // let mut base_guard = self.base.lock().unwrap();
        let mut data_guard = self.data.lock().unwrap();
        let mut private_data_guard = self.private_data.lock().unwrap();

        data_guard.ui_overlay.resize(new_window_size, scale_factor);

        *self.base.window_size.lock().unwrap() = new_window_size;
        let surface_config = {
            let mut surface_config_guard = self.base.surface_config.lock().unwrap();
            surface_config_guard.width = new_window_size.width;
            surface_config_guard.height = new_window_size.height;
            surface_config_guard.clone()
        };

        self.base
            .surface
            .configure(&self.base.device, &surface_config);

        private_data_guard.shading_texture = Texture::create_scaled_surface_texture(
            &self.base,
            data_guard.render_scale,
            "shading_texture",
        );
        private_data_guard.bloom_pingpong_textures = [
            Texture::create_scaled_surface_texture(
                &self.base,
                data_guard.render_scale,
                "bloom_texture_1",
            ),
            Texture::create_scaled_surface_texture(
                &self.base,
                data_guard.render_scale,
                "bloom_texture_2",
            ),
        ];
        private_data_guard.tone_mapping_texture = Texture::create_scaled_surface_texture(
            &self.base,
            data_guard.render_scale,
            "tone_mapping_texture",
        );
        private_data_guard.depth_texture =
            Texture::create_depth_texture(&self.base, data_guard.render_scale, "depth_texture");

        let device = &self.base.device;
        let single_texture_bind_group_layout = &self.base.single_texture_bind_group_layout;
        let two_texture_bind_group_layout = &self.base.two_texture_bind_group_layout;

        let sampler_cache_guard = self.base.sampler_cache.lock().unwrap();
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
                label: Some("shading_texture_bind_group"),
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
                label: Some("tone_mapping_texture_bind_group"),
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
                label: Some("surface_blit_textures_bind_group"),
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
                label: Some("bloom_texture_bind_group_1"),
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
                label: Some("bloom_texture_bind_group_2"),
            }),
        ];
    }

    #[profiling::function]
    pub fn add_debug_nodes(
        &self,
        data: &mut RendererPublicData,
        private_data: &mut RendererPrivateData,
        game_state: &mut GameState,
        culling_frustum_focal_point: Vec3,
        culling_frustum_forward_vector: Vec3,
    ) {
        let scene = &mut game_state.scene;

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
            Self::unbind_transparent_mesh(data, mesh_index);
        }

        if data.draw_node_bounding_spheres {
            let node_ids: Vec<_> = scene.nodes().map(|node| node.id()).collect();
            for node_id in node_ids {
                if let Some(bounding_sphere) = scene.get_node_bounding_sphere(node_id, data) {
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
                                    .mesh(Some(GameNodeMesh {
                                        mesh_type: GameNodeMeshType::Transparent {
                                            color: Vec4::new(0.0, 1.0, 0.0, 0.1),
                                            premultiplied_alpha: false,
                                        },
                                        mesh_indices: vec![self
                                            .transparent_sphere_mesh_index
                                            .try_into()
                                            .unwrap()],
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

        if data.draw_culling_frustum {
            // shrink the frustum along the view direction for the debug view
            let near_plane_distance = 3.0;
            let far_plane_distance = 500.0;

            let window_size = *self.base.window_size.lock().unwrap();
            let aspect_ratio = window_size.width as f32 / window_size.height as f32;

            let culling_frustum_basic_mesh = Frustum::make_frustum_mesh(
                culling_frustum_focal_point,
                culling_frustum_forward_vector,
                near_plane_distance,
                far_plane_distance,
                deg_to_rad(FOV_Y_DEG),
                aspect_ratio,
            );

            private_data.debug_culling_frustum_mesh_index = Some(
                Self::bind_basic_transparent_mesh(&self.base, data, &culling_frustum_basic_mesh),
            );

            let culling_frustum_mesh = GameNodeMesh {
                mesh_type: GameNodeMeshType::Transparent {
                    color: Vec4::new(1.0, 0.0, 0.0, 0.1),
                    premultiplied_alpha: false,
                },
                mesh_indices: vec![private_data.debug_culling_frustum_mesh_index.unwrap()],
                wireframe: false,
                cullable: false,
            };

            let culling_frustum_mesh_wf = GameNodeMesh {
                wireframe: true,
                ..culling_frustum_mesh.clone()
            };

            private_data.debug_culling_frustum_nodes.push(
                scene
                    .add_node(
                        GameNodeDescBuilder::new()
                            .mesh(Some(culling_frustum_mesh))
                            .build(),
                    )
                    .id(),
            );
            private_data.debug_culling_frustum_nodes.push(
                scene
                    .add_node(
                        GameNodeDescBuilder::new()
                            .mesh(Some(culling_frustum_mesh_wf))
                            .build(),
                    )
                    .id(),
            );
        }

        if data.draw_point_light_culling_frusta {
            for point_light in &game_state.point_lights {
                for controlled_direction in build_cubemap_face_camera_view_directions() {
                    // shrink the frustum along the view direction for the debug view
                    let near_plane_distance = 0.5;
                    let far_plane_distance = 5.0;

                    let culling_frustum_basic_mesh = Frustum::make_frustum_mesh(
                        scene
                            .get_global_transform_for_node(point_light.node_id)
                            .position(),
                        controlled_direction.to_vector(),
                        near_plane_distance,
                        far_plane_distance,
                        deg_to_rad(90.0),
                        1.0,
                    );

                    private_data.debug_culling_frustum_mesh_index =
                        Some(Self::bind_basic_transparent_mesh(
                            &self.base,
                            data,
                            &culling_frustum_basic_mesh,
                        ));

                    let culling_frustum_mesh = GameNodeMesh {
                        mesh_type: GameNodeMeshType::Transparent {
                            color: Vec4::new(1.0, 0.0, 0.0, 0.1),
                            premultiplied_alpha: false,
                        },
                        mesh_indices: vec![private_data.debug_culling_frustum_mesh_index.unwrap()],
                        wireframe: false,
                        cullable: false,
                    };

                    let culling_frustum_mesh_wf = GameNodeMesh {
                        wireframe: true,
                        ..culling_frustum_mesh.clone()
                    };

                    private_data.debug_culling_frustum_nodes.push(
                        scene
                            .add_node(
                                GameNodeDescBuilder::new()
                                    .mesh(Some(culling_frustum_mesh))
                                    .build(),
                            )
                            .id(),
                    );
                    private_data.debug_culling_frustum_nodes.push(
                        scene
                            .add_node(
                                GameNodeDescBuilder::new()
                                    .mesh(Some(culling_frustum_mesh_wf))
                                    .build(),
                            )
                            .id(),
                    );
                }
            }
        }
    }

    pub fn set_culling_frustum_lock(
        &self,
        game_state: &GameState,
        lock_mode: CullingFrustumLockMode,
    ) {
        let mut private_data_guard = self.private_data.lock().unwrap();

        if CullingFrustumLockMode::from(private_data_guard.frustum_culling_lock.clone())
            == lock_mode
        {
            return;
        }

        let window_size = *self.base.window_size.lock().unwrap();
        let aspect_ratio = window_size.width as f32 / window_size.height as f32;

        let position = match private_data_guard.frustum_culling_lock {
            CullingFrustumLock::Full((_, locked_position, _)) => locked_position,
            CullingFrustumLock::FocalPoint(locked_position) => locked_position,
            CullingFrustumLock::None => game_state
                .player_controller
                .position(&game_state.physics_state),
        };

        private_data_guard.frustum_culling_lock = match lock_mode {
            CullingFrustumLockMode::Full => CullingFrustumLock::Full((
                game_state
                    .player_controller
                    .view_frustum_with_position(aspect_ratio, position),
                position,
                game_state.player_controller.view_forward_vector(),
            )),
            CullingFrustumLockMode::FocalPoint => CullingFrustumLock::FocalPoint(position),
            CullingFrustumLockMode::None => CullingFrustumLock::None,
        };
    }

    pub fn render(
        &self,
        game_state: &mut GameState,
        window: &winit::window::Window,
        control_flow: &mut winit::event_loop::ControlFlow,
    ) -> anyhow::Result<()> {
        // let mut base_guard = self.base.lock().unwrap();
        let mut data_guard = self.data.lock().unwrap();
        let mut private_data_guard = self.private_data.lock().unwrap();
        let mut profiler_guard = self.profiler.lock().unwrap();

        self.update_internal(
            &self.base,
            &mut data_guard,
            &mut private_data_guard,
            game_state,
            window,
            control_flow,
        );
        self.render_internal(
            &self.base,
            &mut data_guard,
            &mut private_data_guard,
            &mut profiler_guard,
            game_state,
        )
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
        data: &RendererPublicData,
        game_state: &GameState,
        camera_culling_frustum: &Frustum,
        point_lights_frusta: &Vec<Option<Vec<Frustum>>>,
    ) -> u32 {
        assert!(1 + game_state.directional_lights.len() + point_lights_frusta.len() * 6 <= 32,
            "u32 can only store a max of 5 point lights, might be worth using a larger, might be worth using a larger bitvec or a Vec<bool> or something"
        );

        if node.mesh.is_none() {
            return 0;
        }

        /* bounding boxes will be wrong for skinned meshes so we currently can't cull them */
        if node.skin_index.is_some() || !node.mesh.as_ref().unwrap().cullable {
            return u32::MAX;
        }

        let node_bounding_sphere = game_state
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

        if is_touching_frustum(camera_culling_frustum) {
            culling_mask |= 1u32;
        }

        let mut mask_pos = 1; // start at the second bit, first is reserved for camera

        for _ in &game_state.directional_lights {
            // TODO: add support for frustum culling directional lights shadow map gen?
            culling_mask |= 2u32.pow(mask_pos);

            mask_pos += 1;
        }

        for frusta in point_lights_frusta {
            match frusta {
                Some(frusta) => {
                    for frustum in frusta {
                        let is_touching_frustum = is_touching_frustum(frustum);
                        if is_touching_frustum {
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

    /// Prepare and send all data to gpu so it's ready to render
    #[profiling::function]
    fn update_internal(
        &self,
        base: &BaseRenderer,
        data: &mut RendererPublicData,
        private_data: &mut RendererPrivateData,
        game_state: &mut GameState,
        window: &winit::window::Window,
        control_flow: &mut winit::event_loop::ControlFlow,
    ) {
        data.ui_overlay.update(window, control_flow);

        let window_size = *base.window_size.lock().unwrap();
        let aspect_ratio = window_size.width as f32 / window_size.height as f32;
        let camera_position = game_state
            .player_controller
            .position(&game_state.physics_state);
        let camera_view_direction = game_state.player_controller.view_forward_vector();
        let (culling_frustum, culling_frustum_focal_point, culling_frustum_forward_vector) =
            match private_data.frustum_culling_lock {
                CullingFrustumLock::Full(locked) => locked,
                CullingFrustumLock::FocalPoint(locked_position) => (
                    game_state
                        .player_controller
                        .view_frustum_with_position(aspect_ratio, locked_position),
                    locked_position,
                    camera_view_direction,
                ),
                CullingFrustumLock::None => (
                    game_state
                        .player_controller
                        .view_frustum_with_position(aspect_ratio, camera_position),
                    camera_position,
                    camera_view_direction,
                ),
            };

        self.add_debug_nodes(
            data,
            private_data,
            game_state,
            culling_frustum_focal_point,
            culling_frustum_forward_vector,
        );

        // TODO: compute node bounding spheres here too?
        game_state.scene.recompute_global_node_transforms();

        let limits = &base.limits;
        let queue = &base.queue;
        let device = &base.device;
        let bones_and_instances_bind_group_layout = &base.bones_and_instances_bind_group_layout;

        private_data.all_bone_transforms = get_all_bone_data(
            &game_state.scene,
            limits.min_storage_buffer_offset_alignment,
        );
        let previous_bones_buffer_capacity_bytes = private_data.bones_buffer.capacity_bytes();
        let bones_buffer_changed_capacity = private_data.bones_buffer.write(
            device,
            queue,
            &private_data.all_bone_transforms.buffer,
        );
        if bones_buffer_changed_capacity {
            logger_log(&format!(
                "Resized bones instances buffer capacity from {:?} bytes to {:?}, length={:?}, buffer_length={:?}",
                previous_bones_buffer_capacity_bytes,
                private_data.bones_buffer.capacity_bytes(),
                private_data.bones_buffer.length_bytes(),
                private_data.all_bone_transforms.buffer.len(),
            ));
        }

        let mut pbr_mesh_index_to_gpu_instances: HashMap<usize, Vec<GpuPbrMeshInstance>> =
            HashMap::new();
        let mut unlit_mesh_index_to_gpu_instances: HashMap<usize, Vec<GpuUnlitMeshInstance>> =
            HashMap::new();
        let mut wireframe_mesh_index_to_gpu_instances: HashMap<
            usize,
            Vec<GpuWireframeMeshInstance>,
        > = HashMap::new();
        // no instancing for transparent meshes to allow for sorting
        let mut transparent_meshes: Vec<(usize, GpuTransparentMeshInstance, f32)> = Vec::new();

        let point_lights_frusta: Vec<Option<Vec<Frustum>>> = game_state
            .point_lights
            .iter()
            .map(|point_light| {
                game_state
                    .scene
                    .get_node(point_light.node_id)
                    .map(|point_light_node| {
                        build_cubemap_face_camera_frusta(
                            point_light_node.transform.position(),
                            POINT_LIGHT_SHADOW_MAP_FRUSTUM_NEAR_PLANE,
                            POINT_LIGHT_SHADOW_MAP_FRUSTUM_FAR_PLANE,
                        )
                    })
            })
            .collect();

        for node in game_state.scene.nodes() {
            let transform = Mat4::from(
                game_state
                    .scene
                    .get_global_transform_for_node_opt(node.id()),
            );
            if let Some(GameNodeMesh {
                mesh_indices,
                mesh_type,
                wireframe,
                ..
            }) = &node.mesh
            {
                for mesh_index in mesh_indices.iter().copied() {
                    match (mesh_type, data.enable_wireframe_mode, *wireframe) {
                        (GameNodeMeshType::Pbr { material_override }, false, false) => {
                            let culling_mask = Self::get_node_culling_mask(
                                node,
                                data,
                                game_state,
                                &culling_frustum,
                                &point_lights_frusta,
                            );
                            let gpu_instance = GpuPbrMeshInstance::new(
                                transform,
                                material_override.unwrap_or_else(|| {
                                    data.binded_pbr_meshes[mesh_index].dynamic_pbr_params
                                }),
                                culling_mask,
                            );
                            match pbr_mesh_index_to_gpu_instances.entry(mesh_index) {
                                Entry::Occupied(mut entry) => {
                                    entry.get_mut().push(gpu_instance);
                                }
                                Entry::Vacant(entry) => {
                                    entry.insert(vec![gpu_instance]);
                                }
                            }
                        }
                        (mesh_type, is_wireframe_mode_on, is_node_wireframe) => {
                            let (color, is_transparent) = match mesh_type {
                                GameNodeMeshType::Unlit { color } => {
                                    ([color.x, color.y, color.z, 1.0], false)
                                }
                                GameNodeMeshType::Transparent {
                                    color,
                                    premultiplied_alpha,
                                } => {
                                    (
                                        if *premultiplied_alpha {
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
                                GameNodeMeshType::Pbr { material_override } => {
                                    // fancy logic for picking what the wireframe lines
                                    // color will be by checking the pbr material
                                    let fallback_pbr_params =
                                        data.binded_pbr_meshes[mesh_index].dynamic_pbr_params;
                                    let (base_color_factor, emissive_factor) = material_override
                                        .map(|material_override| {
                                            (
                                                material_override.base_color_factor,
                                                material_override.emissive_factor,
                                            )
                                        })
                                        .unwrap_or((
                                            fallback_pbr_params.base_color_factor,
                                            fallback_pbr_params.emissive_factor,
                                        ));
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

                            if is_wireframe_mode_on || is_node_wireframe {
                                // TODO: this search is slow.. but it's only for wireframe mode, so who cares.. ?
                                let wireframe_mesh_index = data
                                    .binded_wireframe_meshes
                                    .iter()
                                    .enumerate()
                                    .find(|(_, wireframe_mesh)| {
                                        wireframe_mesh.source_mesh_index == mesh_index
                                            && MeshType::from(*mesh_type)
                                                == wireframe_mesh.source_mesh_type
                                    })
                                    .unwrap_or_else(|| panic!("Attempted to draw mesh {:?} in wireframe mode without a corresponding wireframe object",
                                        mesh_index))
                                    .0;
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
                                let (scale, _, translation) =
                                    transform.to_scale_rotation_translation();
                                let aabb_world_space = data.binded_transparent_meshes[mesh_index]
                                    .bounding_box
                                    .scale_translate(scale, translation);
                                let closest_point_to_player =
                                    aabb_world_space.find_closest_surface_point(camera_position);
                                let distance_from_player =
                                    closest_point_to_player.distance(camera_position);
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

        let min_storage_buffer_offset_alignment = base.limits.min_storage_buffer_offset_alignment;

        private_data.all_pbr_instances_culling_masks.clear();

        for instances in pbr_mesh_index_to_gpu_instances.values() {
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

        private_data.all_pbr_instances.replace(
            pbr_mesh_index_to_gpu_instances.into_iter(),
            min_storage_buffer_offset_alignment as usize,
        );

        let previous_pbr_instances_buffer_capacity_bytes =
            private_data.pbr_instances_buffer.capacity_bytes();
        let pbr_instances_buffer_changed_capacity = private_data.pbr_instances_buffer.write(
            device,
            queue,
            private_data.all_pbr_instances.buffer(),
        );

        if pbr_instances_buffer_changed_capacity {
            logger_log(&format!(
                "Resized pbr instances buffer capacity from {:?} bytes to {:?}, length={:?}, buffer_length={:?}",
                previous_pbr_instances_buffer_capacity_bytes,
                private_data.pbr_instances_buffer.capacity_bytes(),
                private_data.pbr_instances_buffer.length_bytes(),
                private_data.all_pbr_instances.buffer().len(),
            ));
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
            logger_log(&format!(
                "Resized unlit instances buffer capacity from {:?} bytes to {:?}, length={:?}, buffer_length={:?}",
                previous_unlit_instances_buffer_capacity_bytes,
                private_data.unlit_instances_buffer.capacity_bytes(),
                private_data.unlit_instances_buffer.length_bytes(),
                private_data.all_unlit_instances.buffer().len(),
            ));
        }

        // draw furthest transparent meshes first
        transparent_meshes.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

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
            logger_log(&format!(
                "Resized transparent instances buffer capacity from {:?} bytes to {:?}, length={:?}, buffer_length={:?}",
                previous_transparent_instances_buffer_capacity_bytes,
                private_data.transparent_instances_buffer.capacity_bytes(),
                private_data.transparent_instances_buffer.length_bytes(),
                private_data.all_transparent_instances.buffer().len(),
            ));
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
            logger_log(&format!(
                "Resized wireframe instances buffer capacity from {:?} bytes to {:?}, length={:?}, buffer_length={:?}",
                previous_wireframe_instances_buffer_capacity_bytes,
                private_data.wireframe_instances_buffer.capacity_bytes(),
                private_data.wireframe_instances_buffer.length_bytes(),
                private_data.all_wireframe_instances.buffer().len(),
            ));
        }

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
                label: Some("bones_and_pbr_instances_bind_group"),
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
                label: Some("bones_and_unlit_instances_bind_group"),
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
                label: Some("bones_and_transparent_instances_bind_group"),
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
                label: Some("bones_and_wireframe_instances_bind_group"),
            });

        let mut all_camera_data: Vec<ShaderCameraData> = vec![];

        // collect all camera data

        // main camera
        let player_transform = game_state
            .scene
            .get_global_transform_for_node(game_state.player_node_id);
        all_camera_data.push(ShaderCameraData::from_mat4(
            player_transform.into(),
            window_size.width as f32 / window_size.height as f32,
            NEAR_PLANE_DISTANCE,
            FAR_PLANE_DISTANCE,
            deg_to_rad(FOV_Y_DEG),
            true,
        ));

        // directional lights
        for directional_light in &game_state.directional_lights {
            all_camera_data.push(build_directional_light_camera_view(
                -directional_light.direction,
                100.0,
                100.0,
                1000.0,
            ));
        }

        // point lights
        for point_light in &game_state.point_lights {
            let light_position = game_state
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
                        base.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Camera Buffer"),
                                contents: &contents,
                                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            }),
                    );
                private_data
                    .camera_lights_and_pbr_shader_options_bind_groups
                    .push(base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout:
                            &private_data.camera_lights_and_pbr_shader_options_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: private_data.camera_buffers[i].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: private_data.point_lights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource:
                                    private_data.directional_lights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource:
                                    private_data.pbr_shader_options_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some("camera_lights_and_pbr_shader_options_bind_group"),
                    }));
            } else {
                queue.write_buffer(&private_data.camera_buffers[i], 0, &contents)
            }
        }
        // private_data.camera_buffers

        let _total_instance_buffer_memory_usage = private_data.pbr_instances_buffer.length_bytes()
            + private_data.unlit_instances_buffer.length_bytes()
            + private_data.transparent_instances_buffer.length_bytes()
            + private_data.wireframe_instances_buffer.length_bytes();
        let _total_index_buffer_memory_usage = data
            .binded_pbr_meshes
            .iter()
            .map(|mesh| mesh.geometry_buffers.index_buffer.length_bytes())
            .chain(
                data.binded_unlit_meshes
                    .iter()
                    .map(|mesh| mesh.index_buffer.length_bytes()),
            )
            .chain(
                data.binded_transparent_meshes
                    .iter()
                    .map(|mesh| mesh.index_buffer.length_bytes()),
            )
            .reduce(|acc, val| acc + val);
        let _total_vertex_buffer_memory_usage = data
            .binded_pbr_meshes
            .iter()
            .map(|mesh| mesh.geometry_buffers.vertex_buffer.length_bytes())
            .chain(
                data.binded_unlit_meshes
                    .iter()
                    .map(|mesh| mesh.vertex_buffer.length_bytes()),
            )
            .chain(
                data.binded_transparent_meshes
                    .iter()
                    .map(|mesh| mesh.vertex_buffer.length_bytes()),
            )
            .reduce(|acc, val| acc + val);

        queue.write_buffer(
            &private_data.point_lights_buffer,
            0,
            bytemuck::cast_slice(&make_point_light_uniform_buffer(game_state)),
        );
        queue.write_buffer(
            &private_data.directional_lights_buffer,
            0,
            bytemuck::cast_slice(&make_directional_light_uniform_buffer(
                &game_state.directional_lights,
            )),
        );
        queue.write_buffer(
            &private_data.tone_mapping_config_buffer,
            0,
            bytemuck::cast_slice(&[
                data.tone_mapping_exposure,
                if self.base.surface_config.lock().unwrap().format.is_srgb() {
                    0f32
                } else {
                    1f32
                },
                0f32,
                0f32,
            ]),
        );
        base.queue.write_buffer(
            &private_data.bloom_config_buffers[0],
            0,
            bytemuck::cast_slice(&[0.0f32, data.bloom_threshold, data.bloom_ramp_size]),
        );
        base.queue.write_buffer(
            &private_data.bloom_config_buffers[1],
            0,
            bytemuck::cast_slice(&[1.0f32, data.bloom_threshold, data.bloom_ramp_size]),
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
    }

    #[profiling::function]
    pub fn render_internal(
        &self,
        base: &BaseRenderer,
        data: &mut RendererPublicData,
        private_data: &mut RendererPrivateData,
        profiler: &mut wgpu_profiler::GpuProfiler,
        game_state: &mut GameState,
    ) -> anyhow::Result<()> {
        let surface_texture = base.surface.get_current_texture()?;
        let surface_texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = base
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        if data.enable_shadows {
            game_state
                .directional_lights
                .iter()
                .enumerate()
                .for_each(|(light_index, light)| {
                    let _view_proj_matrices =
                        build_directional_light_camera_view(-light.direction, 100.0, 100.0, 1000.0);
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
                        label: Some("Directional light shadow map"),
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
                        base,
                        data,
                        private_data,
                        profiler,
                        &mut encoder,
                        &shadow_render_pass_desc,
                        &self.directional_shadow_map_pipeline,
                        &private_data.camera_lights_and_pbr_shader_options_bind_groups
                            [1 + light_index],
                        true,
                        2u32.pow((1 + light_index).try_into().unwrap()),
                        None,
                    );
                });
            (0..game_state.point_lights.len()).for_each(|light_index| {
                if let Some(light_node) = game_state
                    .scene
                    .get_node(game_state.point_lights[light_index].node_id)
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
                            (1 + game_state.directional_lights.len()
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
                            label: Some("Point light shadow map"),
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
                            base,
                            data,
                            private_data,
                            profiler,
                            &mut encoder,
                            &shadow_render_pass_desc,
                            &self.point_shadow_map_pipeline,
                            &private_data.camera_lights_and_pbr_shader_options_bind_groups[1
                                + game_state.directional_lights.len()
                                + light_index * 6
                                + face_index],
                            true,
                            culling_mask,
                            Some(face_index.try_into().unwrap()),
                        );
                    });
                }
            });
        }

        let black = wgpu::Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };

        let shading_render_pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Pbr meshes"),
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
            base,
            data,
            private_data,
            profiler,
            &mut encoder,
            &shading_render_pass_desc,
            &self.mesh_pipeline,
            &private_data.camera_lights_and_pbr_shader_options_bind_groups[0],
            false,
            1, // use main camera culling mask
            None,
        );

        {
            let label = "Unlit and wireframe";
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
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

            wgpu_profiler!(label, profiler, &mut render_pass, &base.device, {
                render_pass.set_pipeline(&self.unlit_mesh_pipeline);

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

                    let geometry_buffers = &data.binded_unlit_meshes[binded_unlit_mesh_index];

                    render_pass.set_bind_group(
                        1,
                        &private_data.bones_and_unlit_instances_bind_group,
                        &[0, instances_buffer_start_index],
                    );
                    render_pass
                        .set_vertex_buffer(0, geometry_buffers.vertex_buffer.src().slice(..));
                    render_pass.set_index_buffer(
                        geometry_buffers.index_buffer.src().slice(..),
                        geometry_buffers.index_buffer_format,
                    );
                    render_pass.draw_indexed(
                        0..geometry_buffers.index_buffer.length() as u32,
                        0,
                        0..instance_count as u32,
                    );
                }

                render_pass.set_pipeline(&self.wireframe_pipeline);

                for wireframe_instance_chunk in private_data.all_wireframe_instances.chunks() {
                    let binded_wireframe_mesh_index = wireframe_instance_chunk.id;
                    let instances_buffer_start_index = wireframe_instance_chunk.start_index as u32;
                    let instance_count = (wireframe_instance_chunk.end_index
                        - wireframe_instance_chunk.start_index)
                        / private_data.all_wireframe_instances.stride();

                    let BindedWireframeMesh {
                        source_mesh_type,
                        source_mesh_index,
                        index_buffer,
                        index_buffer_format,
                        ..
                    } = &data.binded_wireframe_meshes[binded_wireframe_mesh_index];

                    let (vertex_buffer, bone_transforms_buffer_start_index) = match source_mesh_type
                    {
                        MeshType::Pbr => {
                            let bone_transforms_buffer_start_index = private_data
                                .all_bone_transforms
                                .animated_bone_transforms
                                .iter()
                                .find(|bone_slice| {
                                    bone_slice.binded_pbr_mesh_index == *source_mesh_index
                                })
                                .map(|bone_slice| bone_slice.start_index.try_into().unwrap())
                                .unwrap_or(0);
                            (
                                &data.binded_pbr_meshes[*source_mesh_index]
                                    .geometry_buffers
                                    .vertex_buffer,
                                bone_transforms_buffer_start_index,
                            )
                        }
                        MeshType::Unlit => (
                            &data.binded_unlit_meshes[*source_mesh_index].vertex_buffer,
                            0,
                        ),
                        MeshType::Transparent => (
                            &data.binded_transparent_meshes[*source_mesh_index].vertex_buffer,
                            0,
                        ),
                    };
                    render_pass.set_bind_group(
                        1,
                        &private_data.bones_and_wireframe_instances_bind_group,
                        &[
                            bone_transforms_buffer_start_index,
                            instances_buffer_start_index,
                        ],
                    );
                    render_pass.set_vertex_buffer(0, vertex_buffer.src().slice(..));
                    render_pass
                        .set_index_buffer(index_buffer.src().slice(..), *index_buffer_format);
                    render_pass.draw_indexed(
                        0..index_buffer.length() as u32,
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
                    label: Some(label),
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

                wgpu_profiler!(label, profiler, &mut render_pass, &base.device, {
                    render_pass.set_pipeline(&self.bloom_threshold_pipeline);
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
                        label: Some(label),
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

                    wgpu_profiler!(label, profiler, &mut render_pass, &base.device, {
                        render_pass.set_pipeline(&self.bloom_blur_pipeline);
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
            wgpu_profiler!(label, profiler, &mut render_pass, &base.device, {});
            private_data.bloom_threshold_cleared = true;
        }

        {
            let label = "Skybox";
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
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

            wgpu_profiler!(label, profiler, &mut render_pass, &base.device, {
                render_pass.set_pipeline(&self.skybox_pipeline);
                render_pass.set_bind_group(0, &private_data.environment_textures_bind_group, &[]);
                render_pass.set_bind_group(
                    1,
                    &private_data.camera_lights_and_pbr_shader_options_bind_groups[private_data
                        .camera_lights_and_pbr_shader_options_bind_groups
                        .len()
                        - 1],
                    &[],
                );
                render_pass.set_vertex_buffer(0, data.skybox_mesh.vertex_buffer.src().slice(..));
                render_pass.set_index_buffer(
                    data.skybox_mesh.index_buffer.src().slice(..),
                    data.skybox_mesh.index_buffer_format,
                );
                render_pass.draw_indexed(
                    0..(data.skybox_mesh.index_buffer.length() as u32),
                    0,
                    0..1,
                );
            });
        }
        {
            let label = "Tone mapping";
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
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
            wgpu_profiler!(label, profiler, &mut render_pass, &base.device, {
                render_pass.set_pipeline(&self.tone_mapping_pipeline);
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
                label: Some(label),
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

            wgpu_profiler!(label, profiler, &mut render_pass, &base.device, {
                render_pass.set_pipeline(&self.transparent_mesh_pipeline);

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

                    let geometry_buffers =
                        &data.binded_transparent_meshes[binded_transparent_mesh_index];

                    render_pass.set_bind_group(
                        1,
                        &private_data.bones_and_transparent_instances_bind_group,
                        &[0, instances_buffer_start_index],
                    );
                    render_pass
                        .set_vertex_buffer(0, geometry_buffers.vertex_buffer.src().slice(..));
                    render_pass.set_index_buffer(
                        geometry_buffers.index_buffer.src().slice(..),
                        geometry_buffers.index_buffer_format,
                    );
                    render_pass.draw_indexed(
                        0..geometry_buffers.index_buffer.length() as u32,
                        0,
                        0..instance_count as u32,
                    );
                }
            });
        }

        // TODO: pass a separate encoder to the ui overlay so it can be profiled
        data.ui_overlay.render(
            &base.device,
            &mut encoder,
            &private_data.tone_mapping_texture.view,
        );

        {
            let label = "Surface blit";
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
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

            wgpu_profiler!(label, profiler, &mut render_pass, &base.device, {
                {
                    render_pass.set_pipeline(&self.surface_blit_pipeline);
                    render_pass.set_bind_group(
                        0,
                        &private_data.tone_mapping_texture_bind_group,
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

        profiler.resolve_queries(&mut encoder);

        base.queue.submit(std::iter::once(encoder.finish()));

        surface_texture.present();

        profiler.end_frame().map_err(|_| anyhow::anyhow!(
            "Something went wrong with wgpu_profiler. Does the crate still not report error details?"
        ))?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn render_pbr_meshes<'a>(
        base: &BaseRenderer,
        data: &RendererPublicData,
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
                    if private_data.all_pbr_instances_culling_masks[pbr_instance_chunk_index]
                        & culling_mask
                        == 0
                    {
                        continue;
                    }

                    let binded_pbr_mesh_index = pbr_instance_chunk.id;
                    let bone_transforms_buffer_start_index = private_data
                        .all_bone_transforms
                        .animated_bone_transforms
                        .iter()
                        .find(|bone_slice| {
                            bone_slice.binded_pbr_mesh_index == binded_pbr_mesh_index
                        })
                        .map(|bone_slice| bone_slice.start_index.try_into().unwrap())
                        .unwrap_or(0);
                    let instances_buffer_start_index = pbr_instance_chunk.start_index as u32;
                    let instance_count = (pbr_instance_chunk.end_index
                        - pbr_instance_chunk.start_index)
                        / private_data.all_pbr_instances.stride();

                    let BindedPbrMesh {
                        geometry_buffers,
                        textures_bind_group,
                        ..
                    } = &data.binded_pbr_meshes[binded_pbr_mesh_index];

                    render_pass.set_bind_group(
                        if is_shadow { 1 } else { 2 },
                        &private_data.bones_and_pbr_instances_bind_group,
                        &[
                            bone_transforms_buffer_start_index,
                            instances_buffer_start_index,
                        ],
                    );
                    if !is_shadow {
                        render_pass.set_bind_group(3, textures_bind_group, &[]);
                    }
                    render_pass
                        .set_vertex_buffer(0, geometry_buffers.vertex_buffer.src().slice(..));
                    render_pass.set_index_buffer(
                        geometry_buffers.index_buffer.src().slice(..),
                        geometry_buffers.index_buffer_format,
                    );
                    render_pass.draw_indexed(
                        0..geometry_buffers.index_buffer.length() as u32,
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
