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
use std::fs::File;
use std::io::BufReader;
use std::num::NonZeroU32;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use glam::f32::{Mat4, Vec3};
use image::Pixel;
use wgpu::util::DeviceExt;
use wgpu::InstanceDescriptor;
use winit::window::Window;

pub const MAX_LIGHT_COUNT: usize = 32;
pub const NEAR_PLANE_DISTANCE: f32 = 0.001;
pub const FAR_PLANE_DISTANCE: f32 = 100000.0;
pub const FOV_Y_DEG: f32 = 45.0;
pub const DEFAULT_WIREFRAME_COLOR: [f32; 4] = [0.0, 1.0, 1.0, 1.0];

/* #[repr(C)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct GpuMatrix4(pub Mat4);

unsafe impl bytemuck::Pod for GpuMatrix4 {}
unsafe impl bytemuck::Zeroable for GpuMatrix4 {} */

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
            color: [0.0, 0.0, 0.0, 1.0],
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
            color: [0.0, 0.0, 0.0, 1.0],
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

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UnlitColorUniform {
    color: [f32; 4],
}

impl From<Vec3> for UnlitColorUniform {
    fn from(color: Vec3) -> Self {
        Self {
            color: [color.x, color.y, color.z, 1.0],
        }
    }
}

pub enum SkyboxBackground<'a> {
    Cube { face_image_paths: [&'a str; 6] },
    Equirectangular { image_path: &'a str },
}

pub enum SkyboxHDREnvironment<'a> {
    Equirectangular { image_path: &'a str },
}

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

#[derive(Debug, PartialEq, Eq)]
pub enum MeshType {
    Pbr,
    Unlit,
}

impl From<GameNodeMeshType> for MeshType {
    fn from(game_node_mesh_type: GameNodeMeshType) -> Self {
        match game_node_mesh_type {
            GameNodeMeshType::Pbr { .. } => MeshType::Pbr,
            GameNodeMeshType::Unlit { .. } => MeshType::Unlit,
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

// TODO: rename to BaseRenderer
pub struct BaseRendererState {
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

impl BaseRendererState {
    pub async fn new(
        window: &winit::window::Window,
        backends: wgpu::Backends,
        present_mode: wgpu::PresentMode,
    ) -> Self {
        let window_size = window.inner_size();

        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends,
            ..Default::default()
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

        let mut features = wgpu::Features::empty();
        features |= wgpu::Features::TEXTURE_COMPRESSION_BC;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features,
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let mut surface_config = surface
            .get_default_config(&adapter, window_size.width, window_size.height)
            .expect("Window surface is incompatible with the graphics adapter");
        surface_config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        surface_config.format = wgpu::TextureFormat::Bgra8UnormSrgb;
        surface_config.alpha_mode = wgpu::CompositeAlphaMode::Auto;
        surface_config.present_mode = present_mode;

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

// TODO: rename to RendererPrivateData
pub struct RendererStatePrivateData {
    // cpu
    all_bone_transforms: AllBoneTransforms,
    all_pbr_instances: ChunkedBuffer<GpuPbrMeshInstance>,
    all_unlit_instances: ChunkedBuffer<GpuUnlitMeshInstance>,
    all_wireframe_instances: ChunkedBuffer<GpuWireframeMeshInstance>,
    debug_nodes: Vec<GameNodeId>,

    bloom_threshold_cleared: bool,

    // gpu
    camera_and_lights_bind_group: wgpu::BindGroup,
    bones_and_pbr_instances_bind_group: wgpu::BindGroup,
    bones_and_unlit_instances_bind_group: wgpu::BindGroup,
    bones_and_wireframe_instances_bind_group: wgpu::BindGroup,
    bloom_config_bind_group: wgpu::BindGroup,
    tone_mapping_config_bind_group: wgpu::BindGroup,

    environment_textures_bind_group: wgpu::BindGroup,
    shading_and_bloom_textures_bind_group: wgpu::BindGroup,
    tone_mapping_texture_bind_group: wgpu::BindGroup,
    shading_texture_bind_group: wgpu::BindGroup,
    bloom_pingpong_texture_bind_groups: [wgpu::BindGroup; 2],

    camera_buffer: wgpu::Buffer,
    point_lights_buffer: wgpu::Buffer,
    directional_lights_buffer: wgpu::Buffer,
    bloom_config_buffer: wgpu::Buffer,
    tone_mapping_config_buffer: wgpu::Buffer,
    bones_buffer: GpuBuffer,
    pbr_instances_buffer: GpuBuffer,
    unlit_instances_buffer: GpuBuffer,
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
    pub binded_wireframe_meshes: Vec<BindedWireframeMesh>,
    pub textures: Vec<Texture>,
}

// TODO: rename to RendererPublicData
pub struct RendererStatePublicData {
    pub binded_pbr_meshes: Vec<BindedPbrMesh>,
    pub binded_unlit_meshes: Vec<BindedUnlitMesh>,
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

    pub ui_overlay: UiOverlay,
}

// TODO: rename to Renderer
pub struct RendererState {
    pub base: Arc<BaseRendererState>,
    pub data: Arc<Mutex<RendererStatePublicData>>,

    // TODO: does this need a mutex?
    private_data: Mutex<RendererStatePrivateData>,

    mesh_pipeline: wgpu::RenderPipeline,
    unlit_mesh_pipeline: wgpu::RenderPipeline,
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
    sphere_mesh_index: i32,
    #[allow(dead_code)]
    plane_mesh_index: i32,
}

impl RendererState {
    pub async fn new(base: BaseRendererState, window: &Window) -> Result<Self> {
        logger_log("Controls:");
        vec![
            "Move Around:             WASD, Space Bar, Ctrl",
            "Look Around:             Mouse",
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
            "Draw Scene SP Tree:      K",
            "Exit:                    Escape",
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
                    std::fs::read_to_string("./src/shaders/unlit_mesh.wgsl")?.into(),
                ),
            });

        let blit_shader = base
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Blit Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    std::fs::read_to_string("./src/shaders/blit.wgsl")?.into(),
                ),
            });

        let textured_mesh_shader = base
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Textured Mesh Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    std::fs::read_to_string("./src/shaders/textured_mesh.wgsl")?.into(),
                ),
            });

        let skybox_shader = base
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Skybox Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    std::fs::read_to_string("./src/shaders/skybox.wgsl")?.into(),
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
                                view_dimension: wgpu::TextureViewDimension::CubeArray,
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
        let camera_and_lights_bind_group_layout =
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
                    ],
                    label: Some("camera_and_lights_uniform_bind_group_layout"),
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
                        &camera_and_lights_bind_group_layout,
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
                        &camera_and_lights_bind_group_layout,
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

        let bloom_threshold_pipeline_layout =
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
            layout: Some(&bloom_threshold_pipeline_layout),
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

        let bloom_blur_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        &base.single_texture_bind_group_layout,
                        &single_uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let bloom_blur_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Blur Pipeline"),
            layout: Some(&bloom_blur_pipeline_layout),
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
                    bind_group_layouts: &[&base.single_texture_bind_group_layout],
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
                entry_point: "fs_main",
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
                        &camera_and_lights_bind_group_layout,
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
                        &camera_and_lights_bind_group_layout,
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

        let cube_mesh = BasicMesh::new("./src/models/cube.obj")?;

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
        let sampler_cache_guard = base.sampler_cache.lock().unwrap();
        let shading_texture_bind_group =
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
                            sampler_cache_guard.get_sampler_by_index(shading_texture.sampler_index),
                        ),
                    },
                ],
                label: Some("shading_texture_bind_group"),
            });
        let tone_mapping_texture_bind_group =
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &base.single_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&tone_mapping_texture.view),
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
        let shading_and_bloom_textures_bind_group =
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
                            sampler_cache_guard.get_sampler_by_index(shading_texture.sampler_index),
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

        let bloom_pingpong_texture_bind_groups = [
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
        drop(sampler_cache_guard);

        let bloom_config_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Bloom Config Buffer"),
                    contents: bytemuck::cast_slice(&[0f32, 0f32, 0f32]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let bloom_config_bind_group = base.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &single_uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bloom_config_buffer.as_entire_binding(),
            }],
            label: Some("bloom_config_bind_group"),
        });

        let tone_mapping_config_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Tone Mapping Config Buffer"),
                    contents: bytemuck::cast_slice(&[0f32]),
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
                let er_skybox_texture_bytes = std::fs::read(image_path)?;
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
                let cubemap_skybox_images = face_image_paths
                    .iter()
                    .map(|path| image::load_from_memory(&std::fs::read(path)?))
                    .collect::<Result<Vec<_>, _>>()?;

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
                let skybox_rad_texture_decoder = {
                    let reader = BufReader::new(File::open(image_path)?);
                    image::codecs::hdr::HdrDecoder::new(reader)?
                };
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

        let camera_buffer = base
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: &vec![0u8; std::mem::size_of::<CameraUniform>()],
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

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

        let camera_and_lights_bind_group =
            base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &camera_and_lights_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: point_lights_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: directional_lights_buffer.as_entire_binding(),
                    },
                ],
                label: Some("camera_and_lights_bind_group"),
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

        let point_shadow_map_textures = Texture::create_cube_depth_texture_array(
            &base,
            1024,
            Some("point_shadow_map_texture"),
            2, // TODO: this currently puts on hard limit on number of point lights at a time
        );

        let directional_shadow_map_textures = Texture::create_depth_texture_array(
            &base,
            2048,
            Some("directional_shadow_map_texture"),
            2, // TODO: this currently puts on hard limit on number of directional lights at a time
        );

        let sampler_cache_guard = base.sampler_cache.lock().unwrap();
        let environment_textures_bind_group =
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
            });
        drop(sampler_cache_guard);

        let ui_overlay = UiOverlay::new(
            window,
            &base.device,
            base.surface_config.lock().unwrap().format,
        );

        let mut data = RendererStatePublicData {
            binded_pbr_meshes: vec![],
            binded_unlit_meshes: vec![],
            binded_wireframe_meshes: vec![],
            textures: vec![],

            skybox_mesh,

            tone_mapping_exposure: INITIAL_TONE_MAPPING_EXPOSURE,
            bloom_threshold: INITIAL_BLOOM_THRESHOLD,
            bloom_ramp_size: INITIAL_BLOOM_RAMP_SIZE,
            render_scale: initial_render_scale,
            enable_bloom: true,
            enable_shadows: true,
            enable_wireframe_mode: false,
            draw_node_bounding_spheres: false,

            ui_overlay,
        };

        let box_mesh_index = Self::bind_basic_unlit_mesh(&base, &mut data, &cube_mesh)
            .try_into()
            .unwrap();

        let sphere_mesh = BasicMesh::new("./src/models/sphere.obj").unwrap();
        let sphere_mesh_index = Self::bind_basic_unlit_mesh(&base, &mut data, &sphere_mesh)
            .try_into()
            .unwrap();

        let plane_mesh = BasicMesh::new("./src/models/plane.obj").unwrap();
        let plane_mesh_index = Self::bind_basic_unlit_mesh(&base, &mut data, &plane_mesh)
            .try_into()
            .unwrap();

        let renderer_state = Self {
            base: Arc::new(base),
            data: Arc::new(Mutex::new(data)),

            private_data: Mutex::new(RendererStatePrivateData {
                all_bone_transforms: AllBoneTransforms {
                    buffer: vec![],
                    animated_bone_transforms: vec![],
                    identity_slice: (0, 0),
                },
                all_pbr_instances: ChunkedBuffer::empty(),
                all_unlit_instances: ChunkedBuffer::empty(),
                all_wireframe_instances: ChunkedBuffer::empty(),
                debug_nodes: vec![],

                bloom_threshold_cleared: true,

                camera_and_lights_bind_group,
                bones_and_pbr_instances_bind_group,
                bones_and_unlit_instances_bind_group,
                bones_and_wireframe_instances_bind_group,
                bloom_config_bind_group,
                tone_mapping_config_bind_group,

                environment_textures_bind_group,
                shading_and_bloom_textures_bind_group,
                tone_mapping_texture_bind_group,
                shading_texture_bind_group,
                bloom_pingpong_texture_bind_groups,

                camera_buffer,
                point_lights_buffer,
                directional_lights_buffer,
                bloom_config_buffer,
                tone_mapping_config_buffer,
                bones_buffer,
                pbr_instances_buffer,
                unlit_instances_buffer,
                wireframe_instances_buffer,

                point_shadow_map_textures,
                directional_shadow_map_textures,
                shading_texture,
                tone_mapping_texture,
                depth_texture,
                bloom_pingpong_textures,
            }),

            mesh_pipeline,
            unlit_mesh_pipeline,
            wireframe_pipeline,
            skybox_pipeline,
            tone_mapping_pipeline,
            surface_blit_pipeline,
            point_shadow_map_pipeline,
            directional_shadow_map_pipeline,
            bloom_threshold_pipeline,
            bloom_blur_pipeline,

            box_mesh_index,
            sphere_mesh_index,
            plane_mesh_index,
        };

        Ok(renderer_state)
    }

    pub fn bind_basic_unlit_mesh(
        base: &BaseRendererState,
        data: &mut RendererStatePublicData,
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

    // returns index of mesh in the RenderScene::binded_pbr_meshes list
    pub fn bind_basic_pbr_mesh(
        base: &BaseRendererState,
        data: &mut RendererStatePublicData,
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
        base: &BaseRendererState,
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
        base: &BaseRendererState,
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

    pub fn resize(&self, new_window_size: winit::dpi::PhysicalSize<u32>, scale_factor: f64) {
        // let mut base_guard = self.base.lock().unwrap();
        let mut data_guard = self.data.lock().unwrap();
        let mut private_data_guard = self.private_data.lock().unwrap();

        data_guard.ui_overlay.resize(new_window_size, scale_factor);

        *self.base.window_size.lock().unwrap() = new_window_size;
        {
            let mut surface_config_guard = self.base.surface_config.lock().unwrap();
            surface_config_guard.width = new_window_size.width;
            surface_config_guard.height = new_window_size.height;

            // TODO: probably don't need to hold onto surface config mutex while runnig this.
            self.base
                .surface
                .configure(&self.base.device, &surface_config_guard);
        }

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

    pub fn clear_debug_nodes(private_data: &mut RendererStatePrivateData, scene: &mut Scene) {
        for node_id in private_data.debug_nodes.iter().copied() {
            scene.remove_node(node_id);
        }
        private_data.debug_nodes = Vec::new();
    }

    pub fn add_debug_nodes(
        data: &mut RendererStatePublicData,
        private_data: &mut RendererStatePrivateData,
        game_state: &mut GameState,
        sphere_mesh_index: i32,
    ) {
        if !data.draw_node_bounding_spheres {
            return;
        }

        let scene = &mut game_state.scene;

        if data.draw_node_bounding_spheres {
            // super slow, but who cares for now
            let nodes: Vec<_> = scene.nodes().cloned().collect();
            for node in nodes {
                if let Some(bounding_sphere) = scene.get_node_bounding_sphere(node.id(), data) {
                    private_data.debug_nodes.push(
                        scene
                            .add_node(
                                GameNodeDescBuilder::new()
                                    .transform(
                                        TransformBuilder::new()
                                            .scale(
                                                bounding_sphere.radius * Vec3::new(1.0, 1.0, 1.0),
                                            )
                                            .position(bounding_sphere.origin)
                                            .build(),
                                    )
                                    .mesh(Some(GameNodeMesh {
                                        mesh_type: GameNodeMeshType::Unlit {
                                            color: Vec3::new(0.0, 1.0, 0.0),
                                        },
                                        mesh_indices: vec![sphere_mesh_index.try_into().unwrap()],
                                        wireframe: true,
                                        cullable: false,
                                    }))
                                    .build(),
                            )
                            .id(),
                    );
                }
            }
        }

        // scene.recompute_node_transforms();
        scene.recompute_global_node_transforms();
    }

    pub fn render(
        &mut self,
        game_state: &mut GameState,
        window: &winit::window::Window,
    ) -> Result<(), wgpu::SurfaceError> {
        // let mut base_guard = self.base.lock().unwrap();
        let mut data_guard = self.data.lock().unwrap();
        let mut private_data_guard = self.private_data.lock().unwrap();
        // TODO: this holds onto these mutexes for quite a while, could it make sense to make them hold for shorter periods?
        self.update_internal(
            &self.base,
            &mut data_guard,
            &mut private_data_guard,
            game_state,
            window,
        );
        self.render_internal(
            &self.base,
            &mut data_guard,
            &mut private_data_guard,
            game_state,
        )
    }

    /// Prepare and send all data to gpu so it's ready to render
    #[profiling::function]
    fn update_internal(
        &self,
        base: &BaseRendererState,
        data: &mut RendererStatePublicData,
        private_data: &mut RendererStatePrivateData,
        game_state: &mut GameState,
        window: &winit::window::Window,
    ) {
        data.ui_overlay.update(window);

        Self::clear_debug_nodes(private_data, &mut game_state.scene);

        // game_state.scene.recompute_node_transforms();
        game_state.scene.recompute_global_node_transforms();

        let window_size = *base.window_size.lock().unwrap();
        let camera_frustum = game_state.player_controller.frustum(
            &game_state.physics_state,
            window_size.width as f32 / window_size.height as f32,
        );

        // private_data: &mut RendererStatePrivateData,
        // game_state: &mut GameState,
        // draw_node_bounding_spheres: bool,
        // sphere_mesh_index: i32,
        Self::add_debug_nodes(data, private_data, game_state, self.sphere_mesh_index);

        let mut frustum_culled_node_list: Vec<GameNodeId> = Vec::new();
        for node in game_state.scene.nodes() {
            if node.mesh.is_none() {
                continue;
            }
            /* bounding boxes will be wrong for skinned meshes so we currently can't cull them */
            if node.skin_index.is_some() || !node.mesh.as_ref().unwrap().cullable {
                frustum_culled_node_list.push(node.id());
                continue;
            }
            if let Some(node_bounding_sphere) = game_state
                .scene
                .get_node_bounding_sphere_opt(node.id(), data)
            {
                match camera_frustum.aabb_intersection_test(node_bounding_sphere.aabb()) {
                    IntersectionResult::FullyContained
                    | IntersectionResult::PartiallyIntersecting => {
                        frustum_culled_node_list.push(node.id());
                    }
                    _ => {}
                }
            }
        }

        let scene = &mut game_state.scene;
        let limits = &base.limits;
        let queue = &base.queue;
        let device = &base.device;
        let bones_and_instances_bind_group_layout = &base.bones_and_instances_bind_group_layout;

        private_data.all_bone_transforms =
            get_all_bone_data(scene, limits.min_storage_buffer_offset_alignment);
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

        for node_id in frustum_culled_node_list {
            let node = scene.get_node_unchecked(node_id);
            let transform = Mat4::from(scene.get_global_transform_for_node_opt(node.id()));
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
                            let gpu_instance = GpuPbrMeshInstance::new(
                                transform,
                                material_override.unwrap_or_else(|| {
                                    data.binded_pbr_meshes[mesh_index].dynamic_pbr_params
                                }),
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
                            let color = match mesh_type {
                                GameNodeMeshType::Unlit { color } => {
                                    [color.x, color.y, color.z, 1.0]
                                }
                                GameNodeMeshType::Pbr { material_override } => {
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
                                    }
                                }
                            };

                            if is_wireframe_mode_on || is_node_wireframe {
                                let wireframe_mesh_index = data
                                    .binded_wireframe_meshes
                                    .iter()
                                    .enumerate()
                                    .find(|(_, wireframe_mesh)| {
                                        wireframe_mesh.source_mesh_index == mesh_index
                                            && MeshType::from(*mesh_type)
                                                == wireframe_mesh.source_mesh_type
                                    })
                                    .unwrap()
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

        private_data.all_pbr_instances = ChunkedBuffer::new(
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

        private_data.all_unlit_instances = ChunkedBuffer::new(
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

        private_data.all_wireframe_instances = ChunkedBuffer::new(
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

        let _total_instance_buffer_memory_usage = private_data.pbr_instances_buffer.length_bytes()
            + private_data.unlit_instances_buffer.length_bytes()
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
            bytemuck::cast_slice(&[data.tone_mapping_exposure]),
        );
    }

    #[profiling::function]
    pub fn render_internal(
        &self,
        base: &BaseRendererState,
        data: &mut RendererStatePublicData,
        private_data: &mut RendererStatePrivateData,
        game_state: &mut GameState,
    ) -> Result<(), wgpu::SurfaceError> {
        let surface_texture = base.surface.get_current_texture()?;
        let surface_texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        if data.enable_shadows {
            game_state
                .directional_lights
                .iter()
                .enumerate()
                .for_each(|(light_index, light)| {
                    let view_proj_matrices =
                        build_directional_light_camera_view(-light.direction, 100.0, 100.0, 1000.0);
                    let texture_view = private_data
                        .directional_shadow_map_textures
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            base_array_layer: light_index.try_into().unwrap(),
                            array_layer_count: NonZeroU32::new(1),
                            ..Default::default()
                        });
                    let shadow_render_pass_desc = wgpu::RenderPassDescriptor {
                        label: Some("Shadow Render Pass"),
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
                    base.queue.write_buffer(
                        &private_data.camera_buffer,
                        0,
                        bytemuck::cast_slice(&[CameraUniform::from(view_proj_matrices)]),
                    );
                    Self::render_pbr_meshes(
                        base,
                        data,
                        private_data,
                        &shadow_render_pass_desc,
                        &self.directional_shadow_map_pipeline,
                        true,
                    );
                });
            (0..game_state.point_lights.len()).for_each(|light_index| {
                if let Some(light_node) = game_state
                    .scene
                    .get_node(game_state.point_lights[light_index].node_id)
                {
                    build_cubemap_face_camera_views(
                        light_node.transform.position(),
                        0.1,
                        1000.0,
                        false,
                    )
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, view_proj_matrices)| {
                        (
                            view_proj_matrices,
                            private_data.point_shadow_map_textures.texture.create_view(
                                &wgpu::TextureViewDescriptor {
                                    dimension: Some(wgpu::TextureViewDimension::D2),
                                    base_array_layer: (6 * light_index + i).try_into().unwrap(),
                                    array_layer_count: NonZeroU32::new(1),
                                    ..Default::default()
                                },
                            ),
                        )
                    })
                    .for_each(
                        |(face_view_proj_matrices, face_texture_view)| {
                            let shadow_render_pass_desc = wgpu::RenderPassDescriptor {
                                label: Some("Shadow Render Pass"),
                                color_attachments: &[],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachment {
                                        view: &face_texture_view,
                                        depth_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(1.0),
                                            store: true,
                                        }),
                                        stencil_ops: None,
                                    },
                                ),
                            };
                            base.queue.write_buffer(
                                &private_data.camera_buffer,
                                0,
                                bytemuck::cast_slice(&[CameraUniform::from(
                                    face_view_proj_matrices,
                                )]),
                            );
                            Self::render_pbr_meshes(
                                base,
                                data,
                                private_data,
                                &shadow_render_pass_desc,
                                &self.point_shadow_map_pipeline,
                                true,
                            );
                        },
                    );
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
            label: Some("Shading Render Pass"),
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

        let player_transform = game_state
            .scene
            .get_global_transform_for_node(game_state.player_node_id);
        let window_size = *base.window_size.lock().unwrap();
        base.queue.write_buffer(
            &private_data.camera_buffer,
            0,
            bytemuck::cast_slice(&[CameraUniform::from(ShaderCameraView::from_mat4(
                player_transform.into(),
                window_size.width as f32 / window_size.height as f32,
                NEAR_PLANE_DISTANCE,
                FAR_PLANE_DISTANCE,
                deg_to_rad(FOV_Y_DEG),
                true,
            ))]),
        );

        Self::render_pbr_meshes(
            base,
            data,
            private_data,
            &shading_render_pass_desc,
            &self.mesh_pipeline,
            false,
        );

        let mut unlit_and_wireframe_encoder =
            base.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Unlit And Wireframe Encoder"),
                });

        {
            let mut render_pass =
                unlit_and_wireframe_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Unlit And Wireframe Render Pass"),
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

            render_pass.set_pipeline(&self.unlit_mesh_pipeline);
            render_pass.set_bind_group(0, &private_data.camera_and_lights_bind_group, &[]);
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
                render_pass.set_vertex_buffer(0, geometry_buffers.vertex_buffer.src().slice(..));
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

                let (vertex_buffer, bone_transforms_buffer_start_index) = match source_mesh_type {
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
                render_pass.set_index_buffer(index_buffer.src().slice(..), *index_buffer_format);
                render_pass.draw_indexed(
                    0..index_buffer.length() as u32,
                    0,
                    0..instance_count as u32,
                );
            }
        }

        base.queue
            .submit(std::iter::once(unlit_and_wireframe_encoder.finish()));

        if data.enable_bloom {
            private_data.bloom_threshold_cleared = false;

            base.queue.write_buffer(
                &private_data.bloom_config_buffer,
                0,
                bytemuck::cast_slice(&[0.0f32, data.bloom_threshold, data.bloom_ramp_size]),
            );

            let mut bloom_threshold_encoder =
                base.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Bloom Threshold Encoder"),
                    });
            {
                let mut bloom_threshold_render_pass =
                    bloom_threshold_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

                bloom_threshold_render_pass.set_pipeline(&self.bloom_threshold_pipeline);
                bloom_threshold_render_pass.set_bind_group(
                    0,
                    &private_data.shading_texture_bind_group,
                    &[],
                );
                bloom_threshold_render_pass.set_bind_group(
                    1,
                    &private_data.bloom_config_bind_group,
                    &[],
                );
                bloom_threshold_render_pass.draw(0..3, 0..1);
            }

            base.queue
                .submit(std::iter::once(bloom_threshold_encoder.finish()));

            let do_bloom_blur_pass = |src_texture: &wgpu::BindGroup,
                                      dst_texture: &wgpu::TextureView,
                                      horizontal: bool| {
                let mut encoder =
                    base.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Bloom Blur Encoder"),
                        });
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
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

                    base.queue.write_buffer(
                        &private_data.bloom_config_buffer,
                        0,
                        bytemuck::cast_slice(&[
                            if horizontal { 0.0f32 } else { 1.0f32 },
                            data.bloom_threshold,
                            data.bloom_ramp_size,
                        ]),
                    );
                    render_pass.set_pipeline(&self.bloom_blur_pipeline);
                    render_pass.set_bind_group(0, src_texture, &[]);
                    render_pass.set_bind_group(1, &private_data.bloom_config_bind_group, &[]);
                    render_pass.draw(0..3, 0..1);
                }

                base.queue.submit(std::iter::once(encoder.finish()));
            };

            // do 10 gaussian blur passes, switching between horizontal and vertical and ping ponging between
            // the two textures, effectively doing 5 full blurs
            let blur_passes = 10;
            (0..blur_passes).for_each(|i| {
                do_bloom_blur_pass(
                    &private_data.bloom_pingpong_texture_bind_groups[i % 2],
                    &private_data.bloom_pingpong_textures[(i + 1) % 2].view,
                    i % 2 == 0,
                );
            });
        } else {
            // clear bloom texture
            let mut encoder = base
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Bloom Clear Encoder"),
                });

            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

            base.queue.submit(std::iter::once(encoder.finish()));
            private_data.bloom_threshold_cleared = true;
        }

        let mut skybox_encoder =
            base.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Skybox Encoder"),
                });
        {
            let mut skybox_render_pass =
                skybox_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
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
            skybox_render_pass.set_pipeline(&self.skybox_pipeline);
            skybox_render_pass.set_bind_group(
                0,
                &private_data.environment_textures_bind_group,
                &[],
            );
            skybox_render_pass.set_bind_group(1, &private_data.camera_and_lights_bind_group, &[]);
            skybox_render_pass.set_vertex_buffer(0, data.skybox_mesh.vertex_buffer.src().slice(..));
            skybox_render_pass.set_index_buffer(
                data.skybox_mesh.index_buffer.src().slice(..),
                data.skybox_mesh.index_buffer_format,
            );
            skybox_render_pass.draw_indexed(
                0..(data.skybox_mesh.index_buffer.length() as u32),
                0,
                0..1,
            );
        }

        base.queue.submit(std::iter::once(skybox_encoder.finish()));

        let mut tone_mapping_encoder =
            base.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Tone Mapping Encoder"),
                });
        {
            let mut tone_mapping_render_pass =
                tone_mapping_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
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
            tone_mapping_render_pass.set_pipeline(&self.tone_mapping_pipeline);
            tone_mapping_render_pass.set_bind_group(
                0,
                &private_data.shading_and_bloom_textures_bind_group,
                &[],
            );
            tone_mapping_render_pass.set_bind_group(
                1,
                &private_data.tone_mapping_config_bind_group,
                &[],
            );
            tone_mapping_render_pass.draw(0..3, 0..1);
        }

        base.queue
            .submit(std::iter::once(tone_mapping_encoder.finish()));

        let mut surface_blit_encoder =
            base.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Surface Blit Encoder"),
                });

        {
            let mut surface_blit_render_pass =
                surface_blit_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
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

            surface_blit_render_pass.set_pipeline(&self.surface_blit_pipeline);
            surface_blit_render_pass.set_bind_group(
                0,
                &private_data.tone_mapping_texture_bind_group,
                &[],
            );
            surface_blit_render_pass.draw(0..3, 0..1);
        }

        data.ui_overlay.render(
            &base.device,
            &mut surface_blit_encoder,
            &surface_texture_view,
        );
        base.queue
            .submit(std::iter::once(surface_blit_encoder.finish()));
        surface_texture.present();

        Ok(())
    }

    fn render_pbr_meshes<'a>(
        base: &BaseRendererState,
        data: &RendererStatePublicData,
        private_data: &RendererStatePrivateData,
        render_pass_descriptor: &wgpu::RenderPassDescriptor<'a, 'a>,
        pipeline: &'a wgpu::RenderPipeline,
        is_shadow: bool,
    ) {
        let device = &base.device;
        let queue = &base.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_pbr_meshes Encoder"),
        });
        {
            let mut render_pass = encoder.begin_render_pass(render_pass_descriptor);
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &private_data.camera_and_lights_bind_group, &[]);
            if !is_shadow {
                render_pass.set_bind_group(1, &private_data.environment_textures_bind_group, &[]);
            }
            for pbr_instance_chunk in private_data.all_pbr_instances.chunks() {
                let binded_pbr_mesh_index = pbr_instance_chunk.id;
                let bone_transforms_buffer_start_index = private_data
                    .all_bone_transforms
                    .animated_bone_transforms
                    .iter()
                    .find(|bone_slice| bone_slice.binded_pbr_mesh_index == binded_pbr_mesh_index)
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
                render_pass.set_vertex_buffer(0, geometry_buffers.vertex_buffer.src().slice(..));
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

        queue.submit(std::iter::once(encoder.finish()));
    }
}
