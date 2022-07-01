use std::{
    num::{NonZeroU32, NonZeroU64},
    time::Instant,
};

use super::*;

use anyhow::Result;

use cgmath::{Deg, Matrix4, One, Vector2, Vector3};
use wgpu::util::DeviceExt;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct GpuMatrix4(pub cgmath::Matrix4<f32>);

unsafe impl bytemuck::Pod for GpuMatrix4 {}
unsafe impl bytemuck::Zeroable for GpuMatrix4 {}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct GpuMatrix3(pub cgmath::Matrix3<f32>);

unsafe impl bytemuck::Pod for GpuMatrix3 {}
unsafe impl bytemuck::Zeroable for GpuMatrix3 {}

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

impl From<&PointLightComponent> for PointLightUniform {
    fn from(light: &PointLightComponent) -> Self {
        let position = light.transform.position();
        let color = light.color;
        let intensity = light.intensity;
        Self {
            position: [position.x, position.y, position.z, 1.0],
            color: [color.x, color.y, color.z, intensity],
        }
    }
}

impl Default for PointLightUniform {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0, 1.0],
            color: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

fn make_point_light_uniform_buffer(lights: &[PointLightComponent]) -> Vec<PointLightUniform> {
    let mut light_uniforms = Vec::new();

    let active_light_count = lights.len();
    let mut active_lights = lights
        .iter()
        .map(PointLightUniform::from)
        .collect::<Vec<_>>();
    light_uniforms.append(&mut active_lights);

    let mut inactive_lights = (0..(MAX_LIGHT_COUNT as usize - active_light_count))
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
            world_space_to_light_space: (view_proj_matrices.proj * view_proj_matrices.view).into(),
            position: [position.x, position.y, position.z, 1.0],
            direction: [direction.x, direction.y, direction.z, 1.0],
            color: [color.x, color.y, color.z, *intensity],
        }
    }
}

impl Default for DirectionalLightUniform {
    fn default() -> Self {
        Self {
            world_space_to_light_space: Matrix4::one().into(),
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

    let mut inactive_lights = (0..(MAX_LIGHT_COUNT as usize - active_light_count))
        .map(|_| DirectionalLightUniform::default())
        .collect::<Vec<_>>();
    light_uniforms.append(&mut inactive_lights);

    light_uniforms
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FlatColorUniform {
    color: [f32; 4],
}

impl From<Vector3<f32>> for FlatColorUniform {
    fn from(color: Vector3<f32>) -> Self {
        Self {
            color: [color.x, color.y, color.z, 1.0],
        }
    }
}

pub const INITIAL_RENDER_SCALE: f32 = 1.0;
pub const INITIAL_TONE_MAPPING_EXPOSURE: f32 = 0.5;
pub const INITIAL_BLOOM_THRESHOLD: f32 = 0.8;
pub const INITIAL_BLOOM_RAMP_SIZE: f32 = 0.2;
pub const ARENA_SIDE_LENGTH: f32 = 50.0;
pub const MAX_LIGHT_COUNT: u8 = 32;
pub const MAX_BONES_BUFFER_SIZE_BYTES: u32 = 8192;
pub const LIGHT_COLOR_A: Vector3<f32> = Vector3::new(0.996, 0.973, 0.663);
pub const LIGHT_COLOR_B: Vector3<f32> = Vector3::new(0.25, 0.973, 0.663);
pub const Z_NEAR: f32 = 0.001;
pub const Z_FAR: f32 = 100000.0;
pub const FOV_Y: Deg<f32> = Deg(45.0);

enum SkyboxBackground<'a> {
    Cube { face_image_paths: [&'a str; 6] },
    Equirectangular { image_path: &'a str },
}

enum SkyboxHDREnvironment<'a> {
    Equirectangular { image_path: &'a str },
}

pub struct RendererState {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    limits: wgpu::Limits,

    tone_mapping_exposure: f32,
    bloom_threshold: f32,
    bloom_ramp_size: f32,

    render_scale: f32,
    state_update_time_accumulator: f32,
    last_frame_instant: Option<Instant>,
    first_frame_instant: Option<Instant>,
    animation_time_acc: f32,
    is_playing_animations: bool,
    pub current_window_size: winit::dpi::PhysicalSize<u32>,
    pub logger: Logger,

    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,

    point_lights_buffer: wgpu::Buffer,
    directional_lights_buffer: wgpu::Buffer,

    bones_buffer: wgpu::Buffer,
    camera_and_lights_bind_group: wgpu::BindGroup,

    mesh_pipeline: wgpu::RenderPipeline,
    flat_color_mesh_pipeline: wgpu::RenderPipeline,
    skybox_pipeline: wgpu::RenderPipeline,
    tone_mapping_pipeline: wgpu::RenderPipeline,
    surface_blit_pipeline: wgpu::RenderPipeline,

    point_shadow_map_pipeline: wgpu::RenderPipeline,
    directional_shadow_map_pipeline: wgpu::RenderPipeline,
    shadow_camera_and_lights_bind_group: wgpu::BindGroup,
    shadow_camera_buffer: wgpu::Buffer,
    point_shadow_map_textures: Texture,
    directional_shadow_map_textures: Texture,

    bones_bind_group: wgpu::BindGroup,

    single_texture_bind_group_layout: wgpu::BindGroupLayout,
    two_texture_bind_group_layout: wgpu::BindGroupLayout,
    bones_bind_group_layout: wgpu::BindGroupLayout,

    shading_texture: Texture,
    tone_mapping_texture: Texture,
    depth_texture: Texture,

    bloom_threshold_pipeline: wgpu::RenderPipeline,
    bloom_blur_pipeline: wgpu::RenderPipeline,
    bloom_pingpong_textures: [Texture; 2],
    bloom_config_bind_group: wgpu::BindGroup,
    bloom_config_buffer: wgpu::Buffer,

    tone_mapping_config_bind_group: wgpu::BindGroup,
    tone_mapping_config_buffer: wgpu::Buffer,

    environment_textures_bind_group: wgpu::BindGroup,
    shading_and_bloom_textures_bind_group: wgpu::BindGroup,
    tone_mapping_texture_bind_group: wgpu::BindGroup,
    shading_texture_bind_group: wgpu::BindGroup,
    bloom_pingpong_texture_bind_groups: [wgpu::BindGroup; 2],

    // store the previous state and next state and interpolate between them
    next_balls: Vec<BallComponent>,
    prev_balls: Vec<BallComponent>,
    actual_balls: Vec<BallComponent>,

    point_lights: Vec<PointLightComponent>,
    directional_lights: Vec<DirectionalLightComponent>,
    test_object_instances: Vec<MeshInstance>,
    plane_instances: Vec<MeshInstance>,

    point_light_mesh: InstancedMeshComponent,
    sphere_mesh: InstancedMeshComponent,
    test_object_mesh: InstancedMeshComponent,
    plane_mesh: InstancedMeshComponent,
    skybox_mesh: MeshComponent, // TODO: always use InstancedMeshComponent?

    scene: Scene,
}

impl RendererState {
    pub async fn new(window: &winit::window::Window) -> Result<Self> {
        // Mountains
        let _skybox_background = SkyboxBackground::Cube {
            face_image_paths: [
                "./src/textures/skybox/right.jpg",
                "./src/textures/skybox/left.jpg",
                "./src/textures/skybox/top.jpg",
                "./src/textures/skybox/bottom.jpg",
                "./src/textures/skybox/front.jpg",
                "./src/textures/skybox/back.jpg",
            ],
        };
        let _skybox_hdr_environment: Option<SkyboxHDREnvironment> = None;

        // Newport Loft
        let skybox_background = SkyboxBackground::Equirectangular {
            image_path: "./src/textures/newport_loft/background.jpg",
        };
        let skybox_hdr_environment: Option<SkyboxHDREnvironment> =
            Some(SkyboxHDREnvironment::Equirectangular {
                image_path: "./src/textures/newport_loft/radiance.hdr",
            });

        // My photosphere pic
        let _skybox_background = SkyboxBackground::Equirectangular {
            image_path: "./src/textures/photosphere_skybox.jpg",
        };
        let _skybox_hdr_environment: Option<SkyboxHDREnvironment> =
            Some(SkyboxHDREnvironment::Equirectangular {
                image_path: "./src/textures/photosphere_skybox_small.jpg",
            });

        // let gltf_path = "/home/david/Downloads/adamHead/adamHead.gltf";
        // let gltf_path = "/home/david/Programming/glTF-Sample-Models/2.0/VC/glTF/VC.gltf";
        // let gltf_path = "./src/models/gltf/TextureCoordinateTest/TextureCoordinateTest.gltf";
        // let gltf_path = "./src/models/gltf/SimpleMeshes/SimpleMeshes.gltf";
        // let gltf_path = "./src/models/gltf/Triangle/Triangle.gltf";
        // let gltf_path = "./src/models/gltf/TriangleWithoutIndices/TriangleWithoutIndices.gltf";
        // let gltf_path = "./src/models/gltf/Sponza/Sponza.gltf";
        // let gltf_path = "./src/models/gltf/EnvironmentTest/EnvironmentTest.gltf";
        // let gltf_path = "./src/models/gltf/Arrow/Arrow.gltf";
        // let gltf_path = "./src/models/gltf/DamagedHelmet/DamagedHelmet.gltf";
        // let gltf_path = "./src/models/gltf/VertexColorTest/VertexColorTest.gltf";
        // let gltf_path =
        //     "/home/david/Programming/glTF-Sample-Models/2.0/BoomBoxWithAxes/glTF/BoomBoxWithAxes.gltf";
        // let gltf_path =
        //     "./src/models/gltf/TextureLinearInterpolationTest/TextureLinearInterpolationTest.glb";
        // let gltf_path = "../glTF-Sample-Models/2.0/RiggedFigure/glTF/RiggedFigure.gltf";
        // let gltf_path = "../glTF-Sample-Models/2.0/RiggedSimple/glTF/RiggedSimple.gltf";
        // let gltf_path = "../glTF-Sample-Models/2.0/CesiumMan/glTF/CesiumMan.gltf";
        // let gltf_path = "../glTF-Sample-Models/2.0/Fox/glTF/Fox.gltf";
        let gltf_path = "../glTF-Sample-Models/2.0/BrainStem/glTF/BrainStem.gltf";
        // let gltf_path =
        //     "/home/david/Programming/glTF-Sample-Models/2.0/BoxAnimated/glTF/BoxAnimated.gltf";
        // let gltf_path = "/home/david/Programming/glTF-Sample-Models/2.0/InterpolationTest/glTF/InterpolationTest.gltf";
        // let gltf_path = "./src/models/gltf/VC/VC.gltf";
        // let gltf_path =
        //     "../glTF-Sample-Models-master/2.0/InterpolationTest/glTF/InterpolationTest.gltf";

        let mut logger = Logger::new();
        // force it to vulkan to get renderdoc to work:
        let backends = if cfg!(target_os = "linux") {
            wgpu::Backends::from(wgpu::Backend::Vulkan)
        } else {
            wgpu::Backends::all()
        };
        let instance = wgpu::Instance::new(backends);
        let size = window.inner_size();
        let surface = unsafe { instance.create_surface(&window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");
        let adapter_info = adapter.get_info();
        let adapter_name = adapter_info.name;
        let adapter_backend = adapter_info.backend;
        logger.log(&format!("Using {adapter_name} ({adapter_backend:?})"));
        logger.log(&format!("Using {adapter_name} ({adapter_backend:?})"));
        logger.log("Controls:");
        vec![
            "Move Around: WASD, Space Bar, Ctrl",
            "Look Around: Mouse",
            "Adjust Speed: Scroll",
            "Adjust Render Scale: Z / X",
            "Adjust Exposure: E / R",
            "Adjust Bloom Threshold: T / Y",
            "Exit: Escape",
        ]
        .iter()
        .for_each(|line| {
            logger.log(&format!("  {line}"));
        });

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let swapchain_format = *surface
            .get_supported_formats(&adapter)
            .get(0)
            .expect("Window surface is incompatible with the graphics adapter");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            // present_mode: wgpu::PresentMode::Immediate,
        };

        surface.configure(&device, &config);

        let flat_color_mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flat Color Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(
                std::fs::read_to_string("./src/shaders/flat_color_mesh.wgsl")?.into(),
            ),
        });

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(
                std::fs::read_to_string("./src/shaders/blit.wgsl")?.into(),
            ),
        });

        let textured_mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Textured Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(
                std::fs::read_to_string("./src/shaders/textured_mesh.wgsl")?.into(),
            ),
        });

        let skybox_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Skybox Shader"),
            source: wgpu::ShaderSource::Wgsl(
                std::fs::read_to_string("./src/shaders/skybox.wgsl")?.into(),
            ),
        });

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
        let five_texture_bind_group_layout =
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
                label: Some("four_texture_bind_group_layout"),
            });

        let single_cube_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("environment_textures_bind_group_layout"),
            });

        let single_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let bones_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("bones_bind_group_layout"),
            });

        let fragment_shader_color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Pipeline Layout"),
            bind_group_layouts: &[
                &camera_and_lights_bind_group_layout,
                &five_texture_bind_group_layout, // TODO: skybox texture actually isn't used here
                &environment_textures_bind_group_layout,
                &bones_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let mesh_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Pipeline"),
            layout: Some(&mesh_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &textured_mesh_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), GpuMeshInstance::desc()],
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
        let mesh_pipeline = device.create_render_pipeline(&mesh_pipeline_descriptor);

        let flat_color_mesh_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Flat Color Mesh Pipeline Layout"),
                bind_group_layouts: &[&camera_and_lights_bind_group_layout],
                push_constant_ranges: &[],
            });
        let mut flat_color_mesh_pipeline_descriptor = mesh_pipeline_descriptor.clone();
        flat_color_mesh_pipeline_descriptor.label = Some("Flat Color Mesh Render Pipeline");
        flat_color_mesh_pipeline_descriptor.layout = Some(&flat_color_mesh_pipeline_layout);
        let flat_color_mesh_pipeline_v_buffers =
            &[Vertex::desc(), GpuFlatColorMeshInstance::desc()];
        flat_color_mesh_pipeline_descriptor.vertex = wgpu::VertexState {
            module: &flat_color_mesh_shader,
            entry_point: "vs_main",
            buffers: flat_color_mesh_pipeline_v_buffers,
        };
        flat_color_mesh_pipeline_descriptor.fragment = Some(wgpu::FragmentState {
            module: &flat_color_mesh_shader,
            entry_point: "fs_main",
            targets: fragment_shader_color_targets,
        });
        let flat_color_mesh_pipeline =
            device.create_render_pipeline(&flat_color_mesh_pipeline_descriptor);

        let bloom_threshold_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &single_texture_bind_group_layout,
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
        let bloom_threshold_pipeline =
            device.create_render_pipeline(&bloom_threshold_pipeline_descriptor);

        let bloom_blur_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &single_texture_bind_group_layout,
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
        let bloom_blur_pipeline = device.create_render_pipeline(&bloom_blur_pipeline_descriptor);

        let surface_blit_color_targets = &[Some(wgpu::ColorTargetState {
            format: config.format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let surface_blit_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&single_texture_bind_group_layout],
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
        let surface_blit_pipeline =
            device.create_render_pipeline(&surface_blit_pipeline_descriptor);

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
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &two_texture_bind_group_layout,
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
        let tone_mapping_pipeline =
            device.create_render_pipeline(&tone_mapping_pipeline_descriptor);

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
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Skybox Render Pipeline Layout"),
                bind_group_layouts: &[
                    &environment_textures_bind_group_layout, // TODO: only using 1 texture here, don't put a bgl with so 8 of them? lol
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
        let skybox_pipeline = device.create_render_pipeline(&skybox_pipeline_descriptor);

        let equirectangular_to_cubemap_color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let equirectangular_to_cubemap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Equirectangular To Cubemap Render Pipeline Layout"),
                bind_group_layouts: &[
                    &single_texture_bind_group_layout,
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
        let equirectangular_to_cubemap_pipeline =
            device.create_render_pipeline(&equirectangular_to_cubemap_pipeline_descriptor);

        let diffuse_env_map_color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let diffuse_env_map_gen_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
        let diffuse_env_map_gen_pipeline =
            device.create_render_pipeline(&diffuse_env_map_gen_pipeline_descriptor);

        let specular_env_map_color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let specular_env_map_gen_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
        let specular_env_map_gen_pipeline =
            device.create_render_pipeline(&specular_env_map_gen_pipeline_descriptor);

        let brdf_lut_gen_color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rg16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let brdf_lut_gen_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
        let brdf_lut_gen_pipeline =
            device.create_render_pipeline(&brdf_lut_gen_pipeline_descriptor);

        let shadow_map_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow Map Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_and_lights_bind_group_layout,
                    &bones_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let point_shadow_map_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Point Shadow Map Pipeline"),
            layout: Some(&shadow_map_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &textured_mesh_shader,
                entry_point: "shadow_map_vs_main",
                buffers: &[Vertex::desc(), GpuMeshInstance::desc()],
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
        let point_shadow_map_pipeline =
            device.create_render_pipeline(&point_shadow_map_pipeline_descriptor);

        let directional_shadow_map_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Directional Shadow Map Pipeline"),
            layout: Some(&shadow_map_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &textured_mesh_shader,
                entry_point: "shadow_map_vs_main",
                buffers: &[Vertex::desc(), GpuMeshInstance::desc()],
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
        let directional_shadow_map_pipeline =
            device.create_render_pipeline(&directional_shadow_map_pipeline_descriptor);

        let (document, buffers, images) = gltf::import(gltf_path)?;
        let scene = build_scene(
            &device,
            &queue,
            &five_texture_bind_group_layout,
            GltfAsset {
                document,
                buffers,
                images,
            },
        )?;
        validate_animation_property_counts(&scene, &mut logger);
        let initial_render_scale = INITIAL_RENDER_SCALE;

        let sphere_mesh = BasicMesh::new("./src/models/sphere.obj")?;
        let cube_mesh = BasicMesh::new("./src/models/cube.obj")?;
        let plane_mesh = BasicMesh::new("./src/models/plane.obj")?;

        let skybox_mesh =
            MeshComponent::new(&cube_mesh, None, &single_uniform_bind_group_layout, &device)?;

        let shading_texture = Texture::create_scaled_surface_texture(
            &device,
            &config,
            initial_render_scale,
            "shading_texture",
        );
        let bloom_pingpong_textures = [
            Texture::create_scaled_surface_texture(
                &device,
                &config,
                initial_render_scale,
                "bloom_texture_1",
            ),
            Texture::create_scaled_surface_texture(
                &device,
                &config,
                initial_render_scale,
                "bloom_texture_2",
            ),
        ];
        let tone_mapping_texture = Texture::create_scaled_surface_texture(
            &device,
            &config,
            initial_render_scale,
            "tone_mapping_texture",
        );
        let shading_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &single_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shading_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shading_texture.sampler),
                },
            ],
            label: Some("shading_texture_bind_group"),
        });
        let tone_mapping_texture_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &single_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&tone_mapping_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&tone_mapping_texture.sampler),
                    },
                ],
                label: Some("tone_mapping_texture_bind_group"),
            });
        let shading_and_bloom_textures_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &two_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&shading_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&shading_texture.sampler),
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
                            &bloom_pingpong_textures[0].sampler,
                        ),
                    },
                ],
                label: Some("surface_blit_textures_bind_group"),
            });
        let bloom_pingpong_texture_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &single_texture_bind_group_layout,
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
                            &bloom_pingpong_textures[0].sampler,
                        ),
                    },
                ],
                label: Some("bloom_texture_bind_group_1"),
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &single_texture_bind_group_layout,
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
                            &bloom_pingpong_textures[1].sampler,
                        ),
                    },
                ],
                label: Some("bloom_texture_bind_group_2"),
            }),
        ];

        let bloom_config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bloom Config Buffer"),
            contents: bytemuck::cast_slice(&[0f32, 0f32, 0f32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bloom_config_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &single_uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bloom_config_buffer.as_entire_binding(),
            }],
            label: Some("bloom_config_bind_group"),
        });

        let tone_mapping_config_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tone Mapping Config Buffer"),
                contents: bytemuck::cast_slice(&[0f32]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let tone_mapping_config_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &single_uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: tone_mapping_config_buffer.as_entire_binding(),
            }],
            label: Some("tone_mapping_config_bind_group"),
        });

        let depth_texture =
            Texture::create_depth_texture(&device, &config, initial_render_scale, "depth_texture");

        // source: https://www.solarsystemscope.com/textures/
        let mars_texture_path = "./src/textures/8k_mars.jpg";
        let mars_texture_bytes = std::fs::read(mars_texture_path)?;
        let mars_texture = Texture::from_encoded_image(
            &device,
            &queue,
            &mars_texture_bytes,
            mars_texture_path,
            None,
            true,
            &Default::default(),
        )?;

        let earth_texture_path = "./src/textures/8k_earth.jpg";
        let earth_texture_bytes = std::fs::read(earth_texture_path)?;
        let earth_texture = Texture::from_encoded_image(
            &device,
            &queue,
            &earth_texture_bytes,
            earth_texture_path,
            None,
            true,
            &Default::default(),
        )?;

        let earth_normal_map_path = "./src/textures/8k_earth_normal_map.jpg";
        let earth_normal_map_bytes = std::fs::read(earth_normal_map_path)?;
        let earth_normal_map = Texture::from_encoded_image(
            &device,
            &queue,
            &earth_normal_map_bytes,
            earth_normal_map_path,
            wgpu::TextureFormat::Rgba8Unorm.into(),
            false,
            &Default::default(),
        )?;

        // let simple_normal_map_path = "./src/textures/simple_normal_map.jpg";
        // let simple_normal_map_bytes = std::fs::read(simple_normal_map_path)?;
        // let simple_normal_map = Texture::from_bytes(
        //     &device,
        //     &queue,
        //     &simple_normal_map_bytes,
        //     simple_normal_map_path,
        //     wgpu::TextureFormat::Rgba8Unorm.into(),
        //     false,
        //     &Default::default(),
        // )?;

        // let brick_normal_map_path = "./src/textures/brick_normal_map.jpg";
        // let brick_normal_map_bytes = std::fs::read(brick_normal_map_path)?;
        // let brick_normal_map = Texture::from_encoded_image(
        //     &device,
        //     &queue,
        //     &brick_normal_map_bytes,
        //     brick_normal_map_path,
        //     wgpu::TextureFormat::Rgba8Unorm.into(),
        //     false,
        //     &Default::default(),
        // )?;

        let skybox_texture = match skybox_background {
            SkyboxBackground::Equirectangular { image_path } => {
                let er_skybox_texture_bytes = std::fs::read(image_path)?;
                let er_skybox_texture = Texture::from_encoded_image(
                    &device,
                    &queue,
                    &er_skybox_texture_bytes,
                    image_path,
                    None,
                    false, // an artifact occurs between the edges of the texture with mipmaps enabled
                    &Default::default(),
                )?;

                Texture::create_cubemap_from_equirectangular(
                    &device,
                    &queue,
                    Some(image_path),
                    &skybox_mesh,
                    &equirectangular_to_cubemap_pipeline,
                    &er_skybox_texture,
                    // TODO: set to true?
                    false,
                )
            }
            SkyboxBackground::Cube { face_image_paths } => {
                let cubemap_skybox_images = face_image_paths
                    .iter()
                    .map(|path| image::load_from_memory(&std::fs::read(path)?))
                    .collect::<Result<Vec<_>, _>>()?;

                Texture::create_cubemap(
                    &device,
                    &queue,
                    CreateCubeMapImagesParam {
                        pos_x: &cubemap_skybox_images[0],
                        neg_x: &cubemap_skybox_images[1],
                        pos_y: &cubemap_skybox_images[2],
                        neg_y: &cubemap_skybox_images[3],
                        pos_z: &cubemap_skybox_images[4],
                        neg_z: &cubemap_skybox_images[5],
                    },
                    Some("cubemap_skybox_texture"),
                    // TODO: set to true?
                    false,
                )
            }
        };

        let er_to_cube_texture;
        let skybox_rad_texture = match skybox_hdr_environment {
            Some(SkyboxHDREnvironment::Equirectangular { image_path }) => {
                let skybox_rad_texture_bytes = std::fs::read(image_path)?;
                let skybox_rad_texture_decoded = stb::image::stbi_loadf_from_memory(
                    &skybox_rad_texture_bytes,
                    stb::image::Channels::RgbAlpha,
                )
                .ok_or_else(|| anyhow::anyhow!("Failed to decode image: {}", image_path))?;
                let skybox_rad_texture_decoded_vec = skybox_rad_texture_decoded.1.into_vec();
                let skybox_rad_texture_decoded_vec: Vec<_> = skybox_rad_texture_decoded_vec
                    .iter()
                    .copied()
                    .map(|v| Float16(half::f16::from_f32(v)))
                    .collect();

                let skybox_rad_texture_er = Texture::from_decoded_image(
                    &device,
                    &queue,
                    bytemuck::cast_slice(&skybox_rad_texture_decoded_vec),
                    (
                        skybox_rad_texture_decoded.0.width as u32,
                        skybox_rad_texture_decoded.0.height as u32,
                    ),
                    image_path.into(),
                    wgpu::TextureFormat::Rgba16Float.into(),
                    false,
                    &Default::default(),
                )?;

                er_to_cube_texture = Texture::create_cubemap_from_equirectangular(
                    &device,
                    &queue,
                    Some(image_path),
                    &skybox_mesh,
                    &equirectangular_to_cubemap_pipeline,
                    &skybox_rad_texture_er,
                    // TODO: set to true?
                    false,
                );

                &er_to_cube_texture
            }
            None => &skybox_texture,
        };

        let diffuse_env_map = Texture::create_diffuse_env_map(
            &device,
            &queue,
            Some("diffuse env map"),
            &skybox_mesh,
            &diffuse_env_map_gen_pipeline,
            skybox_rad_texture,
            // TODO: set to true?
            false,
        );

        let specular_env_map = Texture::create_specular_env_map(
            &device,
            &queue,
            Some("specular env map"),
            &skybox_mesh,
            &specular_env_map_gen_pipeline,
            skybox_rad_texture,
        );

        let brdf_lut = Texture::create_brdf_lut(&device, &queue, &brdf_lut_gen_pipeline);

        let checkerboard_texture_img = {
            let mut img = image::RgbaImage::new(4096, 4096);
            for x in 0..img.width() {
                for y in 0..img.height() {
                    let scale = 10;
                    let x_scaled = x / scale;
                    let y_scaled = y / scale;
                    img.put_pixel(
                        x,
                        y,
                        if (x_scaled + y_scaled) % 2 == 0 {
                            [100, 100, 100, 100].into()
                        } else {
                            [150, 150, 150, 150].into()
                        },
                    );
                }
            }
            img
        };
        let checkerboard_texture = Texture::from_decoded_image(
            &device,
            &queue,
            &checkerboard_texture_img,
            checkerboard_texture_img.dimensions(),
            Some("checkerboard_texture"),
            None,
            true,
            &texture::SamplerDescriptor(wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Nearest,
                ..texture::SamplerDescriptor::default().0
            }),
        )?;

        let directional_lights = vec![DirectionalLightComponent {
            position: Vector3::new(10.0, 5.0, 0.0) * 10.0,
            direction: Vector3::new(-1.0, -0.7, 0.0).normalize(),
            color: LIGHT_COLOR_A,
            intensity: 1.0,
        }];
        // let directional_lights: Vec<DirectionalLightComponent> = vec![];

        let mut point_lights = vec![
            PointLightComponent {
                transform: crate::transform::Transform::new(),
                color: LIGHT_COLOR_A,
                intensity: 1.0,
            },
            PointLightComponent {
                transform: crate::transform::Transform::new(),
                color: LIGHT_COLOR_B,
                intensity: 1.0,
            },
        ];
        // let mut point_lights: Vec<PointLightComponent> = vec![];
        if let Some(point_light_0) = point_lights.get_mut(0) {
            point_light_0
                .transform
                .set_scale(Vector3::new(0.05, 0.05, 0.05));
            point_light_0
                .transform
                .set_position(Vector3::new(0.0, 12.0, 0.0));
        }
        if let Some(point_light_1) = point_lights.get_mut(1) {
            point_light_1
                .transform
                .set_scale(Vector3::new(0.1, 0.1, 0.1));
            point_light_1
                .transform
                .set_position(Vector3::new(0.0, 15.0, 0.0));
        }

        let light_flat_color_instances: Vec<_> = point_lights
            .iter()
            .map(|light| GpuFlatColorMeshInstance::from(light.clone()))
            .collect();

        let point_light_emissive_map = Texture::from_color(
            &device,
            &queue,
            [
                (LIGHT_COLOR_A.x * 255.0).round() as u8,
                (LIGHT_COLOR_A.y * 255.0).round() as u8,
                (LIGHT_COLOR_A.z * 255.0).round() as u8,
                255,
            ],
            // [255, 0, 0, 255],
        )?;
        let point_light_metallic_roughness_map = Texture::from_color(
            &device,
            &queue,
            [255, (0.1 * 255.0f32).round() as u8, 0, 255],
        )?;
        let point_light_ambient_occlusion_map = Texture::from_gray(&device, &queue, 0)?;

        let point_light_mesh = InstancedMeshComponent::new(
            &device,
            &queue,
            &sphere_mesh,
            // TODO: InstancedMeshMaterialParams is tied to the mesh pipeline, not the flat color pipeline... so these values are ultimately ignored
            &InstancedMeshMaterialParams {
                emissive: Some(&point_light_emissive_map),
                metallic_roughness: Some(&point_light_metallic_roughness_map),
                ambient_occlusion: Some(&point_light_ambient_occlusion_map),
                ..Default::default()
            },
            &five_texture_bind_group_layout,
            bytemuck::cast_slice(&light_flat_color_instances),
        )?;

        let mut test_object_instances = vec![MeshInstance::new()];
        test_object_instances[0]
            .transform
            .set_position(Vector3::new(4.0, 10.0, 4.0));

        let test_object_transforms_gpu: Vec<_> = test_object_instances
            .iter()
            .cloned()
            .map(GpuMeshInstance::from)
            .collect();

        // let test_object_diffuse_texture =
        //     Texture::from_color(&device, &queue, [255, 255, 255, 255])?;
        let test_object_metallic_roughness_map = Texture::from_color(
            &device,
            &queue,
            [
                255,
                (0.12 * 255.0f32).round() as u8,
                (0.8 * 255.0f32).round() as u8,
                255,
            ],
        )?;
        let test_object_mesh = InstancedMeshComponent::new(
            &device,
            &queue,
            &sphere_mesh,
            &InstancedMeshMaterialParams {
                diffuse: Some(&earth_texture),
                normal: Some(&earth_normal_map),
                metallic_roughness: Some(&test_object_metallic_roughness_map),
                ..Default::default()
            },
            &five_texture_bind_group_layout,
            bytemuck::cast_slice(&test_object_transforms_gpu),
        )?;

        let mut plane_instances = vec![MeshInstance::new()];
        plane_instances[0].transform.set_scale(Vector3::new(
            ARENA_SIDE_LENGTH,
            1.0,
            ARENA_SIDE_LENGTH,
        ));

        let plane_transforms_gpu: Vec<_> = plane_instances
            .iter()
            .cloned()
            .map(GpuMeshInstance::from)
            .collect();

        let plane_mesh = InstancedMeshComponent::new(
            &device,
            &queue,
            &plane_mesh,
            &InstancedMeshMaterialParams {
                diffuse: Some(&checkerboard_texture),
                ..Default::default()
            },
            &five_texture_bind_group_layout,
            bytemuck::cast_slice(&plane_transforms_gpu),
        )?;

        let camera_controller = CameraController::new(6.0, Camera::new((0.0, 3.0, 4.0).into()));

        let camera_uniform = CameraUniform::from_camera(
            camera_controller.current_pose,
            window,
            Z_NEAR,
            Z_FAR,
            FOV_Y.into(),
        );

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // let point_lights_uniform = make_point_light_uniform_buffer(&point_lights);
        // let point_lights_bytes: &[u8] = bytemuck::cast_slice(&point_lights_uniform);
        // let directional_lights_uniform = make_directional_light_uniform_buffer(&directional_lights);
        // let directional_lights_bytes: &[u8] = bytemuck::cast_slice(&directional_lights_uniform);
        // let all_lights_bytes: Vec<u8> = point_lights_bytes
        //     .iter()
        //     .chain(directional_lights_bytes)
        //     .copied()
        //     .collect();

        let point_lights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Lights Buffer"),
            contents: bytemuck::cast_slice(&make_point_light_uniform_buffer(&point_lights)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let directional_lights_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Directional Lights Buffer"),
                contents: bytemuck::cast_slice(&make_directional_light_uniform_buffer(
                    &directional_lights,
                )),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let initial_bone_transforms_data = (0..MAX_BONES_BUFFER_SIZE_BYTES)
            .map(|_| 0u8)
            .collect::<Vec<_>>();
        let bones_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bones Buffer"),
            contents: &initial_bone_transforms_data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let camera_and_lights_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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

        let bones_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bones_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &bones_buffer,
                    offset: 0,
                    size: NonZeroU64::new(initial_bone_transforms_data.len().try_into().unwrap()),
                }),
            }],
            label: Some("bones_bind_group"),
        });

        // TODO: does there need to be a separate buffer / bind group here for shadows?
        let shadow_camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Shadow Camera Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform::new()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shadow_camera_and_lights_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &camera_and_lights_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shadow_camera_buffer.as_entire_binding(),
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
                label: Some("point_shadow_camera_bind_group"),
            });

        let balls: Vec<_> = (0..500)
            .into_iter()
            .map(|_| {
                BallComponent::new(
                    Vector2::new(
                        -10.0 + rand::random::<f32>() * 20.0,
                        -10.0 + rand::random::<f32>() * 20.0,
                    ),
                    Vector2::new(
                        -1.0 + rand::random::<f32>() * 2.0,
                        -1.0 + rand::random::<f32>() * 2.0,
                    ),
                    0.5 + (rand::random::<f32>() * 0.75),
                    1.0 + (rand::random::<f32>() * 15.0),
                )
            })
            .collect();

        let balls_transforms: Vec<_> = balls
            .iter()
            .map(|ball| GpuMeshInstance::from(ball.instance.clone()))
            .collect();

        let sphere_mesh = InstancedMeshComponent::new(
            &device,
            &queue,
            &sphere_mesh,
            &InstancedMeshMaterialParams {
                diffuse: Some(&mars_texture),
                ..Default::default()
            },
            &five_texture_bind_group_layout,
            bytemuck::cast_slice(&balls_transforms),
        )?;

        let point_light_count: u32 = point_lights.len().try_into().unwrap();
        let point_shadow_map_textures = Texture::create_cube_depth_texture_array(
            &device,
            1024,
            Some("point_shadow_map_texture"),
            point_light_count.max(1),
        );

        let directional_light_count: u32 = directional_lights.len().try_into().unwrap();
        let directional_shadow_map_textures = Texture::create_depth_texture_array(
            &device,
            1024,
            Some("directional_shadow_map_texture"),
            directional_light_count.max(1),
        );

        // dbg!(shadow_map_textures.len());

        let environment_textures_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &environment_textures_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&skybox_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&skybox_texture.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&diffuse_env_map.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&diffuse_env_map.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&specular_env_map.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&specular_env_map.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&brdf_lut.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(&brdf_lut.sampler),
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
                            &point_shadow_map_textures.sampler,
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
                            &directional_shadow_map_textures.sampler,
                        ),
                    },
                ],
                label: Some("skybox_texture_bind_group"),
            });

        let limits = device.limits();

        Ok(Self {
            surface,
            device,
            queue,
            config,

            limits,

            tone_mapping_exposure: INITIAL_TONE_MAPPING_EXPOSURE,
            bloom_threshold: INITIAL_BLOOM_THRESHOLD,
            bloom_ramp_size: INITIAL_BLOOM_RAMP_SIZE,

            render_scale: initial_render_scale,
            state_update_time_accumulator: 0.0,
            last_frame_instant: None,
            first_frame_instant: None,
            animation_time_acc: 0.0,
            is_playing_animations: true,
            logger,
            current_window_size: size,

            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_and_lights_bind_group,

            point_lights_buffer,
            directional_lights_buffer,
            bones_buffer,

            bones_bind_group,
            bones_bind_group_layout,

            mesh_pipeline,
            flat_color_mesh_pipeline,
            skybox_pipeline,
            tone_mapping_pipeline,
            surface_blit_pipeline,

            point_shadow_map_pipeline,
            directional_shadow_map_pipeline,
            shadow_camera_and_lights_bind_group,
            shadow_camera_buffer,
            point_shadow_map_textures,
            directional_shadow_map_textures,

            single_texture_bind_group_layout,
            two_texture_bind_group_layout,

            shading_texture,
            tone_mapping_texture,
            depth_texture,

            environment_textures_bind_group,
            shading_and_bloom_textures_bind_group,
            tone_mapping_texture_bind_group,
            shading_texture_bind_group,
            bloom_pingpong_texture_bind_groups,

            bloom_threshold_pipeline,
            bloom_blur_pipeline,
            bloom_pingpong_textures,
            bloom_config_bind_group,
            bloom_config_buffer,

            tone_mapping_config_bind_group,
            tone_mapping_config_buffer,

            next_balls: balls.clone(),
            prev_balls: balls.clone(),
            actual_balls: balls,

            point_lights,
            directional_lights,
            test_object_instances,
            plane_instances,

            point_light_mesh,
            sphere_mesh,
            test_object_mesh,
            plane_mesh,
            skybox_mesh,

            scene,
        })
    }

    pub fn process_device_input(
        &mut self,
        event: &winit::event::DeviceEvent,
        window: &mut winit::window::Window,
    ) {
        self.camera_controller
            .process_device_events(event, window, &mut self.logger);
    }

    pub fn process_window_input(
        &mut self,
        event: &winit::event::WindowEvent,
        window: &mut winit::window::Window,
    ) {
        if let WindowEvent::KeyboardInput {
            input:
                KeyboardInput {
                    state,
                    virtual_keycode: Some(keycode),
                    ..
                },
            ..
        } = event
        {
            let mut increment_render_scale = |increase: bool, logger: &mut Logger| {
                let delta = 0.1;
                let change = if increase { delta } else { -delta };
                self.render_scale = (self.render_scale + change).max(0.1).min(4.0);
                logger.log(&format!(
                    "Render scale: {:?} ({:?}x{:?})",
                    self.render_scale,
                    (self.config.width as f32 * self.render_scale.sqrt()).round() as u32,
                    (self.config.height as f32 * self.render_scale.sqrt()).round() as u32,
                ));
                self.shading_texture = Texture::create_scaled_surface_texture(
                    &self.device,
                    &self.config,
                    self.render_scale,
                    "shading_texture",
                );
                self.bloom_pingpong_textures = [
                    Texture::create_scaled_surface_texture(
                        &self.device,
                        &self.config,
                        self.render_scale,
                        "bloom_texture_1",
                    ),
                    Texture::create_scaled_surface_texture(
                        &self.device,
                        &self.config,
                        self.render_scale,
                        "bloom_texture_2",
                    ),
                ];
                self.tone_mapping_texture = Texture::create_scaled_surface_texture(
                    &self.device,
                    &self.config,
                    self.render_scale,
                    "tone_mapping_texture",
                );
                self.depth_texture = Texture::create_depth_texture(
                    &self.device,
                    &self.config,
                    self.render_scale,
                    "depth_texture",
                );
                self.shading_texture_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &self.single_texture_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.shading_texture.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    &self.shading_texture.sampler,
                                ),
                            },
                        ],
                        label: Some("shading_texture_bind_group"),
                    });
                self.tone_mapping_texture_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &self.single_texture_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.tone_mapping_texture.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    &self.tone_mapping_texture.sampler,
                                ),
                            },
                        ],
                        label: Some("tone_mapping_texture_bind_group"),
                    });
                self.shading_and_bloom_textures_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &self.two_texture_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.shading_texture.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    &self.shading_texture.sampler,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.bloom_pingpong_textures[0].view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Sampler(
                                    &self.bloom_pingpong_textures[0].sampler,
                                ),
                            },
                        ],
                        label: Some("surface_blit_textures_bind_group"),
                    });
                self.bloom_pingpong_texture_bind_groups = [
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &self.single_texture_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.bloom_pingpong_textures[0].view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    &self.bloom_pingpong_textures[0].sampler,
                                ),
                            },
                        ],
                        label: Some("bloom_texture_bind_group_1"),
                    }),
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &self.single_texture_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.bloom_pingpong_textures[1].view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    &self.bloom_pingpong_textures[1].sampler,
                                ),
                            },
                        ],
                        label: Some("bloom_texture_bind_group_2"),
                    }),
                ];
            };
            let mut increment_exposure = |increase: bool, logger: &mut Logger| {
                let delta = 0.05;
                let change = if increase { delta } else { -delta };
                self.tone_mapping_exposure =
                    (self.tone_mapping_exposure + change).max(0.0).min(20.0);
                logger.log(&format!("Exposure: {:?}", self.tone_mapping_exposure));
            };
            let mut increment_bloom_threshold = |increase: bool, logger: &mut Logger| {
                let delta = 0.05;
                let change = if increase { delta } else { -delta };
                self.bloom_threshold = (self.bloom_threshold + change).max(0.0).min(20.0);
                logger.log(&format!("Bloom Threshold: {:?}", self.bloom_threshold));
            };
            if *state == ElementState::Released {
                match keycode {
                    VirtualKeyCode::Z => {
                        increment_render_scale(false, &mut self.logger);
                    }
                    VirtualKeyCode::X => {
                        increment_render_scale(true, &mut self.logger);
                    }
                    VirtualKeyCode::E => {
                        increment_exposure(false, &mut self.logger);
                    }
                    VirtualKeyCode::R => {
                        increment_exposure(true, &mut self.logger);
                    }
                    VirtualKeyCode::T => {
                        increment_bloom_threshold(false, &mut self.logger);
                    }
                    VirtualKeyCode::Y => {
                        increment_bloom_threshold(true, &mut self.logger);
                    }
                    VirtualKeyCode::P => {
                        self.is_playing_animations = !self.is_playing_animations;
                    }
                    _ => {}
                }
            }
        }
        self.camera_controller
            .process_window_events(event, window, &mut self.logger);
    }

    pub fn resize(&mut self, new_window_size: winit::dpi::PhysicalSize<u32>) {
        self.current_window_size = new_window_size;
        self.config.width = new_window_size.width;
        self.config.height = new_window_size.height;
        self.surface.configure(&self.device, &self.config);
        self.shading_texture = Texture::create_scaled_surface_texture(
            &self.device,
            &self.config,
            self.render_scale,
            "shading_texture",
        );
        self.bloom_pingpong_textures = [
            Texture::create_scaled_surface_texture(
                &self.device,
                &self.config,
                self.render_scale,
                "bloom_texture_1",
            ),
            Texture::create_scaled_surface_texture(
                &self.device,
                &self.config,
                self.render_scale,
                "bloom_texture_2",
            ),
        ];
        self.tone_mapping_texture = Texture::create_scaled_surface_texture(
            &self.device,
            &self.config,
            self.render_scale,
            "tone_mapping_texture",
        );
        self.depth_texture = Texture::create_depth_texture(
            &self.device,
            &self.config,
            self.render_scale,
            "depth_texture",
        );
        // TODO: dry this up? it's repeated three times in this file
        self.shading_texture_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.single_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.shading_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.shading_texture.sampler),
                    },
                ],
                label: Some("shading_texture_bind_group"),
            });
        self.tone_mapping_texture_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.single_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.tone_mapping_texture.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            &self.tone_mapping_texture.sampler,
                        ),
                    },
                ],
                label: Some("tone_mapping_texture_bind_group"),
            });
        self.shading_and_bloom_textures_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.two_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.shading_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.shading_texture.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &self.bloom_pingpong_textures[0].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(
                            &self.bloom_pingpong_textures[0].sampler,
                        ),
                    },
                ],
                label: Some("surface_blit_textures_bind_group"),
            });
        self.bloom_pingpong_texture_bind_groups = [
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.single_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.bloom_pingpong_textures[0].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            &self.bloom_pingpong_textures[0].sampler,
                        ),
                    },
                ],
                label: Some("bloom_texture_bind_group_1"),
            }),
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.single_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.bloom_pingpong_textures[1].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            &self.bloom_pingpong_textures[1].sampler,
                        ),
                    },
                ],
                label: Some("bloom_texture_bind_group_2"),
            }),
        ];
    }

    pub fn update(&mut self, window: &winit::window::Window) {
        let first_frame_instant = self.first_frame_instant.unwrap_or_else(Instant::now);
        let time_seconds = first_frame_instant.elapsed().as_secs_f32();
        self.first_frame_instant = Some(first_frame_instant);

        // results in ~60 state changes per second
        let min_update_timestep_seconds = 1.0 / 60.0;
        // if frametime takes longer than this, we give up on trying to catch up completely
        // prevents the game from getting stuck in a spiral of death
        let max_delay_catchup_seconds = 0.25;
        let frame_instant = Instant::now();
        let mut frame_time_seconds = if let Some(last_frame_instant) = self.last_frame_instant {
            frame_instant
                .duration_since(last_frame_instant)
                .as_secs_f32()
        } else {
            0.0
        };
        if frame_time_seconds > max_delay_catchup_seconds {
            frame_time_seconds = max_delay_catchup_seconds;
        }
        self.last_frame_instant = Some(frame_instant);
        self.state_update_time_accumulator += frame_time_seconds;

        // update ball positions
        while self.state_update_time_accumulator >= min_update_timestep_seconds {
            if self.state_update_time_accumulator < min_update_timestep_seconds * 2.0 {
                self.prev_balls = self.next_balls.clone();
            }
            self.prev_balls = self.next_balls.clone();
            self.next_balls
                .iter_mut()
                .for_each(|ball| ball.update(min_update_timestep_seconds, &mut self.logger));
            self.state_update_time_accumulator -= min_update_timestep_seconds;
        }
        let alpha = self.state_update_time_accumulator / min_update_timestep_seconds;
        self.actual_balls = self
            .prev_balls
            .iter()
            .zip(self.next_balls.iter())
            .map(|(prev_ball, next_ball)| prev_ball.lerp(next_ball, alpha))
            .collect();

        let new_point_light_0 = self.point_lights.get(0).map(|point_light_0| {
            let mut transform = point_light_0.transform;
            transform.set_position(Vector3::new(
                // light_1.transform.position.get().x,
                1.5 * (time_seconds * 0.25 + std::f32::consts::PI).cos(),
                point_light_0.transform.position().y - frame_time_seconds * 0.25,
                1.5 * (time_seconds * 0.25 + std::f32::consts::PI).sin(),
                // light_1.transform.position.get().z,
            ));
            let color = lerp_vec(LIGHT_COLOR_A, LIGHT_COLOR_B, (time_seconds * 2.0).sin());

            PointLightComponent {
                transform,
                color,
                intensity: point_light_0.intensity,
            }
        });
        if let Some(new_point_light_0) = new_point_light_0 {
            self.point_lights[0] = new_point_light_0;
        }

        let new_point_light_1 = self.point_lights.get(1).map(|point_light_1| {
            let transform = point_light_1.transform;
            // transform.set_position(Vector3::new(
            //     1.1 * (time_seconds * 0.25 + std::f32::consts::PI).cos(),
            //     transform.position.get().y,
            //     1.1 * (time_seconds * 0.25 + std::f32::consts::PI).sin(),
            // ));
            let color = lerp_vec(LIGHT_COLOR_B, LIGHT_COLOR_A, (time_seconds * 2.0).sin());

            PointLightComponent {
                transform,
                color,
                intensity: point_light_1.intensity,
            }
        });
        if let Some(new_point_light_1) = new_point_light_1 {
            self.point_lights[1] = new_point_light_1;
        }

        let directional_light_0 = self.directional_lights.get(0).map(|directional_light_0| {
            let direction = directional_light_0.direction;
            // transform.set_position(Vector3::new(
            //     1.1 * (time_seconds * 0.25 + std::f32::consts::PI).cos(),
            //     transform.position.get().y,
            //     1.1 * (time_seconds * 0.25 + std::f32::consts::PI).sin(),
            // ));
            // let color = lerp_vec(LIGHT_COLOR_B, LIGHT_COLOR_A, (time_seconds * 2.0).sin());

            DirectionalLightComponent {
                direction: Vector3::new(direction.x, direction.y + 0.0001, direction.z),
                ..*directional_light_0
            }
        });
        if let Some(directional_light_0) = directional_light_0 {
            self.directional_lights[0] = directional_light_0;
        }

        // let rotational_displacement =
        //     make_quat_from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Rad(frame_time_seconds / 5.0));
        // self.test_object_transforms[0]
        //     .set_rotation(rotational_displacement * self.test_object_transforms[0].rotation.get());

        // self.logger
        //     .log(&format!("Frame time: {:?}", frame_time_seconds));
        // self.logger.log(&format!(
        //     "state_update_time_accumulator: {:?}",
        //     self.state_update_time_accumulator
        // ));

        // to move the boom box with axes around:
        // let meshes: Vec<_> = self.scene.source_asset.document.meshes().collect();
        // let drawable_primitive_groups: Vec<_> = meshes
        //     .iter()
        //     .flat_map(|mesh| mesh.primitives().map(|prim| (&meshes[mesh.index()], prim)))
        //     .filter(|(_, prim)| prim.mode() == gltf::mesh::Mode::Triangles)
        //     .collect();
        // let prim_groups: Vec<_> = drawable_primitive_groups
        //     .iter()
        //     .enumerate()
        //     .filter(|(_, (_, prim))| {
        //         prim.material().alpha_mode() == gltf::material::AlphaMode::Opaque
        //             || prim.material().alpha_mode() == gltf::material::AlphaMode::Mask
        //     })
        //     .collect();
        // let transform_him = |prim_index: usize| {
        //     let BindableMeshData {
        //         instance_buffer,
        //         instances,
        //         ..
        //     } = &self.scene.buffers.bindable_mesh_data[prim_index];
        //     let first_directional_light = &self.directional_lights[0];
        //     let rotation_matrix = look_at_dir(
        //         first_directional_light.position * 0.9,
        //         first_directional_light.direction,
        //         // Vector3::new(10.0, 5.0, 0.0) * 9.0,
        //         // Vector3::new(-1.0, -1.0, 0.0).normalize(),
        //     );
        //     let scale_matrix = make_scale_matrix(Vector3::new(100.0, 100.0, 100.0));
        //     let new_transform =
        //         (rotation_matrix * scale_matrix * instances[0].transform.matrix()).into();
        //     let transformed_instance = GpuMeshInstance::from(MeshInstance {
        //         transform: new_transform,
        //         base_material: instances[0].base_material,
        //     });
        //     self.queue.write_buffer(
        //         &instance_buffer.buffer,
        //         0,
        //         bytemuck::cast_slice(&[transformed_instance]),
        //     );
        // };
        // transform_him(prim_groups[0].0);
        // transform_him(prim_groups[1].0);
        // transform_him(prim_groups[2].0);
        // transform_him(prim_groups[3].0);
        // transform_him(prim_groups[4].0);

        // let scene_meshes: Vec<_> = self.scene.source_asset.document.meshes().collect();
        // let scene_drawable_primitive_groups: Vec<_> = scene_meshes
        //     .iter()
        //     .flat_map(|mesh| mesh.primitives().map(|prim| (&scene_meshes[mesh.index()], prim)))
        //     .filter(|(_, prim)| prim.mode() == gltf::mesh::Mode::Triangles)
        //     .collect();
        // let scene_prim_groups: Vec<_> = scene_drawable_primitive_groups
        //     .iter()
        //     .enumerate()
        //     .filter(|(_, (_, prim))| {
        //         prim.material().alpha_mode() == gltf::material::AlphaMode::Opaque
        //             || prim.material().alpha_mode() == gltf::material::AlphaMode::Mask
        //     })
        //     .collect();
        //     self.scene.buffers.bindable_mesh_data.find(|BindableMeshData {}|)

        // if time_seconds > 5.0 {
        // }

        // do animatons
        if self.is_playing_animations {
            self.animation_time_acc += frame_time_seconds;
            if let Err(err) =
                update_node_transforms_at_moment(&mut self.scene, self.animation_time_acc)
            {
                self.logger
                    .log(&format!("Error: animation computation failed: {:?}", err));
            }
        }
        if let Some(node_0) = self.scene.nodes.get_mut(0) {
            // node_0.transform.set_position(Vector3::new(
            //     node_0.transform.position().x - 0.75 * frame_time_seconds,
            //     0.0,
            //     0.0,
            // ));
            node_0.transform.set_rotation(make_quat_from_axis_angle(
                Vector3::new(0.0, 1.0, 0.0),
                Deg(90.0).into(),
            ));
        }

        // send data to gpu
        let all_bone_transforms =
            get_all_bone_data(&self.scene, self.limits.min_storage_buffer_offset_alignment);
        self.queue
            .write_buffer(&self.bones_buffer, 0, &all_bone_transforms.buffer);
        self.bones_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bones_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.bones_buffer,
                    offset: 0,
                    size: NonZeroU64::new(all_bone_transforms.buffer.len().try_into().unwrap()),
                }),
            }],
            label: Some("bones_bind_group"),
        });
        self.scene.get_drawable_mesh_iterator().for_each(
            |BindableMeshData {
                 instance_buffer,
                 instances,
                 ..
             }| {
                let gpu_instances: Vec<_> = instances
                    .iter()
                    .cloned()
                    .map(
                        |SceneMeshInstance {
                             node_index,
                             base_material,
                             ..
                         }| {
                            let node_ancestry_list = self.scene.get_node_ancestry_list(node_index);
                            let transform = node_ancestry_list
                                .iter()
                                .rev()
                                .fold(crate::transform::Transform::new(), |acc, node_index| {
                                    acc * self.scene.nodes[*node_index].transform
                                });
                            MeshInstance {
                                base_material,
                                transform,
                            }
                        },
                    )
                    .map(GpuMeshInstance::from)
                    .collect();
                self.queue.write_buffer(
                    &instance_buffer.buffer,
                    0,
                    bytemuck::cast_slice(&gpu_instances),
                );
            },
        );
        let balls_transforms: Vec<_> = self
            .actual_balls
            .iter()
            .map(|ball| GpuMeshInstance::from(ball.instance.clone()))
            .collect();
        self.queue.write_buffer(
            &self.sphere_mesh.instance_buffer,
            0,
            bytemuck::cast_slice(&balls_transforms),
        );
        let test_object_transforms_gpu: Vec<_> = self
            .test_object_instances
            .iter()
            .cloned()
            .map(GpuMeshInstance::from)
            .collect();
        self.queue.write_buffer(
            &self.test_object_mesh.instance_buffer,
            0,
            bytemuck::cast_slice(&test_object_transforms_gpu),
        );
        self.camera_controller.update(frame_time_seconds);
        self.camera_uniform = CameraUniform::from_camera(
            self.camera_controller.current_pose,
            window,
            Z_NEAR,
            Z_FAR,
            FOV_Y.into(),
        );
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        let point_light_flat_color_instances: Vec<_> = self
            .point_lights
            .iter()
            .map(|light| GpuFlatColorMeshInstance::from(light.clone()))
            .collect();
        self.queue.write_buffer(
            &self.point_light_mesh.instance_buffer,
            0,
            bytemuck::cast_slice(&point_light_flat_color_instances),
        );

        // let point_lights_uniform = make_point_light_uniform_buffer(&self.point_lights);
        // let point_lights_bytes: &[u8] = bytemuck::cast_slice(&point_lights_uniform);
        // let directional_lights_uniform =
        //     make_directional_light_uniform_buffer(&self.directional_lights);
        // let directional_lights_bytes: &[u8] = bytemuck::cast_slice(&directional_lights_uniform);
        // let all_lights_bytes: Vec<u8> = point_lights_bytes
        //     .iter()
        //     .chain(directional_lights_bytes)
        //     .copied()
        //     .collect();
        // self.logger.log(&format!("{:?}", &self.point_lights));
        self.queue.write_buffer(
            &self.point_lights_buffer,
            0,
            bytemuck::cast_slice(&make_point_light_uniform_buffer(&self.point_lights)),
        );
        self.queue.write_buffer(
            &self.directional_lights_buffer,
            0,
            bytemuck::cast_slice(&make_directional_light_uniform_buffer(
                &self.directional_lights,
            )),
        );
        self.queue.write_buffer(
            &self.tone_mapping_config_buffer,
            0,
            bytemuck::cast_slice(&[self.tone_mapping_exposure]),
        );
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        {
            self.directional_lights
                .iter()
                .enumerate()
                .for_each(|(light_index, light)| {
                    let view_proj_matrices =
                        build_directional_light_camera_view(-light.direction, 100.0, 100.0, 1000.0);
                    let texture_view = self.directional_shadow_map_textures.texture.create_view(
                        &wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            base_array_layer: light_index.try_into().unwrap(),
                            array_layer_count: NonZeroU32::new(1),
                            ..Default::default()
                        },
                    );
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
                    self.queue.write_buffer(
                        &self.shadow_camera_buffer,
                        0,
                        bytemuck::cast_slice(&[CameraUniform::from(view_proj_matrices)]),
                    );
                    self.render_scene(
                        &shadow_render_pass_desc,
                        &self.directional_shadow_map_pipeline,
                        true,
                    );
                });
            (0..self.point_lights.len()).for_each(|light_index| {
                build_cubemap_face_camera_views(
                    self.point_lights[light_index].transform.position(),
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
                        self.point_shadow_map_textures.texture.create_view(
                            &wgpu::TextureViewDescriptor {
                                dimension: Some(wgpu::TextureViewDimension::D2),
                                base_array_layer: (6 * light_index + i).try_into().unwrap(),
                                array_layer_count: NonZeroU32::new(1),
                                ..Default::default()
                            },
                        ),
                    )
                })
                .for_each(|(face_view_proj_matrices, face_texture_view)| {
                    let shadow_render_pass_desc = wgpu::RenderPassDescriptor {
                        label: Some("Shadow Render Pass"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &face_texture_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: true,
                            }),
                            stencil_ops: None,
                        }),
                    };
                    self.queue.write_buffer(
                        &self.shadow_camera_buffer,
                        0,
                        bytemuck::cast_slice(&[CameraUniform::from(face_view_proj_matrices)]),
                    );
                    self.render_scene(
                        &shadow_render_pass_desc,
                        &self.point_shadow_map_pipeline,
                        true,
                    );
                });
            });
        }

        let surface_texture = self.surface.get_current_texture()?;
        let surface_texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let black = wgpu::Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };

        let shading_render_pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Shading Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.shading_texture.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(black),
                    store: true,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        };

        self.render_scene(&shading_render_pass_desc, &self.mesh_pipeline, false);

        let mut lights_flat_shading_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Lights Flat Shading Encoder"),
                });

        {
            let mut lights_flat_shading_render_pass = lights_flat_shading_encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Lights Flat Shading Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.shading_texture.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_texture.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                });

            // render lights
            lights_flat_shading_render_pass.set_pipeline(&self.flat_color_mesh_pipeline);
            lights_flat_shading_render_pass.set_bind_group(
                0,
                &self.camera_and_lights_bind_group,
                &[],
            );
            lights_flat_shading_render_pass
                .set_vertex_buffer(0, self.point_light_mesh.vertex_buffer.slice(..));
            lights_flat_shading_render_pass
                .set_vertex_buffer(1, self.point_light_mesh.instance_buffer.slice(..));
            lights_flat_shading_render_pass.set_index_buffer(
                self.point_light_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            lights_flat_shading_render_pass.draw_indexed(
                0..self.point_light_mesh.num_indices,
                0,
                0..self.point_lights.len() as u32,
            );
        }

        self.queue
            .submit(std::iter::once(lights_flat_shading_encoder.finish()));

        self.queue.write_buffer(
            &self.bloom_config_buffer,
            0,
            bytemuck::cast_slice(&[0.0f32, self.bloom_threshold, self.bloom_ramp_size]),
        );

        let mut bloom_threshold_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Bloom Threshold Encoder"),
                });
        {
            let mut bloom_threshold_render_pass =
                bloom_threshold_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.bloom_pingpong_textures[0].view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(black),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: None,
                });

            bloom_threshold_render_pass.set_pipeline(&self.bloom_threshold_pipeline);
            bloom_threshold_render_pass.set_bind_group(0, &self.shading_texture_bind_group, &[]);
            bloom_threshold_render_pass.set_bind_group(1, &self.bloom_config_bind_group, &[]);
            bloom_threshold_render_pass.draw(0..3, 0..1);
        }

        self.queue
            .submit(std::iter::once(bloom_threshold_encoder.finish()));

        let do_bloom_blur_pass =
            |src_texture: &wgpu::BindGroup, dst_texture: &wgpu::TextureView, horizontal: bool| {
                let mut encoder =
                    self.device
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

                    self.queue.write_buffer(
                        &self.bloom_config_buffer,
                        0,
                        bytemuck::cast_slice(&[
                            if horizontal { 0.0f32 } else { 1.0f32 },
                            self.bloom_threshold,
                            self.bloom_ramp_size,
                        ]),
                    );
                    render_pass.set_pipeline(&self.bloom_blur_pipeline);
                    render_pass.set_bind_group(0, src_texture, &[]);
                    render_pass.set_bind_group(1, &self.bloom_config_bind_group, &[]);
                    render_pass.draw(0..3, 0..1);
                }

                self.queue.submit(std::iter::once(encoder.finish()));
            };

        // do 10 gaussian blur passes, switching between horizontal and vertical and ping ponging between
        // the two textures, effectively doing 5 full blurs
        let blur_passes = 10;
        (0..blur_passes).for_each(|i| {
            do_bloom_blur_pass(
                &self.bloom_pingpong_texture_bind_groups[i % 2],
                &self.bloom_pingpong_textures[(i + 1) % 2].view,
                i % 2 == 0,
            );
        });

        let mut skybox_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Skybox Encoder"),
                });
        {
            let mut skybox_render_pass =
                skybox_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.tone_mapping_texture.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(black),
                            store: true,
                        },
                    })],
                    // depth_stencil_attachment: None,
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_texture.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                });
            skybox_render_pass.set_pipeline(&self.skybox_pipeline);
            skybox_render_pass.set_bind_group(0, &self.environment_textures_bind_group, &[]);
            skybox_render_pass.set_bind_group(1, &self.camera_and_lights_bind_group, &[]);
            skybox_render_pass.set_vertex_buffer(0, self.skybox_mesh.vertex_buffer.slice(..));
            skybox_render_pass.set_index_buffer(
                self.skybox_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            skybox_render_pass.draw_indexed(0..self.skybox_mesh.num_indices, 0, 0..1);
        }

        self.queue.submit(std::iter::once(skybox_encoder.finish()));

        let mut tone_mapping_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Tone Mapping Encoder"),
                });
        {
            let mut tone_mapping_render_pass =
                tone_mapping_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.tone_mapping_texture.view,
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
                &self.shading_and_bloom_textures_bind_group,
                &[],
            );
            tone_mapping_render_pass.set_bind_group(1, &self.tone_mapping_config_bind_group, &[]);
            tone_mapping_render_pass.draw(0..3, 0..1);
        }

        self.queue
            .submit(std::iter::once(tone_mapping_encoder.finish()));

        let mut surface_blit_encoder =
            self.device
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
            surface_blit_render_pass.set_bind_group(0, &self.tone_mapping_texture_bind_group, &[]);
            surface_blit_render_pass.draw(0..3, 0..1);
        }

        self.queue
            .submit(std::iter::once(surface_blit_encoder.finish()));

        surface_texture.present();
        Ok(())
    }

    fn render_scene<'a>(
        &'a self,
        render_pass_descriptor: &wgpu::RenderPassDescriptor<'a, 'a>,
        pipeline: &'a wgpu::RenderPipeline,
        is_shadow: bool,
    ) {
        let all_bone_transforms =
            get_all_bone_data(&self.scene, self.limits.min_storage_buffer_offset_alignment);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_scene Encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(render_pass_descriptor);
            self.scene
                .get_drawable_mesh_iterator()
                .enumerate()
                .for_each(
                    |(
                        drawable_mesh_index,
                        BindableMeshData {
                            vertex_buffer,
                            index_buffer,
                            instance_buffer,
                            textures_bind_group,
                            ..
                        },
                    )| {
                        {
                            render_pass.set_pipeline(pipeline);
                            render_pass.set_bind_group(
                                0,
                                if is_shadow {
                                    &self.shadow_camera_and_lights_bind_group
                                } else {
                                    &self.camera_and_lights_bind_group
                                },
                                &[],
                            );
                            let bone_transforms_buffer_start_index = all_bone_transforms
                                .animated_bone_transforms
                                .iter()
                                .find(|bone_slice| {
                                    bone_slice.drawable_mesh_index == drawable_mesh_index
                                })
                                .map(|bone_slice| bone_slice.start_index.try_into().unwrap())
                                .unwrap_or(0);
                            render_pass.set_bind_group(
                                if is_shadow { 1 } else { 3 },
                                &self.bones_bind_group,
                                &[bone_transforms_buffer_start_index],
                            );
                            if !is_shadow {
                                render_pass.set_bind_group(1, textures_bind_group, &[]);
                                render_pass.set_bind_group(
                                    2,
                                    &self.environment_textures_bind_group,
                                    &[],
                                );
                            }

                            render_pass.set_vertex_buffer(0, vertex_buffer.buffer.slice(..));
                            render_pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));
                            match index_buffer {
                                Some(index_buffer) => {
                                    render_pass.set_index_buffer(
                                        index_buffer.buffer.slice(..),
                                        wgpu::IndexFormat::Uint16,
                                    );
                                    render_pass.draw_indexed(
                                        0..index_buffer.length as u32,
                                        0,
                                        0..instance_buffer.length as u32,
                                    );
                                }
                                None => {
                                    render_pass.draw(
                                        0..vertex_buffer.length as u32,
                                        0..instance_buffer.length as u32,
                                    );
                                }
                            };
                        }
                    },
                );

            render_pass.set_bind_group(if is_shadow { 1 } else { 3 }, &self.bones_bind_group, &[0]);

            // render test object
            if !is_shadow {
                render_pass.set_bind_group(1, &self.test_object_mesh.textures_bind_group, &[]);
            }
            render_pass.set_vertex_buffer(0, self.test_object_mesh.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.test_object_mesh.instance_buffer.slice(..));
            render_pass.set_index_buffer(
                self.test_object_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.draw_indexed(
                0..self.test_object_mesh.num_indices,
                0,
                0..self.test_object_instances.len() as u32,
            );

            // render floor
            if !is_shadow {
                render_pass.set_bind_group(1, &self.plane_mesh.textures_bind_group, &[]);
            }
            render_pass.set_vertex_buffer(0, self.plane_mesh.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.plane_mesh.instance_buffer.slice(..));
            render_pass.set_index_buffer(
                self.plane_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.draw_indexed(
                0..self.plane_mesh.num_indices,
                0,
                0..self.plane_instances.len() as u32,
            );

            // render balls
            // if !is_shadow {
            //     render_pass.set_bind_group(1, &self.sphere_mesh.textures_bind_group, &[]);
            // }
            // render_pass.set_vertex_buffer(0, self.sphere_mesh.vertex_buffer.slice(..));
            // render_pass.set_vertex_buffer(1, self.sphere_mesh.instance_buffer.slice(..));
            // render_pass.set_index_buffer(
            //     self.sphere_mesh.index_buffer.slice(..),
            //     wgpu::IndexFormat::Uint16,
            // );
            // render_pass.draw_indexed(
            //     0..self.sphere_mesh.num_indices,
            //     0,
            //     0..self.actual_balls.len() as u32,
            // );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }
}
