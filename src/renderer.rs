use std::time::Instant;

use super::*;

use anyhow::Result;

use cgmath::{Matrix4, One, Rad, Vector2, Vector3};
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
        let position = light.transform.position.get();
        let color = light.color;
        Self {
            position: [position.x, position.y, position.z, 1.0],
            color: [color.x, color.y, color.z, 1.0],
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

    let mut active_lights = lights
        .iter()
        .map(PointLightUniform::from)
        .collect::<Vec<_>>();
    light_uniforms.append(&mut active_lights);

    let mut inactive_lights = (0..(MAX_LIGHT_COUNT as usize - active_lights.len()))
        .map(|_| PointLightUniform::default())
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

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    rotation_only_view: [[f32; 4]; 4],
    position: [f32; 4],
    near_plane_distance: f32,
    far_plane_distance: f32,
    padding: [f32; 2],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            proj: Matrix4::one().into(),
            view: Matrix4::one().into(),
            rotation_only_view: Matrix4::one().into(),
            position: [0.0; 4],
            near_plane_distance: camera::Z_NEAR,
            far_plane_distance: camera::Z_FAR,
            padding: [0.0; 2],
        }
    }

    fn update_view_proj(&mut self, camera: &Camera, window: &winit::window::Window) {
        let CameraViewProjMatrices {
            proj,
            view,
            rotation_only_view,
            position,
        } = camera.build_view_projection_matrices(window);
        self.proj = proj.into();
        self.view = view.into();
        self.rotation_only_view = rotation_only_view.into();
        self.position = [position.x, position.y, position.z, 1.0];
    }

    pub fn from_view_proj_matrices(
        CameraViewProjMatrices {
            proj,
            view,
            rotation_only_view,
            position,
        }: &CameraViewProjMatrices,
    ) -> Self {
        Self {
            proj: (*proj).into(),
            view: (*view).into(),
            rotation_only_view: (*rotation_only_view).into(),
            position: [position.x, position.y, position.z, 1.0],
            near_plane_distance: 0.0, // TODO: ewwwww
            far_plane_distance: 0.0,  // TODO: ewwwww
            padding: [0.0; 2],
        }
    }
}

const INITIAL_RENDER_SCALE: f32 = 2.0;
pub const ARENA_SIDE_LENGTH: f32 = 50.0;
pub const MAX_LIGHT_COUNT: u8 = 32;
pub const LIGHT_COLOR_A: Vector3<f32> = Vector3::new(0.996, 0.973, 0.663);
pub const LIGHT_COLOR_B: Vector3<f32> = Vector3::new(0.25, 0.973, 0.663);

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
    render_scale: f32,
    state_update_time_accumulator: f32,
    last_frame_instant: Option<Instant>,
    first_frame_instant: Option<Instant>,
    pub current_window_size: winit::dpi::PhysicalSize<u32>,
    pub logger: Logger,

    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,

    lights_buffer: wgpu::Buffer,

    camera_light_bind_group: wgpu::BindGroup,

    mesh_pipeline: wgpu::RenderPipeline,
    flat_color_mesh_pipeline: wgpu::RenderPipeline,
    skybox_pipeline: wgpu::RenderPipeline,
    surface_blit_pipeline: wgpu::RenderPipeline,

    render_texture: Texture,
    depth_texture: Texture,

    skybox_texture_bind_group: wgpu::BindGroup,
    render_texture_bind_group: wgpu::BindGroup,

    // store the previous state and next state and interpolate between them
    next_balls: Vec<BallComponent>,
    prev_balls: Vec<BallComponent>,
    actual_balls: Vec<BallComponent>,

    lights: Vec<PointLightComponent>,
    test_object_transforms: Vec<super::transform::Transform>,
    plane_transforms: Vec<super::transform::Transform>,

    light_mesh: InstancedMeshComponent,
    sphere_mesh: InstancedMeshComponent,
    test_object_mesh: InstancedMeshComponent,
    plane_mesh: InstancedMeshComponent,
    skybox_mesh: MeshComponent, // TODO: always use InstancedMeshComponent

    scene: Scene,
}

impl RendererState {
    pub async fn new(window: &winit::window::Window) -> Result<Self> {
        let mut logger = Logger::new();
        // force it to vulkan to get renderdoc to work:
        // let backends = wgpu::Backends::from(wgpu::Backend::Dx12);
        let backends = wgpu::Backends::all();
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
        logger.log(&format!(
            "Using {} ({:?})",
            adapter_info.name, adapter_info.backend
        ));

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

        let swapchain_format = surface
            .get_preferred_format(&adapter)
            .expect("Window surface is incompatible with the graphics adapter");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width,
            height: size.height,
            // present_mode: wgpu::PresentMode::Fifo,
            present_mode: wgpu::PresentMode::Immediate,
        };

        surface.configure(&device, &config);

        let textured_mesh_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Textured Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(
                std::fs::read_to_string("./src/shaders/textured_mesh.wgsl")?.into(),
            ),
        });

        let flat_color_mesh_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Flat Color Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(
                std::fs::read_to_string("./src/shaders/flat_color_mesh.wgsl")?.into(),
            ),
        });

        let blit_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(
                std::fs::read_to_string("./src/shaders/blit.wgsl")?.into(),
            ),
        });

        let skybox_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
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

        let pbr_env_map_bind_group_layout =
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
                ],
                label: Some("pbr_env_map_bind_group_layout"),
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

        let fragment_shader_color_targets = &[wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        }];

        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Pipeline Layout"),
            bind_group_layouts: &[
                &five_texture_bind_group_layout,
                &two_uniform_bind_group_layout,
                &pbr_env_map_bind_group_layout,
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
                bind_group_layouts: &[&two_uniform_bind_group_layout],
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

        let suface_blit_color_targets = &[wgpu::ColorTargetState {
            format: config.format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        }];
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
                targets: suface_blit_color_targets,
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
                    &pbr_env_map_bind_group_layout,
                    &two_uniform_bind_group_layout,
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

        let equirectangular_to_cubemap_color_targets = &[wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        }];
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

        let diffuse_env_map_color_targets = &[wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        }];
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

        let specular_env_map_color_targets = &[wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        }];
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

        let brdf_lut_gen_color_targets = &[wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rg16Float,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        }];
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

        // let gltf_import_result =
        //     gltf::import("./src/models/gltf/TextureCoordinateTest/TextureCoordinateTest.gltf")?;
        // let gltf_import_result = gltf::import("./src/models/gltf/SimpleMeshes/SimpleMeshes.gltf")?;
        // let gltf_import_result = gltf::import("./src/models/gltf/Triangle/Triangle.gltf")?;
        // let gltf_import_result =
        //     gltf::import("./src/models/gltf/TriangleWithoutIndices/TriangleWithoutIndices.gltf")?;
        // let gltf_import_result = gltf::import(
        //     "./src/models/gltf/TextureLinearInterpolationTest/TextureLinearInterpolationTest.glb",
        // )?;
        let gltf_import_result = gltf::import("./src/models/gltf/Sponza/Sponza.gltf")?;
        // let gltf_import_result =
        //     gltf::import("./src/models/gltf/EnvironmentTest/EnvironmentTest.gltf")?;
        let (document, buffers, images) = gltf_import_result;
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
        // dbg!(&scene);
        // panic!("heyyyyyyy");

        let initial_render_scale = INITIAL_RENDER_SCALE;

        let sphere_mesh = BasicMesh::new("./src/models/sphere.obj")?;
        let cube_mesh = BasicMesh::new("./src/models/cube.obj")?;
        let plane_mesh = BasicMesh::new("./src/models/plane.obj")?;

        let skybox_mesh =
            MeshComponent::new(&cube_mesh, None, &single_uniform_bind_group_layout, &device)?;

        let render_texture = Texture::create_render_texture(
            &device,
            &config,
            initial_render_scale,
            "render_texture",
        );
        let render_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &single_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&render_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&render_texture.sampler),
                },
            ],
            label: Some("render_texture_bind_group"),
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

        // Mountains
        // let skybox_background = SkyboxBackground::Cube {
        //     face_image_paths: [
        //         "./src/textures/skybox/right.jpg",
        //         "./src/textures/skybox/left.jpg",
        //         "./src/textures/skybox/top.jpg",
        //         "./src/textures/skybox/bottom.jpg",
        //         "./src/textures/skybox/front.jpg",
        //         "./src/textures/skybox/back.jpg",
        //     ],
        // };
        // let skybox_hdr_environment: Option<SkyboxHDREnvironment> = None;

        // Newport Loft
        let skybox_background = SkyboxBackground::Equirectangular {
            image_path: "./src/textures/newport_loft/background.jpg",
        };
        let skybox_hdr_environment: Option<SkyboxHDREnvironment> =
            Some(SkyboxHDREnvironment::Equirectangular {
                image_path: "./src/textures/newport_loft/radiance.hdr",
            });

        // My photosphere pic
        // let skybox_background = SkyboxBackground::Equirectangular {
        //     image_path: "./src/textures/photosphere_skybox.jpg",
        // };
        // let skybox_hdr_environment: Option<SkyboxHDREnvironment> = None;

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

        let skybox_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &pbr_env_map_bind_group_layout,
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
            ],
            label: Some("skybox_texture_bind_group"),
        });

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

        let lights = vec![
            PointLightComponent {
                transform: super::transform::Transform::new(),
                color: LIGHT_COLOR_A,
            },
            PointLightComponent {
                transform: super::transform::Transform::new(),
                color: LIGHT_COLOR_B,
            },
        ];
        lights[0]
            .transform
            .set_scale(Vector3::new(0.05, 0.05, 0.05));
        lights[0]
            .transform
            .set_position(Vector3::new(0.0, 2.0, 3.0));
        lights[1].transform.set_scale(Vector3::new(0.1, 0.1, 0.1));
        lights[1]
            .transform
            .set_position(Vector3::new(0.0, 2.0, -3.0));

        let light_flat_color_instances: Vec<_> = lights
            .iter()
            .map(|light| GpuFlatColorMeshInstance::new(light.transform.matrix.get(), light.color))
            .collect();

        let light_emissive_map = Texture::from_color(
            &device,
            &queue,
            [
                (lights[0].color.x * 255.0).round() as u8,
                (lights[0].color.y * 255.0).round() as u8,
                (lights[0].color.z * 255.0).round() as u8,
                255,
            ],
            // [255, 0, 0, 255],
        )?;
        let light_metallic_roughness_map = Texture::from_color(
            &device,
            &queue,
            [255, (0.1 * 255.0f32).round() as u8, 0, 255],
        )?;
        let light_ambient_occlusion_map = Texture::from_gray(&device, &queue, 0)?;

        let light_mesh = InstancedMeshComponent::new(
            &device,
            &queue,
            &sphere_mesh,
            // TODO: InstancedMeshMaterialParams is tied to the mesh pipeline, not the flat color pipeline...
            &InstancedMeshMaterialParams {
                emissive: Some(&light_emissive_map),
                metallic_roughness: Some(&light_metallic_roughness_map),
                ambient_occlusion: Some(&light_ambient_occlusion_map),
                ..Default::default()
            },
            &five_texture_bind_group_layout,
            bytemuck::cast_slice(&light_flat_color_instances),
        )?;

        let test_object_transforms = vec![super::transform::Transform::new()];
        test_object_transforms[0].set_position(Vector3::new(0.0, 1.0, 0.0));

        let test_object_transforms_gpu: Vec<_> = test_object_transforms
            .iter()
            .map(|transform| GpuMeshInstance::new(transform.matrix.get()))
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

        let plane_transforms = vec![super::transform::Transform::new()];
        plane_transforms[0].set_scale(Vector3::new(ARENA_SIDE_LENGTH, 1.0, ARENA_SIDE_LENGTH));

        let plane_transforms_gpu: Vec<_> = plane_transforms
            .iter()
            .map(|transform| GpuMeshInstance::new(transform.matrix.get()))
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

        let camera = Camera::new((0.0, 3.0, 4.0).into());

        let camera_controller = CameraController::new(6.0, &camera);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, window);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let lights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&make_point_light_uniform_buffer(&lights)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &two_uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lights_buffer.as_entire_binding(),
                },
            ],
            label: Some("camera_light_bind_group"),
        });

        let balls: Vec<_> = (0..100)
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
            .map(|ball| GpuMeshInstance::new(ball.transform.matrix.get()))
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

        Ok(Self {
            surface,
            device,
            queue,
            config,
            render_scale: initial_render_scale,
            state_update_time_accumulator: 0.0,
            last_frame_instant: None,
            first_frame_instant: None,
            logger,
            current_window_size: size,

            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_light_bind_group,

            lights_buffer,

            mesh_pipeline,
            flat_color_mesh_pipeline,
            surface_blit_pipeline,
            skybox_pipeline,

            render_texture,
            depth_texture,

            skybox_texture_bind_group,
            render_texture_bind_group,

            next_balls: balls.clone(),
            prev_balls: balls.clone(),
            actual_balls: balls,

            lights,
            test_object_transforms,
            plane_transforms,

            light_mesh,
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
            let mut increment_render_scale = |increase: bool| {
                let delta = 0.1;
                let change = if increase { delta } else { -delta };
                self.render_scale = (self.render_scale + change).max(0.1).min(4.0);
                self.logger
                    .log(&format!("Render scale: {:?}", self.render_scale));
                self.render_texture = Texture::create_render_texture(
                    &self.device,
                    &self.config,
                    self.render_scale,
                    "render_texture",
                );
                self.render_texture_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &self.surface_blit_pipeline.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.render_texture.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    &self.render_texture.sampler,
                                ),
                            },
                        ],
                        label: Some("render_texture_bind_group"),
                    });
                self.depth_texture = Texture::create_depth_texture(
                    &self.device,
                    &self.config,
                    self.render_scale,
                    "depth_texture",
                );
            };
            if *state == ElementState::Released {
                match keycode {
                    VirtualKeyCode::Z => {
                        increment_render_scale(false);
                    }
                    VirtualKeyCode::X => {
                        increment_render_scale(true);
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
        self.render_texture = Texture::create_render_texture(
            &self.device,
            &self.config,
            self.render_scale,
            "render_texture",
        );
        // TODO: dry this up? it's repeated three times in this file
        self.render_texture_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.surface_blit_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.render_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.render_texture.sampler),
                    },
                ],
                label: Some("render_texture_bind_group"),
            });
        self.depth_texture = Texture::create_depth_texture(
            &self.device,
            &self.config,
            self.render_scale,
            "depth_texture",
        );
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

        let light_1 = &mut self.lights[0];
        light_1.transform.set_position(Vector3::new(
            1.1 * (time_seconds * 0.5).cos(),
            light_1.transform.position.get().y,
            1.1 * (time_seconds * 0.5).sin(),
        ));
        light_1.color = lerp_vec(LIGHT_COLOR_A, LIGHT_COLOR_B, (time_seconds * 2.0).sin());

        let light_2 = &mut self.lights[1];
        light_2.transform.set_position(Vector3::new(
            1.1 * (time_seconds * 0.25 + std::f32::consts::PI).cos(),
            light_2.transform.position.get().y,
            1.1 * (time_seconds * 0.25 + std::f32::consts::PI).sin(),
        ));

        light_2.color = lerp_vec(LIGHT_COLOR_B, LIGHT_COLOR_A, (time_seconds * 2.0).sin());

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

        // send data to gpu
        let balls_transforms: Vec<_> = self
            .actual_balls
            .iter()
            .map(|ball| GpuMeshInstance::new(ball.transform.matrix.get()))
            .collect();
        self.queue.write_buffer(
            &self.sphere_mesh.instance_buffer,
            0,
            bytemuck::cast_slice(&balls_transforms),
        );
        let test_object_transforms_gpu: Vec<_> = self
            .test_object_transforms
            .iter()
            .map(|transform| GpuMeshInstance::new(transform.matrix.get()))
            .collect();
        self.queue.write_buffer(
            &self.test_object_mesh.instance_buffer,
            0,
            bytemuck::cast_slice(&test_object_transforms_gpu),
        );
        self.camera_controller
            .update_camera(&mut self.camera, frame_time_seconds);
        self.camera_uniform.update_view_proj(&self.camera, window);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        let light_flat_color_instances: Vec<_> = self
            .lights
            .iter()
            .map(|light| GpuFlatColorMeshInstance::new(light.transform.matrix.get(), light.color))
            .collect();
        self.queue.write_buffer(
            &self.light_mesh.instance_buffer,
            0,
            bytemuck::cast_slice(&light_flat_color_instances),
        );

        let light_uniforms = make_point_light_uniform_buffer(&self.lights);
        self.queue.write_buffer(
            &self.lights_buffer,
            0,
            bytemuck::cast_slice(&light_uniforms),
        );
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let surface_texture = self.surface.get_current_texture()?;
        let surface_texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        let clear_color = wgpu::Color {
            r: 0.0,
            g: 0.0,
            b: 1.0,
            a: 1.0,
        };

        {
            let mut scene_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    // view: &surface_texture,
                    view: &self.render_texture.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: true,
                    },
                }],
                // depth_stencil_attachment: None,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            // render gltf scene
            scene_render_pass.set_pipeline(&self.mesh_pipeline);
            scene_render_pass.set_bind_group(1, &self.camera_light_bind_group, &[]);
            scene_render_pass.set_bind_group(2, &self.skybox_texture_bind_group, &[]);

            let meshes: Vec<_> = self.scene.source_asset.document.meshes().collect();
            let drawable_primitive_groups: Vec<_> = meshes
                .iter()
                .flat_map(|mesh| mesh.primitives().map(|prim| (&meshes[mesh.index()], prim)))
                .filter(|(_, prim)| prim.mode() == gltf::mesh::Mode::Triangles)
                .collect();

            // println!(
            //     "meshes: {:?}",
            //     meshes.iter().map(|mesh| mesh.name()).collect::<Vec<_>>()
            // );
            drawable_primitive_groups
                .iter()
                .enumerate()
                .filter(|(_, (_, prim))| {
                    prim.material().alpha_mode() == gltf::material::AlphaMode::Opaque
                })
                .for_each(|(drawable_prim_index, _)| {
                    let BindableMeshData {
                        vertex_buffer,
                        index_buffer,
                        instance_buffer,
                        textures_bind_group,
                    } = &self.scene.buffers.bindable_mesh_data[drawable_prim_index];
                    scene_render_pass.set_bind_group(0, textures_bind_group, &[]);
                    scene_render_pass.set_vertex_buffer(0, vertex_buffer.buffer.slice(..));
                    scene_render_pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));
                    match index_buffer {
                        Some(index_buffer) => {
                            // println!("Calling draw draw_indexed for mesh: {:?}", mesh.name());
                            scene_render_pass.set_index_buffer(
                                index_buffer.buffer.slice(..),
                                wgpu::IndexFormat::Uint16,
                            );
                            scene_render_pass.draw_indexed(
                                0..index_buffer.length as u32,
                                0,
                                0..instance_buffer.length as u32,
                            );
                        }
                        None => {
                            scene_render_pass.draw(
                                0..vertex_buffer.length as u32,
                                0..instance_buffer.length as u32,
                            );
                        }
                    }
                });

            // render test object
            // scene_render_pass.set_pipeline(&self.mesh_pipeline);
            // scene_render_pass.set_bind_group(0, &self.test_object_mesh.textures_bind_group, &[]);
            // scene_render_pass.set_bind_group(1, &self.camera_light_bind_group, &[]);
            // scene_render_pass.set_bind_group(2, &self.skybox_texture_bind_group, &[]);
            // scene_render_pass.set_vertex_buffer(0, self.test_object_mesh.vertex_buffer.slice(..));
            // scene_render_pass.set_vertex_buffer(1, self.test_object_mesh.instance_buffer.slice(..));
            // scene_render_pass.set_index_buffer(
            //     self.test_object_mesh.index_buffer.slice(..),
            //     wgpu::IndexFormat::Uint16,
            // );
            // scene_render_pass.draw_indexed(
            //     0..self.test_object_mesh.num_indices,
            //     0,
            //     0..self.test_object_transforms.len() as u32,
            // );

            // // render floor
            // scene_render_pass.set_pipeline(&self.mesh_pipeline);
            // scene_render_pass.set_bind_group(0, &self.plane_mesh.textures_bind_group, &[]);
            // scene_render_pass.set_bind_group(1, &self.camera_light_bind_group, &[]);
            // scene_render_pass.set_vertex_buffer(0, self.plane_mesh.vertex_buffer.slice(..));
            // scene_render_pass.set_vertex_buffer(1, self.plane_mesh.instance_buffer.slice(..));
            // scene_render_pass.set_index_buffer(
            //     self.plane_mesh.index_buffer.slice(..),
            //     wgpu::IndexFormat::Uint16,
            // );
            // scene_render_pass.draw_indexed(
            //     0..self.plane_mesh.num_indices,
            //     0,
            //     0..self.plane_transforms.len() as u32,
            // );

            // render balls
            scene_render_pass.set_pipeline(&self.mesh_pipeline);
            scene_render_pass.set_bind_group(0, &self.sphere_mesh.textures_bind_group, &[]);
            scene_render_pass.set_bind_group(1, &self.camera_light_bind_group, &[]);
            scene_render_pass.set_vertex_buffer(0, self.sphere_mesh.vertex_buffer.slice(..));
            scene_render_pass.set_vertex_buffer(1, self.sphere_mesh.instance_buffer.slice(..));
            scene_render_pass.set_index_buffer(
                self.sphere_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            scene_render_pass.draw_indexed(
                0..self.sphere_mesh.num_indices,
                0,
                0..self.actual_balls.len() as u32,
            );

            // render light
            scene_render_pass.set_pipeline(&self.flat_color_mesh_pipeline);
            scene_render_pass.set_bind_group(0, &self.camera_light_bind_group, &[]);
            scene_render_pass.set_vertex_buffer(0, self.light_mesh.vertex_buffer.slice(..));
            scene_render_pass.set_vertex_buffer(1, self.light_mesh.instance_buffer.slice(..));
            scene_render_pass.set_index_buffer(
                self.light_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            scene_render_pass.draw_indexed(
                0..self.light_mesh.num_indices,
                0,
                0..self.lights.len() as u32,
            );

            // render skybox
            // TODO: does it make sense to render the skybox here?
            // doing it in the surface blit pass is faster and might not change the quality when using SSAA
            scene_render_pass.set_pipeline(&self.skybox_pipeline);
            scene_render_pass.set_bind_group(0, &self.skybox_texture_bind_group, &[]);
            scene_render_pass.set_bind_group(1, &self.camera_light_bind_group, &[]);
            scene_render_pass.set_vertex_buffer(0, self.skybox_mesh.vertex_buffer.slice(..));
            scene_render_pass.set_index_buffer(
                self.skybox_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            scene_render_pass.draw_indexed(0..self.skybox_mesh.num_indices, 0, 0..1);
        }

        {
            let mut surface_blit_render_pass =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[wgpu::RenderPassColorAttachment {
                        view: &surface_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: true,
                        },
                    }],
                    depth_stencil_attachment: None,
                });

            surface_blit_render_pass.set_pipeline(&self.surface_blit_pipeline);
            surface_blit_render_pass.set_bind_group(0, &self.render_texture_bind_group, &[]);
            surface_blit_render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
        Ok(())
    }
}
