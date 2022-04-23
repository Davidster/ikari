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
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position: [f32; 4],
}

impl From<Vector3<f32>> for LightUniform {
    fn from(v: Vector3<f32>) -> Self {
        Self {
            position: [v.x, v.y, v.z, 1.0],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    rotation_only_view: [[f32; 4]; 4],
    near_plane_distance: f32,
    far_plane_distance: f32,
    padding: [f32; 2],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            proj: Matrix4::one().into(),
            view: Matrix4::one().into(),
            rotation_only_view: Matrix4::one().into(),
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
        } = camera.build_view_projection_matrices(window);
        self.proj = proj.into();
        self.view = view.into();
        self.rotation_only_view = rotation_only_view.into();
    }
}

pub const ARENA_SIDE_LENGTH: f32 = 1000.0;
pub const USE_PHOTOSPHERE_SKYBOX: bool = false;

pub struct RendererState {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_scale: f32,
    state_update_time_accumulator: f32,
    last_frame_instant: Option<Instant>,
    first_frame_instant: Option<Instant>,
    pub rendered_first_frame: bool,
    pub current_window_size: winit::dpi::PhysicalSize<u32>,
    pub logger: Logger,

    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,

    light: MeshComponent,
    light_buffer: wgpu::Buffer,

    camera_light_bind_group: wgpu::BindGroup,

    textured_mesh_pipeline: wgpu::RenderPipeline,
    normal_mapped_mesh_pipeline: wgpu::RenderPipeline,
    instanced_mesh_pipeline: wgpu::RenderPipeline,
    flat_color_mesh_pipeline: wgpu::RenderPipeline,

    skybox_pipeline: wgpu::RenderPipeline,
    surface_blit_pipeline: wgpu::RenderPipeline,

    skybox_texture: Texture,
    render_texture: Texture,
    depth_texture: Texture,
    // store the previous state and next state and interpolate between them
    next_balls: Vec<BallComponent>,
    prev_balls: Vec<BallComponent>,
    actual_balls: Vec<BallComponent>,

    balls_mesh: InstancedMeshComponent,
    test_object: MeshComponent,
    test_object_normal_map: Texture,
    test_object_textures_bind_group: wgpu::BindGroup,
    floor: MeshComponent,
    skybox_mesh: MeshComponent,
}

impl RendererState {
    pub async fn new(window: &winit::window::Window) -> Result<Self> {
        let mut logger = Logger::new();
        // force it to vulkan to get renderdoc to work:
        let backends = wgpu::Backends::from(wgpu::Backend::Vulkan);
        // let backends = wgpu::Backends::all();
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
            present_mode: wgpu::PresentMode::Fifo,
            // present_mode: wgpu::PresentMode::Immediate,
        };

        surface.configure(&device, &config);

        let textured_mesh_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Textured Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(
                std::fs::read_to_string("./src/textured_mesh_shader.wgsl")?.into(),
            ),
        });

        let flat_color_mesh_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Flat Color Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(
                std::fs::read_to_string("./src/flat_color_mesh_shader.wgsl")?.into(),
            ),
        });

        let blit_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(std::fs::read_to_string("./src/blit.wgsl")?.into()),
        });

        let skybox_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Skybox Shader"),
            source: wgpu::ShaderSource::Wgsl(
                std::fs::read_to_string("./src/skybox_shader.wgsl")?.into(),
            ),
        });

        let photosphere_skybox_shader =
            device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: Some("Photosphere Skybox Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    std::fs::read_to_string("./src/photosphere_skybox_shader.wgsl")?.into(),
                ),
            });

        // TODO: make this be a global variable for the renderer, so everyone can read it
        // and it doesn't get duplicated in mesh.rs
        let diffuse_texture_bind_group_layout =
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
                label: Some("diffuse_texture_bind_group_layout"),
            });
        let render_texture_bind_group_layout =
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
                        // TODO: what difference does this param make? can i set it to NonFiltering?
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("render_texture_bind_group_layout"),
            });
        let normal_mapped_bind_group_layout =
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
                label: Some("normal_mapped_bind_group_layout"),
            });

        // wgpu::BindGroupLayoutEntry {
        //     binding: 0,
        //     visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        //     ty: wgpu::BindingType::Buffer {
        //         ty: wgpu::BufferBindingType::Uniform,
        //         has_dynamic_offset: false,
        //         min_binding_size: None,
        //     },
        //     count: None,
        // },
        let skybox_texture_bind_group_layout =
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
                label: Some("skybox_texture_bind_group_layout"),
            });
        let photosphere_skybox_texture_bind_group_layout =
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
                label: Some("photosphere_skybox_texture_bind_group_layout"),
            });

        let model_trans_uniform_var_bind_group_layout =
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
                label: Some("uniform_var_bind_group_layout"),
            });

        let camera_light_uniform_var_bind_group_layout =
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
                label: Some("camera_light_uniform_var_bind_group_layout"),
            });

        let normal_mapped_mesh_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Normal Mapped Textured Mesh Render Pipeline Layout"),
                bind_group_layouts: &[
                    &normal_mapped_bind_group_layout,
                    &camera_light_uniform_var_bind_group_layout,
                    &model_trans_uniform_var_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let textured_mesh_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Textured Mesh Render Pipeline Layout"),
                bind_group_layouts: &[
                    &diffuse_texture_bind_group_layout,
                    &camera_light_uniform_var_bind_group_layout,
                    &model_trans_uniform_var_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let flat_color_mesh_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Flat Color Mesh Render Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_light_uniform_var_bind_group_layout,
                    &model_trans_uniform_var_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let fragment_shader_color_targets = &[wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        }];
        let textured_mesh_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Textured Mesh Render Pipeline"),
            layout: Some(&textured_mesh_render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &textured_mesh_shader,
                entry_point: "vs_main",
                buffers: &[TexturedVertex::desc()],
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

        // TODO: DRY
        let normal_mapped_mesh_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Normal Mapped Mesh Render Pipeline"),
            layout: Some(&normal_mapped_mesh_render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &textured_mesh_shader,
                entry_point: "vs_main",
                buffers: &[TexturedVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &textured_mesh_shader,
                entry_point: "normal_mapped_fs_main",
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

        // TODO: DRY
        let flat_color_mesh_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Flat Color Mesh Render Pipeline"),
            layout: Some(&flat_color_mesh_render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &flat_color_mesh_shader,
                entry_point: "vs_main",
                buffers: &[TexturedVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &flat_color_mesh_shader,
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

        // instanced pipeline is very similar to non-instanced with a few differences:
        let mut instanced_mesh_pipeline_descriptor = textured_mesh_pipeline_descriptor.clone();
        let instanced_mesh_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Instanced Mesh Render Pipeline Layout"),
                bind_group_layouts: &[
                    &diffuse_texture_bind_group_layout,
                    &camera_light_uniform_var_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let instanced_mesh_pipeline_vertex_buffers_layout =
            &[TexturedVertex::desc(), GpuMeshInstance::desc()];

        instanced_mesh_pipeline_descriptor.label = Some("Instanced Mesh Render Pipeline");
        instanced_mesh_pipeline_descriptor.layout = Some(&instanced_mesh_render_pipeline_layout);
        // instanced_mesh_pipeline_descriptor.vertex.entry_point = "vs_main";
        instanced_mesh_pipeline_descriptor.vertex.entry_point = "instanced_vs_main";
        instanced_mesh_pipeline_descriptor.vertex.buffers =
            instanced_mesh_pipeline_vertex_buffers_layout;

        let suface_blit_color_targets = &[wgpu::ColorTargetState {
            format: config.format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        }];
        let surface_blit_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&render_texture_bind_group_layout],
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

        let textured_mesh_pipeline =
            device.create_render_pipeline(&textured_mesh_pipeline_descriptor);
        let flat_color_mesh_pipeline =
            device.create_render_pipeline(&flat_color_mesh_pipeline_descriptor);
        let normal_mapped_mesh_pipeline =
            device.create_render_pipeline(&normal_mapped_mesh_pipeline_descriptor);
        let instanced_mesh_pipeline =
            device.create_render_pipeline(&instanced_mesh_pipeline_descriptor);
        let surface_blit_pipeline =
            device.create_render_pipeline(&surface_blit_pipeline_descriptor);

        let render_scale = 2.0;

        let render_texture =
            Texture::create_render_texture(&device, &config, render_scale, "render_texture");
        let depth_texture =
            Texture::create_depth_texture(&device, &config, render_scale, "depth_texture");

        // source: https://www.solarsystemscope.com/textures/
        let mars_texture_path = "./src/8k_mars.png";
        let mars_texture_bytes = std::fs::read(mars_texture_path)?;
        let mars_texture = Texture::from_bytes(
            &device,
            &queue,
            &mars_texture_bytes,
            mars_texture_path,
            true,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;

        let earth_texture_path = "./src/8k_earth.png";
        let earth_texture_bytes = std::fs::read(earth_texture_path)?;
        let earth_texture = Texture::from_bytes(
            &device,
            &queue,
            &earth_texture_bytes,
            earth_texture_path,
            true,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;

        let earth_normal_map_path = "./src/8k_earth_normal_map.png";
        let earth_normal_map_bytes = std::fs::read(earth_normal_map_path)?;
        let earth_normal_map = Texture::from_bytes(
            &device,
            &queue,
            &earth_normal_map_bytes,
            earth_normal_map_path,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            wgpu::TextureFormat::Rgba8Unorm.into(),
        )?;

        let simple_normal_map_path = "./src/simple_normal_map.png";
        let simple_normal_map_bytes = std::fs::read(simple_normal_map_path)?;
        let simple_normal_map = Texture::from_bytes(
            &device,
            &queue,
            &simple_normal_map_bytes,
            simple_normal_map_path,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            wgpu::TextureFormat::Rgba8Unorm.into(),
        )?;

        let skybox_fragment_shader_color_targets = &[wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        }];

        let (skybox_pipeline, skybox_texture) = if USE_PHOTOSPHERE_SKYBOX {
            let photosphere_skybox_render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Photosphere Skybox Render Pipeline Layout"),
                    bind_group_layouts: &[
                        &photosphere_skybox_texture_bind_group_layout,
                        &camera_light_uniform_var_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

            let photosphere_skybox_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
                label: Some("Photosphere Skybox Render Pipeline"),
                layout: Some(&photosphere_skybox_render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &photosphere_skybox_shader,
                    entry_point: "vs_main",
                    buffers: &[TexturedVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &photosphere_skybox_shader,
                    entry_point: "fs_main",
                    targets: skybox_fragment_shader_color_targets,
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: wgpu::FrontFace::Cw,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Texture::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    // TODO: should this be LessEqual?
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            };

            let photosphere_skybox_pipeline =
                device.create_render_pipeline(&photosphere_skybox_pipeline_descriptor);
            let photosphere_skybox_texture_path = "./src/photosphere_skybox.png";
            let photosphere_skybox_texture_bytes = std::fs::read(photosphere_skybox_texture_path)?;
            let photosphere_skybox_texture = Texture::from_bytes(
                &device,
                &queue,
                &photosphere_skybox_texture_bytes,
                photosphere_skybox_texture_path,
                false, // an artifact occurs between the edges of the texture with mipmaps enabled
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )?;

            (photosphere_skybox_pipeline, photosphere_skybox_texture)
        } else {
            let skybox_render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Skybox Render Pipeline Layout"),
                    bind_group_layouts: &[
                        &skybox_texture_bind_group_layout,
                        &camera_light_uniform_var_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

            let skybox_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
                label: Some("Skybox Render Pipeline"),
                layout: Some(&skybox_render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &skybox_shader,
                    entry_point: "vs_main",
                    buffers: &[TexturedVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &skybox_shader,
                    entry_point: "fs_main",
                    targets: skybox_fragment_shader_color_targets,
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: wgpu::FrontFace::Cw,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Texture::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    // TODO: should this be LessEqual?
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            };
            let skybox_pipeline = device.create_render_pipeline(&skybox_pipeline_descriptor);

            let cubemap_skybox_images = vec![
                "./src/skybox/right.png",
                "./src/skybox/left.png",
                "./src/skybox/top.png",
                "./src/skybox/bottom.png",
                "./src/skybox/front.png",
                "./src/skybox/back.png",
            ]
            .iter()
            .map(|path| image::load_from_memory(&std::fs::read(path)?))
            .collect::<Result<Vec<_>, _>>()?;
            let cubemap_skybox_texture = Texture::create_cubemap_texture(
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
                Some("skybox_texture"),
                // TODO: set to true!
                false,
            )?;

            (skybox_pipeline, cubemap_skybox_texture)
        };

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
                            [0, 0, 0, 255].into()
                        } else {
                            [255, 255, 255, 255].into()
                        },
                    );
                }
            }
            image::DynamicImage::ImageRgba8(img)
        };
        let checkerboard_texture = Texture::from_image(
            &device,
            &queue,
            &checkerboard_texture_img,
            Some("checkerboard_texture"),
            true,
            None,
            None,
            None,
            wgpu::FilterMode::Nearest.into(),
            None,
            None,
            None,
        )?;

        let sphere_mesh = BasicMesh::new("./src/sphere.obj")?;
        let cube_mesh = BasicMesh::new("./src/cube.obj")?;
        let plane_mesh = BasicMesh::new("./src/plane.obj")?;

        let skybox_mesh = MeshComponent::new(
            &cube_mesh,
            None,
            &model_trans_uniform_var_bind_group_layout,
            &device,
        )?;

        let light = MeshComponent::new(
            &sphere_mesh,
            None,
            &model_trans_uniform_var_bind_group_layout,
            &device,
        )?;
        light.transform.set_scale(Vector3::new(0.05, 0.05, 0.05));
        light.transform.set_position(Vector3::new(0.0, 2.0, 0.0));

        let test_object = MeshComponent::new(
            &sphere_mesh,
            Some(&earth_texture),
            &model_trans_uniform_var_bind_group_layout,
            &device,
        )?;

        // sphere.transform.set_scale(Vector3::new(0.25, 0.25, 0.25));
        test_object
            .transform
            .set_position(Vector3::new(0.0, 1.0, 0.0));
        // let rotational_displacement =
        //     make_quat_from_axis_angle(Vector3::new(0.0, 1.0, 0.0), frame_time_seconds);
        // let new_rotation = rotational_displacement * self.test_object.transform.rotation.get();
        // self.sphere.transform.set_rotation(new_rotation);
        // let rotational_displacement = make_quat_from_axis_angle(
        //     Vector3::new(1.0, 0.0, 0.0),
        //     Rad(-std::f32::consts::PI / 2.0),
        // );
        // test_object
        //     .transform
        //     .set_rotation(rotational_displacement * test_object.transform.rotation.get());

        queue.write_buffer(
            &test_object.transform_buffer,
            0,
            bytemuck::cast_slice(&[GpuMatrix4(test_object.transform.matrix.get())]),
        );
        queue.write_buffer(
            &test_object.normal_rotation_buffer,
            0,
            bytemuck::cast_slice(&[GpuMatrix4(test_object.transform.get_rotation_matrix())]),
        );

        let floor = MeshComponent::new(
            &plane_mesh,
            Some(&checkerboard_texture),
            &model_trans_uniform_var_bind_group_layout,
            &device,
        )?;

        floor.transform.set_position(Vector3::new(0.0, 0.0, 0.0));
        floor
            .transform
            .set_scale(Vector3::new(ARENA_SIDE_LENGTH, 1.0, ARENA_SIDE_LENGTH));

        queue.write_buffer(
            &floor.transform_buffer,
            0,
            bytemuck::cast_slice(&[GpuMatrix4(floor.transform.matrix.get())]),
        );
        queue.write_buffer(
            &floor.normal_rotation_buffer,
            0,
            bytemuck::cast_slice(&[GpuMatrix4(floor.transform.get_rotation_matrix())]),
        );

        let camera = Camera::new((0.0, 20.0, 7.0).into());

        let camera_controller = CameraController::new(6.0, &camera);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, window);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&[LightUniform::from(light.transform.position.get())]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_light_uniform_var_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffer.as_entire_binding(),
                },
            ],
            label: Some("camera_light_bind_group"),
        });

        let test_object_textures_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &normal_mapped_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&earth_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&earth_texture.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&earth_normal_map.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&earth_normal_map.sampler),
                    },
                ],
                label: Some("test_object_textures_bind_group"),
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

        let balls_mesh = InstancedMeshComponent::new(
            &sphere_mesh,
            Some(&mars_texture),
            &model_trans_uniform_var_bind_group_layout,
            &device,
            &balls_transforms,
        )?;

        Ok(Self {
            surface,
            device,
            queue,
            config,
            render_scale,
            state_update_time_accumulator: 0.0,
            last_frame_instant: None,
            first_frame_instant: None,
            rendered_first_frame: false,
            logger,
            current_window_size: size,

            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_light_bind_group,

            light,
            light_buffer,

            textured_mesh_pipeline,
            flat_color_mesh_pipeline,
            normal_mapped_mesh_pipeline,
            instanced_mesh_pipeline,
            surface_blit_pipeline,
            skybox_pipeline,
            skybox_texture,
            render_texture,
            depth_texture,

            next_balls: balls.clone(),
            prev_balls: balls.clone(),
            actual_balls: balls,

            balls_mesh,
            test_object,
            test_object_normal_map: simple_normal_map,
            test_object_textures_bind_group,
            floor,
            skybox_mesh,
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
        self.depth_texture = Texture::create_depth_texture(
            &self.device,
            &self.config,
            self.render_scale,
            "depth_texture",
        );
    }

    pub fn update(&mut self, window: &winit::window::Window) {
        let first_frame_instant = self.first_frame_instant.unwrap_or_else(|| Instant::now());
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

        self.light.transform.set_position(Vector3::new(
            1.05 * time_seconds.cos(),
            self.light.transform.position.get().y,
            1.05 * time_seconds.sin(),
        ));
        let rotational_displacement =
            make_quat_from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Rad(frame_time_seconds / 5.0));
        self.test_object
            .transform
            .set_rotation(rotational_displacement * self.test_object.transform.rotation.get());

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
            &self.balls_mesh.instance_buffer,
            0,
            bytemuck::cast_slice(&balls_transforms),
        );
        self.queue.write_buffer(
            &self.test_object.transform_buffer,
            0,
            bytemuck::cast_slice(&[GpuMatrix4(self.test_object.transform.matrix.get())]),
        );
        self.queue.write_buffer(
            &self.test_object.normal_rotation_buffer,
            0,
            bytemuck::cast_slice(&[GpuMatrix4(self.test_object.transform.get_rotation_matrix())]),
        );
        self.camera_controller
            .update_camera(&mut self.camera, frame_time_seconds);
        self.camera_uniform.update_view_proj(&self.camera, window);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        self.queue.write_buffer(
            &self.light.transform_buffer,
            0,
            bytemuck::cast_slice(&[GpuMatrix4(self.light.transform.matrix.get())]),
        );
        self.queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[LightUniform::from(self.light.transform.position.get())]),
        );
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let surface_texture = self.surface.get_current_texture()?;
        let surface_texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // TODO: can I move these bind groups elsewhere?
        let sky_box_texture_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.skybox_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.skybox_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.skybox_texture.sampler),
                    },
                ],
                label: Some("skybox_texture_bind_group"),
            });
        let render_texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            // render test object
            scene_render_pass.set_pipeline(&self.normal_mapped_mesh_pipeline);
            scene_render_pass.set_bind_group(0, &self.test_object_textures_bind_group, &[]);
            scene_render_pass.set_bind_group(1, &self.camera_light_bind_group, &[]);
            scene_render_pass.set_bind_group(2, &self.test_object.transform_bind_group, &[]);
            scene_render_pass.set_bind_group(3, &self.test_object.normal_rotation_bind_group, &[]);

            scene_render_pass.set_vertex_buffer(0, self.test_object.vertex_buffer.slice(..));
            scene_render_pass.set_index_buffer(
                self.test_object.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            scene_render_pass.draw_indexed(0..self.test_object.num_indices, 0, 0..1);

            // render floor
            scene_render_pass.set_pipeline(&self.textured_mesh_pipeline);
            if let Some(diffuse_texture_bind_group) = &self.floor.diffuse_texture_bind_group {
                scene_render_pass.set_bind_group(0, diffuse_texture_bind_group, &[]);
            }
            scene_render_pass.set_bind_group(1, &self.camera_light_bind_group, &[]);
            scene_render_pass.set_bind_group(2, &self.floor.transform_bind_group, &[]);
            scene_render_pass.set_bind_group(3, &self.floor.normal_rotation_bind_group, &[]);
            scene_render_pass.set_vertex_buffer(0, self.floor.vertex_buffer.slice(..));
            scene_render_pass
                .set_index_buffer(self.floor.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            scene_render_pass.draw_indexed(0..self.floor.num_indices, 0, 0..1);

            // render balls
            scene_render_pass.set_pipeline(&self.instanced_mesh_pipeline);
            scene_render_pass.set_bind_group(1, &self.camera_light_bind_group, &[]);
            if let Some(diffuse_texture_bind_group) = &self.balls_mesh.diffuse_texture_bind_group {
                scene_render_pass.set_bind_group(0, diffuse_texture_bind_group, &[]);
            }
            scene_render_pass.set_vertex_buffer(0, self.balls_mesh.vertex_buffer.slice(..));
            scene_render_pass.set_vertex_buffer(1, self.balls_mesh.instance_buffer.slice(..));
            scene_render_pass.set_index_buffer(
                self.balls_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            scene_render_pass.draw_indexed(
                0..self.balls_mesh.num_indices,
                0,
                0..self.actual_balls.len() as u32,
            );

            // render light
            scene_render_pass.set_pipeline(&self.flat_color_mesh_pipeline);
            scene_render_pass.set_bind_group(0, &self.camera_light_bind_group, &[]);
            scene_render_pass.set_bind_group(1, &self.light.transform_bind_group, &[]);
            scene_render_pass.set_vertex_buffer(0, self.light.vertex_buffer.slice(..));
            scene_render_pass
                .set_index_buffer(self.light.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            scene_render_pass.draw_indexed(0..self.light.num_indices, 0, 0..1);

            // render skybox
            // TODO: does it make sense to render the skybox here?
            // doing it in the surface blit pass is faster and might not change the quality when using SSAA
            scene_render_pass.set_pipeline(&self.skybox_pipeline);
            scene_render_pass.set_bind_group(0, &sky_box_texture_bind_group, &[]);
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
            surface_blit_render_pass.set_bind_group(0, &render_texture_bind_group, &[]);
            surface_blit_render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
        Ok(())
    }
}
