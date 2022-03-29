use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use super::*;

use anyhow::Result;

use cgmath::{Matrix4, One, Vector2, Vector3};
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
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: Matrix4::one().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera, window: &winit::window::Window) {
        self.view_proj = camera.build_view_projection_matrix(&window).into();
    }
}

pub const ARENA_SIDE_LENGTH: f32 = 100.0;

pub struct RendererState {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    last_update_time: Option<Instant>,
    pub rendered_first_frame: bool,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub logger: Logger,

    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,

    textured_mesh_pipeline: wgpu::RenderPipeline,
    instanced_mesh_pipeline: wgpu::RenderPipeline,
    depth_texture: Texture,

    balls: Vec<BallComponent>,
    balls_mesh: InstancedMeshComponent,
    sphere: MeshComponent,
    plane: MeshComponent,
}

impl RendererState {
    pub async fn new(window: &winit::window::Window) -> Result<Self> {
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
        println!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

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
            source: wgpu::ShaderSource::Wgsl(include_str!("textured_mesh_shader.wgsl").into()),
        });

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

        let uniform_var_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_var_bind_group_layout"),
            });

        let textured_mesh_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Textured Mesh Render Pipeline Layout"),
                bind_group_layouts: &[
                    &diffuse_texture_bind_group_layout,
                    &uniform_var_bind_group_layout,
                    &uniform_var_bind_group_layout,
                    &uniform_var_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let fragment_shader_color_targets = &[wgpu::ColorTargetState {
            format: config.format,
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

        // instanced pipeline is very similar to non-instanced with a few differences:
        let mut instanced_mesh_pipeline_descriptor = textured_mesh_pipeline_descriptor.clone();
        let instanced_mesh_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Instanced Mesh Render Pipeline Layout"),
                bind_group_layouts: &[
                    &diffuse_texture_bind_group_layout,
                    &uniform_var_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let instanced_mesh_pipeline_vertex_buffers_layout =
            &[TexturedVertex::desc(), GpuMeshInstance::desc()];

        instanced_mesh_pipeline_descriptor.label = Some("Instanced Mesh Render Pipeline");
        instanced_mesh_pipeline_descriptor.layout = Some(&instanced_mesh_render_pipeline_layout);
        instanced_mesh_pipeline_descriptor.vertex.entry_point = "instanced_vs_main";
        instanced_mesh_pipeline_descriptor.vertex.buffers =
            instanced_mesh_pipeline_vertex_buffers_layout;

        let textured_mesh_pipeline =
            device.create_render_pipeline(&textured_mesh_pipeline_descriptor);

        let instanced_mesh_pipeline =
            device.create_render_pipeline(&instanced_mesh_pipeline_descriptor);

        let depth_texture = Texture::create_depth_texture(&device, &config, "depth_texture");

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
        )?;
        let sphere_mesh = BasicMesh::new("./src/sphere.obj")?;

        let sphere = MeshComponent::new(
            &sphere_mesh,
            &mars_texture,
            &diffuse_texture_bind_group_layout,
            &uniform_var_bind_group_layout,
            &device,
        )?;

        sphere.transform.set_scale(Vector3::new(0.25, 0.25, 0.25));
        sphere.transform.set_position(Vector3::new(0.0, 1.0, -2.0));

        queue.write_buffer(
            &sphere.transform_buffer,
            0,
            bytemuck::cast_slice(&[GpuMatrix4(sphere.transform.matrix.get())]),
        );
        queue.write_buffer(
            &sphere.normal_rotation_buffer,
            0,
            bytemuck::cast_slice(&[GpuMatrix4(sphere.transform.get_rotation_matrix())]),
        );

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
            wgpu::FilterMode::Nearest.into(),
            None,
            None,
        )?;
        let plane_mesh = BasicMesh::new("./src/plane.obj")?;

        let plane = MeshComponent::new(
            &plane_mesh,
            &checkerboard_texture,
            &diffuse_texture_bind_group_layout,
            &uniform_var_bind_group_layout,
            &device,
        )?;

        plane.transform.set_position(Vector3::new(0.0, 0.0, 0.0));
        plane
            .transform
            .set_scale(Vector3::new(ARENA_SIDE_LENGTH, 1.0, ARENA_SIDE_LENGTH));

        queue.write_buffer(
            &plane.transform_buffer,
            0,
            bytemuck::cast_slice(&[GpuMatrix4(plane.transform.matrix.get())]),
        );
        queue.write_buffer(
            &plane.normal_rotation_buffer,
            0,
            bytemuck::cast_slice(&[GpuMatrix4(plane.transform.get_rotation_matrix())]),
        );

        let camera = Camera::new((0.0, 2.0, 7.0).into());

        let camera_controller = CameraController::new(0.1, &camera);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &window);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_var_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let balls: Vec<_> = (0..0)
            .into_iter()
            .map(|_| {
                BallComponent::new(
                    Vector2::new(0.0, 0.0),
                    Vector2::new(
                        -1.0 + rand::random::<f32>() * 2.0,
                        -1.0 + rand::random::<f32>() * 2.0,
                    ),
                    0.5 + (rand::random::<f32>() * 0.75),
                    0.025 + (rand::random::<f32>() * 0.2),
                )
            })
            .collect();

        let balls_transforms: Vec<_> = balls
            .iter()
            .map(|ball| GpuMeshInstance::new(ball.transform.matrix.get()))
            .collect();

        let balls_mesh = InstancedMeshComponent::new(
            &sphere_mesh,
            &mars_texture,
            &diffuse_texture_bind_group_layout,
            &uniform_var_bind_group_layout,
            &device,
            &balls_transforms,
        )?;

        Ok(Self {
            surface,
            device,
            queue,
            config,
            last_update_time: None,
            rendered_first_frame: false,
            size,
            logger: Logger::new(),

            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,

            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,

            textured_mesh_pipeline,
            instanced_mesh_pipeline,
            depth_texture,

            balls,
            balls_mesh,
            sphere,
            plane,
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
        self.camera_controller
            .process_window_events(event, window, &mut self.logger);
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_texture =
            Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
    }

    pub fn update(&mut self, window: &winit::window::Window) {
        let one_milli_in_nanos = 1_000_000.0;
        let sixty_fps_frame_time_nanos = one_milli_in_nanos * 1_000.0 / 60.0;
        let dt = if let Some(last_update_time) = self.last_update_time {
            (last_update_time.elapsed().as_nanos() as f32) / sixty_fps_frame_time_nanos
        } else {
            0.0
        };
        // self.logger.log(&format!("dt: {:?}", dt));
        self.balls
            .iter_mut()
            .for_each(|ball| ball.update(dt, &mut self.logger));
        let balls_transforms: Vec<_> = self
            .balls
            .iter()
            .map(|ball| GpuMeshInstance::new(ball.transform.matrix.get()))
            .collect();
        self.queue.write_buffer(
            &self.balls_mesh.instance_buffer,
            0,
            bytemuck::cast_slice(&balls_transforms),
        );
        self.queue.write_buffer(
            &self.sphere.transform_buffer,
            0,
            bytemuck::cast_slice(&[GpuMatrix4(self.sphere.transform.matrix.get())]),
        );
        self.queue.write_buffer(
            &self.sphere.normal_rotation_buffer,
            0,
            bytemuck::cast_slice(&[GpuMatrix4(self.sphere.transform.get_rotation_matrix())]),
        );
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform.update_view_proj(&self.camera, window);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        self.last_update_time = Some(Instant::now());
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
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
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(&self.textured_mesh_pipeline);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);

            vec![&self.sphere, &self.plane]
                .iter()
                .map(|mesh| *mesh)
                .for_each(
                    |MeshComponent {
                         diffuse_texture_bind_group,
                         vertex_buffer,
                         index_buffer,
                         num_indices,
                         transform_bind_group,
                         normal_rotation_bind_group,
                         ..
                     }| {
                        render_pass.set_bind_group(0, diffuse_texture_bind_group, &[]);
                        render_pass.set_bind_group(2, transform_bind_group, &[]);
                        render_pass.set_bind_group(3, normal_rotation_bind_group, &[]);
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        render_pass
                            .set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                        render_pass.draw_indexed(0..*num_indices, 0, 0..1);
                    },
                );

            render_pass.set_pipeline(&self.instanced_mesh_pipeline);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(0, &self.balls_mesh.diffuse_texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.balls_mesh.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.balls_mesh.instance_buffer.slice(..));
            render_pass.set_index_buffer(
                self.balls_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.draw_indexed(
                0..self.balls_mesh.num_indices,
                0,
                0..self.balls.len() as u32,
            );
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}
