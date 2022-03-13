use std::{collections::HashMap, time::Duration};

use super::*;

use anyhow::Result;

use cgmath::{Matrix4, One, Vector3};
use wgpu::util::DeviceExt;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};

type VertexPosition = [f32; 3];
type VertexNormal = [f32; 3];
type VertexTextureCoords = [f32; 2];

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TexturedVertex {
    position: VertexPosition,
    normal: VertexNormal,
    tex_coords: VertexTextureCoords,
}

impl TexturedVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TexturedVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

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

#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
struct GpuMatrix4(cgmath::Matrix4<f32>);

unsafe impl bytemuck::Pod for GpuMatrix4 {}
unsafe impl bytemuck::Zeroable for GpuMatrix4 {}

#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
struct GpuMatrix3(cgmath::Matrix3<f32>);

unsafe impl bytemuck::Pod for GpuMatrix3 {}
unsafe impl bytemuck::Zeroable for GpuMatrix3 {}

struct MeshComponent {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    _num_vertices: u32,
    _diffuse_texture: Texture,
    diffuse_texture_bind_group: wgpu::BindGroup,
    transform_buffer: wgpu::Buffer,
    transform_bind_group: wgpu::BindGroup,
    normal_rotation_buffer: wgpu::Buffer,
    normal_rotation_bind_group: wgpu::BindGroup,
    transform: super::transform::Transform,
}

pub struct RendererState {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
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
    depth_texture: Texture,

    sphere: MeshComponent,
    plane: MeshComponent,
}

impl MeshComponent {
    pub fn new(
        obj_file_path: &str,
        diffuse_texture_file_path: &str,
        diffuse_texture_bind_group_layout: &wgpu::BindGroupLayout,
        uniform_var_bind_group_layout: &wgpu::BindGroupLayout,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<MeshComponent> {
        let diffuse_texture_bytes = std::fs::read(diffuse_texture_file_path)?;
        let obj_file_string = std::fs::read_to_string(obj_file_path)?;

        let obj = wavefront_obj::obj::parse(obj_file_string)?
            .objects
            .remove(0);

        let vt_indices: Vec<wavefront_obj::obj::VTNIndex> = obj
            .geometry
            .iter()
            .flat_map(|geometry| -> Vec<wavefront_obj::obj::VTNIndex> {
                geometry
                    .shapes
                    .iter()
                    .flat_map(|shape| {
                        if let wavefront_obj::obj::Primitive::Triangle(vti1, vti2, vti3) =
                            shape.primitive
                        {
                            vec![vti1, vti2, vti3]
                        } else {
                            vec![]
                        }
                    })
                    .collect()
            })
            .collect();
        let mut composite_index_map: HashMap<(usize, usize, usize), TexturedVertex> =
            HashMap::new();
        vt_indices.iter().for_each(|vti| {
            let pos_index = vti.0;
            let normal_index = vti.2.expect("Obj file is missing normal");
            let uv_index = vti.1.expect("Obj file is missing uv index");
            let key = (pos_index, normal_index, uv_index);
            if !composite_index_map.contains_key(&key) {
                let wavefront_obj::obj::Vertex {
                    x: p_x,
                    y: p_y,
                    z: p_z,
                } = obj.vertices[pos_index];
                let wavefront_obj::obj::Normal {
                    x: n_x,
                    y: n_y,
                    z: n_z,
                } = obj.normals[normal_index];
                let wavefront_obj::obj::TVertex { u, v, .. } = obj.tex_vertices[uv_index];
                composite_index_map.insert(
                    key,
                    TexturedVertex {
                        position: [p_x as f32, p_y as f32, p_z as f32],
                        normal: [n_x as f32, n_y as f32, n_z as f32],
                        tex_coords: [u as f32, 1.0 - v as f32],
                    },
                );
            }
        });
        let mut index_map: HashMap<(usize, usize, usize), usize> = HashMap::new();
        let mut mesh_vertices: Vec<TexturedVertex> = Vec::new();
        composite_index_map
            .iter()
            .enumerate()
            .for_each(|(i, (key, vertex))| {
                index_map.insert(*key, i);
                mesh_vertices.push(*vertex);
            });
        let mesh_indices: Vec<_> = vt_indices
            .iter()
            .flat_map(|vti| {
                let pos_index = vti.0;
                let normal_index = vti.2.expect("Obj file is missing normal");
                let uv_index = vti.1.unwrap();
                let key = (pos_index, normal_index, uv_index);
                index_map.get(&key).map(|final_index| *final_index as u16)
            })
            .collect();

        let sphere_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MeshComponent Vertex Buffer"),
            contents: bytemuck::cast_slice(&mesh_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let sphere_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MeshComponent Index Buffer"),
            contents: bytemuck::cast_slice(&mesh_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let vertex_count = mesh_vertices.len() as u32;
        let index_count = mesh_indices.len() as u32;

        let diffuse_texture = Texture::from_bytes(
            &device,
            &queue,
            &diffuse_texture_bytes,
            diffuse_texture_file_path,
        )?;

        let diffuse_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &diffuse_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some("MeshComponent diffuse_texture_bind_group"),
        });

        let transform = super::transform::Transform::new();

        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MeshComponent Transform Buffer"),
            contents: bytemuck::cast_slice(&[GpuMatrix4(transform.matrix.get())]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_var_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: transform_buffer.as_entire_binding(),
            }],
            label: Some("MeshComponent transform_bind_group"),
        });

        let normal_rotation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Normal Rotation Buffer"),
            contents: bytemuck::cast_slice(&[GpuMatrix4(transform.get_rotation_matrix())]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let normal_rotation_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_var_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: normal_rotation_buffer.as_entire_binding(),
            }],
            label: Some("MeshComponent normal_rotation_bind_group"),
        });

        Ok(MeshComponent {
            vertex_buffer: sphere_vertex_buffer,
            index_buffer: sphere_index_buffer,
            num_indices: index_count,
            _num_vertices: vertex_count,
            _diffuse_texture: diffuse_texture,
            diffuse_texture_bind_group,
            transform_buffer,
            transform_bind_group,
            normal_rotation_buffer,
            normal_rotation_bind_group,
            transform,
        })
    }
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

        let swapchain_format = surface.get_preferred_format(&adapter).unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            // TODO: switching to immediate requires to scale the movement by deltaT
            // present_mode: wgpu::PresentMode::Immediate,
        };

        surface.configure(&device, &config);

        let texture_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Texture Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("texture_shader.wgsl").into()),
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

        let sphere_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sphere Render Pipeline Layout"),
                bind_group_layouts: &[
                    &diffuse_texture_bind_group_layout,
                    &uniform_var_bind_group_layout,
                    &uniform_var_bind_group_layout,
                    &uniform_var_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let depth_texture = Texture::create_depth_texture(&device, &config, "depth_texture");

        let textured_mesh_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Sphere Render Pipeline"),
                layout: Some(&sphere_render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &texture_shader,
                    entry_point: "vs_main",
                    buffers: &[TexturedVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &texture_shader,
                    entry_point: "fs_main",
                    targets: &[wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }],
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
                    depth_compare: wgpu::CompareFunction::Less, // 1.
                    stencil: wgpu::StencilState::default(),     // 2.
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        let sphere = MeshComponent::new(
            "./src/sphere.obj",
            "./src/2k_mars.png",
            &diffuse_texture_bind_group_layout,
            &uniform_var_bind_group_layout,
            &device,
            &queue,
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

        let plane = MeshComponent::new(
            "./src/plane.obj",
            "./src/2k_checkerboard.png",
            &diffuse_texture_bind_group_layout,
            &uniform_var_bind_group_layout,
            &device,
            &queue,
        )?;

        plane.transform.set_position(Vector3::new(0.0, 0.0, -2.0));
        plane.transform.set_scale(Vector3::new(10.0, 1.0, 10.0));

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

        Ok(Self {
            surface,
            device,
            queue,
            config,
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
            depth_texture,

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
        self.camera_controller.process_window_events(event, window);
        // TODO: move out of renderer
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
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
        enum RotationAxis {
            PITCH,
            YAW,
            _ROLL,
        }
        let increment_rotation = |axis: RotationAxis, val: f32| {
            let mut rotation = self.sphere.transform.rotation.get();
            match axis {
                RotationAxis::PITCH => {
                    rotation.y += val;
                }
                RotationAxis::YAW => {
                    rotation.x += val;
                }
                RotationAxis::_ROLL => {
                    rotation.z += val;
                }
            }
            self.sphere.transform.set_rotation(rotation);
        };
        if self.is_right_pressed {
            increment_rotation(RotationAxis::YAW, 0.1);
        } else if self.is_left_pressed {
            increment_rotation(RotationAxis::YAW, -0.1);
        }
        if self.is_forward_pressed {
            increment_rotation(RotationAxis::PITCH, 0.1);
        } else if self.is_backward_pressed {
            increment_rotation(RotationAxis::PITCH, -0.1);
        }
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
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera, window);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
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

            vec![&self.sphere, &self.plane].iter().for_each(
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
                    render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
                    render_pass.set_bind_group(2, transform_bind_group, &[]);
                    render_pass.set_bind_group(3, normal_rotation_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..*num_indices, 0, 0..1);
                },
            );
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}
