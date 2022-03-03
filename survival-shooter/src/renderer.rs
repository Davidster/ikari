use std::collections::{HashMap, HashSet};

use super::*;

use anyhow::Result;

use wgpu::util::DeviceExt;

type VertexPosition = [f32; 3];
type VertexTextureCoords = [f32; 2];

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TexturedVertex {
    position: VertexPosition,
    tex_coords: VertexTextureCoords,
}

impl TexturedVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

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
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

struct MeshComponent {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    num_vertices: u32,
    diffuse_texture: texture::Texture,
    diffuse_texture_bind_group: wgpu::BindGroup,
}

pub struct RendererState {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,

    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    sphere: MeshComponent,
}

impl RendererState {
    pub async fn new(window: &winit::window::Window) -> Result<Self> {
        let sphere_obj = wavefront_obj::obj::parse(std::fs::read_to_string("./src/sphere.obj")?)?
            .objects
            .remove(0);

        let before = std::time::Instant::now();

        let vt_indices: Vec<wavefront_obj::obj::VTNIndex> = sphere_obj
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
        let mut pos_uv_map: HashMap<(usize, usize), TexturedVertex> = HashMap::new();
        vt_indices.iter().for_each(|vti| {
            let pos_index = vti.0;
            let uv_index = vti.1.unwrap();
            let key = (pos_index, uv_index);
            if !pos_uv_map.contains_key(&key) {
                let wavefront_obj::obj::Vertex { x, y, z } = sphere_obj.vertices[pos_index];
                let wavefront_obj::obj::TVertex { u, v, .. } = sphere_obj.tex_vertices[uv_index];
                pos_uv_map.insert(
                    key,
                    TexturedVertex {
                        position: [x as f32, y as f32, z as f32],
                        tex_coords: [u as f32, 1.0 - v as f32],
                    },
                );
            }
        });
        let mut index_map: HashMap<(usize, usize), usize> = HashMap::new();
        let mut final_vertices: Vec<TexturedVertex> = Vec::new();
        pos_uv_map
            .iter()
            .enumerate()
            .for_each(|(i, (key, vertex))| {
                index_map.insert(*key, i);
                final_vertices.push(*vertex);
            });
        let final_indices: Vec<_> = vt_indices
            .iter()
            .flat_map(|vti| {
                let pos_index = vti.0;
                let uv_index = vti.1.unwrap();
                let key = (pos_index, uv_index);
                index_map.get(&key).map(|final_index| *final_index as u16)
            })
            .collect();

        dbg!(final_vertices.len());
        dbg!(sphere_obj.vertices.len());
        dbg!(before.elapsed());

        // sphere_obj
        //     .geometry
        //     .iter()
        //     .for_each(|geometry| -> Vec<wavefront_obj::obj::VTNIndex> {
        //         geometry
        //             .shapes
        //             .iter()
        //             .for_each(|shape| match shape.primitive {
        //                 wavefront_obj::obj::Primitive::Triangle(f1, f2, f3) => {
        //                     sphere_vertices_2.push([f1, f2, f3].to_vec().iter().map(|face| TexturedVertex))
        //                 },
        //                 _ => Vec::new(),
        //             })
        //             .collect()
        //     });
        let mut extra_vertices: Vec<wavefront_obj::obj::VTNIndex> = Vec::new();
        // let mut sphere_indices: Vec<u16> =
        //     sphere_vtindices.iter().map(|face| face.0 as u16).collect();
        // let mut sphere_vertices: Vec<TexturedVertex> = sphere_obj
        //     .vertices
        //     .iter()
        //     .enumerate()
        //     .map(|(index, wavefront_obj::obj::Vertex { x, y, z })| {
        //         // TODO: is this correct? this might be the wrong index to use
        //         // let wavefront_obj::obj::TVertex { u, v, .. } = sphere_obj.tex_vertices[index];
        //         let unique_face_vertices: Vec<wavefront_obj::obj::VTNIndex> = sphere_vtindices
        //             .iter()
        //             .filter(|vtindex| vtindex.0 == index)
        //             .cloned()
        //             .collect::<HashSet<wavefront_obj::obj::VTNIndex>>()
        //             .iter()
        //             .cloned()
        //             .collect();
        //         let to_textured_vertex = |face: &wavefront_obj::obj::VTNIndex| {
        //             let wavefront_obj::obj::TVertex { u, v, .. } =
        //                 sphere_obj.tex_vertices[face.1.unwrap()];
        //             TexturedVertex {
        //                 position: [*x as f32, *y as f32, *z as f32],
        //                 tex_coords: [u as f32, 1.0 - v as f32],
        //             }
        //         };
        //         let mut vtindex_iterator = unique_face_vertices.iter();
        //         let first_face_vertex = vtindex_iterator.next().unwrap();
        //         extra_vertices.append(&mut vtindex_iterator.cloned().collect());
        //         to_textured_vertex(first_face_vertex)
        //     })
        //     .collect();
        // extra_vertices.iter().for_each(|extra_vertex|);
        // let sphere_count
        // sphere_vertices.append(extra_vertices);
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
        };

        surface.configure(&device, &config);

        let sphere_texture_bytes = include_bytes!("2k_mars.png");
        let sphere_texture =
            texture::Texture::from_bytes(&device, &queue, sphere_texture_bytes, "2k_mars.png")
                .unwrap();

        let sphere_texture_bind_group_layout =
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
                label: Some("sphere texture_bind_group_layout"),
            });

        let star_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &sphere_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&sphere_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sphere_texture.sampler),
                },
            ],
            label: Some("sphere diffuse_bind_group"),
        });

        let texture_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Texture Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("texture_shader.wgsl").into()),
        });

        let camera = Camera {
            eye: (0.0, 1.5, 3.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let camera_controller = CameraController::new(0.2);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
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
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let sphere_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sphere Render Pipeline Layout"),
                bind_group_layouts: &[&sphere_texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let sphere_render_pipeline =
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
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        let sphere_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Vertex Buffer"),
            contents: bytemuck::cast_slice(&final_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let sphere_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Star Index Buffer"),
            contents: bytemuck::cast_slice(&final_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let sphere_vertex_count = final_vertices.len() as u32;
        let sphere_index_count = final_indices.len() as u32;

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,

            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,

            sphere: MeshComponent {
                pipeline: sphere_render_pipeline,
                vertex_buffer: sphere_vertex_buffer,
                index_buffer: sphere_index_buffer,
                num_indices: sphere_index_count,
                num_vertices: sphere_vertex_count,
                diffuse_texture: sphere_texture,
                diffuse_texture_bind_group: star_texture_bind_group,
            },
        })
    }

    pub fn process_input(&mut self, event: &winit::event::WindowEvent) -> bool {
        self.camera_controller.process_events(event);
        false
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        // Reconfigure the surface with the new size
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
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
                depth_stencil_attachment: None,
            });

            let MeshComponent {
                pipeline,
                diffuse_texture_bind_group,
                vertex_buffer,
                index_buffer,
                num_indices,
                ..
            } = &self.sphere;

            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, diffuse_texture_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..*num_indices, 0, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}
