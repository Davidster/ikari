use std::collections::{hash_map, HashMap};

use super::*;

use anyhow::Result;
use cgmath::{Matrix4, Vector2, Vector3};
use wgpu::util::DeviceExt;

type VertexPosition = [f32; 3];
type VertexNormal = [f32; 3];
type VertexTextureCoords = [f32; 2];

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TexturedVertex {
    position: VertexPosition,
    normal: VertexNormal,
    tex_coords: VertexTextureCoords,
    tangent: VertexNormal,
    bitangent: VertexNormal,
}

impl TexturedVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Float32x3,
        4 => Float32x3
    ];

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TexturedVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMeshInstance {
    model_transform: GpuMatrix4,
}

impl GpuMeshInstance {
    const ATTRIBS: [wgpu::VertexAttribute; 8] = wgpu::vertex_attr_array![
        5 => Float32x4,  6 => Float32x4,  7 => Float32x4,  8 => Float32x4,
        9 => Float32x4, 10 => Float32x4, 11 => Float32x4, 12 => Float32x4
    ];

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuMeshInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }

    pub fn new(model_transform: Matrix4<f32>) -> GpuMeshInstance {
        GpuMeshInstance {
            model_transform: GpuMatrix4(model_transform),
        }
    }
}

pub struct MeshComponent {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,

    pub num_indices: u32,
    pub _num_vertices: u32,

    pub diffuse_texture_bind_group: Option<wgpu::BindGroup>,

    pub normal_rotation_buffer: wgpu::Buffer,
    pub normal_rotation_bind_group: wgpu::BindGroup,

    pub transform_buffer: wgpu::Buffer,
    pub transform_bind_group: wgpu::BindGroup,
    pub transform: super::transform::Transform,
}

impl MeshComponent {
    pub fn new(
        mesh: &BasicMesh,
        diffuse_texture: Option<&Texture>,
        uniform_var_bind_group_layout: &wgpu::BindGroupLayout,
        device: &wgpu::Device,
    ) -> Result<MeshComponent> {
        let sphere_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MeshComponent Vertex Buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let sphere_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MeshComponent Index Buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let vertex_count = mesh.vertices.len() as u32;
        let index_count = mesh.indices.len() as u32;

        let diffuse_texture_bind_group = if let Some(diffuse_texture) = diffuse_texture {
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
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            }))
        } else {
            None
        };

        let transform = super::transform::Transform::new();

        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MeshComponent Transform Buffer"),
            contents: bytemuck::cast_slice(&[GpuMatrix4(transform.matrix.get())]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: uniform_var_bind_group_layout,
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
            layout: uniform_var_bind_group_layout,
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
            diffuse_texture_bind_group,
            transform_buffer,
            transform_bind_group,
            normal_rotation_buffer,
            normal_rotation_bind_group,
            transform,
        })
    }
}

pub struct InstancedMeshComponent {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub _num_vertices: u32,
    pub textures_bind_group: wgpu::BindGroup,
}

impl InstancedMeshComponent {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &BasicMesh,
        diffuse_texture: Option<&Texture>,
        normal_map: Option<&Texture>,
        uniform_var_bind_group_layout: &wgpu::BindGroupLayout,
        instances: &[GpuMeshInstance],
    ) -> Result<InstancedMeshComponent> {
        // TODO: a bunch of stuff from the mesh component isn't used;
        //       this should be DRY'd in a different way
        let mesh_component =
            MeshComponent::new(mesh, diffuse_texture, uniform_var_bind_group_layout, device)?;

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("InstancedMeshComponent instance_buffer"),
            contents: bytemuck::cast_slice(instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let one_pixel_white_image = {
            let mut img = image::RgbaImage::new(1, 1);
            img.put_pixel(0, 0, image::Rgba([255, 255, 255, 255]));
            image::DynamicImage::ImageRgba8(img)
        };
        let one_pixel_white_texture = Texture::from_image(
            device,
            queue,
            &one_pixel_white_image,
            Some("one_pixel_white_texture"),
            None,
            false,
            &texture::SamplerDescriptor(wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..texture::SamplerDescriptor::default().0
            }),
        )?;

        let one_pixel_up_image = {
            let mut img = image::RgbaImage::new(1, 1);
            img.put_pixel(0, 0, image::Rgba([127, 127, 255, 255]));
            image::DynamicImage::ImageRgba8(img)
        };
        let one_pixel_up_texture = Texture::from_image(
            device,
            queue,
            &one_pixel_up_image,
            Some("one_pixel_up_image"),
            wgpu::TextureFormat::Rgba8Unorm.into(),
            false,
            &texture::SamplerDescriptor(wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..texture::SamplerDescriptor::default().0
            }),
        )?;

        let diffuse_texture = diffuse_texture.unwrap_or(&one_pixel_white_texture);
        let normal_map = normal_map.unwrap_or(&one_pixel_up_texture);

        // TODO: copied in renderer.rs
        let textures_bind_group_layout =
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
                label: Some("InstancedMeshComponent textures_bind_group_layout"),
            });
        let textures_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &textures_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&normal_map.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&normal_map.sampler),
                },
            ],
            label: Some("InstancedMeshComponent textures_bind_group"),
        });

        Ok(InstancedMeshComponent {
            vertex_buffer: mesh_component.vertex_buffer,
            index_buffer: mesh_component.index_buffer,
            num_indices: mesh_component.num_indices,
            _num_vertices: mesh_component._num_vertices,
            textures_bind_group,
            instance_buffer,
        })
    }
}

pub struct BasicMesh {
    vertices: Vec<TexturedVertex>,
    indices: Vec<u16>,
}

impl BasicMesh {
    pub fn new(obj_file_path: &str) -> Result<Self> {
        let obj_file_string = std::fs::read_to_string(obj_file_path)?;

        let obj = wavefront_obj::obj::parse(obj_file_string)?
            .objects
            .remove(0);

        let triangles: Vec<(
            wavefront_obj::obj::VTNIndex,
            wavefront_obj::obj::VTNIndex,
            wavefront_obj::obj::VTNIndex,
        )> = obj
            .geometry
            .iter()
            .flat_map(
                |geometry| -> Vec<(
                    wavefront_obj::obj::VTNIndex,
                    wavefront_obj::obj::VTNIndex,
                    wavefront_obj::obj::VTNIndex,
                )> {
                    geometry
                        .shapes
                        .iter()
                        .flat_map(|shape| {
                            if let wavefront_obj::obj::Primitive::Triangle(vti1, vti2, vti3) =
                                shape.primitive
                            {
                                Some((vti1, vti2, vti3))
                            } else {
                                None
                            }
                        })
                        .collect()
                },
            )
            .collect();
        let mut composite_index_map: HashMap<(usize, usize, usize), TexturedVertex> =
            HashMap::new();
        triangles.iter().for_each(|(vti1, vti2, vti3)| {
            let points = vec![vti1, vti2, vti3];
            let points_with_attribs: Vec<_> = points
                .iter()
                .map(|vti| {
                    let pos_index = vti.0;
                    let normal_index = vti.2.expect("Obj file is missing normal");
                    let uv_index = vti.1.expect("Obj file is missing uv index");
                    let key = (pos_index, normal_index, uv_index);
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
                    let position = Vector3::new(p_x as f32, p_y as f32, p_z as f32);
                    let normal = Vector3::new(n_x as f32, n_y as f32, n_z as f32);
                    // convert uv format into 0->1 range
                    let tex_coords = Vector2::new(u as f32, 1.0 - v as f32);
                    (key, position, normal, tex_coords)
                })
                .collect();

            let edge_1 = points_with_attribs[1].1 - points_with_attribs[0].1;
            let edge_2 = points_with_attribs[2].1 - points_with_attribs[0].1;

            let delta_uv_1 = points_with_attribs[1].3 - points_with_attribs[0].3;
            let delta_uv_2 = points_with_attribs[2].3 - points_with_attribs[0].3;

            let f = 1.0 / (delta_uv_1.x * delta_uv_2.y - delta_uv_2.x * delta_uv_1.y);

            let tangent = Vector3::new(
                f * (delta_uv_2.y * edge_1.x - delta_uv_1.y * edge_2.x),
                f * (delta_uv_2.y * edge_1.y - delta_uv_1.y * edge_2.y),
                f * (delta_uv_2.y * edge_1.z - delta_uv_1.y * edge_2.z),
            );

            let bitangent = Vector3::new(
                f * (-delta_uv_2.x * edge_1.x + delta_uv_1.x * edge_2.x),
                f * (-delta_uv_2.x * edge_1.y + delta_uv_1.x * edge_2.y),
                f * (-delta_uv_2.x * edge_1.z + delta_uv_1.x * edge_2.z),
            );

            points_with_attribs
                .iter()
                .for_each(|(key, position, normal, tex_coords)| {
                    if let hash_map::Entry::Vacant(vacant_entry) = composite_index_map.entry(*key) {
                        let to_arr = |vec: &Vector3<f32>| [vec.x, vec.y, vec.z];
                        vacant_entry.insert(TexturedVertex {
                            position: to_arr(position),
                            normal: to_arr(normal),
                            tex_coords: [tex_coords.x, tex_coords.y],
                            tangent: to_arr(&tangent),
                            bitangent: to_arr(&bitangent),
                        });
                    }
                });
        });
        let mut index_map: HashMap<(usize, usize, usize), usize> = HashMap::new();
        let mut vertices: Vec<TexturedVertex> = Vec::new();
        composite_index_map
            .iter()
            .enumerate()
            .for_each(|(i, (key, vertex))| {
                index_map.insert(*key, i);
                vertices.push(*vertex);
            });
        let indices: Vec<_> = triangles
            .iter()
            .flat_map(|(vti1, vti2, vti3)| {
                vec![vti1, vti2, vti3]
                    .iter()
                    .flat_map(|vti| {
                        let pos_index = vti.0;
                        let normal_index = vti.2.expect("Obj file is missing normal");
                        let uv_index = vti.1.unwrap();
                        let key = (pos_index, normal_index, uv_index);
                        index_map.get(&key).map(|final_index| *final_index as u16)
                    })
                    .collect::<Vec<u16>>()
            })
            .collect();
        Ok(BasicMesh { vertices, indices })
    }
}
