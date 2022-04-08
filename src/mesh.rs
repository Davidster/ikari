use std::collections::{hash_map, HashMap};

use super::*;

use anyhow::Result;
use cgmath::Matrix4;
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
}

impl TexturedVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

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
    pub diffuse_texture_bind_group: Option<wgpu::BindGroup>,
}

impl InstancedMeshComponent {
    pub fn new(
        mesh: &BasicMesh,
        diffuse_texture: Option<&Texture>,
        uniform_var_bind_group_layout: &wgpu::BindGroupLayout,
        device: &wgpu::Device,
        instances: &[GpuMeshInstance],
    ) -> Result<InstancedMeshComponent> {
        let mesh_component =
            MeshComponent::new(mesh, diffuse_texture, uniform_var_bind_group_layout, device)?;

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("InstancedMeshComponent instance_buffer"),
            contents: bytemuck::cast_slice(instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Ok(InstancedMeshComponent {
            vertex_buffer: mesh_component.vertex_buffer,
            index_buffer: mesh_component.index_buffer,
            num_indices: mesh_component.num_indices,
            _num_vertices: mesh_component._num_vertices,
            diffuse_texture_bind_group: mesh_component.diffuse_texture_bind_group,
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
            if let hash_map::Entry::Vacant(vacant_entry) = composite_index_map.entry(key) {
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
                vacant_entry.insert(TexturedVertex {
                    position: [p_x as f32, p_y as f32, p_z as f32],
                    normal: [n_x as f32, n_y as f32, n_z as f32],
                    tex_coords: [u as f32, 1.0 - v as f32],
                });
            }
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
        let indices: Vec<_> = vt_indices
            .iter()
            .flat_map(|vti| {
                let pos_index = vti.0;
                let normal_index = vti.2.expect("Obj file is missing normal");
                let uv_index = vti.1.unwrap();
                let key = (pos_index, normal_index, uv_index);
                index_map.get(&key).map(|final_index| *final_index as u16)
            })
            .collect();
        Ok(BasicMesh { vertices, indices })
    }
}
