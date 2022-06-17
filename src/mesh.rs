use std::collections::{hash_map, HashMap};

use super::*;

use anyhow::Result;
use cgmath::{Matrix4, Vector2, Vector3, Vector4};
use wgpu::util::DeviceExt;

type VertexPosition = [f32; 3];
type VertexNormal = [f32; 3];
type VertexColor = [f32; 4];
type VertexTextureCoords = [f32; 2];

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: VertexPosition,
    pub normal: VertexNormal,
    pub tex_coords: VertexTextureCoords,
    pub tangent: VertexNormal,
    pub bitangent: VertexNormal,
    pub color: VertexColor,
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Float32x3,
        4 => Float32x3,
        5 => Float32x4,
    ];

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMeshInstance {
    model_transform: GpuMatrix4,
    base_color_factor: [f32; 4],
    emissive_factor: [f32; 4],
    mrno: [f32; 4], // metallic_factor, roughness_factor, normal scale, occlusion strength
    alpha_cutoff: f32,
    padding: [f32; 3], // TODO: is this needed?
}

impl GpuMeshInstance {
    const ATTRIBS: [wgpu::VertexAttribute; 8] = wgpu::vertex_attr_array![
        6 => Float32x4,  7 => Float32x4,  8 => Float32x4,  9 => Float32x4, // transform
        10 => Float32x4,
        11 => Float32x4,
        12 => Float32x4,
        13 => Float32,
    ];

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuMeshInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }

    pub fn new(instance: &MeshInstance) -> GpuMeshInstance {
        let MeshInstance {
            transform,
            base_material,
        } = instance;
        let BaseMaterial {
            base_color_factor,
            emissive_factor,
            metallic_factor,
            roughness_factor,
            normal_scale,
            occlusion_strength,
            alpha_cutoff,
        } = base_material;
        GpuMeshInstance {
            model_transform: GpuMatrix4(transform.matrix()),
            base_color_factor: (*base_color_factor).into(),
            emissive_factor: [
                emissive_factor[0],
                emissive_factor[1],
                emissive_factor[2],
                1.0,
            ],
            mrno: [
                *metallic_factor,
                *roughness_factor,
                *normal_scale,
                *occlusion_strength,
            ],
            alpha_cutoff: *alpha_cutoff,
            padding: [0.0, 0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFlatColorMeshInstance {
    model_transform: GpuMatrix4,
    color: [f32; 4],
}

impl GpuFlatColorMeshInstance {
    const ATTRIBS: [wgpu::VertexAttribute; 9] = wgpu::vertex_attr_array![
        6 => Float32x4,  7 => Float32x4,  8 => Float32x4,  9 => Float32x4,
        10 => Float32x4, 11 => Float32x4, 12 => Float32x4, 13 => Float32x4,
        14 => Float32x4
    ];

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuFlatColorMeshInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }

    pub fn new(model_transform: Matrix4<f32>, color: Vector3<f32>) -> GpuFlatColorMeshInstance {
        GpuFlatColorMeshInstance {
            model_transform: GpuMatrix4(model_transform),
            color: [color.x, color.y, color.z, 1.0],
        }
    }
}

#[derive(Clone, Debug)]
pub struct MeshInstance {
    pub transform: transform::Transform,
    pub base_material: BaseMaterial,
}

impl MeshInstance {
    pub fn new() -> MeshInstance {
        MeshInstance {
            transform: transform::Transform::new(),
            base_material: Default::default(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BaseMaterial {
    pub base_color_factor: Vector4<f32>,
    pub emissive_factor: Vector3<f32>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha_cutoff: f32,
}

impl Default for BaseMaterial {
    fn default() -> Self {
        BaseMaterial {
            base_color_factor: Vector4::new(1.0, 1.0, 1.0, 1.0),
            emissive_factor: Vector3::new(0.0, 0.0, 0.0),
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            alpha_cutoff: -1.0,
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
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MeshComponent Vertex Buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
            contents: bytemuck::cast_slice(&[GpuMatrix4(transform.matrix())]),
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
            vertex_buffer,
            index_buffer,
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

#[derive(Default)]
pub struct InstancedMeshMaterialParams<'a> {
    pub diffuse: Option<&'a Texture>,
    pub normal: Option<&'a Texture>,
    pub metallic_roughness: Option<&'a Texture>,
    pub emissive: Option<&'a Texture>,
    pub ambient_occlusion: Option<&'a Texture>,
}

impl InstancedMeshComponent {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &BasicMesh,
        material: &InstancedMeshMaterialParams,
        textures_bind_group_layout: &wgpu::BindGroupLayout,
        initial_buffer_contents: &[u8],
    ) -> Result<InstancedMeshComponent> {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("InstancedMeshComponent Vertex Buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("InstancedMeshComponent Index Buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let vertex_count = mesh.vertices.len() as u32;
        let index_count = mesh.indices.len() as u32;

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("InstancedMeshComponent instance_buffer"),
            contents: initial_buffer_contents,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let textures_bind_group =
            get_textures_bind_group(material, device, queue, textures_bind_group_layout)?;

        Ok(InstancedMeshComponent {
            vertex_buffer,
            index_buffer,
            num_indices: index_count,
            _num_vertices: vertex_count,
            textures_bind_group,
            instance_buffer,
        })
    }
}

pub fn get_textures_bind_group(
    material: &InstancedMeshMaterialParams,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    textures_bind_group_layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::BindGroup, anyhow::Error> {
    let auto_generated_diffuse_texture;
    let diffuse_texture = match material.diffuse {
        Some(diffuse_texture) => diffuse_texture,
        None => {
            auto_generated_diffuse_texture =
                Texture::from_color(device, queue, [255, 255, 255, 255])?;
            &auto_generated_diffuse_texture
        }
    };
    let auto_generated_normal_map;
    let normal_map = match material.normal {
        Some(normal_map) => normal_map,
        None => {
            auto_generated_normal_map = Texture::flat_normal_map(device, queue)?;
            &auto_generated_normal_map
        }
    };
    let auto_generated_metallic_roughness_map;
    let metallic_roughness_map = match material.metallic_roughness {
        Some(metallic_roughness_map) => metallic_roughness_map,
        None => {
            auto_generated_metallic_roughness_map =
                Texture::from_color(device, queue, [255, 127, 0, 255])?;
            &auto_generated_metallic_roughness_map
        }
    };
    let auto_generated_emissive_map;
    let emissive_map = match material.emissive {
        Some(emissive_map) => emissive_map,
        None => {
            auto_generated_emissive_map = Texture::from_color(device, queue, [0, 0, 0, 255])?;
            &auto_generated_emissive_map
        }
    };
    let auto_generated_ambient_occlusion_map;
    let ambient_occlusion_map = match material.ambient_occlusion {
        Some(ambient_occlusion_map) => ambient_occlusion_map,
        None => {
            auto_generated_ambient_occlusion_map =
                Texture::from_color(device, queue, [255, 255, 255, 255])?;
            &auto_generated_ambient_occlusion_map
        }
    };
    let textures_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: textures_bind_group_layout,
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
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&metallic_roughness_map.view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&metallic_roughness_map.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&emissive_map.view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler(&emissive_map.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(&ambient_occlusion_map.view),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::Sampler(&ambient_occlusion_map.sampler),
            },
        ],
        label: Some("InstancedMeshComponent textures_bind_group"),
    });
    Ok(textures_bind_group)
}

pub struct BasicMesh {
    vertices: Vec<Vertex>,
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
        let mut composite_index_map: HashMap<(usize, usize, usize), Vertex> = HashMap::new();
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
                        vacant_entry.insert(Vertex {
                            position: to_arr(position),
                            normal: to_arr(normal),
                            tex_coords: [tex_coords.x, tex_coords.y],
                            tangent: to_arr(&tangent),
                            bitangent: to_arr(&bitangent),
                            color: [1.0, 1.0, 1.0, 1.0],
                        });
                    }
                });
        });
        let mut index_map: HashMap<(usize, usize, usize), usize> = HashMap::new();
        let mut vertices: Vec<Vertex> = Vec::new();
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
