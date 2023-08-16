use crate::{
    file_loader::{FileLoader, GameFilePath},
    texture::*,
};

use std::collections::{hash_map, HashMap};

use anyhow::Result;
use glam::{
    f32::{Vec2, Vec3, Vec4},
    Mat4,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
    pub color: [f32; 4],
    pub bone_indices: [u32; 4],
    pub bone_weights: [f32; 4],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 8] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Float32x3,
        4 => Float32x3,
        5 => Float32x4,
        6 => Uint32x4,
        7 => Float32x4,
    ];

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: Default::default(),
            normal: [0.0, 1.0, 0.0],
            tex_coords: Default::default(),
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            bone_indices: Default::default(),
            bone_weights: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPbrMeshInstance {
    pub model_transform: Mat4,
    pub base_color_factor: [f32; 4],
    pub emissive_factor: [f32; 4],
    pub mrno: [f32; 4], // metallic_factor, roughness_factor, normal scale, occlusion strength
    pub alpha_cutoff: f32,
    pub culling_mask: u32,
    pub padding: [f32; 2],
}

impl GpuPbrMeshInstance {
    pub fn new(transform: Mat4, pbr_params: DynamicPbrParams, culling_mask: u32) -> Self {
        let DynamicPbrParams {
            base_color_factor,
            emissive_factor,
            metallic_factor,
            roughness_factor,
            normal_scale,
            occlusion_strength,
            alpha_cutoff,
        } = pbr_params;
        Self {
            model_transform: transform,
            base_color_factor: base_color_factor.into(),
            emissive_factor: [
                emissive_factor[0],
                emissive_factor[1],
                emissive_factor[2],
                1.0,
            ],
            mrno: [
                metallic_factor,
                roughness_factor,
                normal_scale,
                occlusion_strength,
            ],
            alpha_cutoff,
            culling_mask,
            padding: [0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuUnlitMeshInstance {
    pub model_transform: Mat4,
    pub color: [f32; 4],
}

pub type GpuWireframeMeshInstance = GpuUnlitMeshInstance;

pub type GpuTransparentMeshInstance = GpuUnlitMeshInstance;

#[derive(Copy, Clone, Debug)]
pub struct DynamicPbrParams {
    pub base_color_factor: Vec4,
    pub emissive_factor: Vec3,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha_cutoff: f32,
}

impl Default for DynamicPbrParams {
    fn default() -> Self {
        DynamicPbrParams {
            base_color_factor: Vec4::new(1.0, 1.0, 1.0, 1.0),
            emissive_factor: Vec3::new(0.0, 0.0, 0.0),
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            alpha_cutoff: -1.0,
        }
    }
}

#[derive(Debug, Default, Hash, PartialEq, Eq, Clone)]
pub struct IndexedPbrMaterial {
    pub base_color: Option<usize>,
    pub normal: Option<usize>,
    pub metallic_roughness: Option<usize>,
    pub emissive: Option<usize>,
    pub ambient_occlusion: Option<usize>,
}

#[derive(Default)]
pub struct PbrMaterial<'a> {
    pub base_color: Option<&'a Texture>,
    pub normal: Option<&'a Texture>,
    pub metallic_roughness: Option<&'a Texture>,
    pub emissive: Option<&'a Texture>,
    pub ambient_occlusion: Option<&'a Texture>,
}

pub struct BasicMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

impl BasicMesh {
    pub async fn new(obj_file_path: &GameFilePath) -> Result<Self> {
        let obj_file_string = FileLoader::read_to_string(obj_file_path).await?;

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
            let points = [vti1, vti2, vti3];
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
                    let position = Vec3::new(p_x as f32, p_y as f32, p_z as f32);
                    let normal = Vec3::new(n_x as f32, n_y as f32, n_z as f32);
                    // convert uv format into 0->1 range
                    let tex_coords = Vec2::new(u as f32, 1.0 - v as f32);
                    (key, position, normal, tex_coords)
                })
                .collect();

            let edge_1 = points_with_attribs[1].1 - points_with_attribs[0].1;
            let edge_2 = points_with_attribs[2].1 - points_with_attribs[0].1;

            let delta_uv_1 = points_with_attribs[1].3 - points_with_attribs[0].3;
            let delta_uv_2 = points_with_attribs[2].3 - points_with_attribs[0].3;

            let f = 1.0 / (delta_uv_1.x * delta_uv_2.y - delta_uv_2.x * delta_uv_1.y);

            let tangent = Vec3::new(
                f * (delta_uv_2.y * edge_1.x - delta_uv_1.y * edge_2.x),
                f * (delta_uv_2.y * edge_1.y - delta_uv_1.y * edge_2.y),
                f * (delta_uv_2.y * edge_1.z - delta_uv_1.y * edge_2.z),
            );

            let bitangent = Vec3::new(
                f * (-delta_uv_2.x * edge_1.x + delta_uv_1.x * edge_2.x),
                f * (-delta_uv_2.x * edge_1.y + delta_uv_1.x * edge_2.y),
                f * (-delta_uv_2.x * edge_1.z + delta_uv_1.x * edge_2.z),
            );

            points_with_attribs
                .iter()
                .for_each(|(key, position, normal, tex_coords)| {
                    if let hash_map::Entry::Vacant(vacant_entry) = composite_index_map.entry(*key) {
                        let to_arr = |vec: &Vec3| [vec.x, vec.y, vec.z];
                        vacant_entry.insert(Vertex {
                            position: to_arr(position),
                            normal: to_arr(normal),
                            tex_coords: [tex_coords.x, tex_coords.y],
                            tangent: to_arr(&tangent),
                            bitangent: to_arr(&bitangent),
                            ..Default::default()
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
                [vti1, vti2, vti3]
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
