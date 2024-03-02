use crate::{renderer::Float16, texture::*};

use std::{
    collections::{hash_map, HashMap},
    io::{BufReader, Cursor},
};

use anyhow::{bail, Result};
use glam::{
    f32::{Vec2, Vec3, Vec4},
    Mat4, UVec4, Vec2Swizzles, Vec3Swizzles,
};
use obj::raw::parse_obj;

pub struct Vertex {
    pub position: Vec3,
    pub bone_weights: Vec4,
    pub normal: Vec3,
    pub tangent: Vec3,
    pub tex_coords: Vec2,
    pub color: Vec4,
    pub bone_indices: UVec4,
}

// TODO: vertex data can be compressed a lot.
// use 2 values instad of 3 for normal, tangent.
// use less precision for normal, tangent, bone weights?
// don't store bitangent
// use [u8; 3] for color
// use u8 for bone indices,
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderVertex {
    pub position: [f32; 3],
    pub bone_weights: [Float16; 4],
    pub normal: [Float16; 2],
    pub tangent: [Float16; 2],
    pub tex_coords: [u16; 2],
    pub color: [u8; 4],
    pub bone_indices: [u8; 4],
}

impl ShaderVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![
        0 => Float32x3, // position
        1 => Float16x4, // bone_weights
        2 => Float16x2, // normal
        3 => Float16x2, // tangent
        4 => Unorm16x2, // tex_coords
        5 => Unorm8x4,  // color
        6 => Uint8x4,  // bone_indices
    ];

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ShaderVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
// https://cesium.com/blog/2015/05/18/vertex-compression/
// https://jcgt.org/published/0003/02/01/
pub fn oct_encode_unit_vector(mut n: Vec3) -> [Float16; 2] {
    n = n / (n.x.abs() + n.y.abs() + n.z.abs());
    let mut result_vec2 = if n.z >= 0.0 { n.xy() } else { oct_wrap(n.xy()) };
    result_vec2 = result_vec2 * Vec2::splat(0.5) + Vec2::splat(0.5);
    [
        Float16(half::f16::from_f32(result_vec2.x)),
        Float16(half::f16::from_f32(result_vec2.y)),
    ]
}

fn oct_wrap(v: Vec2) -> Vec2 {
    (Vec2::splat(1.0) - v.yx().abs())
        * Vec2::new(
            if v.x >= 0.0 { 1.0 } else { -1.0 },
            if v.y >= 0.0 { 1.0 } else { -1.0 },
        )
}

impl Default for ShaderVertex {
    fn default() -> Self {
        Self {
            position: Default::default(),
            normal: oct_encode_unit_vector([0.0, 1.0, 0.0].into()),
            tangent: oct_encode_unit_vector([1.0, 0.0, 0.0].into()),
            tex_coords: Default::default(),
            color: [255, 255, 255, 255],
            bone_indices: Default::default(),
            bone_weights: [
                Float16(half::f16::from_f32(1.0)),
                Float16(half::f16::from_f32(0.0)),
                Float16(half::f16::from_f32(0.0)),
                Float16(half::f16::from_f32(0.0)),
            ],
        }
    }
}

impl From<Vertex> for ShaderVertex {
    fn from(value: Vertex) -> Self {
        let convert_bone_index = |bone_index: u32| -> u8 {
            bone_index.try_into().unwrap_or_else(|_| {
                log::error!("Failed to convert bone index {} from u32 to u8. Does the character have more than 255 bones?", bone_index);
                0
            })
        };

        Self {
            position: value.position.into(),
            normal: oct_encode_unit_vector(value.normal),
            tangent: oct_encode_unit_vector(value.tangent),
            tex_coords: [
                ((value.tex_coords.x * (u16::MAX as f32)).round() % (u16::MAX as f32 + 1.0)) as u16,
                ((value.tex_coords.y * (u16::MAX as f32)).round() % (u16::MAX as f32 + 1.0)) as u16,
            ],
            color: [
                (value.color.x * u8::MAX as f32).round() as u8,
                (value.color.y * u8::MAX as f32).round() as u8,
                (value.color.z * u8::MAX as f32).round() as u8,
                (value.color.w * u8::MAX as f32).round() as u8,
            ],
            bone_indices: [
                convert_bone_index(value.bone_indices.x),
                convert_bone_index(value.bone_indices.y),
                convert_bone_index(value.bone_indices.z),
                convert_bone_index(value.bone_indices.w),
            ],
            bone_weights: [
                Float16(half::f16::from_f32(value.bone_weights.x)),
                Float16(half::f16::from_f32(value.bone_weights.y)),
                Float16(half::f16::from_f32(value.bone_weights.z)),
                Float16(half::f16::from_f32(value.bone_weights.w)),
            ],
        }
    }
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: Default::default(),
            normal: [0.0, 1.0, 0.0].into(),
            tex_coords: Default::default(),
            tangent: [1.0, 0.0, 0.0].into(),
            color: [1.0, 1.0, 1.0, 1.0].into(),
            bone_indices: Default::default(),
            bone_weights: [1.0, 0.0, 0.0, 0.0].into(),
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
    pub padding: [f32; 3],
}

impl GpuPbrMeshInstance {
    pub fn new(transform: Mat4, pbr_params: DynamicPbrParams) -> Self {
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
            padding: [0.0, 0.0, 0.0],
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
pub struct IndexedPbrTextures {
    pub base_color: Option<usize>,
    pub normal: Option<usize>,
    pub metallic_roughness: Option<usize>,
    pub emissive: Option<usize>,
    pub ambient_occlusion: Option<usize>,
}

#[derive(Default)]
pub struct PbrTextures<'a> {
    pub base_color: Option<&'a Texture>,
    pub normal: Option<&'a Texture>,
    pub metallic_roughness: Option<&'a Texture>,
    pub emissive: Option<&'a Texture>,
    pub ambient_occlusion: Option<&'a Texture>,
}

pub struct BasicMesh {
    pub vertices: Vec<ShaderVertex>,
    pub indices: Vec<u16>,
}

impl BasicMesh {
    pub fn new(obj_file_bytes: &[u8]) -> Result<Self> {
        let obj = parse_obj(BufReader::new(Cursor::new(obj_file_bytes)))?;

        let mut triangles: Vec<[(usize, usize, usize); 3]> = vec![];

        for polygon in obj.polygons.iter() {
            match polygon {
                obj::raw::object::Polygon::PTN(points) => {
                    if points.len() < 3 {
                        bail!("BasicMesh requires that all polygons have at least 3 vertices");
                    }

                    let last_elem = points.last().unwrap();

                    triangles.extend(
                        points[..points.len() - 1]
                            .iter()
                            .zip(points[1..points.len() - 1].iter())
                            .map(|(&x, &y)| [*last_elem, x, y]),
                    );
                }
                _ => {
                    bail!("BasicMesh requires that all points have a position, uv and normal");
                }
            }
        }

        let mut composite_index_map: HashMap<(usize, usize, usize), ShaderVertex> = HashMap::new();
        triangles.iter().for_each(|points| {
            let points_with_attribs: Vec<_> = points
                .iter()
                .map(|vti| {
                    let key = (vti.0, vti.2, vti.1);
                    let pos = obj.positions[vti.0];
                    let normal = obj.normals[vti.2];
                    let uv = obj.tex_coords[vti.1];
                    let position = Vec3::new(pos.0, pos.1, pos.2);
                    let normal = Vec3::from(normal);
                    // convert uv format into 0->1 range
                    let tex_coords = Vec2::new(uv.0, 1.0 - uv.1);
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

            let _bitangent = Vec3::new(
                f * (-delta_uv_2.x * edge_1.x + delta_uv_1.x * edge_2.x),
                f * (-delta_uv_2.x * edge_1.y + delta_uv_1.x * edge_2.y),
                f * (-delta_uv_2.x * edge_1.z + delta_uv_1.x * edge_2.z),
            );

            points_with_attribs
                .iter()
                .copied()
                .for_each(|(key, position, normal, tex_coords)| {
                    if let hash_map::Entry::Vacant(vacant_entry) = composite_index_map.entry(key) {
                        vacant_entry.insert(ShaderVertex::from(Vertex {
                            position,
                            normal,
                            tangent,
                            tex_coords,
                            ..Default::default()
                        }));
                    }
                });
        });
        let mut index_map: HashMap<(usize, usize, usize), usize> = HashMap::new();
        let mut vertices: Vec<ShaderVertex> = Vec::new();
        composite_index_map
            .iter()
            .enumerate()
            .for_each(|(i, (key, vertex))| {
                index_map.insert(*key, i);
                vertices.push(*vertex);
            });
        let indices: Vec<_> = triangles
            .iter()
            .flat_map(|points| {
                points
                    .iter()
                    .flat_map(|vti| {
                        let key = (vti.0, vti.2, vti.1);
                        index_map.get(&key).map(|final_index| *final_index as u16)
                    })
                    .collect::<Vec<u16>>()
            })
            .collect();
        Ok(BasicMesh { vertices, indices })
    }
}
