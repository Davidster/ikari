use std::collections::HashMap;
use std::iter::Map;

use anyhow::bail;
use anyhow::Result;
use cgmath::Vector2;
use cgmath::Vector3;
use wgpu::util::DeviceExt;

use super::*;

pub struct GltfAsset {
    pub document: gltf::Document,
    pub buffers: Vec<gltf::buffer::Data>,
    pub images: Vec<gltf::image::Data>,
}

pub struct Scene {
    pub source_asset: GltfAsset,
    pub buffers: SceneBuffers,
}

pub struct SceneBuffers {
    // same order as the meshes in src
    pub geometry: Vec<GeometryBuffer>,
    // same order as the textures in src
    pub textures: Vec<Texture>,
}

pub struct GeometryBuffer {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: Option<wgpu::Buffer>,
}

// TODO: return result instead of panic
pub fn build_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gltf_asset: GltfAsset,
) -> Result<Scene> {
    let GltfAsset {
        document,
        buffers,
        images,
    } = &gltf_asset;

    let geometry = document
        .meshes()
        .map(|mesh| {
            dbg!(mesh.name());
            if mesh
                .primitives()
                .any(|prim| prim.mode() != gltf::mesh::Mode::Triangles)
            {
                bail!("Only triangle primitives are supported");
            }
            let first_triangles_prim = mesh
                .primitives()
                .find(|prim| prim.mode() == gltf::mesh::Mode::Triangles)
                .ok_or_else(|| anyhow::anyhow!("No triangle primitives found"))?;
            // let triangles: Vec<_> = mesh
            //     .primitives()
            //     .filter(|prim| prim.mode() == gltf::mesh::Mode::Triangles)
            //     .collect();

            let get_buffer_slice_from_accessor = |accessor: gltf::Accessor| {
                let buffer_view = accessor.view().unwrap();
                let buffer = &buffers[buffer_view.buffer().index()];
                let byte_range_start = buffer_view.offset() + accessor.offset();
                let byte_range_end = byte_range_start + (accessor.size() * accessor.count());
                let byte_range = byte_range_start..byte_range_end;
                &buffer[byte_range]
            };

            // let triangles_with_positions: Vec<Vec<[f32; 3]>> = triangles
            //     .iter()
            //     .map(|triangle| {
            //         let (_, accessor) = triangle
            //             .attributes()
            //             .find(|(semantic, _)| *semantic == gltf::Semantic::Positions)
            //             .ok_or_else(|| anyhow::anyhow!("No positions found"))?;
            //         let data_type = accessor.data_type();
            //         let dimensions = accessor.dimensions();
            //         if dimensions != gltf::accessor::Dimensions::Vec3 {
            //             bail!("Expected vec3 data but found: {:?}", dimensions);
            //         }
            //         if data_type != gltf::accessor::DataType::F32 {
            //             bail!("Expected f32 data but found: {:?}", data_type);
            //         }
            //         anyhow::Ok(
            //             bytemuck::cast_slice(get_buffer_slice_from_accessor(accessor)).to_vec(),
            //         )
            //     })
            //     .collect::<Result<Vec<_>, _>>()?;
            // let triangle_count = triangles_with_positions.len();
            // let vertex_positions: Vec<_> = triangles_with_positions.iter().flatten().collect();
            // let vertex_count = triangle_count * 3;

            // TODO: use Vector3<f32> instead of [f32; 3]
            let vertex_positions: Vec<[f32; 3]> = {
                let (_, accessor) = first_triangles_prim
                    .attributes()
                    .find(|(semantic, _)| *semantic == gltf::Semantic::Positions)
                    .ok_or_else(|| anyhow::anyhow!("No positions found"))?;
                let data_type = accessor.data_type();
                let dimensions = accessor.dimensions();
                if dimensions != gltf::accessor::Dimensions::Vec3 {
                    bail!("Expected vec3 data but found: {:?}", dimensions);
                }
                if data_type != gltf::accessor::DataType::F32 {
                    bail!("Expected f32 data but found: {:?}", data_type);
                }
                anyhow::Ok(bytemuck::cast_slice(get_buffer_slice_from_accessor(accessor)).to_vec())
            }?;
            let vertex_count = vertex_positions.len();
            let triangle_count = vertex_count / 3;

            // let all_triangles_have_indices = !triangles
            //     .iter()
            //     .any(|triangle| triangle.indices().is_none());
            // let no_triangles_have_indices = !triangles
            //     .iter()
            //     .any(|triangle| triangle.indices().is_some());
            // if !all_triangles_have_indices && !no_triangles_have_indices {
            //     bail!("Expected all triangles to have indices or none to have indices");
            // }
            // let triangles_with_indices: Option<Vec<Vec<u16>>> = if no_triangles_have_indices {
            //     None
            // } else {
            //     triangles
            //         .iter()
            //         .map(|triangle| {
            //             triangle.indices().map(|accessor| {
            //                 // let buffer = &buffers[accessor.view().unwrap().index()];
            //                 // let buffer_slice: &[u8] =
            //                 //     &buffer[accessor.offset()..(accessor.count() * accessor.size())];
            //                 let data_type = accessor.data_type();
            //                 let buffer_slice = get_buffer_slice_from_accessor(accessor);

            //                 match data_type {
            //                     gltf::accessor::DataType::U16 => {
            //                         bytemuck::cast_slice(buffer_slice).to_vec()
            //                     }
            //                     gltf::accessor::DataType::U8 => bytemuck::cast_slice(
            //                         &buffer_slice.iter().map(|&x| x as u16).collect::<Vec<_>>(),
            //                     )
            //                     .to_vec(),
            //                     data_type => {
            //                         panic!("Expected u16 or u8 indices but found: {:?}", data_type)
            //                     }
            //                 }
            //             })
            //         })
            //         .collect()
            // };
            // let indices: Vec<_> = triangles_with_indices.iter().flatten().collect();
            let indices: Option<Vec<u16>> = first_triangles_prim
                .indices()
                .map(|accessor| {
                    let data_type = accessor.data_type();
                    let buffer_slice = get_buffer_slice_from_accessor(accessor);

                    let indices: Vec<u16> = match data_type {
                        gltf::accessor::DataType::U16 => {
                            anyhow::Ok(bytemuck::cast_slice(buffer_slice).to_vec())
                        }
                        gltf::accessor::DataType::U8 => anyhow::Ok(
                            bytemuck::cast_slice(
                                &buffer_slice.iter().map(|&x| x as u16).collect::<Vec<_>>(),
                            )
                            .to_vec(),
                        ),
                        data_type => {
                            bail!("Expected u16 or u8 indices but found: {:?}", data_type)
                        }
                    }?;
                    anyhow::Ok(indices)
                })
                .map_or(Ok(None), |v| v.map(Some))?;

            // let triangles_indices: Vec<_> = match &triangles_with_indices {
            //     Some(triangle_indices) => triangle_indices
            //         .iter()
            //         .enumerate()
            //         .map(|(triangle_index, triangle_indices)| {
            //             (
            //                 triangle_indices[0] as usize,
            //                 triangle_indices[1] as usize,
            //                 triangle_indices[2] as usize,
            //             )
            //         })
            //         .collect(),
            //     None => (0..triangle_count)
            //         .map(|triangle_index| {
            //             let first_vertex_index = triangle_index * 3;
            //             (
            //                 first_vertex_index,
            //                 first_vertex_index + 1,
            //                 first_vertex_index + 2,
            //             )
            //         })
            //         .collect(),
            // };
            let triangles_as_index_tuples: Vec<_> = (0..triangle_count)
                .map(|triangle_index| {
                    let i_left = triangle_index * 3;
                    match &indices {
                        Some(indices) => (
                            indices[i_left] as usize,
                            indices[i_left + 1] as usize,
                            indices[i_left + 2] as usize,
                        ),
                        None => (i_left, i_left + 1, i_left + 2),
                    }
                })
                .collect();

            // let all_triangles_have_tcs = !triangles.iter().any(|triangle| {
            //     !triangle
            //         .attributes()
            //         .any(|(semantic, _)| semantic == gltf::Semantic::TexCoords(0))
            // });
            // let no_triangles_have_tcs = !triangles.iter().any(|triangle| {
            //     triangle
            //         .attributes()
            //         .any(|(semantic, _)| semantic == gltf::Semantic::TexCoords(0))
            // });
            // if !all_triangles_have_tcs && !no_triangles_have_tcs {
            //     bail!("Expected all triangles to have normals or none to have normals");
            // }
            // let triangles_with_tex_coords: Vec<Vec<[f32; 2]>> = if all_triangles_have_tcs {
            //     triangles
            //         .iter()
            //         .map(|triangle| {
            //             let (_, accessor) = triangle
            //                 .attributes()
            //                 .find(|(semantic, _)| *semantic == gltf::Semantic::TexCoords(0))
            //                 .unwrap();
            //             let data_type = accessor.data_type();
            //             let dimensions = accessor.dimensions();
            //             if dimensions != gltf::accessor::Dimensions::Vec2 {
            //                 bail!("Expected vec2 data but found: {:?}", dimensions);
            //             }
            //             if data_type != gltf::accessor::DataType::F32 {
            //                 bail!("Expected f32 data but found: {:?}", data_type);
            //             }
            //             Ok(bytemuck::cast_slice(get_buffer_slice_from_accessor(accessor)).to_vec())
            //         })
            //         .collect::<Result<Vec<_>, _>>()?
            // } else {
            //     (0..triangle_count)
            //         .map(|_| vec![[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
            //         .collect()
            // };
            // let vertex_tex_coords: Vec<_> = triangles_with_tex_coords.iter().flatten().collect();

            let vertex_tex_coords: Vec<[f32; 2]> = first_triangles_prim
                .attributes()
                .find(|(semantic, _)| *semantic == gltf::Semantic::TexCoords(0))
                .map(|(_, accessor)| {
                    let data_type = accessor.data_type();
                    let dimensions = accessor.dimensions();
                    if dimensions != gltf::accessor::Dimensions::Vec2 {
                        bail!("Expected vec2 data but found: {:?}", dimensions);
                    }
                    if data_type != gltf::accessor::DataType::F32 {
                        bail!("Expected f32 data but found: {:?}", data_type);
                    }
                    Ok(bytemuck::cast_slice(get_buffer_slice_from_accessor(accessor)).to_vec())
                })
                .map_or(Ok(None), |v| v.map(Some))?
                .unwrap_or_else(|| (0..vertex_count).map(|_| [0.5, 0.5]).collect());

            // let all_triangles_have_normals = !triangles.iter().any(|triangle| {
            //     !triangle
            //         .attributes()
            //         .any(|(semantic, _)| semantic == gltf::Semantic::Normals)
            // });
            // let no_triangles_have_normals = !triangles.iter().any(|triangle| {
            //     triangle
            //         .attributes()
            //         .any(|(semantic, _)| semantic == gltf::Semantic::Normals)
            // });
            // if !all_triangles_have_normals && !no_triangles_have_normals {
            //     bail!("Expected all triangles to have normals or none to have normals");
            // }
            // let triangles_with_normals: Vec<Vec<[f32; 3]>> = if all_triangles_have_normals {
            //     triangles
            //         .iter()
            //         .map(|triangle| {
            //             let (_, accessor) = triangle
            //                 .attributes()
            //                 .find(|(semantic, _)| *semantic == gltf::Semantic::Normals)
            //                 .unwrap();
            //             let data_type = accessor.data_type();
            //             let dimensions = accessor.dimensions();
            //             if dimensions != gltf::accessor::Dimensions::Vec3 {
            //                 bail!("Expected vec3 data but found: {:?}", dimensions);
            //             }
            //             if data_type != gltf::accessor::DataType::F32 {
            //                 bail!("Expected f32 data but found: {:?}", data_type);
            //             }
            //             Ok(bytemuck::cast_slice(get_buffer_slice_from_accessor(accessor)).to_vec())
            //         })
            //         .collect::<Result<Vec<_>, _>>()?
            // } else {
            //     // compute normals
            //     // key is flattened vertex position, value is accumulated normal and count
            //     let mut vertex_normal_accumulators: HashMap<usize, ([f32; 3], usize)> =
            //         HashMap::new();
            //     // let triangles: Vec<_> = match &triangle_indices {
            //     //     Some(triangle_indices) => triangle_indices
            //     //         .iter()
            //     //         .enumerate()
            //     //         .map(|(triangle_index, triangle_indices)| {
            //     //             (
            //     //                 triangle_index,
            //     //                 triangle_indices
            //     //                     .iter()
            //     //                     .map(|&index| vertex_positions[index as usize])
            //     //                     .collect::<Vec<_>>(),
            //     //             )
            //     //         })
            //     //         .collect(),
            //     //     None => (0..triangle_count)
            //     //         .map(|triangle_index| {
            //     //             let first_vertex_index = triangle_index * 3;
            //     //             (
            //     //                 triangle_index,
            //     //                 (first_vertex_index..(first_vertex_index + 3))
            //     //                     .map(|index| vertex_positions[index])
            //     //                     .collect::<Vec<_>>(),
            //     //             )
            //     //         })
            //     //         .collect(),
            //     // };
            //     triangles_as_index_tuples.iter().copied().for_each(
            //         |(index_a, index_b, index_c)| {
            //             let a = vertex_positions[index_a];
            //             let b = vertex_positions[index_b];
            //             let c = vertex_positions[index_c];
            //             let a_to_b =
            //                 Vector3::new(b[0], b[1], b[2]) - Vector3::new(a[0], a[1], a[2]);
            //             let a_to_c =
            //                 Vector3::new(c[0], c[1], c[2]) - Vector3::new(a[0], a[1], a[2]);
            //             let normal = a_to_b.cross(a_to_c).normalize();
            //             vec![index_a, index_b, index_c]
            //                 .iter()
            //                 .for_each(|vertex_index| {
            //                     let (accumulated_normal, count) = vertex_normal_accumulators
            //                         .entry(*vertex_index)
            //                         .or_insert(([0.0, 0.0, 0.0], 0));
            //                     *accumulated_normal = [
            //                         accumulated_normal[0] + normal[0],
            //                         accumulated_normal[1] + normal[1],
            //                         accumulated_normal[2] + normal[2],
            //                     ];
            //                     *count += 1;
            //                 });
            //         },
            //     );
            //     (0..triangle_count)
            //         .map(|triangle_index| {
            //             (0..3)
            //                 .map(|index| {
            //                     let vertex_index = triangle_index * 3 + index;
            //                     let (accumulated_normal, count) =
            //                         vertex_normal_accumulators.get(&vertex_index).unwrap();
            //                     [
            //                         accumulated_normal[0] / *count as f32,
            //                         accumulated_normal[1] / *count as f32,
            //                         accumulated_normal[2] / *count as f32,
            //                     ]
            //                 })
            //                 .collect::<Vec<_>>()
            //         })
            //         .collect()
            // };
            // let vertex_normals: Vec<_> = triangles_with_normals.iter().flatten().collect();

            let vertex_normals: Vec<[f32; 3]> = first_triangles_prim
                .attributes()
                .find(|(semantic, _)| *semantic == gltf::Semantic::Normals)
                .map(|(_, accessor)| {
                    let data_type = accessor.data_type();
                    let dimensions = accessor.dimensions();
                    if dimensions != gltf::accessor::Dimensions::Vec3 {
                        bail!("Expected vec3 data but found: {:?}", dimensions);
                    }
                    if data_type != gltf::accessor::DataType::F32 {
                        bail!("Expected f32 data but found: {:?}", data_type);
                    }
                    Ok(bytemuck::cast_slice(get_buffer_slice_from_accessor(accessor)).to_vec())
                })
                .map_or(Ok(None), |v| v.map(Some))?
                .unwrap_or_else(|| {
                    // compute normals
                    // key is flattened vertex position, value is accumulated normal and count
                    let mut vertex_normal_accumulators: HashMap<usize, ([f32; 3], usize)> =
                        HashMap::new();
                    triangles_as_index_tuples.iter().copied().for_each(
                        |(index_a, index_b, index_c)| {
                            let a = vertex_positions[index_a];
                            let b = vertex_positions[index_b];
                            let c = vertex_positions[index_c];
                            let a_to_b =
                                Vector3::new(b[0], b[1], b[2]) - Vector3::new(a[0], a[1], a[2]);
                            let a_to_c =
                                Vector3::new(c[0], c[1], c[2]) - Vector3::new(a[0], a[1], a[2]);
                            let normal = a_to_b.cross(a_to_c).normalize();
                            vec![index_a, index_b, index_c]
                                .iter()
                                .for_each(|vertex_index| {
                                    let (accumulated_normal, count) = vertex_normal_accumulators
                                        .entry(*vertex_index)
                                        .or_insert(([0.0, 0.0, 0.0], 0));
                                    *accumulated_normal = [
                                        accumulated_normal[0] + normal[0],
                                        accumulated_normal[1] + normal[1],
                                        accumulated_normal[2] + normal[2],
                                    ];
                                    *count += 1;
                                });
                        },
                    );
                    (0..vertex_count)
                        .map(|vertex_index| {
                            let (accumulated_normal, count) =
                                vertex_normal_accumulators.get(&vertex_index).unwrap();
                            [
                                accumulated_normal[0] / *count as f32,
                                accumulated_normal[1] / *count as f32,
                                accumulated_normal[2] / *count as f32,
                            ]
                        })
                        .collect()
                });

            let triangles_with_tangents_and_bitangents: Vec<_> = triangles_as_index_tuples
                .iter()
                .copied()
                .map(|(index_a, index_b, index_c)| {
                    let points_with_attribs: Vec<_> = vec![index_a, index_b, index_c]
                        .iter()
                        .copied()
                        .map(|index| {
                            let pos = vertex_positions[index];
                            let norm = vertex_normals[index];
                            let tc = vertex_tex_coords[index];
                            (
                                Vector3::new(pos[0], pos[1], pos[2]),
                                Vector3::new(norm[0], norm[1], norm[2]),
                                Vector2::new(tc[0], tc[1]),
                            )
                        })
                        .collect();

                    let edge_1 = points_with_attribs[1].0 - points_with_attribs[0].0;
                    let edge_2 = points_with_attribs[2].0 - points_with_attribs[0].0;

                    let delta_uv_1 = points_with_attribs[1].2 - points_with_attribs[0].2;
                    let delta_uv_2 = points_with_attribs[2].2 - points_with_attribs[0].2;

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

                    (tangent, bitangent)
                })
                .collect();

            let triangles_with_all_data: Vec<_> = (0..triangle_count)
                .map(|triangle_index| {
                    let (tangent, bitangent) =
                        triangles_with_tangents_and_bitangents[triangle_index];
                    (0..3)
                        .map(|index| {
                            let vertex_index = triangle_index * 3 + index;
                            let to_arr = |vec: &Vector3<f32>| [vec.x, vec.y, vec.z];
                            Vertex {
                                position: vertex_positions[vertex_index],
                                normal: vertex_normals[vertex_index],
                                tex_coords: vertex_tex_coords[vertex_index],
                                tangent: to_arr(&tangent),
                                bitangent: to_arr(&bitangent),
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let vertices_with_all_data: Vec<_> =
                triangles_with_all_data.iter().flatten().cloned().collect();
            // let yo: Vec<Vertex> = vertices_with_all_data.iter().cloned().collect();

            // let normals: Vec<Vec<[f32; 3]>> = triangles
            //     .iter()
            //     .map(|triangle| {
            //     })
            //     .collect();

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("InstancedMeshComponent Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices_with_all_data),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = indices.map(|indices| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("InstancedMeshComponent Index Buffer"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                })
            });

            Ok(GeometryBuffer {
                vertex_buffer,
                index_buffer,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let textures = document
        .textures()
        .map(|texture| {
            let source_image_index = texture.source().index();
            let image = &images[source_image_index];
            Texture::from_decoded_image(
                device,
                queue,
                &image.pixels,
                (image.width, image.height),
                texture.name(),
                wgpu::TextureFormat::Rgba8Unorm.into(),
                false,
                &Default::default(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Scene {
        source_asset: gltf_asset,
        buffers: SceneBuffers { geometry, textures },
    })
}
