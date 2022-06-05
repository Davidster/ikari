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
    // TODO: add bind groups
}

pub struct SceneBuffers {
    // same order as the meshes in src
    pub bindable_mesh_data: Vec<BindableMeshData>,
    // same order as the textures in src
    pub textures: Vec<Texture>,
}

pub struct BindableMeshData {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: Option<wgpu::Buffer>,
    pub textures_bind_group: wgpu::BindGroup,
}

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

    let bindable_mesh_data = document
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
            let (vertex_buffer, index_buffer) =
                build_geometry_buffers(device, &first_triangles_prim, buffers)?;
            let textures_bind_group =
                build_textures_bind_group(device, queue, &first_triangles_prim, &textures)?;
            anyhow::Ok(BindableMeshData {
                vertex_buffer,
                index_buffer,
                textures_bind_group,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Scene {
        source_asset: gltf_asset,
        buffers: SceneBuffers {
            bindable_mesh_data,
            textures,
        },
    })
}

fn build_textures_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    triangles_prim: &gltf::mesh::Primitive,
    textures: &[Texture],
) -> Result<wgpu::BindGroup> {
    todo!()
    // let material = triangles_prim.material();
    // let alpha_mode = material.alpha_mode();
    // if alpha_mode != gltf::material::AlphaMode::Opaque {
    //     bail!("Only opaque  alpha mode is supported");
    // }
    // // TODO: support double-sided
    // // TODO: support base values, like base color, metallic, roughness, etc.

    // let material_normal_map = material.normal_texture().map(|info| info.texture());
    // let auto_generated_normal_map;
    // let normal_map = match material_normal_map {
    //     Some(normal_map) => &textures[normal_map.index()],
    //     None => {
    //         auto_generated_normal_map = Texture::flat_normal_map(device, queue)?;
    //         &auto_generated_normal_map
    //     }
    // };

    // let pbr_info = material.pbr_metallic_roughness();

    // let material_diffuse_texture = pbr_info.base_color_texture().map(|info| info.texture());
    // let auto_generated_diffuse_texture;
    // let diffuse_texture = match material_diffuse_texture {
    //     Some(diffuse_texture) => &textures[diffuse_texture.index()],
    //     None => {
    //         auto_generated_diffuse_texture =
    //             Texture::from_color(device, queue, [255, 255, 255, 255])?;
    //         &auto_generated_diffuse_texture
    //     }
    // };

    // let material_metallic_roughness_map = pbr_info
    //     .metallic_roughness_texture()
    //     .map(|info| info.texture());
    // let auto_generated_metallic_map;
    // let metallic_map = match material.metallic {
    //     Some(metallic_map) => metallic_map,
    //     None => {
    //         auto_generated_metallic_map = Texture::from_gray(device, queue, 0)?;
    //         &auto_generated_metallic_map
    //     }
    // };

    // let auto_generated_roughness_map;
    // let roughness_map = match material.roughness {
    //     Some(roughness_map) => roughness_map,
    //     None => {
    //         auto_generated_roughness_map = Texture::from_gray(device, queue, 127)?;
    //         &auto_generated_roughness_map
    //     }
    // };

    // let auto_generated_emissive_map;
    // let emissive_map = match material.emissive {
    //     Some(emissive_map) => emissive_map,
    //     None => {
    //         auto_generated_emissive_map = Texture::from_color(device, queue, [0, 0, 0, 255])?;
    //         &auto_generated_emissive_map
    //     }
    // };

    // let auto_generated_ambient_occlusion_map;
    // let ambient_occlusion_map = match material.ambient_occlusion {
    //     Some(ambient_occlusion_map) => ambient_occlusion_map,
    //     None => {
    //         auto_generated_ambient_occlusion_map = Texture::from_gray(device, queue, 255)?;
    //         &auto_generated_ambient_occlusion_map
    //     }
    // };

    // let textures_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    //     layout: textures_bind_group_layout,
    //     entries: &[
    //         wgpu::BindGroupEntry {
    //             binding: 0,
    //             resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
    //         },
    //         wgpu::BindGroupEntry {
    //             binding: 1,
    //             resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
    //         },
    //         wgpu::BindGroupEntry {
    //             binding: 2,
    //             resource: wgpu::BindingResource::TextureView(&normal_map.view),
    //         },
    //         wgpu::BindGroupEntry {
    //             binding: 3,
    //             resource: wgpu::BindingResource::Sampler(&normal_map.sampler),
    //         },
    //         wgpu::BindGroupEntry {
    //             binding: 4,
    //             resource: wgpu::BindingResource::TextureView(&metallic_map.view),
    //         },
    //         wgpu::BindGroupEntry {
    //             binding: 5,
    //             resource: wgpu::BindingResource::Sampler(&metallic_map.sampler),
    //         },
    //         wgpu::BindGroupEntry {
    //             binding: 6,
    //             resource: wgpu::BindingResource::TextureView(&roughness_map.view),
    //         },
    //         wgpu::BindGroupEntry {
    //             binding: 7,
    //             resource: wgpu::BindingResource::Sampler(&roughness_map.sampler),
    //         },
    //         wgpu::BindGroupEntry {
    //             binding: 8,
    //             resource: wgpu::BindingResource::TextureView(&emissive_map.view),
    //         },
    //         wgpu::BindGroupEntry {
    //             binding: 9,
    //             resource: wgpu::BindingResource::Sampler(&emissive_map.sampler),
    //         },
    //         wgpu::BindGroupEntry {
    //             binding: 10,
    //             resource: wgpu::BindingResource::TextureView(&ambient_occlusion_map.view),
    //         },
    //         wgpu::BindGroupEntry {
    //             binding: 11,
    //             resource: wgpu::BindingResource::Sampler(&ambient_occlusion_map.sampler),
    //         },
    //     ],
    //     label: Some("InstancedMeshComponent textures_bind_group"),
    // });
}

pub fn build_geometry_buffers(
    device: &wgpu::Device,
    triangles_prim: &gltf::mesh::Primitive,
    buffers: &[gltf::buffer::Data],
) -> Result<(wgpu::Buffer, Option<wgpu::Buffer>)> {
    let get_buffer_slice_from_accessor = |accessor: gltf::Accessor| {
        let buffer_view = accessor.view().unwrap();
        let buffer = &buffers[buffer_view.buffer().index()];
        let byte_range_start = buffer_view.offset() + accessor.offset();
        let byte_range_end = byte_range_start + (accessor.size() * accessor.count());
        let byte_range = byte_range_start..byte_range_end;
        &buffer[byte_range]
    };

    let slice_3_to_vec_3 = |slice: &[f32; 3]| Vector3::new(slice[0], slice[1], slice[2]);

    let vertex_positions: Vec<Vector3<f32>> = {
        let (_, accessor) = triangles_prim
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
        let positions: &[[f32; 3]] = bytemuck::cast_slice(get_buffer_slice_from_accessor(accessor));
        anyhow::Ok(positions.to_vec().iter().map(slice_3_to_vec_3).collect())
    }?;
    let vertex_count = vertex_positions.len();
    let triangle_count = vertex_count / 3;

    let indices: Option<Vec<u16>> = triangles_prim
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

    let vertex_tex_coords: Vec<[f32; 2]> = triangles_prim
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

    let vertex_normals: Vec<Vector3<f32>> = triangles_prim
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
            let normals: &[[f32; 3]] =
                bytemuck::cast_slice(get_buffer_slice_from_accessor(accessor));
            Ok(normals.to_vec().iter().map(slice_3_to_vec_3).collect())
        })
        .map_or(Ok(None), |v| v.map(Some))?
        .unwrap_or_else(|| {
            // compute normals
            // key is flattened vertex position, value is accumulated normal and count
            let mut vertex_normal_accumulators: HashMap<usize, (Vector3<f32>, usize)> =
                HashMap::new();
            triangles_as_index_tuples
                .iter()
                .copied()
                .for_each(|(index_a, index_b, index_c)| {
                    let a = vertex_positions[index_a];
                    let b = vertex_positions[index_b];
                    let c = vertex_positions[index_c];
                    let a_to_b = Vector3::new(b[0], b[1], b[2]) - Vector3::new(a[0], a[1], a[2]);
                    let a_to_c = Vector3::new(c[0], c[1], c[2]) - Vector3::new(a[0], a[1], a[2]);
                    let normal = a_to_b.cross(a_to_c).normalize();
                    vec![index_a, index_b, index_c]
                        .iter()
                        .for_each(|vertex_index| {
                            let (accumulated_normal, count) = vertex_normal_accumulators
                                .entry(*vertex_index)
                                .or_insert((Vector3::new(0.0, 0.0, 0.0), 0));
                            *accumulated_normal += normal;
                            *count += 1;
                        });
                });
            (0..vertex_count)
                .map(|vertex_index| {
                    let (accumulated_normal, count) =
                        vertex_normal_accumulators.get(&vertex_index).unwrap();
                    accumulated_normal / (*count as f32)
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
            let (tangent, bitangent) = triangles_with_tangents_and_bitangents[triangle_index];
            (0..3)
                .map(|index| {
                    let vertex_index = triangle_index * 3 + index;
                    let to_arr = |vec: &Vector3<f32>| [vec.x, vec.y, vec.z];
                    Vertex {
                        position: vertex_positions[vertex_index].into(),
                        normal: vertex_normals[vertex_index].into(),
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

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scene Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices_with_all_data),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = indices.map(|indices| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scene Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        })
    });

    Ok((vertex_buffer, index_buffer))
}
