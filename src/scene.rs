use std::collections::HashMap;
use std::collections::HashSet;
use std::iter::Map;

use anyhow::bail;
use anyhow::Result;
use cgmath::abs_diff_eq;
use cgmath::ulps_eq;
use cgmath::Matrix4;
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
    pub vertex_buffer: BufferAndLength,

    pub index_buffer: Option<BufferAndLength>,

    pub instance_buffer: BufferAndLength,

    pub textures_bind_group: wgpu::BindGroup,
}

pub struct BufferAndLength {
    pub buffer: wgpu::Buffer,
    pub length: usize,
}

pub fn build_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    five_texture_bind_group_layout: &wgpu::BindGroupLayout,
    gltf_asset: GltfAsset,
) -> Result<Scene> {
    let GltfAsset {
        document,
        buffers,
        images,
    } = &gltf_asset;

    let scene_index = document
        .default_scene()
        .map(|scene| scene.index())
        .unwrap_or(0);

    let textures = document
        .textures()
        .map(|texture| {
            let source_image_index = texture.source().index();
            let image_data = &images[source_image_index];
            dbg!("Creating texture: {:?}", texture.name());

            let (image_pixels, texture_format) = get_image_pixels(image_data)?;

            Texture::from_decoded_image(
                device,
                queue,
                &image_pixels,
                (image_data.width, image_data.height),
                texture.name(),
                texture_format.into(),
                false,
                &Default::default(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let node_transforms: Vec<_> = {
        // node index -> parent node index
        let parent_index_map: HashMap<usize, usize> = document
            .nodes()
            .flat_map(|parent_node| {
                let parent_node_index = parent_node.index();
                parent_node
                    .children()
                    .map(move |child_node| (child_node.index(), parent_node_index))
            })
            .collect();
        let nodes: Vec<_> = document.nodes().collect();
        nodes
            .iter()
            .map(|node| {
                let node_ancestry_list = get_ancestry_list(node.index(), &parent_index_map);
                node_ancestry_list
                    .iter()
                    .fold(Matrix4::identity(), |acc, node_index| {
                        let node = &nodes[*node_index];
                        let node_transform = gltf_transform_to_mat4(node.transform());
                        acc * node_transform
                    })
            })
            .collect()
    };

    let scene_nodes: Vec<_> = document
        .scenes()
        .find(|scene| scene.index() == scene_index)
        .ok_or_else(|| anyhow::anyhow!("Expected scene with index: {:?}", scene_index))?
        .nodes()
        .collect();

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
            let mesh_transforms: Vec<_> = scene_nodes
                .iter()
                .filter(|node| {
                    node.mesh().is_some() && node.mesh().unwrap().index() == mesh.index()
                })
                .map(|node| GpuMeshInstance::new(node_transforms[node.index()]))
                .collect();

            // TODO: create instance buffer from mesh_transforms
            let instance_buffer = BufferAndLength {
                buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("InstancedMeshComponent instance_buffer"),
                    contents: bytemuck::cast_slice(&mesh_transforms),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }),
                length: mesh_transforms.len(),
            };

            let textures_bind_group = build_textures_bind_group(
                device,
                queue,
                &first_triangles_prim,
                &textures,
                five_texture_bind_group_layout,
            )?;
            anyhow::Ok(BindableMeshData {
                vertex_buffer,
                index_buffer,
                instance_buffer,
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

fn get_image_pixels(image_data: &gltf::image::Data) -> Result<(Vec<u8>, wgpu::TextureFormat)> {
    let image_pixels = &image_data.pixels;
    if image_data.format == gltf::image::Format::R8G8B8 {
        let image =
            image::RgbImage::from_raw(image_data.width, image_data.height, image_pixels.to_vec())
                .ok_or_else(|| anyhow::anyhow!("Failed to decode R8G8B8 image"))?;
        let image_pixels_conv = image::DynamicImage::ImageRgb8(image).to_rgba8().to_vec();
        Ok((image_pixels_conv, wgpu::TextureFormat::Rgba8Unorm))
    } else {
        let texture_format = texture_format_to_wgpu(image_data.format)?;
        Ok((image_pixels.to_vec(), texture_format))
    }
}

fn texture_format_to_wgpu(format: gltf::image::Format) -> Result<wgpu::TextureFormat> {
    match format {
        gltf::image::Format::R8 => Ok(wgpu::TextureFormat::R8Unorm),
        gltf::image::Format::R8G8 => Ok(wgpu::TextureFormat::Rg8Unorm),
        gltf::image::Format::R8G8B8A8 => Ok(wgpu::TextureFormat::Rgba8Unorm),
        gltf::image::Format::R16 => Ok(wgpu::TextureFormat::R16Unorm),
        gltf::image::Format::R16G16 => Ok(wgpu::TextureFormat::Rg16Unorm),
        gltf::image::Format::R16G16B16A16 => Ok(wgpu::TextureFormat::Rgba16Unorm),
        _ => bail!("Unsupported texture format: {:?}", format),
    }
}

fn get_ancestry_list(node_index: usize, parent_index_map: &HashMap<usize, usize>) -> Vec<usize> {
    get_ancestry_list_impl(node_index, parent_index_map, Vec::new())
}

fn get_ancestry_list_impl(
    node_index: usize,
    parent_index_map: &HashMap<usize, usize>,
    _acc: Vec<usize>,
) -> Vec<usize> {
    let with_self: Vec<_> = _acc
        .iter()
        .chain(vec![node_index].iter())
        .copied()
        .collect();
    match parent_index_map.get(&node_index) {
        Some(parent_index) => {
            get_ancestry_list_impl(*parent_index, parent_index_map, with_self).to_vec()
        }
        None => with_self,
    }
}

fn gltf_transform_to_mat4(gltf_transform: gltf::scene::Transform) -> Matrix4<f32> {
    match gltf_transform {
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => {
            let transform = transform::Transform::new();
            transform.set_position(translation.into());
            transform.set_rotation(rotation.into());
            transform.set_scale(scale.into());
            transform.matrix.get()
        }
        gltf::scene::Transform::Matrix { matrix } => Matrix4::from(matrix),
    }
}

fn build_textures_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    triangles_prim: &gltf::mesh::Primitive,
    textures: &[Texture],
    five_texture_bind_group_layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::BindGroup> {
    let material = triangles_prim.material();
    // TODO: support alpha modes
    // let alpha_mode = material.alpha_mode();
    // if alpha_mode != gltf::material::AlphaMode::Opaque {
    //     bail!("Only opaque alpha mode is supported");
    // }
    // TODO: support double-sided
    // TODO: support base values, like base color, metallic, roughness, etc.

    let pbr_info = material.pbr_metallic_roughness();

    let material_diffuse_texture = pbr_info.base_color_texture().map(|info| info.texture());
    let auto_generated_diffuse_texture;
    let diffuse_texture = match material_diffuse_texture {
        Some(diffuse_texture) => &textures[diffuse_texture.index()],
        None => {
            auto_generated_diffuse_texture =
                Texture::from_color(device, queue, [255, 255, 255, 255])?;
            &auto_generated_diffuse_texture
        }
    };

    let material_metallic_roughness_map = pbr_info
        .metallic_roughness_texture()
        .map(|info| info.texture());
    let auto_generated_metallic_roughness_map;
    let metallic_roughness_map = match material_metallic_roughness_map {
        Some(metallic_roughness_map) => &textures[metallic_roughness_map.index()],
        None => {
            auto_generated_metallic_roughness_map =
                Texture::from_color(device, queue, [255, 127, 0, 255])?;
            &auto_generated_metallic_roughness_map
        }
    };

    let material_normal_map = material.normal_texture().map(|info| info.texture());
    let auto_generated_normal_map;
    let normal_map = match material_normal_map {
        Some(normal_map) => &textures[normal_map.index()],
        None => {
            auto_generated_normal_map = Texture::flat_normal_map(device, queue)?;
            &auto_generated_normal_map
        }
    };

    let material_emissive_map = material.emissive_texture().map(|info| info.texture());
    let auto_generated_emissive_map;
    let emissive_map = match material_emissive_map {
        Some(emissive_map) => &textures[emissive_map.index()],
        None => {
            auto_generated_emissive_map = Texture::from_color(device, queue, [0, 0, 0, 255])?;
            &auto_generated_emissive_map
        }
    };

    let material_ambient_occlusion_map = material.occlusion_texture().map(|info| info.texture());
    let auto_generated_ambient_occlusion_map;
    let ambient_occlusion_map = match material_ambient_occlusion_map {
        Some(ambient_occlusion_map) => &textures[ambient_occlusion_map.index()],
        None => {
            auto_generated_ambient_occlusion_map =
                Texture::from_color(device, queue, [255, 255, 255, 255])?;
            &auto_generated_ambient_occlusion_map
        }
    };

    let textures_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: five_texture_bind_group_layout,
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

pub fn build_geometry_buffers(
    device: &wgpu::Device,
    triangles_prim: &gltf::mesh::Primitive,
    buffers: &[gltf::buffer::Data],
) -> Result<(BufferAndLength, Option<BufferAndLength>)> {
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
    let vertex_position_count = vertex_positions.len();

    let indices: Option<Vec<u16>> = triangles_prim
        .indices()
        .map(|accessor| {
            let data_type = accessor.data_type();
            let buffer_slice = get_buffer_slice_from_accessor(accessor);

            let indices: Vec<u16> = match data_type {
                gltf::accessor::DataType::U16 => {
                    anyhow::Ok(bytemuck::cast_slice(buffer_slice).to_vec())
                }
                gltf::accessor::DataType::U8 => {
                    anyhow::Ok(buffer_slice.iter().map(|&x| x as u16).collect::<Vec<u16>>())
                }
                gltf::accessor::DataType::U32 => anyhow::Ok(
                    bytemuck::cast_slice::<_, u32>(buffer_slice)
                        .iter()
                        .map(|&x| x as u16)
                        .collect(),
                ),
                data_type => {
                    bail!("Expected u16 or u8 indices but found: {:?}", data_type)
                }
            }?;
            anyhow::Ok(indices)
        })
        .map_or(Ok(None), |v| v.map(Some))?;

    let triangle_count = indices
        .as_ref()
        .map(|indices| indices.len() / 3)
        .unwrap_or(vertex_position_count / 3);

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
        .unwrap_or_else(|| (0..vertex_position_count).map(|_| [0.5, 0.5]).collect());
    let vertex_tex_coord_count = vertex_tex_coords.len();

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
            (0..vertex_position_count)
                .map(|vertex_index| {
                    let (accumulated_normal, count) =
                        vertex_normal_accumulators.get(&vertex_index).unwrap();
                    accumulated_normal / (*count as f32)
                })
                .collect()
        });
    let vertex_normal_count = vertex_normals.len();

    if vertex_normal_count != vertex_position_count {
        bail!(
            "Expected vertex normals for every vertex but found: vertex_position_count({:?}) != vertex_normal_count({:?})",
            vertex_position_count,
            vertex_normal_count
        );
    }
    if vertex_tex_coord_count != vertex_position_count {
        bail!(
            "Expected vertex normals for every vertex but found: vertex_position_count({:?}) != vertex_tex_coord_count({:?})",
            vertex_position_count,
            vertex_tex_coord_count
        );
    }

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

            if abs_diff_eq!(delta_uv_1.x, 0.0, epsilon = 0.00001)
                && abs_diff_eq!(delta_uv_2.x, 0.0, epsilon = 0.00001)
                && abs_diff_eq!(delta_uv_1.y, 0.0, epsilon = 0.00001)
                && abs_diff_eq!(delta_uv_2.y, 0.0, epsilon = 0.00001)
            {
                return (Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0));
            }

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

    let triangles_with_all_data: Vec<_> = triangles_as_index_tuples
        .iter()
        .copied()
        .enumerate()
        .map(|(triangle_index, (index_a, index_b, index_c))| {
            let (tangent, bitangent) = triangles_with_tangents_and_bitangents[triangle_index];
            vec![index_a, index_b, index_c]
                .iter()
                .map(|index| {
                    // let vertex_index = triangle_index * 3 + index;
                    let to_arr = |vec: &Vector3<f32>| [vec.x, vec.y, vec.z];
                    Vertex {
                        position: vertex_positions[*index].into(),
                        normal: vertex_normals[*index].into(),
                        tex_coords: vertex_tex_coords[*index],
                        tangent: to_arr(&tangent),
                        bitangent: to_arr(&bitangent),
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let indices: Option<Vec<u16>> = Some((0..(triangle_count * 3)).map(|i| i as u16).collect());

    let vertices_with_all_data: Vec<_> =
        triangles_with_all_data.iter().flatten().cloned().collect();

    // dbg!(&vertex_positions);
    // dbg!(&vertices_with_all_data);
    // dbg!(&indices);
    // panic!();

    let vertex_buffer = BufferAndLength {
        buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scene Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices_with_all_data),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        length: vertices_with_all_data.len(),
    };

    let index_buffer = indices.map(|indices| BufferAndLength {
        buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scene Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        }),
        length: indices.len(),
    });

    Ok((vertex_buffer, index_buffer))
}
