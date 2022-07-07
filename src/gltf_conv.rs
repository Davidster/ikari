use std::collections::HashMap;

use anyhow::{bail, Result};
use cgmath::{abs_diff_eq, Matrix4, Vector2, Vector3, Vector4};
use wgpu::util::DeviceExt;

use super::*;

pub fn build_scene(
    base_renderer_state: &BaseRendererState,
    (document, buffers, images): (
        &gltf::Document,
        &Vec<gltf::buffer::Data>,
        &Vec<gltf::image::Data>,
    ),
) -> Result<(GameScene, RenderScene)> {
    let device = &base_renderer_state.device;
    let queue = &base_renderer_state.queue;
    let pbr_textures_bind_group_layout = &base_renderer_state.pbr_textures_bind_group_layout;

    let scene_index = document
        .default_scene()
        .map(|scene| scene.index())
        .unwrap_or(0);

    let materials: Vec<_> = document.materials().collect();

    let textures = document
        .textures()
        .map(|texture| {
            let source_image_index = texture.source().index();
            let image_data = &images[source_image_index];

            let srgb = materials.iter().any(|material| {
                vec![
                    material.emissive_texture(),
                    material.pbr_metallic_roughness().base_color_texture(),
                ]
                .iter()
                .flatten()
                .any(|texture_info| texture_info.texture().index() == texture.index())
            });

            let (image_pixels, texture_format) = get_image_pixels(image_data, srgb)?;

            let gltf_sampler = texture.sampler();
            let default_sampler = SamplerDescriptor::default();
            let address_mode_u = sampler_wrapping_mode_to_wgpu(gltf_sampler.wrap_s());
            let address_mode_v = sampler_wrapping_mode_to_wgpu(gltf_sampler.wrap_t());
            let mag_filter = gltf_sampler
                .mag_filter()
                .map(|gltf_mag_filter| match gltf_mag_filter {
                    gltf::texture::MagFilter::Nearest => wgpu::FilterMode::Nearest,
                    gltf::texture::MagFilter::Linear => wgpu::FilterMode::Linear,
                })
                .unwrap_or(default_sampler.mag_filter);
            let (min_filter, mipmap_filter) = gltf_sampler
                .min_filter()
                .map(|gltf_min_filter| match gltf_min_filter {
                    gltf::texture::MinFilter::Nearest => {
                        (wgpu::FilterMode::Nearest, default_sampler.mipmap_filter)
                    }
                    gltf::texture::MinFilter::Linear => {
                        (wgpu::FilterMode::Linear, default_sampler.mipmap_filter)
                    }
                    gltf::texture::MinFilter::NearestMipmapNearest => {
                        (wgpu::FilterMode::Nearest, wgpu::FilterMode::Nearest)
                    }
                    gltf::texture::MinFilter::LinearMipmapNearest => {
                        (wgpu::FilterMode::Linear, wgpu::FilterMode::Nearest)
                    }
                    gltf::texture::MinFilter::NearestMipmapLinear => {
                        (wgpu::FilterMode::Nearest, wgpu::FilterMode::Linear)
                    }
                    gltf::texture::MinFilter::LinearMipmapLinear => {
                        (wgpu::FilterMode::Linear, wgpu::FilterMode::Linear)
                    }
                })
                .unwrap_or((default_sampler.min_filter, default_sampler.mipmap_filter));

            Texture::from_decoded_image(
                device,
                queue,
                &image_pixels,
                (image_data.width, image_data.height),
                texture.name(),
                texture_format.into(),
                true,
                // &SamplerDescriptor(wgpu::SamplerDescriptor {
                //     address_mode_u: wgpu::AddressMode::ClampToEdge,
                //     address_mode_v: wgpu::AddressMode::ClampToEdge,
                //     address_mode_w: wgpu::AddressMode::ClampToEdge,
                //     mag_filter: wgpu::FilterMode::Nearest,
                //     min_filter: wgpu::FilterMode::Nearest,
                //     mipmap_filter: wgpu::FilterMode::Nearest,
                //     ..Default::default()
                // }),
                &SamplerDescriptor(wgpu::SamplerDescriptor {
                    address_mode_u,
                    address_mode_v,
                    mag_filter,
                    min_filter,
                    mipmap_filter,
                    ..Default::default()
                }),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

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

    let skins: Vec<_> = document
        .skins()
        .map(|skin| {
            let bone_node_indices: Vec<_> = skin.joints().map(|joint| joint.index()).collect();
            anyhow::Ok(Skin {
                bone_inverse_bind_matrices: skin
                    .inverse_bind_matrices()
                    .map(|accessor| {
                        let data_type = accessor.data_type();
                        let dimensions = accessor.dimensions();
                        if dimensions != gltf::accessor::Dimensions::Mat4 {
                            bail!("Expected mat4 data but found: {:?}", dimensions);
                        }
                        if data_type != gltf::accessor::DataType::F32 {
                            bail!("Expected f32 data but found: {:?}", data_type);
                        }
                        let matrices_u8 = get_buffer_slice_from_accessor(accessor, buffers);
                        Ok(bytemuck::cast_slice::<_, [[f32; 4]; 4]>(&matrices_u8)
                            .to_vec()
                            .iter()
                            .cloned()
                            .map(Matrix4::from)
                            .collect())
                    })
                    .transpose()?
                    .unwrap_or_else(|| {
                        (0..bone_node_indices.len())
                            .map(|_| Matrix4::one())
                            .collect()
                    }),
                bone_node_indices,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let scene_nodes: Vec<_> = get_full_node_list(
        document
            .scenes()
            .find(|scene| scene.index() == scene_index)
            .ok_or_else(|| anyhow::anyhow!("Expected scene with index: {:?}", scene_index))?,
    );

    let meshes: Vec<_> = document.meshes().collect();

    let mut binded_pbr_meshes: Vec<BindedPbrMesh> = Vec::new();
    // gltf node index -> game node
    let mut node_mesh_links: HashMap<usize, Vec<usize>> = HashMap::new();

    for (binded_pbr_mesh_index, (mesh, primitive_group)) in meshes
        .iter()
        .flat_map(|mesh| mesh.primitives().map(|prim| (&meshes[mesh.index()], prim)))
        .filter(|(_, prim)| {
            prim.mode() == gltf::mesh::Mode::Triangles
                && (prim.material().alpha_mode() == gltf::material::AlphaMode::Opaque
                    || prim.material().alpha_mode() == gltf::material::AlphaMode::Mask)
        })
        .enumerate()
    {
        let (textures_bind_group, dynamic_pbr_params) = build_textures_bind_group(
            device,
            queue,
            &primitive_group.material(),
            &textures,
            pbr_textures_bind_group_layout,
        )?;

        let (vertex_buffer, index_buffer) =
            build_geometry_buffers(device, &primitive_group, buffers)?;
        let initial_instances: Vec<_> = scene_nodes
            .iter()
            .filter(|node| node.mesh().is_some() && node.mesh().unwrap().index() == mesh.index())
            .collect();
        let initial_instance_buffer: Vec<u8> = (0..(initial_instances.len()
            * std::mem::size_of::<GpuPbrMeshInstance>()))
            .map(|_| 0u8)
            .collect();

        let instance_buffer = BufferAndLength {
            buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("InstancedMeshComponent instance_buffer"),
                contents: bytemuck::cast_slice(&initial_instance_buffer),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
            length: initial_instances.len(),
        };

        let primitive_mode = crate::render_scene::PrimitiveMode::Triangles;

        let alpha_mode = match primitive_group.material().alpha_mode() {
            gltf::material::AlphaMode::Opaque => crate::render_scene::AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => crate::render_scene::AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => {
                todo!("Alpha blending isn't yet supported")
            }
        };

        for gltf_node in initial_instances.iter() {
            let binded_pbr_mesh_indices =
                node_mesh_links.entry(gltf_node.index()).or_insert(vec![]);
            binded_pbr_mesh_indices.push(binded_pbr_mesh_index);
        }

        binded_pbr_meshes.push(BindedPbrMesh {
            vertex_buffer,
            index_buffer,
            dynamic_pbr_params,
            instance_buffer,
            textures_bind_group,
            primitive_mode,
            alpha_mode,
        });
    }

    // it is important that the node indices from the gltf document are preserved
    // for any of the other stuff that refers to the nodes by index such as the animations
    let nodes: Vec<_> = document
        .nodes()
        .map(|node| GameNode {
            transform: crate::transform::Transform::from(node.transform()),
            renderer_skin_index: node.skin().map(|skin| skin.index()),
            mesh: node_mesh_links
                .get(&node.index())
                .map(|mesh_indices| GameNodeMesh::Pbr {
                    mesh_indices: mesh_indices.clone(),
                    material_override: None,
                }),
        })
        .collect();

    let animations = get_animations(document, buffers)?;

    Ok((
        GameScene {
            parent_index_map,
            nodes,
        },
        RenderScene {
            buffers: SceneBuffers {
                binded_pbr_meshes,
                binded_unlit_meshes: vec![],
                textures,
            },

            skins,
            animations,
        },
    ))
}

fn get_image_pixels(
    image_data: &gltf::image::Data,
    srgb: bool,
) -> Result<(Vec<u8>, wgpu::TextureFormat)> {
    let image_pixels = &image_data.pixels;
    match (image_data.format, srgb) {
        (gltf::image::Format::R8G8B8, srgb) => {
            let image = image::RgbImage::from_raw(
                image_data.width,
                image_data.height,
                image_pixels.to_vec(),
            )
            .ok_or_else(|| anyhow::anyhow!("Failed to decode R8G8B8 image"))?;
            let image_pixels_conv = image::DynamicImage::ImageRgb8(image).to_rgba8().to_vec();
            Ok((
                image_pixels_conv,
                if srgb {
                    wgpu::TextureFormat::Rgba8UnormSrgb
                } else {
                    wgpu::TextureFormat::Rgba8Unorm
                },
            ))
        }
        (gltf::image::Format::R8G8, true) => {
            // srgb is true meaning this is a color image, so the red channel is luma and g is alpha
            Ok((
                image_pixels
                    .chunks(2)
                    .flat_map(|pixel| [pixel[0], pixel[0], pixel[0], pixel[1]])
                    .collect(),
                wgpu::TextureFormat::Rgba8UnormSrgb,
            ))
        }
        _ => {
            let texture_format = texture_format_to_wgpu(image_data.format, srgb)?;
            Ok((image_pixels.to_vec(), texture_format))
        }
    }
}

fn texture_format_to_wgpu(format: gltf::image::Format, srgb: bool) -> Result<wgpu::TextureFormat> {
    match (format, srgb) {
        (gltf::image::Format::R8, false) => Ok(wgpu::TextureFormat::R8Unorm),
        (gltf::image::Format::R8G8, false) => Ok(wgpu::TextureFormat::Rg8Unorm),
        (gltf::image::Format::R8G8B8A8, false) => Ok(wgpu::TextureFormat::Rgba8Unorm),
        (gltf::image::Format::R8G8B8A8, true) => Ok(wgpu::TextureFormat::Rgba8UnormSrgb),
        (gltf::image::Format::R16, false) => Ok(wgpu::TextureFormat::R16Unorm),
        (gltf::image::Format::R16G16, false) => Ok(wgpu::TextureFormat::Rg16Unorm),
        (gltf::image::Format::R16G16B16A16, false) => Ok(wgpu::TextureFormat::Rgba16Unorm),
        _ => bail!(
            "Unsupported texture format combo: {:?}, srgb={:?}",
            format,
            srgb
        ),
    }
}

fn sampler_wrapping_mode_to_wgpu(wrapping_mode: gltf::texture::WrappingMode) -> wgpu::AddressMode {
    match wrapping_mode {
        gltf::texture::WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        gltf::texture::WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        gltf::texture::WrappingMode::Repeat => wgpu::AddressMode::Repeat,
    }
}

fn get_full_node_list(scene: gltf::scene::Scene) -> Vec<gltf::scene::Node> {
    scene
        .nodes()
        .flat_map(|node| get_full_node_list_impl(node, Vec::new()))
        .collect()
}

fn get_full_node_list_impl<'a>(
    node: gltf::scene::Node<'a>,
    acc: Vec<gltf::scene::Node<'a>>,
) -> Vec<gltf::scene::Node<'a>> {
    let acc_with_self: Vec<_> = acc
        .iter()
        .chain(vec![node.clone()].iter())
        .cloned()
        .collect();
    if node.children().count() == 0 {
        acc_with_self
    } else {
        node.children()
            .flat_map(|child| get_full_node_list_impl(child, acc_with_self.clone()))
            .collect()
    }
}

fn build_textures_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    material: &gltf::material::Material,
    textures: &[Texture],
    five_texture_bind_group_layout: &wgpu::BindGroupLayout,
) -> Result<(wgpu::BindGroup, DynamicPbrParams)> {
    let pbr_info = material.pbr_metallic_roughness();

    let material_diffuse_texture = pbr_info.base_color_texture().map(|info| info.texture());
    // material_diffuse_texture = None;
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
    // material_metallic_roughness_map = None;
    let auto_generated_metallic_roughness_map;
    let metallic_roughness_map = match material_metallic_roughness_map {
        Some(metallic_roughness_map) => &textures[metallic_roughness_map.index()],
        None => {
            auto_generated_metallic_roughness_map =
                Texture::from_color(device, queue, [255, 255, 255, 255])?;
            &auto_generated_metallic_roughness_map
        }
    };

    let material_normal_map = material.normal_texture().map(|info| info.texture());
    // material_normal_map = None;
    let auto_generated_normal_map;
    let normal_map = match material_normal_map {
        Some(normal_map) => &textures[normal_map.index()],
        None => {
            auto_generated_normal_map = Texture::flat_normal_map(device, queue)?;
            &auto_generated_normal_map
        }
    };

    let material_emissive_map = material.emissive_texture().map(|info| info.texture());
    // material_emissive_map = None;
    let auto_generated_emissive_map;
    let emissive_map = match material_emissive_map {
        Some(emissive_map) => &textures[emissive_map.index()],
        None => {
            auto_generated_emissive_map = Texture::from_color(device, queue, [255, 255, 255, 255])?;
            &auto_generated_emissive_map
        }
    };

    let material_ambient_occlusion_map = material.occlusion_texture().map(|info| info.texture());
    // material_ambient_occlusion_map = None;
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

    let dynamic_pbr_params = DynamicPbrParams {
        base_color_factor: Vector4::from(pbr_info.base_color_factor()),
        emissive_factor: Vector3::from(material.emissive_factor()),
        metallic_factor: pbr_info.metallic_factor(),
        roughness_factor: pbr_info.roughness_factor(),
        normal_scale: material
            .normal_texture()
            .map(|info| info.scale())
            .unwrap_or(DynamicPbrParams::default().normal_scale),
        occlusion_strength: material
            .occlusion_texture()
            .map(|info| info.strength())
            .unwrap_or(DynamicPbrParams::default().occlusion_strength),
        alpha_cutoff: match material.alpha_mode() {
            gltf::material::AlphaMode::Mask => material.alpha_cutoff().unwrap_or(0.5),
            _ => DynamicPbrParams::default().alpha_cutoff,
        },
    };

    Ok((textures_bind_group, dynamic_pbr_params))
}

pub fn get_buffer_slice_from_accessor(
    accessor: gltf::Accessor,
    buffers: &[gltf::buffer::Data],
) -> Vec<u8> {
    let buffer_view = accessor.view().unwrap();
    let buffer = &buffers[buffer_view.buffer().index()];
    let first_byte_offset = buffer_view.offset() + accessor.offset();
    let stride = buffer_view.stride().unwrap_or_else(|| accessor.size());
    (0..accessor.count())
        .flat_map(|i| {
            let byte_range_start = first_byte_offset + i * stride;
            let byte_range_end = byte_range_start + accessor.size();
            let byte_range = byte_range_start..byte_range_end;
            (&buffer[byte_range]).to_vec()
        })
        .collect()
}

pub fn build_geometry_buffers(
    device: &wgpu::Device,
    primitive_group: &gltf::mesh::Primitive,
    buffers: &[gltf::buffer::Data],
) -> Result<(BufferAndLength, Option<BufferAndLength>)> {
    // let get_buffer_slice_from_accessor = |accessor: gltf::Accessor| {
    //     let buffer_view = accessor.view().unwrap();
    //     let buffer = &buffers[buffer_view.buffer().index()];
    //     let first_byte_offset = buffer_view.offset() + accessor.offset();
    //     let stride = buffer_view.stride().unwrap_or_else(|| accessor.size());
    //     (0..accessor.count())
    //         .flat_map(|i| {
    //             let byte_range_start = first_byte_offset + i * stride;
    //             let byte_range_end = byte_range_start + accessor.size();
    //             let byte_range = byte_range_start..byte_range_end;
    //             (&buffer[byte_range]).to_vec()
    //         })
    //         .collect::<Vec<_>>()
    // };

    let vertex_positions: Vec<Vector3<f32>> = {
        let (_, accessor) = primitive_group
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
        let positions_u8 = get_buffer_slice_from_accessor(accessor, buffers);
        let positions: &[[f32; 3]] = bytemuck::cast_slice(&positions_u8);
        anyhow::Ok(
            positions
                .to_vec()
                .iter()
                .copied()
                .map(Vector3::from)
                .collect(),
        )
    }?;
    let vertex_position_count = vertex_positions.len();
    let _bounding_box = {
        let max_x = vertex_positions
            .iter()
            .map(|pos| pos.x)
            .max_by(|a, b| a.partial_cmp(b).unwrap());
        let max_y = vertex_positions
            .iter()
            .map(|pos| pos.y)
            .max_by(|a, b| a.partial_cmp(b).unwrap());
        let max_z = vertex_positions
            .iter()
            .map(|pos| pos.z)
            .max_by(|a, b| a.partial_cmp(b).unwrap());
        let min_x = vertex_positions
            .iter()
            .map(|pos| pos.x)
            .min_by(|a, b| a.partial_cmp(b).unwrap());
        let min_y = vertex_positions
            .iter()
            .map(|pos| pos.y)
            .min_by(|a, b| a.partial_cmp(b).unwrap());
        let min_z = vertex_positions
            .iter()
            .map(|pos| pos.z)
            .min_by(|a, b| a.partial_cmp(b).unwrap());
        (
            Vector3::new(min_x, min_y, min_z),
            Vector3::new(max_x, max_y, max_z),
        )
    };

    let indices: Option<Vec<u16>> = primitive_group
        .indices()
        .map(|accessor| {
            let data_type = accessor.data_type();
            let buffer_slice = get_buffer_slice_from_accessor(accessor, buffers);

            let indices: Vec<u16> = match data_type {
                gltf::accessor::DataType::U16 => {
                    anyhow::Ok(bytemuck::cast_slice(&buffer_slice).to_vec())
                }
                gltf::accessor::DataType::U8 => {
                    anyhow::Ok(buffer_slice.iter().map(|&x| x as u16).collect::<Vec<u16>>())
                }
                gltf::accessor::DataType::U32 => {
                    let as_u32 = bytemuck::cast_slice::<_, u32>(&buffer_slice);
                    let as_u16: Vec<_> = as_u32
                        .iter()
                        .map(|&x| u16::try_from(x))
                        .collect::<Result<Vec<_>, _>>()?;
                    anyhow::Ok(as_u16)
                }
                data_type => {
                    bail!("Expected u32, u16 or u8 indices but found: {:?}", data_type)
                }
            }?;
            anyhow::Ok(indices)
        })
        .transpose()?;

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

    let vertex_tex_coords: Vec<[f32; 2]> = primitive_group
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
            Ok(bytemuck::cast_slice(&get_buffer_slice_from_accessor(accessor, buffers)).to_vec())
        })
        .transpose()?
        .unwrap_or_else(|| (0..vertex_position_count).map(|_| [0.5, 0.5]).collect());
    let vertex_tex_coord_count = vertex_tex_coords.len();

    let vertex_colors: Vec<[f32; 4]> = primitive_group
        .attributes()
        .find(|(semantic, _)| *semantic == gltf::Semantic::Colors(0))
        .map(|(_, accessor)| {
            let data_type = accessor.data_type();
            let dimensions = accessor.dimensions();
            if dimensions != gltf::accessor::Dimensions::Vec4 {
                bail!("Expected vec4 data but found: {:?}", dimensions);
            }
            let buffer_slice = match data_type {
                gltf::accessor::DataType::F32 => {
                    Some(get_buffer_slice_from_accessor(accessor, buffers))
                }
                gltf::accessor::DataType::U8 => Some(
                    bytemuck::cast_slice::<_, u8>(
                        &get_buffer_slice_from_accessor(accessor, buffers)
                            .iter()
                            .map(|res| *res as f32 / 255.0)
                            .collect::<Vec<_>>(),
                    )
                    .to_vec(),
                ),
                gltf::accessor::DataType::U16 => Some(
                    bytemuck::cast_slice::<_, u8>(
                        &bytemuck::cast_slice::<_, u16>(&get_buffer_slice_from_accessor(
                            accessor, buffers,
                        ))
                        .iter()
                        .map(|res| *res as f32 / 255.0)
                        .collect::<Vec<_>>(),
                    )
                    .to_vec(),
                ),
                _ => None,
            }
            .ok_or_else(|| {
                anyhow::anyhow!("Expected f32, u8, or u16 data but found: {:?}", data_type)
            })?;
            Ok(bytemuck::cast_slice(&buffer_slice).to_vec())
        })
        .transpose()?
        .unwrap_or_else(|| {
            (0..vertex_position_count)
                .map(|_| [1.0, 1.0, 1.0, 1.0])
                .collect()
        });
    let vertex_color_count = vertex_colors.len();

    let vertex_normals: Vec<Vector3<f32>> = primitive_group
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
            let normals_u8 = get_buffer_slice_from_accessor(accessor, buffers);
            let normals: &[[f32; 3]] = bytemuck::cast_slice(&normals_u8);
            Ok(normals
                .to_vec()
                .iter()
                .copied()
                .map(Vector3::from)
                .collect())
        })
        .transpose()?
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

    let vertex_bone_indices: Vec<[u32; 4]> = primitive_group
        .attributes()
        .find(|(semantic, _)| *semantic == gltf::Semantic::Joints(0))
        .map(|(_, accessor)| {
            let data_type = accessor.data_type();
            let dimensions = accessor.dimensions();
            if dimensions != gltf::accessor::Dimensions::Vec4 {
                bail!("Expected vec4 data but found: {:?}", dimensions);
            }
            let bone_indices_u16 = match data_type {
                gltf::accessor::DataType::U16 => Some(
                    bytemuck::cast_slice::<_, u16>(&get_buffer_slice_from_accessor(
                        accessor, buffers,
                    ))
                    .to_vec(),
                ),
                gltf::accessor::DataType::U8 => Some(
                    get_buffer_slice_from_accessor(accessor, buffers)
                        .iter()
                        .map(|res| *res as u16)
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            }
            .ok_or_else(|| anyhow::anyhow!("Expected u8 or u16 data but found: {:?}", data_type))?;
            let bone_indices_u16_grouped = bytemuck::cast_slice::<_, [u16; 4]>(&bone_indices_u16);
            Ok(bone_indices_u16_grouped
                .to_vec()
                .iter()
                .map(|indices| {
                    [
                        indices[0] as u32,
                        indices[1] as u32,
                        indices[2] as u32,
                        indices[3] as u32,
                    ]
                })
                .collect())
        })
        .transpose()?
        .unwrap_or_else(|| (0..vertex_position_count).map(|_| [0, 0, 0, 0]).collect());
    let vertex_bone_indices_count = vertex_bone_indices.len();

    let vertex_bone_weights: Vec<[f32; 4]> = primitive_group
        .attributes()
        .find(|(semantic, _)| *semantic == gltf::Semantic::Weights(0))
        .map(|(_, accessor)| {
            let data_type = accessor.data_type();
            let dimensions = accessor.dimensions();
            if dimensions != gltf::accessor::Dimensions::Vec4 {
                bail!("Expected vec4 data but found: {:?}", dimensions);
            }
            if data_type != gltf::accessor::DataType::F32 {
                bail!("Expected f32 data but found: {:?}", data_type);
            }
            let bone_weights_u8 = get_buffer_slice_from_accessor(accessor, buffers);
            Ok(bytemuck::cast_slice::<_, [f32; 4]>(&bone_weights_u8).to_vec())
        })
        .transpose()?
        .unwrap_or_else(|| {
            (0..vertex_position_count)
                .map(|_| [1.0, 0.0, 0.0, 0.0])
                .collect()
        });
    let vertex_bone_weights_count = vertex_bone_weights.len();

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
    if vertex_color_count != vertex_position_count {
        bail!(
          "Expected vertex colors for every vertex but found: vertex_position_count({:?}) != vertex_color_count({:?})",
          vertex_position_count,
          vertex_color_count
      );
    }
    if vertex_bone_indices_count != vertex_position_count {
        bail!(
          "Expected vertex bone indices for every vertex but found: vertex_position_count({:?}) != vertex_bone_indices_count({:?})",
          vertex_position_count,
          vertex_bone_indices_count
      );
    }
    if vertex_bone_weights_count != vertex_position_count {
        bail!(
          "Expected vertex bone weights for every vertex but found: vertex_position_count({:?}) != vertex_bone_weights_count({:?})",
          vertex_position_count,
          vertex_bone_weights_count
      );
    }

    let vertex_tangents_and_bitangents: Vec<_> = primitive_group
        .attributes()
        .find(|(semantic, _)| *semantic == gltf::Semantic::Tangents)
        .map(|(_, accessor)| {
            let data_type = accessor.data_type();
            let dimensions = accessor.dimensions();
            if dimensions != gltf::accessor::Dimensions::Vec4 {
                bail!("Expected vec3 data but found: {:?}", dimensions);
            }
            if data_type != gltf::accessor::DataType::F32 {
                bail!("Expected f32 data but found: {:?}", data_type);
            }
            let tangents_u8 = get_buffer_slice_from_accessor(accessor, buffers);
            let tangents: &[[f32; 4]] = bytemuck::cast_slice(&tangents_u8);

            Ok(tangents
                .to_vec()
                .iter()
                .enumerate()
                .map(|(vertex_index, tangent_slice)| {
                    let normal = vertex_normals[vertex_index];
                    let tangent =
                        Vector3::new(tangent_slice[0], tangent_slice[1], tangent_slice[2]);
                    // handedness is stored in w component: http://foundationsofgameenginedev.com/FGED2-sample.pdf
                    let coordinate_system_handedness =
                        if tangent_slice[3] > 0.0 { -1.0 } else { 1.0 };
                    let bitangent = coordinate_system_handedness * normal.cross(tangent);
                    (tangent, bitangent)
                })
                .collect())
        })
        .transpose()?
        .unwrap_or_else(|| {
            triangles_as_index_tuples
                .iter()
                .copied()
                .flat_map(|(index_a, index_b, index_c)| {
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

                    let (tangent, bitangent) = {
                        if abs_diff_eq!(delta_uv_1.x, 0.0, epsilon = 0.00001)
                            && abs_diff_eq!(delta_uv_2.x, 0.0, epsilon = 0.00001)
                            && abs_diff_eq!(delta_uv_1.y, 0.0, epsilon = 0.00001)
                            && abs_diff_eq!(delta_uv_2.y, 0.0, epsilon = 0.00001)
                        {
                            (Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0))
                        } else {
                            let f =
                                1.0 / (delta_uv_1.x * delta_uv_2.y - delta_uv_2.x * delta_uv_1.y);

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
                        }
                    };

                    (0..3).map(|_| (tangent, bitangent)).collect::<Vec<_>>()
                })
                .collect()
        });

    let vertices_with_all_data: Vec<_> = (0..(vertex_position_count))
        .map(|index| {
            let to_arr = |vec: &Vector3<f32>| [vec.x, vec.y, vec.z];
            let (tangent, bitangent) = vertex_tangents_and_bitangents[index];
            Vertex {
                position: vertex_positions[index].into(),
                normal: vertex_normals[index].into(),
                tex_coords: vertex_tex_coords[index],
                tangent: to_arr(&tangent),
                bitangent: to_arr(&bitangent),
                color: vertex_colors[index],
                bone_indices: vertex_bone_indices[index],
                bone_weights: vertex_bone_weights[index],
            }
        })
        .collect();

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
