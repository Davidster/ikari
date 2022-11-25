use std::collections::HashMap;

use anyhow::{bail, Result};
use cgmath::{abs_diff_eq, Matrix4, Vector2, Vector3, Vector4};

use super::*;

pub fn build_scene(
    base_renderer_state: &mut BaseRendererState,
    (document, buffers, images): (
        &gltf::Document,
        &Vec<gltf::buffer::Data>,
        &Vec<gltf::image::Data>,
    ),
    logger: &mut Logger,
) -> Result<(Scene, RenderBuffers)> {
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
                &base_renderer_state.device,
                &base_renderer_state.queue,
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

    let scene_nodes: Vec<_> = get_full_node_list(
        document
            .scenes()
            .find(|scene| scene.index() == scene_index)
            .ok_or_else(|| anyhow::anyhow!("Expected scene with index: {:?}", scene_index))?,
    );

    let meshes: Vec<_> = document.meshes().collect();

    let mut binded_pbr_meshes: Vec<BindedPbrMesh> = Vec::new();
    let mut binded_wireframe_meshes: Vec<BindedWireframeMesh> = Vec::new();
    let mut pbr_mesh_vertices: Vec<Vec<Vertex>> = Vec::new();
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
        let dynamic_pbr_params = get_dynamic_pbr_params(&primitive_group.material());
        let pbr_material = get_pbr_material(&primitive_group.material(), &textures);
        let textures_bind_group =
            base_renderer_state.make_pbr_textures_bind_group(&pbr_material, true)?;

        let (
            vertices,
            vertex_buffer,
            index_buffer,
            index_buffer_format,
            wireframe_index_buffer,
            wireframe_index_buffer_format,
            bounding_box,
        ) = build_geometry_buffers(&base_renderer_state.device, &primitive_group, buffers)?;
        let initial_instances: Vec<_> = scene_nodes
            .iter()
            .filter(|node| node.mesh().is_some() && node.mesh().unwrap().index() == mesh.index())
            .collect();

        let instance_buffer = GpuBuffer::empty(
            &base_renderer_state.device,
            initial_instances.len(),
            std::mem::size_of::<GpuPbrMeshInstance>(),
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );

        let wireframe_instance_buffer = GpuBuffer::empty(
            &base_renderer_state.device,
            1,
            std::mem::size_of::<GpuWireframeMeshInstance>(),
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );

        let primitive_mode = crate::renderer::PrimitiveMode::Triangles;

        let alpha_mode = match primitive_group.material().alpha_mode() {
            gltf::material::AlphaMode::Opaque => crate::renderer::AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => crate::renderer::AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => {
                todo!("Alpha blending isn't yet supported")
            }
        };

        for gltf_node in initial_instances.iter() {
            #[allow(clippy::or_fun_call)]
            let binded_pbr_mesh_indices =
                node_mesh_links.entry(gltf_node.index()).or_insert(vec![]);
            binded_pbr_mesh_indices.push(binded_pbr_mesh_index);
        }

        binded_pbr_meshes.push(BindedPbrMesh {
            geometry_buffers: GeometryBuffers {
                vertex_buffer,
                index_buffer,
                index_buffer_format,
                instance_buffer,
                bounding_box,
            },
            dynamic_pbr_params,
            textures_bind_group,
            primitive_mode,
            alpha_mode,
        });

        binded_wireframe_meshes.push(BindedWireframeMesh {
            source_mesh_type: MeshType::Pbr,
            source_mesh_index: binded_pbr_mesh_index,
            index_buffer: wireframe_index_buffer,
            index_buffer_format: wireframe_index_buffer_format,
            instance_buffer: wireframe_instance_buffer,
        });

        pbr_mesh_vertices.push(vertices);
    }

    // it is important that the node indices from the gltf document are preserved
    // for any of the other stuff that refers to the nodes by index such as the animations
    let nodes: Vec<_> = document
        .nodes()
        .map(|node| GameNodeDesc {
            transform: crate::transform::Transform::from(node.transform()),
            skin_index: node.skin().map(|skin| skin.index()),
            mesh: node_mesh_links
                .get(&node.index())
                .map(|mesh_indices| GameNodeMesh {
                    mesh_indices: mesh_indices.clone(),
                    mesh_type: GameNodeMeshType::Pbr {
                        material_override: None,
                    },
                    wireframe: false,
                }),
            name: node.name().map(|name| name.to_string()),
        })
        .collect();

    let animations = get_animations(document, buffers)?;

    let skins: Vec<_> = document
        .skins()
        .enumerate()
        .map(|(skin_index, skin)| {
            let bone_node_indices: Vec<_> = skin.joints().map(|joint| joint.index()).collect();
            let bone_inverse_bind_matrices: Vec<_> = skin
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
                });

            let skeleton_skin_node = nodes
                .iter()
                .find(|node| node.skin_index == Some(skin_index))
                .unwrap();
            let skeleton_mesh_index = skeleton_skin_node.mesh.as_ref().unwrap().mesh_indices[0];
            let skeleton_mesh_vertices = &pbr_mesh_vertices[skeleton_mesh_index];

            let bone_bounding_box_transforms: Vec<_> = (0..bone_inverse_bind_matrices.len())
                .map(|bone_index| {
                    let bone_inv_bind_matrix = bone_inverse_bind_matrices[bone_index];
                    let vertex_weight_threshold = 0.5f32;
                    let vertex_positions_for_node: Vec<_> = skeleton_mesh_vertices
                        .iter()
                        .filter(|vertex| {
                            vertex
                                .bone_indices
                                .iter()
                                .zip(vertex.bone_weights.iter())
                                .any(|(v_bone_index, v_bone_weight)| {
                                    *v_bone_index as usize == bone_index
                                        && *v_bone_weight > vertex_weight_threshold
                                })
                        })
                        .map(|vertex| {
                            let position = Vector4::new(
                                vertex.position[0],
                                vertex.position[1],
                                vertex.position[2],
                                1.0,
                            );
                            bone_inv_bind_matrix * position
                        })
                        .collect();
                    if vertex_positions_for_node.is_empty() {
                        return TransformBuilder::new()
                            .scale(Vector3::new(0.0, 0.0, 0.0))
                            .build();
                    }
                    let mut min_point = Vector3::new(
                        vertex_positions_for_node[0].x,
                        vertex_positions_for_node[0].y,
                        vertex_positions_for_node[0].z,
                    );
                    let mut max_point = min_point;
                    for pos in vertex_positions_for_node {
                        min_point.x = min_point.x.min(pos.x);
                        min_point.y = min_point.y.min(pos.y);
                        min_point.z = min_point.z.min(pos.z);
                        max_point.x = max_point.x.max(pos.x);
                        max_point.y = max_point.y.max(pos.y);
                        max_point.z = max_point.z.max(pos.z);
                    }
                    TransformBuilder::new()
                        .scale((max_point - min_point) / 2.0)
                        .position((max_point + min_point) / 2.0)
                        .build()
                })
                .collect();

            anyhow::Ok(IndexedSkin {
                bone_inverse_bind_matrices,
                bone_node_indices,
                bone_bounding_box_transforms,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let render_buffers = RenderBuffers {
        binded_pbr_meshes,
        binded_unlit_meshes: vec![],
        binded_wireframe_meshes,
        textures,
    };

    logger.log("Scene loaded:");

    // nodes: Vec<(Option<GameNode>, usize)>, // (node, generation number). None means the node was removed from the scene
    // pub skins: Vec<Skin>,
    // pub animations: Vec<Animation>,
    // // node index -> parent node index
    // parent_index_map: HashMap<usize, usize>,
    // // skeleton skin node index -> parent_index_map
    // skeleton_parent_index_maps: HashMap<usize, HashMap<usize, usize>>,

    logger.log(&format!("  - node count: {:?}", nodes.len()));
    logger.log(&format!("  - skin count: {:?}", skins.len()));
    logger.log(&format!("  - animation count: {:?}", animations.len()));
    logger.log("  Render buffers:");
    logger.log(&format!(
        "    - PBR mesh count: {:?}",
        render_buffers.binded_pbr_meshes.len()
    ));
    logger.log(&format!(
        "    - Unlit mesh count: {:?}",
        render_buffers.binded_unlit_meshes.len()
    ));
    logger.log(&format!(
        "    - Wireframe mesh count: {:?}",
        render_buffers.binded_wireframe_meshes.len()
    ));
    logger.log(&format!(
        "    - Texture count: {:?}",
        render_buffers.textures.len()
    ));

    let scene = Scene::new(nodes, skins, animations, parent_index_map);

    Ok((scene, render_buffers))
}

pub fn get_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<IndexedAnimation>> {
    document
        .animations()
        .map(|animation| {
            let channel_timings: Vec<_> = animation
                .channels()
                .map(|channel| get_keyframe_times(&channel.sampler(), buffers))
                .collect::<Result<Vec<_>, _>>()?;
            let length_seconds = *channel_timings
                .iter()
                .map(|keyframe_times| keyframe_times.last().unwrap())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let channels: Vec<_> = animation
                .channels()
                .enumerate()
                .map(|(channel_index, channel)| {
                    validate_channel_data_type(&channel)?;
                    let sampler = channel.sampler();
                    let accessor = sampler.output();
                    anyhow::Ok(IndexedChannel {
                        node_index: channel.target().node().index(),
                        property: channel.target().property(),
                        interpolation_type: sampler.interpolation(),
                        keyframe_timings: channel_timings[channel_index].clone(),
                        keyframe_values_u8: get_buffer_slice_from_accessor(accessor, buffers),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            anyhow::Ok(IndexedAnimation {
                length_seconds,
                channels,
            })
        })
        .collect::<Result<Vec<_>, _>>()
}

fn validate_channel_data_type(channel: &gltf::animation::Channel) -> Result<()> {
    let accessor = channel.sampler().output();
    let data_type = accessor.data_type();
    let dimensions = accessor.dimensions();
    match channel.target().property() {
        gltf::animation::Property::Translation | gltf::animation::Property::Scale => {
            if dimensions != gltf::accessor::Dimensions::Vec3 {
                bail!("Expected vec3 data but found: {:?}", dimensions);
            }
            if data_type != gltf::accessor::DataType::F32 {
                bail!("Expected f32 data but found: {:?}", data_type);
            }
        }
        gltf::animation::Property::Rotation => {
            if dimensions != gltf::accessor::Dimensions::Vec4 {
                bail!("Expected vec4 data but found: {:?}", dimensions);
            }
            if data_type != gltf::accessor::DataType::F32 {
                bail!("Expected f32 data but found: {:?}", data_type);
            }
        }
        gltf::animation::Property::MorphTargetWeights => {
            bail!("MorphTargetWeights not supported")
        }
    };
    Ok(())
}

fn get_keyframe_times(
    sampler: &gltf::animation::Sampler,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<f32>> {
    let keyframe_times = {
        let accessor = sampler.input();
        let data_type = accessor.data_type();
        let dimensions = accessor.dimensions();
        if dimensions != gltf::accessor::Dimensions::Scalar {
            bail!("Expected scalar data but found: {:?}", dimensions);
        }
        if data_type != gltf::accessor::DataType::F32 {
            bail!("Expected f32 data but found: {:?}", data_type);
        }
        let result_u8 = get_buffer_slice_from_accessor(accessor, buffers);
        bytemuck::cast_slice::<_, f32>(&result_u8).to_vec()
    };

    Ok(keyframe_times)
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

fn get_pbr_material<'a>(
    material: &gltf::material::Material,
    textures: &'a [Texture],
) -> PbrMaterial<'a> {
    let pbr_info = material.pbr_metallic_roughness();

    let get_texture = |texture: Option<gltf::texture::Texture>| {
        texture
            .map(|texture| texture.index())
            .map(|texture_index| &textures[texture_index])
    };

    PbrMaterial {
        base_color: get_texture(pbr_info.base_color_texture().map(|info| info.texture())),
        normal: get_texture(material.normal_texture().map(|info| info.texture())),
        metallic_roughness: get_texture(
            pbr_info
                .metallic_roughness_texture()
                .map(|info| info.texture()),
        ),
        emissive: get_texture(material.emissive_texture().map(|info| info.texture())),
        ambient_occlusion: get_texture(material.occlusion_texture().map(|info| info.texture())),
    }
}

fn get_dynamic_pbr_params(material: &gltf::material::Material) -> DynamicPbrParams {
    let pbr_info = material.pbr_metallic_roughness();

    DynamicPbrParams {
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
    }
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
            buffer[byte_range].to_vec()
        })
        .collect()
}

// TODO: this tuple is getting out of hand lol. return a GeometryBuffers or something
pub fn build_geometry_buffers(
    device: &wgpu::Device,
    primitive_group: &gltf::mesh::Primitive,
    buffers: &[gltf::buffer::Data],
) -> Result<(
    Vec<Vertex>,
    GpuBuffer,
    GpuBuffer,
    wgpu::IndexFormat,
    GpuBuffer,
    wgpu::IndexFormat,
    (Vector3<f32>, Vector3<f32>),
)> {
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
    let bounding_box = {
        let mut min_point = vertex_positions[0];
        let mut max_point = min_point;
        for pos in &vertex_positions {
            min_point.x = min_point.x.min(pos.x);
            min_point.y = min_point.y.min(pos.y);
            min_point.z = min_point.z.min(pos.z);
            max_point.x = max_point.x.max(pos.x);
            max_point.y = max_point.y.max(pos.y);
            max_point.z = max_point.z.max(pos.z);
        }
        (min_point, max_point)
    };

    let indices: Vec<u32> = primitive_group
        .indices()
        .map(|accessor| {
            let data_type = accessor.data_type();
            let buffer_slice = get_buffer_slice_from_accessor(accessor, buffers);

            let indices: Vec<u32> = match data_type {
                gltf::accessor::DataType::U16 => anyhow::Ok(
                    bytemuck::cast_slice::<_, u16>(&buffer_slice)
                        .iter()
                        .map(|&x| x as u32)
                        .collect::<Vec<_>>(),
                ),
                gltf::accessor::DataType::U8 => {
                    anyhow::Ok(buffer_slice.iter().map(|&x| x as u32).collect::<Vec<_>>())
                }
                gltf::accessor::DataType::U32 => {
                    anyhow::Ok(bytemuck::cast_slice::<_, u32>(&buffer_slice).to_vec())
                }
                data_type => {
                    bail!("Expected u32, u16 or u8 indices but found: {:?}", data_type)
                }
            }?;
            anyhow::Ok(indices)
        })
        .unwrap_or_else(|| {
            let vertex_position_count_u32 = u32::try_from(vertex_position_count)?;
            Ok((0..vertex_position_count_u32).collect())
        })?;

    let triangle_count = indices.len() / 3;

    let triangles_as_index_tuples: Vec<_> = (0..triangle_count)
        .map(|triangle_index| {
            let i_left = triangle_index * 3;
            (
                indices[i_left] as usize,
                indices[i_left + 1] as usize,
                indices[i_left + 2] as usize,
            )
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

                    let f = 1.0 / (delta_uv_1.x * delta_uv_2.y - delta_uv_2.x * delta_uv_1.y);

                    let (tangent, bitangent) = {
                        if abs_diff_eq!(f, 0.0, epsilon = 0.00001) || !f.is_finite() {
                            (Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0))
                        } else {
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

    let vertex_buffer = GpuBuffer::from_bytes(
        device,
        bytemuck::cast_slice(&vertices_with_all_data),
        std::mem::size_of::<Vertex>(),
        wgpu::BufferUsages::VERTEX,
    );

    let into_index_buffer = |indices: &Vec<u32>| {
        let index_buffer_u16_result: Result<Vec<_>, _> =
            indices.iter().map(|index| u16::try_from(*index)).collect();
        let index_buffer_u16;
        let (index_buffer_bytes, index_buffer_format) = match index_buffer_u16_result {
            Ok(as_u16) => {
                index_buffer_u16 = as_u16;
                (
                    bytemuck::cast_slice::<u16, u8>(&index_buffer_u16),
                    wgpu::IndexFormat::Uint16,
                )
            }
            Err(_) => (
                bytemuck::cast_slice::<u32, u8>(indices),
                wgpu::IndexFormat::Uint32,
            ),
        };

        (
            GpuBuffer::from_bytes(
                device,
                index_buffer_bytes,
                match index_buffer_format {
                    wgpu::IndexFormat::Uint16 => std::mem::size_of::<u16>(),
                    wgpu::IndexFormat::Uint32 => std::mem::size_of::<u32>(),
                },
                wgpu::BufferUsages::INDEX,
            ),
            index_buffer_format,
        )
    };

    let (index_buffer, index_buffer_format) = into_index_buffer(&indices);

    let (wireframe_index_buffer, wireframe_index_buffer_format) = into_index_buffer(
        &indices
            .chunks(3)
            .flat_map(|triangle| {
                vec![
                    triangle[0],
                    triangle[1],
                    triangle[1],
                    triangle[2],
                    triangle[2],
                    triangle[0],
                ]
            })
            .collect(),
    );

    Ok((
        vertices_with_all_data,
        vertex_buffer,
        index_buffer,
        index_buffer_format,
        wireframe_index_buffer,
        wireframe_index_buffer_format,
        bounding_box,
    ))
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct ChannelPropertyStr<'a>(&'a str);

impl From<gltf::animation::Property> for ChannelPropertyStr<'_> {
    fn from(prop: gltf::animation::Property) -> Self {
        Self(match prop {
            gltf::animation::Property::Translation => "Translation",
            gltf::animation::Property::Scale => "Scale",
            gltf::animation::Property::Rotation => "Rotation",
            gltf::animation::Property::MorphTargetWeights => "MorphTargetWeights",
        })
    }
}

pub fn validate_animation_property_counts(gltf_document: &gltf::Document, logger: &mut Logger) {
    let property_counts: HashMap<(usize, ChannelPropertyStr), usize> = gltf_document
        .animations()
        .flat_map(|animation| animation.channels())
        .fold(HashMap::new(), |mut acc, channel| {
            let count = acc
                .entry((
                    channel.target().node().index(),
                    channel.target().property().into(),
                ))
                .or_insert(0);
            *count += 1;
            acc
        });
    for ((node_index, property), count) in property_counts {
        if count > 1 {
            logger.log(&format!(
                "Warning: expected no more than 1 animated property but found {:?} (node_index={:?}, node_name={:?}, property={:?})",
                count,
                node_index,
                gltf_document
                    .nodes()
                    .find(|node| node.index() == node_index)
                    .and_then(|node| node.name()),
                property.0
            ))
        }
    }
}
