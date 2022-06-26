use std::collections::HashMap;

use anyhow::bail;
use anyhow::Result;
use cgmath::abs_diff_eq;
use cgmath::Matrix4;
use cgmath::Vector2;
use cgmath::Vector3;
use cgmath::Vector4;
use wgpu::util::DeviceExt;

use super::*;

#[derive(Debug)]
pub struct GltfAsset {
    pub document: gltf::Document,
    pub buffers: Vec<gltf::buffer::Data>,
    pub images: Vec<gltf::image::Data>,
}

#[derive(Debug)]
pub struct Scene {
    pub source_asset: GltfAsset,
    pub buffers: SceneBuffers,
    // same order as the nodes list
    pub node_transforms: Vec<crate::transform::Transform>,
    // node index -> parent node index
    pub parent_index_map: HashMap<usize, usize>,
}

#[derive(Debug)]
pub struct SceneBuffers {
    // same order as the drawable_primitive_groups vec
    pub bindable_mesh_data: Vec<BindableMeshData>,
    // same order as the textures in src
    pub textures: Vec<Texture>,
}

#[derive(Debug)]
pub struct BindableMeshData {
    pub vertex_buffer: BufferAndLength,

    pub index_buffer: Option<BufferAndLength>,

    pub instance_buffer: BufferAndLength,
    pub instances: Vec<SceneMeshInstance>,

    pub textures_bind_group: wgpu::BindGroup,
}

#[derive(Debug)]
pub struct BufferAndLength {
    pub buffer: wgpu::Buffer,
    pub length: usize,
}

#[derive(Debug, Clone)]
pub struct SceneMeshInstance {
    pub node_index: usize,
    pub transform: crate::transform::Transform,
    pub base_material: BaseMaterial,
}

impl Scene {
    pub fn get_drawable_mesh_iterator(&self) -> impl Iterator<Item = &BindableMeshData> {
        let drawable_prim_indices: Vec<_> = self
            .source_asset
            .document
            .meshes()
            .flat_map(|mesh| mesh.primitives())
            .filter(|prim| prim.mode() == gltf::mesh::Mode::Triangles)
            .enumerate()
            .filter(|(_, prim)| {
                prim.material().alpha_mode() == gltf::material::AlphaMode::Opaque
                    || prim.material().alpha_mode() == gltf::material::AlphaMode::Mask
            })
            .map(|(prim_index, _)| prim_index)
            .collect();
        (0..drawable_prim_indices.len())
            .map(move |i| &self.buffers.bindable_mesh_data[drawable_prim_indices[i]])
    }
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

            dbg!(texture.name());

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

    let node_transforms: Vec<_> = document
        .nodes()
        .map(|node| crate::transform::Transform::from(node.transform()))
        .collect();

    let scene_nodes: Vec<_> = get_full_node_list(
        document
            .scenes()
            .find(|scene| scene.index() == scene_index)
            .ok_or_else(|| anyhow::anyhow!("Expected scene with index: {:?}", scene_index))?,
    );

    println!("scene_nodes len: {:?}", scene_nodes.len());
    println!(
        "scene_nodes info: {:?}",
        scene_nodes
            .iter()
            .map(|node| (
                node.name(),
                node.children()
                    .map(|child| child.index())
                    .collect::<Vec<_>>(),
                node.transform(),
                node.mesh().map(|mesh| mesh.index())
            ))
            .collect::<Vec<_>>(),
    );

    let meshes: Vec<_> = document.meshes().collect();

    let drawable_primitive_groups: Vec<_> = meshes
        .iter()
        .flat_map(|mesh| mesh.primitives().map(|prim| (&meshes[mesh.index()], prim)))
        .filter(|(_, prim)| prim.mode() == gltf::mesh::Mode::Triangles)
        .collect();

    let bindable_mesh_data = drawable_primitive_groups
        .iter()
        .enumerate()
        .map(|(_, (mesh, primitive_group))| {
            let (textures_bind_group, base_material) = build_textures_bind_group(
                device,
                queue,
                primitive_group,
                &textures,
                five_texture_bind_group_layout,
            )?;

            let (vertex_buffer, index_buffer) =
                build_geometry_buffers(device, primitive_group, buffers)?;
            let instances: Vec<_> = scene_nodes
                .iter()
                .filter(|node| {
                    node.mesh().is_some() && node.mesh().unwrap().index() == mesh.index()
                })
                .map(|node| {
                    let node_ancestry_list =
                        get_node_ancestry_list(node.index(), &parent_index_map);
                    let transform = node_ancestry_list
                        .iter()
                        .rev()
                        .fold(crate::transform::Transform::new(), |acc, node_index| {
                            acc * node_transforms[*node_index]
                        });
                    SceneMeshInstance {
                        node_index: node.index(),
                        transform,
                        base_material,
                    }
                })
                .collect();
            let gpu_instances: Vec<_> = instances
                .iter()
                .cloned()
                .map(MeshInstance::from)
                .map(GpuMeshInstance::from)
                .collect();

            println!(
                "{:?}: mesh_transforms len: {:?}",
                mesh.name(),
                instances.len()
            );

            let instance_buffer = BufferAndLength {
                buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("InstancedMeshComponent instance_buffer"),
                    contents: bytemuck::cast_slice(&gpu_instances),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }),
                length: instances.len(),
            };

            anyhow::Ok(BindableMeshData {
                vertex_buffer,
                index_buffer,
                instance_buffer,
                instances,
                textures_bind_group,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let runtime_node_transforms: Vec<_> = node_transforms
        .iter()
        .map(|transform| crate::transform::Transform::from(*transform))
        .collect();

    // runtime_node_transforms[0] = TransformBuilder::new()
    //     .scale(Vector3::new(1.0, 1.0, 1.0))
    //     .build();

    Ok(Scene {
        source_asset: gltf_asset,
        buffers: SceneBuffers {
            bindable_mesh_data,
            textures,
        },
        parent_index_map,
        node_transforms,
    })
}

fn get_image_pixels(
    image_data: &gltf::image::Data,
    srgb: bool,
) -> Result<(Vec<u8>, wgpu::TextureFormat)> {
    let image_pixels = &image_data.pixels;
    dbg!(image_data.format, srgb);
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

pub fn get_node_ancestry_list(
    node_index: usize,
    parent_index_map: &HashMap<usize, usize>,
) -> Vec<usize> {
    get_node_ancestry_list_impl(node_index, parent_index_map, Vec::new())
}

fn get_node_ancestry_list_impl(
    node_index: usize,
    parent_index_map: &HashMap<usize, usize>,
    acc: Vec<usize>,
) -> Vec<usize> {
    let with_self: Vec<_> = acc.iter().chain(vec![node_index].iter()).copied().collect();
    match parent_index_map.get(&node_index) {
        Some(parent_index) => {
            get_node_ancestry_list_impl(*parent_index, parent_index_map, with_self).to_vec()
        }
        None => with_self,
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
    triangles_prim: &gltf::mesh::Primitive,
    textures: &[Texture],
    five_texture_bind_group_layout: &wgpu::BindGroupLayout,
) -> Result<(wgpu::BindGroup, BaseMaterial)> {
    let material = triangles_prim.material();

    // println!("alpha mode: {:?}", material.alpha_mode());

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

    let base_material = BaseMaterial {
        base_color_factor: Vector4::from(pbr_info.base_color_factor()),
        emissive_factor: Vector3::from(material.emissive_factor()),
        metallic_factor: pbr_info.metallic_factor(),
        roughness_factor: pbr_info.roughness_factor(),
        normal_scale: material
            .normal_texture()
            .map(|info| info.scale())
            .unwrap_or(BaseMaterial::default().normal_scale),
        occlusion_strength: material
            .occlusion_texture()
            .map(|info| info.strength())
            .unwrap_or(BaseMaterial::default().occlusion_strength),
        alpha_cutoff: match material.alpha_mode() {
            gltf::material::AlphaMode::Mask => material.alpha_cutoff().unwrap_or(0.5),
            _ => BaseMaterial::default().alpha_cutoff,
        },
    };

    Ok((textures_bind_group, base_material))
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
    dbg!(vertex_position_count);
    let bounding_box = {
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
    dbg!(bounding_box);

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
                    let as_u16: Vec<_> = as_u32.iter().map(|&x| x as u16).collect();
                    let as_u16_u32: Vec<_> = as_u16.iter().map(|&x| x as u32).collect();
                    println!("as_u32 == as_u16: {}", as_u32.to_vec() == as_u16_u32);
                    anyhow::Ok(as_u16)
                }
                data_type => {
                    bail!("Expected u32, u16 or u8 indices but found: {:?}", data_type)
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
        .map_or(Ok(None), |v| v.map(Some))?
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
            if data_type != gltf::accessor::DataType::F32 {
                bail!("Expected f32 data but found: {:?}", data_type);
            }
            Ok(bytemuck::cast_slice(&get_buffer_slice_from_accessor(accessor, buffers)).to_vec())
        })
        .map_or(Ok(None), |v| v.map(Some))?
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
    if vertex_color_count != vertex_position_count {
        bail!(
            "Expected vertex colors for every vertex but found: vertex_position_count({:?}) != vertex_tex_coord_count({:?})",
            vertex_position_count,
            vertex_color_count
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
        .map_or(Ok(None), |v| v.map(Some))?
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
