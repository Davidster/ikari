use super::*;
use std::{num::NonZeroU32, ops::Deref};

use anyhow::*;
use cgmath::Vector3;
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub size: wgpu::Extent3d,
}

pub struct CreateCubeMapImagesParam<'a> {
    pub pos_x: &'a image::DynamicImage,
    pub neg_x: &'a image::DynamicImage,
    pub pos_y: &'a image::DynamicImage,
    pub neg_y: &'a image::DynamicImage,
    pub pos_z: &'a image::DynamicImage,
    pub neg_z: &'a image::DynamicImage,
}

pub struct SamplerDescriptor<'a>(pub wgpu::SamplerDescriptor<'a>);

impl<'a> Deref for SamplerDescriptor<'a> {
    type Target = wgpu::SamplerDescriptor<'a>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Default for SamplerDescriptor<'_> {
    fn default() -> Self {
        SamplerDescriptor(wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        })
    }
}

// TODO: maybe implement some functions on the BaseRendererState so we have the device and queue for free?
impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    // supports jpg and png
    pub fn from_encoded_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img_bytes: &[u8],
        label: &str,
        format: Option<wgpu::TextureFormat>,
        generate_mipmaps: bool,
        sampler_descriptor: &SamplerDescriptor,
    ) -> Result<Self> {
        let img = image::load_from_memory(img_bytes)?;
        let img_as_rgba = img.to_rgba8();
        Self::from_decoded_image(
            device,
            queue,
            &img_as_rgba,
            img_as_rgba.dimensions(),
            Some(label),
            format,
            generate_mipmaps,
            sampler_descriptor,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_decoded_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img_bytes: &[u8],
        dimensions: (u32, u32),
        label: Option<&str>,
        format: Option<wgpu::TextureFormat>,
        generate_mipmaps: bool,
        sampler_descriptor: &SamplerDescriptor,
    ) -> Result<Self> {
        let size = wgpu::Extent3d {
            width: dimensions.0 as u32,
            height: dimensions.1 as u32,
            depth_or_array_layers: 1,
        };
        let mip_level_count = if generate_mipmaps {
            size.max_mips(wgpu::TextureDimension::D2)
        } else {
            1
        };
        let format = format.unwrap_or(wgpu::TextureFormat::Rgba8UnormSrgb);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
        });

        let format_info = format.describe();

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            img_bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(format_info.block_size as u32 * dimensions.0),
                rows_per_image: NonZeroU32::new(dimensions.1),
            },
            size,
        );

        let view = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&sampler_descriptor.0);

        let mip_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mip_encoder"),
        });

        if generate_mipmaps {
            generate_mipmaps_for_texture(
                device,
                queue,
                mip_encoder,
                &texture,
                mip_level_count,
                format,
            )?;
        }

        Ok(Self {
            texture,
            view,
            sampler,
            size,
        })
    }

    pub fn _from_color_srgb(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color: [u8; 4],
    ) -> Result<Self> {
        let one_pixel_image = {
            let mut img = image::RgbaImage::new(1, 1);
            img.put_pixel(0, 0, image::Rgba(color));
            img
        };
        Texture::from_decoded_image(
            device,
            queue,
            &one_pixel_image,
            one_pixel_image.dimensions(),
            Some("from_color texture"),
            wgpu::TextureFormat::Rgba8UnormSrgb.into(),
            false,
            &SamplerDescriptor(wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..SamplerDescriptor::default().0
            }),
        )
    }

    pub fn from_color(device: &wgpu::Device, queue: &wgpu::Queue, color: [u8; 4]) -> Result<Self> {
        let one_pixel_image = {
            let mut img = image::RgbaImage::new(1, 1);
            img.put_pixel(0, 0, image::Rgba(color));
            img
        };
        Texture::from_decoded_image(
            device,
            queue,
            &one_pixel_image,
            one_pixel_image.dimensions(),
            Some("from_color texture"),
            wgpu::TextureFormat::Rgba8Unorm.into(),
            false,
            &SamplerDescriptor(wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..SamplerDescriptor::default().0
            }),
        )
    }

    pub fn _from_gray(device: &wgpu::Device, queue: &wgpu::Queue, gray_value: u8) -> Result<Self> {
        let one_pixel_gray_image = {
            let mut img = image::GrayImage::new(1, 1);
            img.put_pixel(0, 0, image::Luma([gray_value]));
            img
        };
        Texture::from_decoded_image(
            device,
            queue,
            &one_pixel_gray_image,
            one_pixel_gray_image.dimensions(),
            Some("from_gray texture"),
            wgpu::TextureFormat::R8Unorm.into(),
            false,
            &SamplerDescriptor(wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..SamplerDescriptor::default().0
            }),
        )
    }

    pub fn flat_normal_map(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self> {
        let one_pixel_up_image = {
            let mut img = image::RgbaImage::new(1, 1);
            img.put_pixel(0, 0, image::Rgba([127, 127, 255, 255]));
            img
        };
        Texture::from_decoded_image(
            device,
            queue,
            &one_pixel_up_image,
            one_pixel_up_image.dimensions(),
            Some("flat_normal_map texture"),
            wgpu::TextureFormat::Rgba8Unorm.into(),
            false,
            &SamplerDescriptor(wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..SamplerDescriptor::default().0
            }),
        )
    }

    pub fn create_scaled_surface_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        render_scale: f32,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: ((config.width as f32) * render_scale.sqrt()).round() as u32,
            height: ((config.height as f32) * render_scale.sqrt()).round() as u32,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
        });

        let view = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            size,
        }
    }

    pub fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        render_scale: f32,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: ((config.width as f32) * render_scale.sqrt()).round() as u32,
            height: ((config.height as f32) * render_scale.sqrt()).round() as u32,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Texture::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let view = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::GreaterEqual),
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            size,
        }
    }

    pub fn create_cube_depth_texture_array(
        device: &wgpu::Device,
        size: u32,
        label: Option<&str>,
        length: u32,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6 * length,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Texture::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            // compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            size,
        }
    }

    pub fn create_depth_texture_array(
        device: &wgpu::Device,
        size: u32,
        label: Option<&str>,
        length: u32,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: length,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Texture::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            // compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            size,
        }
    }

    pub fn create_cubemap_from_equirectangular(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: Option<&str>,
        skybox_buffers: &GeometryBuffers,
        er_to_cubemap_pipeline: &wgpu::RenderPipeline,
        er_texture: &Texture,
        generate_mipmaps: bool,
    ) -> Self {
        let size = wgpu::Extent3d {
            // TODO: is divide by 3 the right move?
            width: er_texture.size.width / 3,
            height: er_texture.size.width / 3,
            depth_or_array_layers: 6,
        };

        let mip_level_count = if generate_mipmaps {
            size.max_mips(wgpu::TextureDimension::D2)
        } else {
            1
        };

        // TODO: dry
        let single_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("single_uniform_bind_group_layout"),
            });

        // TODO: dry
        let single_texture_bind_group_layout =
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
                label: Some("single_texture_bind_group_layout"),
            });

        let sampler = device.create_sampler(&SamplerDescriptor::default().0);

        let cubemap_texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cubemap Generation Camera Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform::new()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &single_uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("cubemap_gen_camera_bind_group"),
        });

        let faces: Vec<_> = build_cubemap_face_camera_views(
            Vector3::new(0.0, 0.0, 0.0),
            NEAR_PLANE_DISTANCE,
            FAR_PLANE_DISTANCE,
            true,
        )
        .iter()
        .copied()
        .enumerate()
        .map(|(i, view_proj_matrices)| {
            (
                view_proj_matrices,
                cubemap_texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i as u32,
                    array_layer_count: NonZeroU32::new(1),
                    ..Default::default()
                }),
            )
        })
        .collect();

        for (face_view_proj_matrices, face_texture_view) in faces {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("create_cubemap_texture_from_equirectangular encoder"),
            });
            let er_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &single_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&er_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
                label: None,
            });
            queue.write_buffer(
                &camera_buffer,
                0,
                bytemuck::cast_slice(&[CameraUniform::from(face_view_proj_matrices)]),
            );
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &face_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: None,
                });
                rpass.set_pipeline(er_to_cubemap_pipeline);
                rpass.set_bind_group(0, &er_texture_bind_group, &[]);
                rpass.set_bind_group(1, &camera_bind_group, &[]);
                rpass.set_vertex_buffer(0, skybox_buffers.vertex_buffer.src().slice(..));
                rpass.set_index_buffer(
                    skybox_buffers.index_buffer.src().slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                rpass.draw_indexed(0..(skybox_buffers.index_buffer.length() as u32), 0, 0..1);
            }
            queue.submit(Some(encoder.finish()));
        }

        if generate_mipmaps {
            todo!("Call generate_mipmaps_for_texture for each side of the cubemap");
        }

        let view = cubemap_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            texture: cubemap_texture,
            view,
            sampler,
            size,
        }
    }

    pub fn create_diffuse_env_map(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: Option<&str>,
        skybox_buffers: &GeometryBuffers,
        env_map_gen_pipeline: &wgpu::RenderPipeline,
        skybox_rad_texture: &Texture,
        generate_mipmaps: bool,
    ) -> Self {
        // let texture_size = texture.texture.
        let size = wgpu::Extent3d {
            width: 128,
            height: 128,
            depth_or_array_layers: 6,
        };

        let mip_level_count = if generate_mipmaps {
            size.max_mips(wgpu::TextureDimension::D2)
        } else {
            1
        };

        // TODO: dry
        let single_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("single_uniform_bind_group_layout"),
            });

        // TODO: dry
        let single_cube_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::Cube,
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
                label: Some("single_cube_texture_bind_group_layout"),
            });

        let env_map = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Env map Generation Camera Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform::new()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &single_uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("env_map_gen_camera_bind_group"),
        });

        let faces: Vec<_> = build_cubemap_face_camera_views(
            Vector3::new(0.0, 0.0, 0.0),
            NEAR_PLANE_DISTANCE,
            FAR_PLANE_DISTANCE,
            true,
        )
        .iter()
        .copied()
        .enumerate()
        .map(|(i, view_proj_matrices)| {
            (
                view_proj_matrices,
                env_map.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i as u32,
                    array_layer_count: NonZeroU32::new(1),
                    ..Default::default()
                }),
            )
        })
        .collect();

        for (face_view_proj_matrices, face_texture_view) in faces {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("create_env_map encoder"),
            });
            let skybox_ir_texture_bind_group =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &single_cube_texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&skybox_rad_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&skybox_rad_texture.sampler),
                        },
                    ],
                    label: None,
                });
            queue.write_buffer(
                &camera_buffer,
                0,
                bytemuck::cast_slice(&[CameraUniform::from(face_view_proj_matrices)]),
            );
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &face_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: None,
                });
                rpass.set_pipeline(env_map_gen_pipeline);
                rpass.set_bind_group(0, &skybox_ir_texture_bind_group, &[]);
                rpass.set_bind_group(1, &camera_bind_group, &[]);
                rpass.set_vertex_buffer(0, skybox_buffers.vertex_buffer.src().slice(..));
                rpass.set_index_buffer(
                    skybox_buffers.index_buffer.src().slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                rpass.draw_indexed(0..(skybox_buffers.index_buffer.length() as u32), 0, 0..1);
            }
            queue.submit(Some(encoder.finish()));
        }

        if generate_mipmaps {
            todo!("Call generate_mipmaps_for_texture for each side of the cubemap");
        }

        let view = env_map.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            texture: env_map,
            view,
            sampler,
            size,
        }
    }

    pub fn create_specular_env_map(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: Option<&str>,
        skybox_buffers: &GeometryBuffers,
        env_map_gen_pipeline: &wgpu::RenderPipeline,
        skybox_rad_texture: &Texture,
    ) -> Self {
        // let texture_size = texture.texture.
        let size = wgpu::Extent3d {
            width: skybox_rad_texture.size.width,
            height: skybox_rad_texture.size.height,
            depth_or_array_layers: 6,
        };

        // TODO: dry
        let two_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("single_uniform_bind_group_layout"),
            });

        // TODO: dry
        let single_cube_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::Cube,
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
                label: Some("single_cube_texture_bind_group_layout"),
            });

        let mip_level_count = 5;

        let env_map = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Env map Generation Camera Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform::new()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let roughness_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Env map Generation Roughness Buffer"),
            contents: bytemuck::cast_slice(&[0.0f32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_roughness_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &two_uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: roughness_buffer.as_entire_binding(),
                },
            ],
            label: Some("env_map_gen_camera_bind_group"),
        });

        let camera_projection_matrices = build_cubemap_face_camera_views(
            Vector3::new(0.0, 0.0, 0.0),
            NEAR_PLANE_DISTANCE,
            FAR_PLANE_DISTANCE,
            true,
        );

        // TODO: level 0 doesn't really need to be done since roughness = 0 basically copies the skybox plainly
        //       but we'll need to write the contents of skybox_rad_texture to the first mip level of the cubemap above
        (0..mip_level_count)
            .map(|i| (i, i as f32 * (1.0 / (mip_level_count - 1) as f32)))
            .for_each(|(mip_level, roughness_level)| {
                camera_projection_matrices
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, view_proj_matrices)| {
                        (
                            view_proj_matrices,
                            env_map.create_view(&wgpu::TextureViewDescriptor {
                                dimension: Some(wgpu::TextureViewDimension::D2),
                                base_array_layer: i as u32,
                                array_layer_count: NonZeroU32::new(1),
                                base_mip_level: mip_level as u32,
                                mip_level_count: NonZeroU32::new(1),
                                ..Default::default()
                            }),
                        )
                    })
                    .for_each(|(face_view_proj_matrices, face_texture_view)| {
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("create_env_map encoder"),
                            });
                        let skybox_ir_texture_bind_group =
                            device.create_bind_group(&wgpu::BindGroupDescriptor {
                                layout: &single_cube_texture_bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(
                                            &skybox_rad_texture.view,
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Sampler(
                                            &skybox_rad_texture.sampler,
                                        ),
                                    },
                                ],
                                label: None,
                            });
                        queue.write_buffer(
                            &camera_buffer,
                            0,
                            bytemuck::cast_slice(&[CameraUniform::from(face_view_proj_matrices)]),
                        );
                        queue.write_buffer(
                            &roughness_buffer,
                            0,
                            bytemuck::cast_slice(&[roughness_level]),
                        );
                        {
                            let mut rpass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: None,
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &face_texture_view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                            store: true,
                                        },
                                    })],
                                    depth_stencil_attachment: None,
                                });
                            rpass.set_pipeline(env_map_gen_pipeline);
                            rpass.set_bind_group(0, &skybox_ir_texture_bind_group, &[]);
                            rpass.set_bind_group(1, &camera_roughness_bind_group, &[]);
                            rpass
                                .set_vertex_buffer(0, skybox_buffers.vertex_buffer.src().slice(..));
                            rpass.set_index_buffer(
                                skybox_buffers.index_buffer.src().slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            rpass.draw_indexed(
                                0..(skybox_buffers.index_buffer.length() as u32),
                                0,
                                0..1,
                            );
                        }
                        queue.submit(Some(encoder.finish()));
                    });
            });

        let view = env_map.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            texture: env_map,
            view,
            sampler,
            size,
        }
    }

    // each image should have the same dimensions!
    pub fn create_cubemap(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        images: CreateCubeMapImagesParam,
        label: Option<&str>,
        generate_mipmaps: bool,
    ) -> Self {
        // order of the images for a cubemap is documented here:
        // https://www.khronos.org/opengl/wiki/Cubemap_Texture
        let images_as_rgba = vec![
            images.pos_x,
            images.neg_x,
            images.pos_y,
            images.neg_y,
            images.pos_z,
            images.neg_z,
        ]
        .iter()
        .map(|img| img.to_rgba8())
        .collect::<Vec<_>>();
        let dimensions = images_as_rgba[0].dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 6,
        };

        let mip_level_count = if generate_mipmaps {
            size.max_mips(wgpu::TextureDimension::D2)
        } else {
            1
        };

        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label,
                size,
                mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            },
            // pack images into one big byte array
            &images_as_rgba
                .iter()
                .flat_map(|image| image.to_vec())
                .collect::<Vec<_>>(),
        );

        if generate_mipmaps {
            todo!("Call generate_mipmaps_for_texture for each side of the cubemap");
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            size,
        }
    }

    pub fn create_brdf_lut(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        brdf_lut_gen_pipeline: &wgpu::RenderPipeline,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Brdf Lut"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("create_brdf_lut encoder"),
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            rpass.set_pipeline(brdf_lut_gen_pipeline);
            rpass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));

        Self {
            texture,
            view,
            sampler,
            size,
        }
    }
}

fn generate_mipmaps_for_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut mip_encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    mip_level_count: u32,
    format: wgpu::TextureFormat,
) -> Result<()> {
    let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            std::fs::read_to_string("./src/shaders/blit.wgsl")?.into(),
        ),
    });
    let mip_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mip_render_pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &blit_shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &blit_shader,
            entry_point: "fs_main",
            targets: &[Some(format.into())],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview: None,
    });
    let mip_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("mip_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    // TODO: dry
    let single_texture_bind_group_layout =
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
            label: Some("single_texture_bind_group_layout"),
        });
    let mip_texure_views = (0..mip_level_count)
        .map(|mip| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("mip"),
                format: None,
                dimension: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: NonZeroU32::new(1),
                base_array_layer: 0,
                array_layer_count: None,
            })
        })
        .collect::<Vec<_>>();

    for target_mip in 1..mip_level_count as usize {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &single_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&mip_texure_views[target_mip - 1]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&mip_sampler),
                },
            ],
            label: None,
        });

        let mut rpass = mip_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &mip_texure_views[target_mip],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });
        rpass.set_pipeline(&mip_render_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
    queue.submit(Some(mip_encoder.finish()));
    Ok(())
}
