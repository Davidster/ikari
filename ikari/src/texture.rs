use crate::camera::*;
use crate::renderer::BaseRenderer;
use crate::renderer::RendererConstantData;
use crate::renderer::FAR_PLANE_DISTANCE;
use crate::renderer::NEAR_PLANE_DISTANCE;
use crate::renderer::USE_LABELS;
use crate::sampler_cache::*;

use anyhow::*;
use glam::f32::Vec3;

use image::RgbaImage;
use serde::Deserialize;
use serde::Serialize;
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler_index: usize,
    pub size: wgpu::Extent3d,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RawImage {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_count: u32,
    pub raw: Vec<u8>,
}

impl From<image::RgbaImage> for RawImage {
    fn from(image: RgbaImage) -> Self {
        Self {
            width: image.width(),
            height: image.height(),
            depth: 1,
            mip_count: 1,
            raw: image.into_raw(),
        }
    }
}

impl RawImage {
    pub fn slice(&self) -> RawImageSlice {
        RawImageSlice {
            width: self.width,
            height: self.height,
            depth: self.depth,
            mip_count: self.mip_count,
            raw: &self.raw,
        }
    }
}

#[derive(Debug)]
pub struct RawImageSlice<'a> {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_count: u32,
    pub raw: &'a [u8],
}

// TODO: maybe implement some functions on the BaseRendererState so we have the device and queue for free?
//       or maybe store an Arc<Device> and Arc<Queue>
impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn unpadded_bytes_per_row(&self, mip_level: Option<u32>) -> u32 {
        (self.size.width >> mip_level.unwrap_or(0))
            * self.texture.format().block_size(None).unwrap()
    }

    pub fn padded_bytes_per_row(&self, mip_level: Option<u32>) -> u32 {
        let unpadded_bytes_per_row = self.unpadded_bytes_per_row(mip_level);
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        unpadded_bytes_per_row + padded_bytes_per_row_padding
    }

    // supports jpg and png
    pub fn from_encoded_image(
        base_renderer: &BaseRenderer,
        img_bytes: &[u8],
        label: Option<&str>,
        format: Option<wgpu::TextureFormat>,
        generate_mipmaps: bool,
        sampler_descriptor: &SamplerDescriptor,
    ) -> Result<Self> {
        let img = image::load_from_memory(img_bytes)?;
        let img_as_rgba = img.to_rgba8();
        Self::from_decoded_image(
            base_renderer,
            &img_as_rgba,
            img_as_rgba.dimensions(),
            1,
            label,
            format,
            generate_mipmaps,
            sampler_descriptor,
        )
    }

    // TODO: take RawImage as argument instead
    #[allow(clippy::too_many_arguments)]
    pub fn from_decoded_image(
        base_renderer: &BaseRenderer,
        img_bytes: &[u8],
        dimensions: (u32, u32),
        baked_mip_levels: u32,
        label: Option<&str>,
        format: Option<wgpu::TextureFormat>,
        generate_mipmaps: bool,
        sampler_descriptor: &SamplerDescriptor,
    ) -> Result<Self> {
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        if generate_mipmaps && baked_mip_levels != 1 {
            panic!("Generating mips on textures that have baked mips is not supported");
        }

        let format = format.unwrap_or(wgpu::TextureFormat::Rgba8UnormSrgb);
        let texture = if generate_mipmaps {
            let mip_level_count = size.max_mips(wgpu::TextureDimension::D2);
            let texture = base_renderer
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: if USE_LABELS { label } else { None },
                    size,
                    mip_level_count,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
            base_renderer.queue.write_texture(
                wgpu::ImageCopyTexture {
                    aspect: wgpu::TextureAspect::All,
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                },
                img_bytes,
                wgpu::ImageDataLayout {
                    offset: 0,
                    // queue.write_texture is exempt from COPY_BYTES_PER_ROW_ALIGNMENT requirement
                    bytes_per_row: Some(format.block_size(None).unwrap() * dimensions.0),
                    rows_per_image: Some(dimensions.1),
                },
                size,
            );

            let mip_encoder =
                base_renderer
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: USE_LABELS.then_some("mip_encoder"),
                    });

            generate_mipmaps_for_texture(
                base_renderer,
                mip_encoder,
                &texture,
                mip_level_count,
                format,
            )?;

            texture
        } else {
            base_renderer.device.create_texture_with_data(
                &base_renderer.queue,
                &wgpu::TextureDescriptor {
                    label: if USE_LABELS { label } else { None },
                    size,
                    mip_level_count: baked_mip_levels,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                img_bytes,
            )
        };

        let view = texture.create_view(&Default::default());
        let sampler_index = base_renderer
            .sampler_cache
            .lock()
            .unwrap()
            .get_sampler_index(&base_renderer.device, sampler_descriptor);

        Ok(Self {
            texture,
            view,
            sampler_index,
            size,
        })
    }

    pub fn _from_color_srgb(base_renderer: &BaseRenderer, color: [u8; 4]) -> Result<Self> {
        let one_pixel_image = {
            let mut img = image::RgbaImage::new(1, 1);
            img.put_pixel(0, 0, image::Rgba(color));
            img
        };
        Texture::from_decoded_image(
            base_renderer,
            &one_pixel_image,
            one_pixel_image.dimensions(),
            1,
            Some("from_color texture"),
            wgpu::TextureFormat::Rgba8UnormSrgb.into(),
            false,
            &SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            },
        )
    }

    pub fn from_color(base_renderer: &BaseRenderer, color: [u8; 4]) -> Result<Self> {
        let one_pixel_image = {
            let mut img = image::RgbaImage::new(1, 1);
            img.put_pixel(0, 0, image::Rgba(color));
            img
        };
        Texture::from_decoded_image(
            base_renderer,
            &one_pixel_image,
            one_pixel_image.dimensions(),
            1,
            Some("from_color texture"),
            wgpu::TextureFormat::Rgba8Unorm.into(),
            false,
            &SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            },
        )
    }

    pub fn _from_gray(base_renderer: &BaseRenderer, gray_value: u8) -> Result<Self> {
        let one_pixel_gray_image = {
            let mut img = image::GrayImage::new(1, 1);
            img.put_pixel(0, 0, image::Luma([gray_value]));
            img
        };
        Texture::from_decoded_image(
            base_renderer,
            &one_pixel_gray_image,
            one_pixel_gray_image.dimensions(),
            1,
            Some("from_gray texture"),
            wgpu::TextureFormat::R8Unorm.into(),
            false,
            &SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            },
        )
    }

    pub fn _flat_normal_map(base_renderer: &BaseRenderer) -> Result<Self> {
        Self::from_color(base_renderer, [127, 127, 255, 255])
    }

    /// only works if the texture had the COPY_SRC usage set at creation
    /// returns one byte array per image (to account for arrays/cubemaps) including mips if any
    pub async fn to_bytes(&self, base_renderer: &BaseRenderer) -> Result<Vec<Vec<u8>>> {
        let mut result = vec![];
        for depth in 0..self.size.depth_or_array_layers {
            let mut depth_level_bytes = vec![];
            for mip_level in 0..self.texture.mip_level_count() {
                let width = self.size.width >> mip_level;
                let height = self.size.height >> mip_level;
                let unpadded_bytes_per_row = self.unpadded_bytes_per_row(Some(mip_level));
                let padded_bytes_per_row = self.padded_bytes_per_row(Some(mip_level));

                let mut encoder =
                    base_renderer
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: USE_LABELS.then_some("to_bytes encoder"),
                        });

                let output_buffer = base_renderer.device.create_buffer(&wgpu::BufferDescriptor {
                    size: (padded_bytes_per_row * height) as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    label: USE_LABELS.then_some("to_bytes output buffer"),
                    mapped_at_creation: false,
                });

                encoder.copy_texture_to_buffer(
                    wgpu::ImageCopyTexture {
                        aspect: wgpu::TextureAspect::All,
                        texture: &self.texture,
                        mip_level,
                        origin: wgpu::Origin3d {
                            z: depth,
                            ..wgpu::Origin3d::ZERO
                        },
                    },
                    wgpu::ImageCopyBuffer {
                        buffer: &output_buffer,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(padded_bytes_per_row),
                            rows_per_image: None,
                        },
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );

                base_renderer.queue.submit(Some(encoder.finish()));

                {
                    let buffer_slice = output_buffer.slice(..);

                    // TODO: make sure this works on the web
                    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
                    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                        tx.send(result).unwrap();
                    });
                    base_renderer.device.poll(wgpu::Maintain::Wait);
                    rx.receive().await.unwrap()?;

                    let data = &buffer_slice.get_mapped_range();
                    depth_level_bytes.reserve((unpadded_bytes_per_row * height) as usize);
                    for i in 0..height {
                        let start_index = (i * padded_bytes_per_row) as usize;
                        let end_index = start_index + unpadded_bytes_per_row as usize;
                        depth_level_bytes.extend_from_slice(&data[start_index..end_index]);
                    }
                }
                output_buffer.unmap();
            }
            result.push(depth_level_bytes);
        }

        Ok(result)
    }

    pub fn create_scaled_surface_texture(
        base_renderer: &BaseRenderer,
        (width, height): (u32, u32),
        render_scale: f32,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: ((width as f32 * render_scale.sqrt()).round() as u32).max(1),
            height: ((height as f32 * render_scale.sqrt()).round() as u32).max(1),
            depth_or_array_layers: 1,
        };
        let texture = base_renderer
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: USE_LABELS.then_some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

        let view = texture.create_view(&Default::default());
        let sampler_index = base_renderer
            .sampler_cache
            .lock()
            .unwrap()
            .get_sampler_index(
                &base_renderer.device,
                &SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                },
            );

        Self {
            texture,
            view,
            sampler_index,
            size,
        }
    }

    pub fn create_depth_texture(
        base_renderer: &BaseRenderer,
        (width, height): (u32, u32),
        render_scale: f32,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: ((width as f32 * render_scale.sqrt()).round() as u32).max(1),
            height: ((height as f32 * render_scale.sqrt()).round() as u32).max(1),
            depth_or_array_layers: 1,
        };
        let texture = base_renderer
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: USE_LABELS.then_some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Texture::DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

        let view = texture.create_view(&Default::default());
        let sampler_index = base_renderer
            .sampler_cache
            .lock()
            .unwrap()
            .get_sampler_index(
                &base_renderer.device,
                &SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    compare: Some(wgpu::CompareFunction::GreaterEqual),
                    ..Default::default()
                },
            );

        Self {
            texture,
            view,
            sampler_index,
            size,
        }
    }

    pub fn create_cube_depth_texture_array(
        base_renderer: &BaseRenderer,
        size: u32,
        label: Option<&str>,
        length: u32,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6 * length,
        };

        let texture = base_renderer
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: if USE_LABELS { label } else { None },
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Texture::DEPTH_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        });

        let sampler_index = base_renderer
            .sampler_cache
            .lock()
            .unwrap()
            .get_sampler_index(
                &base_renderer.device,
                &SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::Repeat,
                    address_mode_w: wgpu::AddressMode::Repeat,
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    // compare: Some(wgpu::CompareFunction::LessEqual),
                    ..Default::default()
                },
            );

        Self {
            texture,
            view,
            sampler_index,
            size,
        }
    }

    pub fn create_depth_texture_array(
        base_renderer: &BaseRenderer,
        size: (u32, u32),
        label: Option<&str>,
        length: u32,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: length,
        };

        let texture = base_renderer
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: if USE_LABELS { label } else { None },
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Texture::DEPTH_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let sampler_index = base_renderer
            .sampler_cache
            .lock()
            .unwrap()
            .get_sampler_index(
                &base_renderer.device,
                &SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    // compare: Some(wgpu::CompareFunction::LessEqual),
                    ..Default::default()
                },
            );

        Self {
            texture,
            view,
            sampler_index,
            size,
        }
    }

    pub fn create_cubemap_from_equirectangular(
        base_renderer: &BaseRenderer,
        renderer_constant_data: &RendererConstantData,
        format: wgpu::TextureFormat,
        label: Option<&str>,
        er_texture: &Texture,
        generate_mipmaps: bool,
    ) -> Result<Self> {
        let equirectangular_to_cubemap_pipeline = match format {
            wgpu::TextureFormat::Rgba8UnormSrgb => {
                &renderer_constant_data.equirectangular_to_cubemap_pipeline
            },
            wgpu::TextureFormat::Rgba16Float => {
                &renderer_constant_data.equirectangular_to_cubemap_hdr_pipeline
            }
            _ => anyhow::bail!("create_cubemap_from_equirectangular called with unsupported texture format: {format:?}"),
        };

        // make sure it's a multiple of 4 (cheap trick to support Bc7 compression)
        let size = ((((er_texture.size.width as f32 / 3.0) / 4.0).ceil() * 4.0) as u32).max(4);

        let size = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6,
        };

        let mip_level_count = if generate_mipmaps {
            size.max_mips(wgpu::TextureDimension::D2)
        } else {
            1
        };

        let cubemap_texture = base_renderer
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: if USE_LABELS { label } else { None },
                size,
                mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

        let camera_buffer =
            base_renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: USE_LABELS.then_some("Cubemap Generation Camera Buffer"),
                    contents: bytemuck::cast_slice(&[SkyboxShaderCameraRaw::default()]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
        let camera_bind_group =
            base_renderer
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &renderer_constant_data.single_uniform_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    }],
                    label: USE_LABELS.then_some("cubemap_gen_camera_bind_group"),
                });

        let faces: Vec<_> = build_cubemap_face_camera_views(
            Vec3::new(0.0, 0.0, 0.0),
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
                    array_layer_count: Some(1),
                    ..Default::default()
                }),
            )
        })
        .collect();

        for (face_view_proj_matrices, face_texture_view) in faces {
            let mut encoder =
                base_renderer
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: USE_LABELS
                            .then_some("create_cubemap_texture_from_equirectangular encoder"),
                    });
            let er_texture_bind_group;
            {
                let mut sampler_cache_guard = base_renderer.sampler_cache.lock().unwrap();
                let sampler = sampler_cache_guard.get_sampler(
                    &base_renderer.device,
                    &SamplerDescriptor {
                        address_mode_u: wgpu::AddressMode::ClampToEdge,
                        address_mode_v: wgpu::AddressMode::ClampToEdge,
                        address_mode_w: wgpu::AddressMode::ClampToEdge,
                        mag_filter: wgpu::FilterMode::Linear,
                        min_filter: wgpu::FilterMode::Linear,
                        mipmap_filter: wgpu::FilterMode::Nearest,
                        ..Default::default()
                    },
                );
                er_texture_bind_group =
                    base_renderer
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: &renderer_constant_data.single_texture_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(&er_texture.view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(sampler),
                                },
                            ],
                            label: None,
                        });
            }

            base_renderer.queue.write_buffer(
                &camera_buffer,
                0,
                bytemuck::cast_slice(&[SkyboxShaderCameraRaw::from(face_view_proj_matrices)]),
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
                rpass.set_pipeline(equirectangular_to_cubemap_pipeline);
                rpass.set_bind_group(0, &er_texture_bind_group, &[]);
                rpass.set_bind_group(1, &camera_bind_group, &[]);
                rpass.set_vertex_buffer(
                    0,
                    renderer_constant_data
                        .skybox_mesh
                        .vertex_buffer
                        .src()
                        .slice(..),
                );
                rpass.set_index_buffer(
                    renderer_constant_data
                        .skybox_mesh
                        .index_buffer
                        .buffer
                        .src()
                        .slice(..),
                    renderer_constant_data.skybox_mesh.index_buffer.format,
                );
                rpass.draw_indexed(
                    0..(renderer_constant_data
                        .skybox_mesh
                        .index_buffer
                        .buffer
                        .length() as u32),
                    0,
                    0..1,
                );
            }
            base_renderer.queue.submit(Some(encoder.finish()));
        }

        if generate_mipmaps {
            todo!("Call generate_mipmaps_for_texture for each side of the cubemap");
        }

        let view = cubemap_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let sampler_index = base_renderer
            .sampler_cache
            .lock()
            .unwrap()
            .get_sampler_index(
                &base_renderer.device,
                &SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::Repeat,
                    address_mode_w: wgpu::AddressMode::Repeat,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                },
            );

        Ok(Self {
            texture: cubemap_texture,
            view,
            sampler_index,
            size,
        })
    }

    /// Each image should have the same dimensions!
    pub fn create_cubemap(
        base_renderer: &BaseRenderer,
        image: RawImageSlice,
        label: Option<&str>,
        format: wgpu::TextureFormat,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: image.width,
            height: image.height,
            depth_or_array_layers: 6,
        };

        let texture = base_renderer.device.create_texture_with_data(
            &base_renderer.queue,
            &wgpu::TextureDescriptor {
                label: if USE_LABELS { label } else { None },
                size,
                mip_level_count: image.mip_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            image.raw,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let sampler_index = base_renderer
            .sampler_cache
            .lock()
            .unwrap()
            .get_sampler_index(
                &base_renderer.device,
                &SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                },
            );

        Self {
            texture,
            view,
            sampler_index,
            size,
        }
    }

    pub fn create_diffuse_env_map(
        base_renderer: &BaseRenderer,
        renderer_constant_data: &RendererConstantData,
        label: Option<&str>,
        skybox_rad_texture: &Texture,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: 128,
            height: 128,
            depth_or_array_layers: 6,
        };

        let env_map = base_renderer
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: if USE_LABELS { label } else { None },
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

        let camera_buffer =
            base_renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: USE_LABELS.then_some("Env map Generation Camera Buffer"),
                    contents: bytemuck::cast_slice(&[SkyboxShaderCameraRaw::default()]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
        let camera_bind_group =
            base_renderer
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &renderer_constant_data.single_uniform_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    }],
                    label: USE_LABELS.then_some("env_map_gen_camera_bind_group"),
                });

        let faces: Vec<_> = build_cubemap_face_camera_views(
            Vec3::new(0.0, 0.0, 0.0),
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
                    array_layer_count: Some(1),
                    ..Default::default()
                }),
            )
        })
        .collect();

        for (face_view_proj_matrices, face_texture_view) in faces {
            let mut encoder =
                base_renderer
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: USE_LABELS.then_some("create_env_map encoder"),
                    });
            let skybox_ir_texture_bind_group =
                base_renderer
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &renderer_constant_data.single_cube_texture_bind_group_layout,
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
                                    base_renderer
                                        .sampler_cache
                                        .lock()
                                        .unwrap()
                                        .get_sampler_by_index(skybox_rad_texture.sampler_index),
                                ),
                            },
                        ],
                        label: None,
                    });

            base_renderer.queue.write_buffer(
                &camera_buffer,
                0,
                bytemuck::cast_slice(&[SkyboxShaderCameraRaw::from(face_view_proj_matrices)]),
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
                rpass.set_pipeline(&renderer_constant_data.diffuse_env_map_gen_pipeline);
                rpass.set_bind_group(0, &skybox_ir_texture_bind_group, &[]);
                rpass.set_bind_group(1, &camera_bind_group, &[]);
                rpass.set_vertex_buffer(
                    0,
                    renderer_constant_data
                        .skybox_mesh
                        .vertex_buffer
                        .src()
                        .slice(..),
                );
                rpass.set_index_buffer(
                    renderer_constant_data
                        .skybox_mesh
                        .index_buffer
                        .buffer
                        .src()
                        .slice(..),
                    renderer_constant_data.skybox_mesh.index_buffer.format,
                );
                rpass.draw_indexed(
                    0..(renderer_constant_data
                        .skybox_mesh
                        .index_buffer
                        .buffer
                        .length() as u32),
                    0,
                    0..1,
                );
            }
            base_renderer.queue.submit(Some(encoder.finish()));
        }

        let view = env_map.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let sampler_index = base_renderer
            .sampler_cache
            .lock()
            .unwrap()
            .get_sampler_index(
                &base_renderer.device,
                &SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::Repeat,
                    address_mode_w: wgpu::AddressMode::Repeat,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                },
            );

        Self {
            texture: env_map,
            view,
            sampler_index,
            size,
        }
    }

    pub fn create_specular_env_map(
        base_renderer: &BaseRenderer,
        renderer_constant_data: &RendererConstantData,
        label: Option<&str>,
        skybox_rad_texture: &Texture,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: skybox_rad_texture.size.width,
            height: skybox_rad_texture.size.height,
            depth_or_array_layers: 6,
        };

        let mip_level_count = 5.min(size.max_mips(wgpu::TextureDimension::D2));

        let env_map = base_renderer
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: if USE_LABELS { label } else { None },
                size,
                mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

        let camera_buffer =
            base_renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: USE_LABELS.then_some("Env map Generation Camera Buffer"),
                    contents: bytemuck::cast_slice(&[SkyboxShaderCameraRaw::default()]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let roughness_buffer =
            base_renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: USE_LABELS.then_some("Env map Generation Roughness Buffer"),
                    contents: bytemuck::cast_slice(&[0.0f32]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
        let camera_roughness_bind_group =
            base_renderer
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &renderer_constant_data.two_uniform_bind_group_layout,
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
                    label: USE_LABELS.then_some("spec_env_map_gen_roughness_bind_group"),
                });

        let camera_projection_matrices = build_cubemap_face_camera_views(
            Vec3::new(0.0, 0.0, 0.0),
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
                                array_layer_count: Some(1),
                                base_mip_level: mip_level,
                                mip_level_count: Some(1),
                                ..Default::default()
                            }),
                        )
                    })
                    .for_each(|(face_view_proj_matrices, face_texture_view)| {
                        let mut encoder = base_renderer.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: USE_LABELS.then_some("create_env_map encoder"),
                            },
                        );
                        let skybox_ir_texture_bind_group =
                            base_renderer
                                .device
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    layout: &renderer_constant_data
                                        .single_cube_texture_bind_group_layout,
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
                                                base_renderer
                                                    .sampler_cache
                                                    .lock()
                                                    .unwrap()
                                                    .get_sampler_by_index(
                                                        skybox_rad_texture.sampler_index,
                                                    ),
                                            ),
                                        },
                                    ],
                                    label: None,
                                });
                        base_renderer.queue.write_buffer(
                            &camera_buffer,
                            0,
                            bytemuck::cast_slice(&[SkyboxShaderCameraRaw::from(
                                face_view_proj_matrices,
                            )]),
                        );
                        base_renderer.queue.write_buffer(
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
                            rpass.set_pipeline(
                                &renderer_constant_data.specular_env_map_gen_pipeline,
                            );
                            rpass.set_bind_group(0, &skybox_ir_texture_bind_group, &[]);
                            rpass.set_bind_group(1, &camera_roughness_bind_group, &[]);
                            rpass.set_vertex_buffer(
                                0,
                                renderer_constant_data
                                    .skybox_mesh
                                    .vertex_buffer
                                    .src()
                                    .slice(..),
                            );
                            rpass.set_index_buffer(
                                renderer_constant_data
                                    .skybox_mesh
                                    .index_buffer
                                    .buffer
                                    .src()
                                    .slice(..),
                                renderer_constant_data.skybox_mesh.index_buffer.format,
                            );
                            rpass.draw_indexed(
                                0..(renderer_constant_data
                                    .skybox_mesh
                                    .index_buffer
                                    .buffer
                                    .length() as u32),
                                0,
                                0..1,
                            );
                        }
                        base_renderer.queue.submit(Some(encoder.finish()));
                    });
            });

        let view = env_map.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let sampler_index = base_renderer
            .sampler_cache
            .lock()
            .unwrap()
            .get_sampler_index(
                &base_renderer.device,
                &SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::Repeat,
                    address_mode_w: wgpu::AddressMode::Repeat,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                },
            );

        Self {
            texture: env_map,
            view,
            sampler_index,
            size,
        }
    }

    pub fn create_brdf_lut(
        base_renderer: &BaseRenderer,
        brdf_lut_gen_pipeline: &wgpu::RenderPipeline,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        };

        let texture = base_renderer
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: USE_LABELS.then_some("Brdf Lut"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });

        let sampler_index = base_renderer
            .sampler_cache
            .lock()
            .unwrap()
            .get_sampler_index(
                &base_renderer.device,
                &SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                },
            );

        let mut encoder =
            base_renderer
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: USE_LABELS.then_some("create_brdf_lut encoder"),
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
        base_renderer.queue.submit(Some(encoder.finish()));

        Self {
            texture,
            view,
            sampler_index,
            size,
        }
    }
}

#[profiling::function]
fn generate_mipmaps_for_texture(
    base_renderer: &BaseRenderer,
    mut mip_encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    mip_level_count: u32,
    format: wgpu::TextureFormat,
) -> Result<()> {
    // TODO: don't crate the pipeline each time we generate mipmaps! 😠
    let blit_shader = base_renderer
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/blit.wgsl").into()),
        });

    let single_texture_bind_group_layout =
        base_renderer
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: USE_LABELS.then_some("single_texture_bind_group_layout"),
            });

    let mip_pipeline_layout =
        base_renderer
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: USE_LABELS.then_some("Mesh Pipeline Layout"),
                bind_group_layouts: &[&single_texture_bind_group_layout],
                push_constant_ranges: &[],
            });

    let mip_render_pipeline =
        base_renderer
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: USE_LABELS.then_some("mip_render_pipeline"),
                layout: Some(&mip_pipeline_layout),
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

    let mip_texure_views = (0..mip_level_count)
        .map(|mip| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: USE_LABELS.then_some("mip"),
                format: None,
                dimension: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
            })
        })
        .collect::<Vec<_>>();

    for target_mip in 1..mip_level_count as usize {
        let bind_group;
        {
            let mut sampler_cache_guard = base_renderer.sampler_cache.lock().unwrap();
            let mip_sampler = sampler_cache_guard.get_sampler(
                &base_renderer.device,
                &SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                },
            );
            bind_group = base_renderer
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &single_texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &mip_texure_views[target_mip - 1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(mip_sampler),
                        },
                    ],
                    label: None,
                });
        }

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
    base_renderer.queue.submit(Some(mip_encoder.finish()));
    Ok(())
}
