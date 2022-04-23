use std::num::{NonZeroU32, NonZeroU8};

use anyhow::*;
use image::GenericImageView;
use wgpu::util::DeviceExt;

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

pub struct CreateCubeMapImagesParam<'a> {
    pub pos_x: &'a image::DynamicImage,
    pub neg_x: &'a image::DynamicImage,
    pub pos_y: &'a image::DynamicImage,
    pub neg_y: &'a image::DynamicImage,
    pub pos_z: &'a image::DynamicImage,
    pub neg_z: &'a image::DynamicImage,
}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: &str,
        generate_mipmaps: bool,
        lod_min_clamp: Option<f32>,
        lod_max_clamp: Option<f32>,
        mipmap_filter: Option<wgpu::FilterMode>,
        mag_filter: Option<wgpu::FilterMode>,
        min_filter: Option<wgpu::FilterMode>,
        anisotropy_clamp: Option<NonZeroU8>,
        format: Option<wgpu::TextureFormat>,
    ) -> Result<Self> {
        let img = image::load_from_memory(bytes)?;
        Self::from_image(
            device,
            queue,
            &img,
            Some(label),
            generate_mipmaps,
            lod_min_clamp,
            lod_max_clamp,
            mipmap_filter,
            mag_filter,
            min_filter,
            anisotropy_clamp,
            format,
        )
    }

    pub fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &image::DynamicImage,
        label: Option<&str>,
        generate_mipmaps: bool,
        lod_min_clamp: Option<f32>,
        lod_max_clamp: Option<f32>,
        mipmap_filter: Option<wgpu::FilterMode>,
        mag_filter: Option<wgpu::FilterMode>,
        min_filter: Option<wgpu::FilterMode>,
        anisotropy_clamp: Option<NonZeroU8>,
        format: Option<wgpu::TextureFormat>,
    ) -> Result<Self> {
        let img_as_rgba = img.as_rgba8().ok_or_else(|| {
            anyhow!("Failed to convert image into rgba8. Is the image missing the alpha channel?")
        })?;

        let dimensions = img.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let mip_level_count = if generate_mipmaps { size.max_mips() } else { 1 };
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

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0, // TODO: could this affect things?
                origin: wgpu::Origin3d::ZERO,
            },
            img_as_rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * dimensions.0),
                rows_per_image: NonZeroU32::new(dimensions.1),
            },
            size,
        );

        // apply defaults to sampler
        let sampler_mag_filter = mag_filter.unwrap_or(wgpu::FilterMode::Linear);
        let sampler_min_filter = min_filter.unwrap_or(wgpu::FilterMode::Linear);
        let sampler_anisotropy_clamp = anisotropy_clamp.or(NonZeroU8::new(8));
        let mipmap_filter = mipmap_filter.unwrap_or(wgpu::FilterMode::Nearest);
        let lod_min_clamp =
            lod_min_clamp.unwrap_or(wgpu::SamplerDescriptor::default().lod_min_clamp);
        let lod_max_clamp =
            lod_max_clamp.unwrap_or(wgpu::SamplerDescriptor::default().lod_max_clamp);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: sampler_mag_filter,
            min_filter: sampler_min_filter,
            mipmap_filter,
            anisotropy_clamp: sampler_anisotropy_clamp,
            lod_min_clamp,
            lod_max_clamp,
            ..Default::default()
        });

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
            );
        }

        Ok(Self {
            texture,
            view,
            sampler,
        })
    }

    pub fn create_render_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        render_scale: f32,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: ((config.width as f32) * render_scale).round() as u32,
            height: ((config.height as f32) * render_scale).round() as u32,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
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
        }
    }

    pub fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        render_scale: f32,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: ((config.width as f32) * render_scale).round() as u32,
            height: ((config.height as f32) * render_scale).round() as u32,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }

    // each image should have the same dimensions!
    pub fn create_cubemap_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        images: CreateCubeMapImagesParam,
        label: Option<&str>,
        generate_mipmaps: bool,
    ) -> Result<Self> {
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
        .map(|img| {
            img.as_rgba8().ok_or_else(|| {
                anyhow!(
                    "Failed to convert image into rgba8. Is the image missing the alpha channel?"
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
        let dimensions = images_as_rgba[0].dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 6,
        };

        let layer_size = wgpu::Extent3d {
            depth_or_array_layers: 1,
            ..size
        };

        let mip_level_count = if generate_mipmaps {
            layer_size.max_mips()
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
            ..wgpu::TextureViewDescriptor::default()
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

        Ok(Self {
            texture,
            view,
            sampler,
        })
    }
}

fn generate_mipmaps_for_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut mip_encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    mip_level_count: u32,
    format: wgpu::TextureFormat,
) {
    let blit_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
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
            targets: &[format.into()],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
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
    let mip_bind_group_layout = mip_render_pipeline.get_bind_group_layout(0);
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
            layout: &mip_bind_group_layout,
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
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &mip_texure_views[target_mip],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        rpass.set_pipeline(&mip_render_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
    queue.submit(Some(mip_encoder.finish()));
}
