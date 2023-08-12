use std::path::PathBuf;

use ikari::{
    renderer::{
        BaseRenderer, BindedSkybox, Renderer, SkyboxBackgroundPath, SkyboxHDREnvironmentPath,
    },
    texture::{RawImage, Texture},
    texture_compression::TextureCompressionArgs,
};
use image::{ImageBuffer, Rgba};

use crate::texture_compressor::TextureCompressorArgs;

const DXC_PATH: &str = "dxc/";

pub struct SkyboxProcessorArgs {
    pub background_path: PathBuf,
    pub environment_hdr_path: Option<PathBuf>,
    pub out_folder: PathBuf,
}

pub fn run(args: SkyboxProcessorArgs) {
    if let Err(err) = ikari::block_on(run_internal(args)) {
        log::error!("Error: {err}\n{}", err.backtrace());
    }
}

pub async fn run_internal(args: SkyboxProcessorArgs) -> anyhow::Result<()> {
    let backends = if cfg!(target_os = "windows") {
        wgpu::Backends::from(wgpu::Backend::Dx12)
        // wgpu::Backends::PRIMARY
    } else {
        wgpu::Backends::PRIMARY
    };

    let base_renderer = BaseRenderer::offscreen(backends, Some(DXC_PATH.into())).await?;
    let mut renderer =
        Renderer::new(base_renderer, wgpu::TextureFormat::Bgra8Unorm, (1, 1)).await?;

    let bindable_skybox = ikari::asset_loader::make_bindable_skybox(
        SkyboxBackgroundPath::Equirectangular(
            args.background_path
                .to_str()
                .expect("background_path was not valid unicode"),
        ),
        args.environment_hdr_path
            .as_ref()
            .map(|environment_hdr_path| {
                SkyboxHDREnvironmentPath::Equirectangular(
                    environment_hdr_path
                        .to_str()
                        .expect("environment_hdr_path was not valid unicode"),
                )
            }),
    )
    .await?;

    let binded_skybox =
        ikari::asset_loader::bind_skybox(&renderer.base, &renderer.constant_data, bindable_skybox)?;

    log::info!("Done processing skybox");

    let BindedSkybox {
        background,
        diffuse_environment_map,
        specular_environment_map,
    } = binded_skybox;

    let compressor = ikari::texture_compression::TextureCompressor;

    let cube_texture_names = ["pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z"];

    std::fs::create_dir_all(&args.out_folder)?;

    {
        // TODO: join into a single file like with diffuse/spec env maps
        let folder = std::path::Path::join(&args.out_folder, "background");
        std::fs::create_dir_all(&folder)?;

        let texture = background;

        if texture.texture.mip_level_count() != 1 {
            log::error!("Skybox background texture contained mipmaps which will be ignored");
        }

        let all_texture_bytes = texture.to_bytes(&renderer.base).await?;

        for (texture_bytes, file_name) in all_texture_bytes.iter().zip(cube_texture_names.iter()) {
            let compressed_img_bytes = compressor.compress_raw_image(TextureCompressionArgs {
                img_bytes: &texture_bytes,
                img_width: texture.size.width,
                img_height: texture.size.height,
                img_channel_count: 4,
                generate_mipmaps: false,
                is_normal_map: false,
                is_srgb: true,
                thread_count: num_cpus::get() as u32,
            })?;

            let full_file_path =
                std::path::Path::join(&folder, format!("{file_name}_compressed.bin"));

            std::fs::write(&full_file_path, compressed_img_bytes)?;
            log::info!("Done compressing: {:?}", full_file_path.canonicalize()?);
        }
    }

    {
        let texture = diffuse_environment_map;
        let all_texture_bytes = texture.to_bytes(&renderer.base).await?;

        let compressed_img_bytes = compressor.compress_raw_float_image(RawImage {
            width: texture.size.width,
            height: texture.size.height,
            depth: all_texture_bytes.len() as u32,
            mip_count: texture.texture.mip_level_count(),
            raw: all_texture_bytes.iter().flatten().copied().collect(),
        })?;

        let full_file_path =
            std::path::Path::join(&args.out_folder, "diffuse_environment_map_compressed.bin");

        std::fs::write(&full_file_path, compressed_img_bytes)?;
        log::info!("Done compressing: {:?}", full_file_path.canonicalize()?);
    }

    {
        let texture = specular_environment_map;
        let all_texture_bytes = texture.to_bytes(&renderer.base).await?;

        let compressed_img_bytes = compressor.compress_raw_float_image(RawImage {
            width: texture.size.width,
            height: texture.size.height,
            depth: all_texture_bytes.len() as u32,
            mip_count: texture.texture.mip_level_count(),
            raw: all_texture_bytes.iter().flatten().copied().collect(),
        })?;

        let full_file_path =
            std::path::Path::join(&args.out_folder, "specular_environment_map_compressed.bin");

        std::fs::write(&full_file_path, compressed_img_bytes)?;
        log::info!("Done compressing: {:?}", full_file_path.canonicalize()?);
    }

    Ok(())
}
