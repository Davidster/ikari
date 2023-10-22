use std::path::PathBuf;

use ikari::{
    file_manager::native_fs,
    renderer::{
        BaseRenderer, BindedSkybox, Renderer, SkyboxBackgroundPath, SkyboxEnvironmentHDRPath,
    },
    texture::RawImage,
    texture_compression::TextureCompressionArgs,
};

use crate::PATH_MAKER;

const DXC_PATH: &str = "dxc/";

pub struct SkyboxProcessorArgs {
    pub background_path: PathBuf,
    pub environment_hdr_path: Option<PathBuf>,
    pub out_folder: PathBuf,
}

pub async fn run(args: SkyboxProcessorArgs) {
    if let Err(err) = run_internal(args).await {
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
    let renderer = Renderer::new(base_renderer, wgpu::TextureFormat::Bgra8Unorm, (1, 1)).await?;

    let bindable_skybox = ikari::asset_loader::make_bindable_skybox(
        &SkyboxBackgroundPath::Equirectangular(PATH_MAKER.make(args.background_path)),
        args.environment_hdr_path
            .as_ref()
            .map(|environment_hdr_path| {
                SkyboxEnvironmentHDRPath::Equirectangular(PATH_MAKER.make(environment_hdr_path))
            })
            .as_ref(),
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

    native_fs::create_dir_all(&args.out_folder)?;

    {
        let folder = std::path::Path::join(&args.out_folder, "background");
        native_fs::create_dir_all(&folder)?;

        let texture = background;

        if texture.texture.mip_level_count() != 1 {
            log::error!("Skybox background texture contained mipmaps which will be ignored");
        }

        let all_texture_bytes = texture.to_bytes(&renderer.base).await?;

        let cube_texture_names = ["pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z"];
        for (texture_bytes, file_name) in all_texture_bytes.iter().zip(cube_texture_names.iter()) {
            // also save a PNG for use on the web
            let png_compressed_img_bytes = image::RgbaImage::from_raw(
                texture.size.width,
                texture.size.height,
                texture_bytes.clone(),
            )
            .ok_or_else(|| anyhow::anyhow!("Failed to decode raw background image"))?;

            let png_file_path = std::path::Path::join(&folder, format!("{file_name}.png"));

            png_compressed_img_bytes.save(png_file_path.clone())?;

            let gpu_compressed_img_bytes =
                compressor.compress_raw_image(TextureCompressionArgs {
                    img_bytes: texture_bytes,
                    img_width: texture.size.width,
                    img_height: texture.size.height,
                    img_channel_count: 4,
                    generate_mipmaps: false,
                    is_normal_map: false,
                    is_srgb: true,
                    thread_count: num_cpus::get() as u32,
                })?;

            let gpu_compressed_file_path =
                std::path::Path::join(&folder, format!("{file_name}_compressed.bin"));

            native_fs::write(&gpu_compressed_file_path, gpu_compressed_img_bytes)?;
            log::info!("Done compressing: {:?}", png_file_path.canonicalize()?);
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

        native_fs::write(&full_file_path, compressed_img_bytes)?;
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

        native_fs::write(&full_file_path, compressed_img_bytes)?;
        log::info!("Done compressing: {:?}", full_file_path.canonicalize()?);
    }

    Ok(())
}
