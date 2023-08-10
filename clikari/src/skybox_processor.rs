use std::path::PathBuf;

use ikari::{
    renderer::{
        BaseRenderer, BindedSkybox, Renderer, SkyboxBackgroundPath, SkyboxHDREnvironmentPath,
    },
    texture::Texture,
};
use image::{ImageBuffer, Rgba};

const DXC_PATH: &str = "dxc/";

pub struct SkyboxProcessorArgs {
    pub background_path: PathBuf,
    pub environment_hdr_path: Option<PathBuf>,
    pub out_path: PathBuf,
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

    renderer.base.device.start_capture();

    let bindable_skybox = ikari::asset_loader::make_bindable_skybox(
        SkyboxBackgroundPath::Equirectangular {
            image_path: args
                .background_path
                .to_str()
                .expect("background_path was not valid unicode"),
        },
        args.environment_hdr_path
            .as_ref()
            .map(
                |environment_hdr_path| SkyboxHDREnvironmentPath::Equirectangular {
                    image_path: environment_hdr_path
                        .to_str()
                        .expect("environment_hdr_path was not valid unicode"),
                },
            ),
    )
    .await?;

    let binded_skybox =
        ikari::asset_loader::bind_skybox(&renderer.base, &renderer.constant_data, bindable_skybox)?;

    let BindedSkybox {
        background,
        diffuse_environment_map,
        specular_environment_map,
    } = binded_skybox;

    {
        let texture = background;
        let path = "background.png";

        let texture_bytes = texture.to_bytes(&renderer.base).await?;

        // TODO: use ikari:: texture compression module to compress it as srgb
        // basis_universal doesn't support BC6 at the moment

        // let buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(
        //     texture.size.width,
        //     texture.size.height * texture.size.depth_or_array_layers,
        //     texture
        //         .to_bytes(&renderer.base)
        //         .await?
        //         .iter()
        //         .flatten()
        //         .copied()
        //         .collect::<Vec<_>>(),
        // )
        // .unwrap();
        // buffer.save(path).unwrap();
    }

    renderer.base.device.stop_capture();

    /* {
        let texture = diffuse_environment_map;
        let path = "diffuse_environment_map.png";
        let buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(
            texture.size.width * texture.size.depth_or_array_layers,
            texture.size.height,
            texture
                .to_bytes(&renderer.base)
                .await?
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
        )
        .unwrap();
        buffer.save(path).unwrap();
    }

    {
        let texture = specular_environment_map;
        let path = "specular_environment_map.png";
        let buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(
            texture.size.width * texture.size.depth_or_array_layers,
            texture.size.height,
            texture
                .to_bytes(&renderer.base)
                .await?
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
        )
        .unwrap();
        buffer.save(path).unwrap();
    } */

    Ok(())
}
