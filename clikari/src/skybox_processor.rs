use std::path::PathBuf;

pub struct SkyboxProcessorArgs {
    pub background_path: PathBuf,
    pub environment_hdr_path: Option<PathBuf>,
}

pub fn run(args: SkyboxProcessorArgs) {
    // let base_render_state = {
    //     let backends = if cfg!(target_os = "windows") {
    //         wgpu::Backends::from(wgpu::Backend::Dx12)
    //         // wgpu::Backends::PRIMARY
    //     } else {
    //         wgpu::Backends::PRIMARY
    //     };
    //     BaseRenderer::new(&window, backends, wgpu::PresentMode::AutoNoVsync).await
    // };
}
