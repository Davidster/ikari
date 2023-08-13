use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{mpsc::channel, Arc},
};

use threadpool::ThreadPool;
use walkdir::WalkDir;

use ikari::texture_compression::{texture_path_to_compressed_path, TextureCompressionArgs};

use crate::PATH_MAKER;

const DEFAULT_THREADS_PER_TEXTURE: u32 = 4;

pub struct TextureCompressorArgs {
    pub search_folder: PathBuf,
    pub threads_per_texture: Option<u32>,
    pub force: bool,
}

pub fn run(args: TextureCompressorArgs) {
    if !args.search_folder.exists() {
        log::error!(
            "Error: search folder {:?} does not exist",
            args.search_folder
        );
        return;
    }

    let threads_per_texture = args
        .threads_per_texture
        .unwrap_or(DEFAULT_THREADS_PER_TEXTURE);

    let worker_count = (num_cpus::get() as f32 / threads_per_texture as f32).ceil() as usize;
    log::info!("worker_count={worker_count}");

    let texture_paths = {
        let mut texture_paths = find_gltf_texture_paths(&args.search_folder).unwrap();

        let gltf_exlusion_map: HashSet<PathBuf> = texture_paths
            .iter()
            .cloned()
            .map(|(path, _, _)| path.canonicalize().unwrap())
            .collect();

        // interpret all dangling textures as srgb color maps
        texture_paths.extend(
            find_dangling_texture_paths(&args.search_folder, gltf_exlusion_map)
                .unwrap()
                .iter()
                .cloned()
                .map(|path| (path, true, false)),
        );

        let mut filtered_texture_paths = vec![];

        for item in &texture_paths {
            let (path, _, _) = item;
            let compressed_path = texture_path_to_compressed_path(&PATH_MAKER.make(path));

            // remove all paths that have already been processed
            if !args.force && compressed_path.resolve().exists() {
                log::info!("{compressed_path:?} already exists. skipping");
                continue;
            }

            filtered_texture_paths.push(item.clone());
        }

        filtered_texture_paths
    };

    let texture_paths = Arc::new(texture_paths);

    let texture_count = texture_paths.len();

    let pool = ThreadPool::new(worker_count);

    let (tx, rx) = channel();
    for texture_index in 0..texture_count {
        let tx = tx.clone();
        let texture_paths = texture_paths.clone();
        pool.execute(move || {
            let (path, is_srgb, is_normal_map) = &texture_paths[texture_index];
            log::info!("start {path:?} (srgb={is_srgb:?}, is_normal_map={is_normal_map:?})");
            compress_file(
                path.as_path(),
                *is_srgb,
                *is_normal_map,
                threads_per_texture,
            )
            .unwrap();
            tx.send(texture_index).unwrap();
        });
    }
    let mut done_count = 0;
    for _ in 0..texture_count {
        let texture_index = rx.recv().unwrap();
        done_count += 1;
        log::info!(
            "done {:?} ({:?}/{:?})",
            texture_paths[texture_index].0,
            done_count,
            texture_count
        );
    }
}

fn find_gltf_texture_paths(search_folder: &Path) -> anyhow::Result<Vec<(PathBuf, bool, bool)>> {
    let mut result = Vec::new();
    let gltf_paths: Vec<_> = WalkDir::new(search_folder)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| !e.file_type().is_dir())
        .filter(|e| e.path().extension().is_some())
        .filter(|e| {
            e.path().extension().unwrap() == "gltf" || e.path().extension().unwrap() == "glb"
        })
        .map(|e| e.path().to_path_buf())
        .collect();
    for path in gltf_paths {
        let gltf = gltf::Gltf::open(&path)?;

        for texture in gltf.textures() {
            let is_srgb = gltf.materials().any(|material| {
                [
                    material.emissive_texture(),
                    material.pbr_metallic_roughness().base_color_texture(),
                ]
                .iter()
                .flatten()
                .any(|texture_info| texture_info.texture().index() == texture.index())
            });
            let is_normal_map = !is_srgb
                && gltf.materials().any(|material| {
                    material.normal_texture().is_some()
                        && material.normal_texture().unwrap().texture().index() == texture.index()
                });

            match texture.source().source() {
                gltf::image::Source::View { .. } => {
                    log::warn!("Warning: found inline texture in gltf file {:?}, texture index {:?}. This texture wont be compressed", path, texture.index());
                }
                gltf::image::Source::Uri { uri, .. } => {
                    let path = path.parent().unwrap().join(PathBuf::from(uri));
                    result.push((path, is_srgb, is_normal_map));
                }
            };
        }
    }
    Ok(result)
}

fn find_dangling_texture_paths(
    search_folder: &Path,
    exclude_list: HashSet<PathBuf>,
) -> anyhow::Result<Vec<PathBuf>> {
    Ok(WalkDir::new(search_folder)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| !e.file_type().is_dir())
        .filter(|e| !exclude_list.contains(&e.path().canonicalize().unwrap()))
        .filter(|e| e.path().extension().is_some())
        .filter(|e| {
            e.path().extension().unwrap() == "jpg" || e.path().extension().unwrap() == "png"
        })
        .map(|e| e.path().to_path_buf())
        .collect())
}

fn compress_file(
    img_path: &Path,
    is_srgb: bool,
    is_normal_map: bool,
    threads: u32,
) -> anyhow::Result<()> {
    let img_bytes = std::fs::read(img_path)?;
    let img_decoded = image::load_from_memory(&img_bytes)?.to_rgba8();
    let (img_width, img_height) = img_decoded.dimensions();
    let img_channel_count = 4;

    let compressor = ikari::texture_compression::TextureCompressor;
    let compressed_img_bytes = compressor.compress_raw_image(TextureCompressionArgs {
        img_bytes: &img_decoded,
        img_width,
        img_height,
        img_channel_count,
        generate_mipmaps: true,
        is_srgb,
        is_normal_map,
        thread_count: threads,
    })?;

    log::debug!(
        "path: {:?} jpg: {:?}, decoded: {:?}, compressed: {:?}",
        img_path,
        img_bytes.len(),
        img_decoded.len(),
        compressed_img_bytes.len()
    );

    // TODO: wrap fs::write in FileLoader?
    std::fs::write(
        texture_path_to_compressed_path(&PATH_MAKER.make(img_path)).resolve(),
        compressed_img_bytes,
    )?;

    Ok(())
}
