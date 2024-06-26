use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{mpsc::channel, Arc},
};

use anyhow::bail;
use threadpool::ThreadPool;
use walkdir::WalkDir;

use ikari::{
    block_on,
    file_manager::{native_fs, FileManager},
    texture_compression::{texture_path_to_compressed_path, TextureCompressionArgs},
};

use crate::PATH_MAKER;

pub struct TextureCompressorArgs {
    pub search_folder: PathBuf,
    pub max_thread_count: Option<usize>,
    pub force: bool,
}

pub async fn run(args: TextureCompressorArgs) {
    if let Err(err) = run_internal(args).await {
        log::error!("Error: {err}\n{}", err.backtrace());
    }
}

async fn run_internal(args: TextureCompressorArgs) -> anyhow::Result<()> {
    if !args.search_folder.exists() {
        bail!(
            "Error: search folder {:?} does not exist",
            args.search_folder
        );
    }

    let texture_paths = {
        let mut texture_paths = find_gltf_texture_paths(&args.search_folder)?;

        let mut gltf_exlusion_map: HashSet<PathBuf> = HashSet::new();
        for (path, _, _) in &texture_paths {
            gltf_exlusion_map.insert(path.canonicalize()?);
        }

        // interpret all dangling textures as srgb color maps
        texture_paths.extend(
            find_dangling_texture_paths(&args.search_folder, gltf_exlusion_map)?
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

    if texture_count == 0 {
        log::warn!("No work to do. Aborting");
        return Ok(());
    }

    let max_thread_count = args.max_thread_count.unwrap_or(num_cpus::get());
    dbg!(max_thread_count);
    let threads_per_texture = (max_thread_count / texture_count).max(1);
    dbg!(threads_per_texture);
    let worker_count = max_thread_count / threads_per_texture;
    dbg!(worker_count);
    log::info!("max_thread_count={max_thread_count}, threads_per_texture={threads_per_texture}, worker_count={worker_count}");

    let pool = ThreadPool::new(worker_count);

    let (tx, rx) = channel();
    for texture_index in 0..texture_count {
        let tx = tx.clone();
        let texture_paths = texture_paths.clone();
        pool.execute(move || {
            let (path, is_srgb, is_normal_map) = &texture_paths[texture_index];
            log::info!("start {path:?} (srgb={is_srgb:?}, is_normal_map={is_normal_map:?})");
            block_on(async {
                if let Err(error) = compress_file(
                    path.as_path(),
                    *is_srgb,
                    *is_normal_map,
                    threads_per_texture as u32,
                )
                .await
                {
                    log::error!("Failed to compress texture: {path:?}\n{error:?}");
                }
            });

            tx.send(texture_index)
                .expect("Failed to send texture out of thread pool worker");
        });
    }
    let mut done_count = 0;
    for _ in 0..texture_count {
        let texture_index = rx.recv()?;
        done_count += 1;
        log::info!(
            "done {:?} ({:?}/{:?})",
            texture_paths[texture_index].0,
            done_count,
            texture_count
        );
    }

    Ok(())
}

fn find_gltf_texture_paths(search_folder: &Path) -> anyhow::Result<Vec<(PathBuf, bool, bool)>> {
    let mut result = Vec::new();
    let gltf_paths: Vec<_> = WalkDir::new(search_folder)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| !e.file_type().is_dir())
        .filter(|e| match e.path().extension() {
            Some(ext) => ext == "gltf" || ext == "glb",
            None => false,
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
                    let Some(normal_texture) = material.normal_texture() else {
                        return false;
                    };
                    normal_texture.texture().index() == texture.index()
                });

            match texture.source().source() {
                gltf::image::Source::View { .. } => {
                    log::warn!("Found inline texture in gltf file {:?}, texture index {:?}. This texture won't be compressed", path, texture.index());
                }
                gltf::image::Source::Uri { uri, .. } => {
                    let Some(parent) = path.parent() else {
                        log::warn!("Found uri texture but gltf file path doesn't have a parent. uri={uri}, path={path:?}");
                        continue;
                    };
                    let path = parent.join(PathBuf::from(uri));
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
        .filter(|e| match e.path().canonicalize() {
            Ok(absolute_path) => !exclude_list.contains(&absolute_path),
            Err(_) => false,
        })
        .filter(|e| match e.path().extension() {
            Some(ext) => ext == "jpg" || ext == "png",
            None => false,
        })
        .map(|e| e.path().to_path_buf())
        .collect())
}

async fn compress_file(
    img_path: &Path,
    is_srgb: bool,
    is_normal_map: bool,
    threads: u32,
) -> anyhow::Result<()> {
    let img_bytes = FileManager::read(&PATH_MAKER.make(img_path)).await?;
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

    native_fs::write(
        texture_path_to_compressed_path(&PATH_MAKER.make(img_path)).resolve(),
        &compressed_img_bytes,
    )?;

    Ok(())
}
