use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{mpsc::channel, Arc},
};

use threadpool::ThreadPool;
use walkdir::WalkDir;

use ikari::texture_compression::texture_path_to_compressed_path;

const DATA_FOLDER: &str = "./src";
const COMPRESSION_THREAD_COUNT: usize = 2;

fn main() {
    let mut texture_paths = find_gltf_texture_paths().unwrap();

    let gltf_exlusion_map: HashSet<PathBuf> = texture_paths
        .iter()
        .cloned()
        .map(|(path, _)| path.canonicalize().unwrap())
        .collect();

    // interpret all dangling textures as color (normal map = false)
    texture_paths.extend(
        find_dangling_texture_paths(gltf_exlusion_map)
            .unwrap()
            .iter()
            .cloned()
            .map(|path| (path, true)),
    );

    // remove all paths that have already been processed
    texture_paths = texture_paths
        .iter()
        .cloned()
        .filter(|(path, _)| !texture_path_to_compressed_path(path).try_exists().unwrap())
        .collect();

    let worker_count = num_cpus::get() / COMPRESSION_THREAD_COUNT;
    let texture_count = texture_paths.len();

    let pool = ThreadPool::new(worker_count);

    let (tx, rx) = channel();
    for texture_index in 0..texture_count {
        let tx = tx.clone();
        let texture_paths = texture_paths.clone();
        pool.execute(move || {
            let (path, is_normal_map) = &texture_paths[texture_index];
            compress_file(path.as_path(), *is_normal_map).unwrap();
            tx.send(texture_index).unwrap();
        });
    }
    let mut done_count = 0;
    for _ in 0..texture_count {
        let texture_index = rx.recv().unwrap();
        done_count += 1;
        println!(
            "done {:?} ({:?}/{:?})",
            texture_paths[texture_index], done_count, texture_count
        );
    }
}

fn find_gltf_texture_paths() -> anyhow::Result<Vec<(PathBuf, bool)>> {
    let mut result = Vec::new();
    let gltf_paths: Vec<_> = WalkDir::new(DATA_FOLDER)
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
        // let images: Vec<_> = gltf.images().collect();

        for texture in gltf.textures() {
            let is_normal_res = gltf.materials().find(|material| {
                material.normal_texture().is_some()
                    && material.normal_texture().unwrap().texture().index() == texture.index()
            });
            match texture.source().source() {
                gltf::image::Source::View { .. } => {
                    println!("Warning: found inline texture in gltf file {:?}, texture index {:?}. This texture wont be compressed", path, texture.index());
                }
                gltf::image::Source::Uri { uri, .. } => {
                    let path = path.parent().unwrap().join(PathBuf::from(uri));
                    // dbg!(&is_normal_res);
                    result.push((path, is_normal_res.is_some()));
                }
            };
        }
    }
    Ok(result)
}

// TODO: exclude list doesn't work because the relative path is different between the gltf file and the CWD of this binary
fn find_dangling_texture_paths(exclude_list: HashSet<PathBuf>) -> anyhow::Result<Vec<PathBuf>> {
    Ok(WalkDir::new(DATA_FOLDER)
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

fn compress_file(img_path: &Path, is_normal_map: bool) -> anyhow::Result<()> {
    let img_bytes = std::fs::read(img_path)?;
    let img_decoded = image::load_from_memory(&img_bytes)?.to_rgba8();
    let (img_width, img_height) = img_decoded.dimensions();
    let img_channels = 4;

    let compressor = ikari::texture_compression::TextureCompressor::new();
    let compressed_img_bytes = unsafe {
        compressor.compress_raw_image(
            &img_decoded,
            img_width,
            img_height,
            img_channels,
            is_normal_map,
            COMPRESSION_THREAD_COUNT as u32,
        )
    }?;

    // println!(
    //     "path: {:?} jpg: {:?}, decoded: {:?}, compressed: {:?}",
    //     img_path,
    //     img_bytes.len(),
    //     img_decoded.len(),
    //     compressed_img_bytes.len()
    // );

    std::fs::write(
        texture_path_to_compressed_path(img_path),
        compressed_img_bytes,
    )?;

    Ok(())
}
