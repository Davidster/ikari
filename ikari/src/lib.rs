#[cfg(not(target_arch = "wasm32"))]
pub use pollster::block_on;
#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_futures::spawn_local as block_on;

pub mod animation;
pub mod asset_loader;
pub mod audio;
pub mod buffer;
pub mod camera;
pub mod collisions;
pub mod engine_state;
pub mod file_manager;
pub mod gameloop;
pub mod gltf_loader;
pub mod math;
pub mod mesh;
pub mod mutex;
pub mod physics;
pub mod player_controller;
pub mod profile_dump;
pub mod raw_image;
pub mod renderer;
pub mod sampler_cache;
pub mod scene;
pub mod scene_tree;
pub mod skinning;
pub mod texture;
pub mod texture_compression;
pub mod thread;
pub mod time;
pub mod time_tracker;
pub mod transform;
pub mod ui;
pub mod wasm_not_sync;
pub mod web_canvas_manager;
