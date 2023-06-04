/* #[cfg(target_arch = "wasm32")]
use crate::game::*;
#[cfg(target_arch = "wasm32")]
use crate::renderer::*;
#[cfg(target_arch = "wasm32")]
use crate::scene::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*; */

pub mod animation;
pub mod asset_loader;
pub mod audio;
pub mod ball;
pub mod buffer;
pub mod camera;
pub mod character;
pub mod collisions;
pub mod file_loader;
pub mod game;
pub mod game_state;
pub mod gameloop;
pub mod gltf_loader;
pub mod light;
pub mod logger;
pub mod math;
pub mod mesh;
pub mod physics;
pub mod physics_ball;
pub mod player_controller;
pub mod profile_dump;
pub mod renderer;
pub mod revolver;
pub mod sampler_cache;
pub mod scene;
pub mod scene_tree;
pub mod skinning;
pub mod texture;
#[cfg(not(target_arch = "wasm32"))]
pub mod texture_compression;
pub mod time;
pub mod time_tracker;
pub mod transform;
pub mod ui_overlay;
