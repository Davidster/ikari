use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ikari::asset_loader::{AssetBinder, AssetId, AssetLoader};
use ikari::scene::GameNodeId;
use ikari::ui::IkariUiContainer;
use ikari::wasm_not_sync::WasmNotArc;
use ikari::{mesh::BasicMesh, physics::rapier3d_f64::prelude::*};

use crate::ui_overlay::UiOverlay;
use crate::{
    ball::BallComponent, character::Character, physics_ball::PhysicsBall, revolver::Revolver,
};

pub struct GameState {
    pub state_update_time_accumulator: f64,
    pub is_playing_animations: bool,

    pub bgm_sound_index: Option<usize>,
    pub gunshot_sound_index: Option<usize>,
    // pub gunshot_sound_data: SoundData,
    pub point_light_node_ids: Vec<GameNodeId>,

    // store the previous state and next state and interpolate between them
    pub next_balls: Vec<BallComponent>,
    pub prev_balls: Vec<BallComponent>,
    pub actual_balls: Vec<BallComponent>,
    pub ball_node_ids: Vec<GameNodeId>,
    pub ball_pbr_mesh_index: usize,

    pub ball_spawner_acc: f64,

    pub test_object_node_id: GameNodeId,
    pub crosshair_node_id: Option<GameNodeId>,
    pub revolver: Option<Revolver>,

    pub bouncing_ball_node_id: GameNodeId,
    pub bouncing_ball_body_handle: RigidBodyHandle,

    pub physics_balls: Vec<PhysicsBall>,

    pub character: Option<Character>,

    pub cube_mesh: BasicMesh,

    pub asset_loader: Arc<AssetLoader>,
    pub asset_binder: WasmNotArc<AssetBinder>,

    pub ui_overlay: IkariUiContainer<UiOverlay>,

    pub asset_id_map: Arc<Mutex<HashMap<String, AssetId>>>,
}

impl ikari::gameloop::GameState<UiOverlay> for GameState {
    fn get_ui_container(&mut self) -> &mut IkariUiContainer<UiOverlay> {
        &mut self.ui_overlay
    }
}
