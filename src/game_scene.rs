use super::*;

#[derive(Debug)]
pub struct GameScene {
    pub nodes: Vec<GameNode>,
}

#[derive(Debug)]
pub struct GameNode {
    pub transform: crate::transform::Transform,
    pub render_node_index: usize,
}
