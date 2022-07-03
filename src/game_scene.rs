use std::collections::HashMap;

use super::*;

#[derive(Debug)]
pub struct GameScene {
    pub nodes: Vec<GameNode>,
    // node index -> parent node index
    pub parent_index_map: HashMap<usize, usize>,
}

#[derive(Debug)]
pub struct GameNode {
    pub transform: crate::transform::Transform,
    pub renderer_scene_skin_index: Option<usize>,
}

impl GameScene {
    pub fn get_model_root_if_in_skeleton(&self, node_index: usize) -> Option<usize> {
        self.get_node_ancestry_list(node_index)
            .iter()
            .find_map(|node_index| {
                self.nodes[*node_index]
                    .renderer_scene_skin_index
                    .map(|_| *node_index)
            })
    }

    // TODO: remove
    pub fn _node_is_part_of_skeleton(&self, node_index: usize) -> bool {
        let ancestry_list = self.get_node_ancestry_list(node_index);
        ancestry_list
            .iter()
            .any(|node_index| self.nodes[*node_index].renderer_scene_skin_index.is_some())
    }

    pub fn get_node_ancestry_list(&self, node_index: usize) -> Vec<usize> {
        get_node_ancestry_list(node_index, &self.parent_index_map)
    }
}

pub fn get_node_ancestry_list(
    node_index: usize,
    parent_index_map: &HashMap<usize, usize>,
) -> Vec<usize> {
    get_node_ancestry_list_impl(node_index, parent_index_map, Vec::new())
}

fn get_node_ancestry_list_impl(
    node_index: usize,
    parent_index_map: &HashMap<usize, usize>,
    acc: Vec<usize>,
) -> Vec<usize> {
    let with_self: Vec<_> = acc.iter().chain(vec![node_index].iter()).copied().collect();
    match parent_index_map.get(&node_index) {
        Some(parent_index) => {
            get_node_ancestry_list_impl(*parent_index, parent_index_map, with_self).to_vec()
        }
        None => with_self,
    }
}
