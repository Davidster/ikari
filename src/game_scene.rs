use std::collections::HashMap;

use super::*;

#[derive(Debug)]
pub struct GameScene {
    pub nodes: Vec<GameNode>,
    // node index -> parent node index
    pub parent_index_map: HashMap<usize, usize>,
}

#[derive(Debug, Clone)]
pub struct GameNode {
    pub transform: crate::transform::Transform,
    pub renderer_skin_index: Option<usize>,
    // many meshes can share the same transform
    pub binded_mesh_indices: Option<Vec<usize>>,
    pub dynamic_material_params: Option<DynamicMaterialParams>,
}

impl GameScene {
    pub fn get_model_root_if_in_skeleton(&self, node_index: usize) -> Option<usize> {
        self.get_node_ancestry_list(node_index)
            .iter()
            .find_map(|node_index| {
                self.nodes[*node_index]
                    .renderer_skin_index
                    .map(|_| *node_index)
            })
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

impl Default for GameNode {
    fn default() -> Self {
        Self {
            transform: crate::transform::Transform::new(),
            renderer_skin_index: None,
            binded_mesh_indices: None,
            dynamic_material_params: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GameNodeBuilder {
    transform: crate::transform::Transform,
    renderer_skin_index: Option<usize>,
    binded_mesh_indices: Option<Vec<usize>>,
    dynamic_material_params: Option<DynamicMaterialParams>,
}

impl GameNodeBuilder {
    pub fn new() -> Self {
        let GameNode {
            transform,
            renderer_skin_index,
            binded_mesh_indices,
            dynamic_material_params,
        } = GameNode::default();
        Self {
            transform,
            renderer_skin_index,
            binded_mesh_indices,
            dynamic_material_params,
        }
    }

    pub fn transform(mut self, transform: crate::transform::Transform) -> Self {
        self.transform = transform;
        self
    }

    #[allow(dead_code)]
    pub fn renderer_skin_index(mut self, renderer_skin_index: Option<usize>) -> Self {
        self.renderer_skin_index = renderer_skin_index;
        self
    }

    #[allow(dead_code)]
    pub fn binded_mesh_indices(mut self, binded_mesh_indices: Option<Vec<usize>>) -> Self {
        self.binded_mesh_indices = binded_mesh_indices;
        self
    }

    #[allow(dead_code)]
    pub fn dynamic_material_params(
        mut self,
        dynamic_material_params: Option<DynamicMaterialParams>,
    ) -> Self {
        self.dynamic_material_params = dynamic_material_params;
        self
    }

    pub fn build(self) -> GameNode {
        GameNode {
            transform: self.transform,
            renderer_skin_index: self.renderer_skin_index,
            binded_mesh_indices: self.binded_mesh_indices,
            dynamic_material_params: self.dynamic_material_params,
        }
    }
}
