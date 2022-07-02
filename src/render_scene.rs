use std::collections::HashMap;

use cgmath::Matrix4;

use super::*;

// TODO: clean up this structure if needed
#[derive(Debug)]
pub struct RenderScene {
    pub buffers: SceneBuffers,
    // same order as the nodes list
    pub nodes: Vec<Node>,
    pub skins: Vec<Skin>,
    // node index -> parent node index
    pub parent_index_map: HashMap<usize, usize>,
    // same order as the animations list in the source asset
    pub animations: Vec<Animation>,
}

#[derive(Debug)]
pub struct SceneBuffers {
    // same order as the drawable_primitive_groups vec
    pub bindable_mesh_data: Vec<BindableMeshData>,
    // same order as the textures in src
    pub textures: Vec<Texture>,
}

#[derive(Debug)]
pub struct BindableMeshData {
    pub vertex_buffer: BufferAndLength,

    pub index_buffer: Option<BufferAndLength>,

    pub instance_buffer: BufferAndLength,
    pub instances: Vec<SceneMeshInstance>,

    pub textures_bind_group: wgpu::BindGroup,

    pub alpha_mode: AlphaMode,
    pub primitive_mode: PrimitiveMode,
}

#[derive(Debug)]
pub struct BufferAndLength {
    pub buffer: wgpu::Buffer,
    pub length: usize,
}

#[derive(Debug, Clone)]
pub struct SceneMeshInstance {
    pub node_index: usize,
    pub base_material: BaseMaterial,
}

#[derive(Debug, Clone)]
pub struct Node {
    // TODO: put this transform into the game scene and stop storing it here?
    pub transform: crate::transform::Transform,
    pub skin_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct Skin {
    pub bone_inverse_bind_matrices: Vec<Matrix4<f32>>,
    pub bone_node_indices: Vec<usize>,
}

#[derive(Debug)]
pub enum AlphaMode {
    Opaque,
    Mask,
}

#[derive(Debug)]
pub enum PrimitiveMode {
    Triangles,
}

#[derive(Debug)]
pub struct Animation {
    pub length_seconds: f32,
    pub channels: Vec<Channel>,
}

#[derive(Debug)]
pub struct Channel {
    pub node_index: usize,
    pub property: gltf::animation::Property,
    pub interpolation_type: gltf::animation::Interpolation,
    pub keyframe_timings: Vec<f32>,
    pub keyframe_values_u8: Vec<u8>,
}

impl RenderScene {
    pub fn get_drawable_mesh_iterator(&self) -> impl Iterator<Item = &BindableMeshData> {
        self.buffers.bindable_mesh_data.iter()
    }

    pub fn get_model_root_if_in_skeleton(&self, node_index: usize) -> Option<usize> {
        self.get_node_ancestry_list(node_index)
            .iter()
            .find_map(|node_index| self.nodes[*node_index].skin_index.map(|_| *node_index))
    }

    // TODO: remove
    pub fn _node_is_part_of_skeleton(&self, node_index: usize) -> bool {
        let ancestry_list = self.get_node_ancestry_list(node_index);
        ancestry_list
            .iter()
            .any(|node_index| self.nodes[*node_index].skin_index.is_some())
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
