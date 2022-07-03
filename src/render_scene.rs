use cgmath::Matrix4;

use super::*;

// TODO: clean up this structure if needed
#[derive(Debug)]
pub struct RenderScene {
    pub buffers: SceneBuffers,
    pub skins: Vec<Skin>,
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
    // TODO: remove?
    pub fn get_drawable_mesh_iterator(&self) -> impl Iterator<Item = &BindableMeshData> {
        self.buffers.bindable_mesh_data.iter()
    }
}
