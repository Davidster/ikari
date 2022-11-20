use super::*;

pub struct TextureManager {
    textures: Vec<Texture>,
}

pub enum DefaultTexture {
    BaseColor,
    Normal,
    MetallicRoughness,
    MetallicRoughnessGLTF,
    Emissive,
    EmissiveGLTF,
    AmbientOcclusion,
}

impl TextureManager {
    pub fn new() -> Self {
        Self {
            textures: Vec::new(),
        }
    }

    pub fn add_texture(&mut self, texture: Texture) -> usize {
        self.textures.push(texture);
        self.textures.len() - 1
    }

    pub fn get_default_texture(default_texture: DefaultTexture) {}
}
