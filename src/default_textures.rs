use super::*;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

pub struct DefaultTextures {
    textures: HashMap<DefaultTexture, Texture>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefaultTexture {
    BaseColor,
    Normal,
    MetallicRoughness,
    MetallicRoughnessGLTF,
    Emissive,
    EmissiveGLTF,
    AmbientOcclusion,
}

impl DefaultTextures {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
        }
    }

    pub fn get_default_texture<'a>(&self, default_texture: DefaultTexture) -> &'a Texture {
        match self.textures.entry(default_texture) {
            Entry::Occupied(texture) => texture,
            Entry::Vacant(entry) => panic!(),
        }
    }
}
