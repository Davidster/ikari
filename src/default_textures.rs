use super::*;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::rc::Rc;

pub struct DefaultTextures {
    textures: HashMap<DefaultTextureType, Rc<Texture>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefaultTextureType {
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

    pub fn get_default_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        default_texture_type: DefaultTextureType,
    ) -> anyhow::Result<Rc<Texture>> {
        match self.textures.entry(default_texture_type) {
            Entry::Occupied(texture) => Ok(texture.get().clone()),
            Entry::Vacant(entry) => {
                let color: [u8; 4] = match default_texture_type {
                    DefaultTextureType::BaseColor => [255, 255, 255, 255],
                    DefaultTextureType::Normal => [127, 127, 255, 255],
                    DefaultTextureType::MetallicRoughness => [255, 255, 255, 255],
                    DefaultTextureType::MetallicRoughnessGLTF => [255, 127, 0, 255],
                    DefaultTextureType::Emissive => [0, 0, 0, 255],
                    DefaultTextureType::EmissiveGLTF => [255, 255, 255, 255],
                    DefaultTextureType::AmbientOcclusion => [255, 255, 255, 255],
                };
                let texture = Rc::new(Texture::from_color(device, queue, color)?);
                Ok(entry.insert(texture).clone())
            }
        }
    }
}
