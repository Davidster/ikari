use wgpu::{Device, Sampler};

// same as wgpu::SamplerDescriptor but without the label
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SamplerDescriptor {
    pub address_mode_u: wgpu::AddressMode,
    pub address_mode_v: wgpu::AddressMode,
    pub address_mode_w: wgpu::AddressMode,
    pub mag_filter: wgpu::FilterMode,
    pub min_filter: wgpu::FilterMode,
    pub mipmap_filter: wgpu::FilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub compare: Option<wgpu::CompareFunction>,
    pub anisotropy_clamp: u16,
    pub border_color: Option<wgpu::SamplerBorderColor>,
}

impl Default for SamplerDescriptor {
    fn default() -> Self {
        let def = wgpu::SamplerDescriptor::default();
        Self {
            address_mode_u: def.address_mode_u,
            address_mode_v: def.address_mode_v,
            address_mode_w: def.address_mode_w,
            mag_filter: def.mag_filter,
            min_filter: def.min_filter,
            mipmap_filter: def.mipmap_filter,
            lod_min_clamp: def.lod_min_clamp,
            lod_max_clamp: def.lod_max_clamp,
            compare: def.compare,
            anisotropy_clamp: def.anisotropy_clamp,
            border_color: def.border_color,
        }
    }
}

impl SamplerDescriptor {
    pub fn into_wgpu(self) -> wgpu::SamplerDescriptor<'static> {
        wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: self.address_mode_u,
            address_mode_v: self.address_mode_v,
            address_mode_w: self.address_mode_w,
            mag_filter: self.mag_filter,
            min_filter: self.min_filter,
            mipmap_filter: self.mipmap_filter,
            lod_min_clamp: self.lod_min_clamp,
            lod_max_clamp: self.lod_max_clamp,
            compare: self.compare,
            anisotropy_clamp: self.anisotropy_clamp,
            border_color: self.border_color,
        }
    }
}

#[derive(Default, Debug)]
pub struct SamplerCache {
    samplers: Vec<(SamplerDescriptor, Sampler)>,
}

impl SamplerCache {
    pub fn get_sampler_index(
        &mut self,
        device: &Device,
        a_descriptor: &SamplerDescriptor,
    ) -> usize {
        if let Some(existing_sampler_index) = self
            .samplers
            .iter()
            .enumerate()
            .find_map(|(i, (descriptor, _))| descriptor.eq(a_descriptor).then_some(i))
        {
            return existing_sampler_index;
        }
        let new_sampler_index = self.samplers.len();
        let new_sampler = device.create_sampler(&a_descriptor.into_wgpu());
        self.samplers.push((*a_descriptor, new_sampler));
        new_sampler_index
    }

    pub fn get_sampler_by_index(&self, sampler_index: usize) -> &Sampler {
        let (_, sampler) = &self.samplers[sampler_index];
        sampler
    }

    pub fn get_sampler(&mut self, device: &Device, a_descriptor: &SamplerDescriptor) -> &Sampler {
        let sampler_index = self.get_sampler_index(device, a_descriptor);
        self.get_sampler_by_index(sampler_index)
    }
}
