use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RawImage {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_count: u32,
    pub format: wgpu::TextureFormat,
    pub bytes: Vec<u8>,
}

impl RawImage {
    pub fn from_dynamic_image(image: image::DynamicImage, is_srgb: bool) -> Self {
        let pre_srgb_format = match image {
            image::DynamicImage::ImageLuma8(_) => wgpu::TextureFormat::R8Unorm,
            image::DynamicImage::ImageLumaA8(_) => wgpu::TextureFormat::Rg8Unorm,
            image::DynamicImage::ImageRgb8(_) => wgpu::TextureFormat::Rgba8Unorm,
            image::DynamicImage::ImageRgba8(_) => wgpu::TextureFormat::Rgba8Unorm,
            image::DynamicImage::ImageLuma16(_) => wgpu::TextureFormat::R16Unorm,
            image::DynamicImage::ImageLumaA16(_) => wgpu::TextureFormat::Rg16Unorm,
            image::DynamicImage::ImageRgb16(_) => wgpu::TextureFormat::Rgba16Unorm,
            image::DynamicImage::ImageRgba16(_) => wgpu::TextureFormat::Rgba16Unorm,
            image::DynamicImage::ImageRgb32F(_) => wgpu::TextureFormat::Rgba32Float,
            image::DynamicImage::ImageRgba32F(_) => wgpu::TextureFormat::Rgba32Float,
            _ => wgpu::TextureFormat::Rgba8Unorm,
        };
        let format = if is_srgb {
            pre_srgb_format.add_srgb_suffix()
        } else {
            pre_srgb_format
        };

        if is_srgb && format == pre_srgb_format {
            log::error!("Interpreting texture as srgb with a texture format that doesn't support srgb ({pre_srgb_format:?})");
        }

        let bytes = if matches!(
            image,
            image::DynamicImage::ImageRgb8(_)
                | image::DynamicImage::ImageRgb16(_)
                | image::DynamicImage::ImageRgb32F(_)
        ) {
            image.to_rgba8().into_raw()
        } else {
            image.as_bytes().to_vec()
        };

        Self {
            width: image.width(),
            height: image.height(),
            depth: 1,
            mip_count: 1,
            format,
            bytes,
        }
    }
}

impl FromIterator<RawImage> for RawImage {
    fn from_iter<T: IntoIterator<Item = RawImage>>(raw_images: T) -> Self {
        let mut raw_images = raw_images.into_iter();

        let first_img = raw_images.next();

        if first_img.is_none() {
            return Self {
                width: 0,
                height: 0,
                depth: 0,
                mip_count: 1,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                bytes: vec![],
            };
        }

        let mut joiner = RawImageDepthJoiner::new(first_img.unwrap());

        for image in raw_images {
            joiner.append_image(image);
        }

        joiner.complete()
    }
}

/// useful for joining several images into a texture array / cubemap
pub struct RawImageDepthJoiner {
    width: u32,
    height: u32,
    depth: u32,
    mip_count: u32,
    format: wgpu::TextureFormat,
    bytes: Vec<u8>,
}

impl RawImageDepthJoiner {
    pub fn new(raw_image: RawImage) -> Self {
        assert_eq!(
            raw_image.depth, 1,
            "Starting with an image of depth != 1 is not supported"
        );

        Self {
            width: raw_image.width,
            height: raw_image.height,
            depth: 1,
            mip_count: raw_image.mip_count,
            format: raw_image.format,
            bytes: raw_image.bytes,
        }
    }

    pub fn with_capacity(raw_image: RawImage, capacity: usize) -> Self {
        let mut joiner = Self::new(raw_image);

        if capacity > 1 {
            joiner
                .bytes
                .reserve_exact((capacity - 1) * joiner.bytes.len());
        }

        joiner
    }

    pub fn append_image(&mut self, raw_image: RawImage) {
        assert_eq!(
            self.width, raw_image.width,
            "Images should have the same width if they are to be joined with depth"
        );
        assert_eq!(
            self.height, raw_image.height,
            "Images should have the same width if they are to be joined with depth"
        );
        assert_eq!(
            self.format, raw_image.format,
            "Images should have the same format if they are to be joined with depth"
        );
        assert_eq!(
            self.mip_count, raw_image.mip_count,
            "Images should have the same mip count if they are to be joined with depth"
        );

        self.depth += 1;
        self.bytes.extend_from_slice(&raw_image.bytes);
    }

    pub fn complete(self) -> RawImage {
        RawImage {
            width: self.width,
            height: self.height,
            depth: self.depth,
            mip_count: self.mip_count,
            format: self.format,
            bytes: self.bytes,
        }
    }
}
