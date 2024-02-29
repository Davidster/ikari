use rmp_serde::Serializer;
use serde::Serialize;

use crate::{file_manager::GameFilePath, raw_image::RawImage};

#[cfg(not(target_arch = "wasm32"))]
const BASISU_COMPRESSION_FORMAT: basis_universal::BasisTextureFormat =
    basis_universal::BasisTextureFormat::UASTC4x4;

const MINIZ_OXIDE_COMPRESSION_LEVEL: u8 = 6;

#[derive(Debug)]
pub struct TextureCompressor;

// TODO: there's a bunch of wasm32 checks here. use the module style instead like in file_manager.rs

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct TextureCompressionArgs<'a> {
    pub img_bytes: &'a [u8],
    pub img_width: u32,
    pub img_height: u32,
    pub img_channel_count: u8,
    pub generate_mipmaps: bool,
    pub is_normal_map: bool,
    pub is_srgb: bool,
    pub thread_count: u32,
}

/// Warning: this will give the wrong result for Astc-formatted hdr textures
#[cfg(not(target_arch = "wasm32"))]
pub fn basis_u_texture_format_to_wgpu(
    format: basis_universal::transcoding::TranscoderTextureFormat,
    is_srgb: bool,
) -> wgpu::TextureFormat {
    let pre_srgb_format = match format {
        basis_universal::TranscoderTextureFormat::ETC2_RGBA => wgpu::TextureFormat::Etc2Rgba8Unorm,
        basis_universal::TranscoderTextureFormat::BC3_RGBA => wgpu::TextureFormat::Bc3RgbaUnorm,
        basis_universal::TranscoderTextureFormat::BC4_R => wgpu::TextureFormat::Bc4RUnorm,
        basis_universal::TranscoderTextureFormat::BC5_RG => wgpu::TextureFormat::Bc5RgUnorm,
        basis_universal::TranscoderTextureFormat::BC7_RGBA => wgpu::TextureFormat::Bc7RgbaUnorm,

        basis_universal::TranscoderTextureFormat::ASTC_4x4_RGBA => wgpu::TextureFormat::Astc {
            block: wgpu::AstcBlock::B4x4,
            channel: wgpu::AstcChannel::Unorm,
        },
        basis_universal::TranscoderTextureFormat::ETC1_RGB => unimplemented!(),
        basis_universal::TranscoderTextureFormat::BC1_RGB => unimplemented!(),
        basis_universal::TranscoderTextureFormat::PVRTC1_4_RGB => unimplemented!(),
        basis_universal::TranscoderTextureFormat::PVRTC1_4_RGBA => unimplemented!(),
        basis_universal::TranscoderTextureFormat::ATC_RGB => unimplemented!(),
        basis_universal::TranscoderTextureFormat::ATC_RGBA => unimplemented!(),
        basis_universal::TranscoderTextureFormat::FXT1_RGB => unimplemented!(),
        basis_universal::TranscoderTextureFormat::PVRTC2_4_RGB => unimplemented!(),
        basis_universal::TranscoderTextureFormat::PVRTC2_4_RGBA => unimplemented!(),
        basis_universal::TranscoderTextureFormat::ETC2_EAC_R11 => unimplemented!(),
        basis_universal::TranscoderTextureFormat::ETC2_EAC_RG11 => unimplemented!(),
        basis_universal::TranscoderTextureFormat::RGBA32 => unimplemented!(),
        basis_universal::TranscoderTextureFormat::RGB565 => unimplemented!(),
        basis_universal::TranscoderTextureFormat::BGR565 => unimplemented!(),
        basis_universal::TranscoderTextureFormat::RGBA4444 => unimplemented!(),
    };

    let format = if is_srgb {
        pre_srgb_format.add_srgb_suffix()
    } else {
        pre_srgb_format
    };

    if is_srgb && format == pre_srgb_format {
        log::error!("Interpreting texture as srgb with a texture format that doesn't support srgb ({pre_srgb_format:?})");
    }

    format
}

impl TextureCompressor {
    // floating point texture format isn't yet supported by basisu. see
    // https://github.com/BinomialLLC/basis_universal/issues/310
    pub fn compress_raw_float_image(&self, image: RawImage) -> anyhow::Result<Vec<u8>> {
        let mut image_serialized = vec![];
        image.serialize(&mut Serializer::new(&mut image_serialized))?;

        // 0 = default compression level
        let miniz_encoded_data = miniz_oxide::deflate::compress_to_vec(
            image_serialized.as_slice(),
            MINIZ_OXIDE_COMPRESSION_LEVEL,
        );

        Ok(miniz_encoded_data)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn compress_raw_image(&self, args: TextureCompressionArgs) -> anyhow::Result<Vec<u8>> {
        basis_universal::encoder_init();

        let TextureCompressionArgs {
            img_bytes,
            img_width,
            img_height,
            img_channel_count,
            generate_mipmaps,
            is_normal_map,
            is_srgb,
            thread_count,
        } = args;

        let mut basisu_compressor = basis_universal::Compressor::new(thread_count);

        let mut params = basis_universal::CompressorParams::new();
        params.set_basis_format(BASISU_COMPRESSION_FORMAT);
        params.set_uastc_quality_level(3); // level 3 takes longer to compress but is higher quality
        params.set_rdo_uastc(Some(1.0)); // default
        params.set_generate_mipmaps(generate_mipmaps);
        params.set_mipmap_smallest_dimension(1); // default
        params.set_color_space(if is_srgb {
            basis_universal::ColorSpace::Srgb
        } else {
            basis_universal::ColorSpace::Linear
        });
        params.set_print_status_to_stdout(false);

        if is_normal_map {
            params.tune_for_normal_maps();
        }

        let mut source_image = params.source_image_mut(0);
        source_image.init(img_bytes, img_width, img_height, img_channel_count);

        if is_normal_map {
            let pixel_count = source_image.pixel_data_u32_mut().len();
            let image_bytes = source_image.pixel_data_u8_mut();
            for pixel_index in 0..pixel_count {
                image_bytes[pixel_index * 4 + 3] = image_bytes[pixel_index * 4 + 1];
                image_bytes[pixel_index * 4 + 1] = image_bytes[pixel_index * 4];
                image_bytes[pixel_index * 4 + 2] = image_bytes[pixel_index * 4];
            }
        }

        // Safety
        //
        // Compressing with invalid parameters may cause undefined behavior.
        // (The underlying C++ library does not thoroughly validate parameters)
        // see https://docs.rs/basis-universal/0.3.1/basis_universal/encoding/struct.Compressor.html#method.process
        unsafe {
            basisu_compressor.init(&params);

            if let Err(error_code) = basisu_compressor.process() {
                anyhow::bail!("Error compressing img to basisu {:?}", error_code);
            }
        }

        // 0 = default compression level
        let miniz_encoded_data = miniz_oxide::deflate::compress_to_vec(
            basisu_compressor.basis_file(),
            MINIZ_OXIDE_COMPRESSION_LEVEL,
        );

        Ok(miniz_encoded_data)
    }

    // floating point texture format isn't yet supported by basisu. see
    // https://github.com/BinomialLLC/basis_universal/issues/310
    #[profiling::function]
    pub fn transcode_float_image(&self, img_bytes: &[u8]) -> anyhow::Result<RawImage> {
        let miniz_decoded_data = miniz_oxide::inflate::decompress_to_vec(img_bytes)
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        Ok(rmp_serde::from_slice(&miniz_decoded_data)?)
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[profiling::function]
    pub fn transcode_image(
        &self,
        img_bytes: &[u8],
        is_srgb: bool,
        is_normal_map: bool,
    ) -> anyhow::Result<RawImage> {
        basis_universal::transcoder_init();

        let miniz_decoded_data = miniz_oxide::inflate::decompress_to_vec(img_bytes)
            .map_err(|err| anyhow::anyhow!("{err}"))?;

        let mut basisu_transcoder = basis_universal::Transcoder::new();

        if !basisu_transcoder.validate_header(&miniz_decoded_data) {
            anyhow::bail!("Image data failed basisu validation");
        }

        if let Err(prep_err) = basisu_transcoder.prepare_transcoding(&miniz_decoded_data) {
            anyhow::bail!("Error calling prepare_transcoding: {:?}", prep_err);
        }

        let mip_levels = basisu_transcoder.image_level_count(&miniz_decoded_data, 0);

        let basis_universal::ImageLevelDescription {
            original_width: img_width,
            original_height: img_height,
            ..
        } = basisu_transcoder
            .image_level_description(&miniz_decoded_data, 0, 0)
            .unwrap();

        let gpu_texture_format = if is_normal_map {
            basis_universal::transcoding::TranscoderTextureFormat::BC5_RG
        } else {
            basis_universal::transcoding::TranscoderTextureFormat::BC7_RGBA
        };

        // full mip chain uses 33% more memory
        // https://en.wikipedia.org/wiki/1/4_%2B_1/16_%2B_1/64_%2B_1/256_%2B_%E2%8B%AF
        let mut full_mip_chain_bytes =
            Vec::with_capacity((img_width as f32 * img_height as f32 * 1.34).ceil() as usize);
        for mip_level in 0..mip_levels {
            match basisu_transcoder.transcode_image_level(
                &miniz_decoded_data,
                gpu_texture_format,
                basis_universal::transcoding::TranscodeParameters {
                    image_index: 0,
                    level_index: mip_level,
                    decode_flags: Some(basis_universal::transcoding::DecodeFlags::HIGH_QUALITY),
                    output_row_pitch_in_blocks_or_pixels: None,
                    output_rows_in_pixels: None,
                },
            ) {
                Ok(mip_level) => {
                    full_mip_chain_bytes.extend(mip_level);
                }
                Err(transcode_error) => {
                    anyhow::bail!("Error transcoding img from basisu {:?}", transcode_error);
                }
            };
        }

        Ok(RawImage {
            width: img_width,
            height: img_height,
            depth: 1,
            mip_count: mip_levels,
            bytes: full_mip_chain_bytes,
            format: basis_u_texture_format_to_wgpu(gpu_texture_format, is_srgb),
        })
    }
}

pub fn texture_path_to_compressed_path(path: &GameFilePath) -> GameFilePath {
    let mut new_path = path.clone();
    new_path.relative_path.set_file_name(format!(
        "{:}_compressed.bin",
        new_path
            .relative_path
            .file_stem()
            .unwrap()
            .to_string_lossy()
    ));

    new_path
}
