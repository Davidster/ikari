use crate::file_manager::GameFilePath;
use crate::time::Instant;

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use glam::f32::Vec3;
use oddio::{
    FixedGain, FramesSignal, Gain, Handle, Mixer, SpatialBuffered, SpatialOptions, SpatialScene,
    Stop,
};
use symphonia::core::{
    audio::SampleBuffer, codecs::CODEC_TYPE_NULL, io::MediaSource, io::MediaSourceStream,
    probe::Hint,
};

pub struct AudioStreams {
    _spatial_scene_output_stream: cpal::Stream,
    _mixer_output_stream: cpal::Stream,
}

pub struct AudioManager {
    master_volume: f32,
    device_sample_rate: u32,

    spatial_scene_handle: Handle<SpatialScene>,
    mixer_handle: Handle<Mixer<[f32; 2]>>,
    sounds: Vec<Option<Sound>>,
}

const CHANNEL_COUNT: usize = 2;
pub const AUDIO_STREAM_BUFFER_LENGTH_SECONDS: f32 = 2.5;

#[derive(Debug, Clone, Default)]
pub struct SoundData(pub Vec<[f32; CHANNEL_COUNT]>);

pub struct Sound {
    volume: f32,
    is_playing: bool,
    signal_handle: SoundSignalHandle,
    data: SoundData,
    length_seconds: Option<f32>,
    file_path: GameFilePath,
    /// position within the track in milliseconds,
    last_pause_pos_seconds: f32,
    last_resume_time: Option<Instant>,
    buffered_to_pos_seconds: f32,
}

pub enum SoundSignal {
    Mono { signal: FramesSignal<f32> },
    Stereo { signal: FramesSignal<[f32; 2]> },
}

pub enum SoundSignalHandle {
    Spacial {
        signal_handle: Handle<SpatialBuffered<Stop<Gain<FramesSignal<f32>>>>>,
    },
    Ambient {
        signal_handle: Handle<Stop<Gain<FramesSignal<[f32; 2]>>>>,
    },
    AmbientFixed {
        signal_handle: Handle<Stop<FixedGain<FramesSignal<[f32; 2]>>>>,
    },
    Streamed {
        signal_handle: Handle<Stop<Gain<oddio::Stream<[f32; 2]>>>>,
    },
}

#[derive(Debug, Copy, Clone)]
pub enum AudioFileFormat {
    Mp3,
    Wav,
}

#[derive(Debug, Copy, Clone)]
pub struct SpacialParams {
    initial_position: Vec3,
    initial_velocity: Vec3,
}

#[derive(Debug, Clone)]
pub struct SoundParams {
    pub initial_volume: f32,
    pub fixed_volume: bool,
    pub spacial_params: Option<SpacialParams>,
    pub stream: bool,
}

#[cfg(target_arch = "wasm32")]
async fn get_media_source(file_path: &GameFilePath) -> Result<Box<dyn MediaSource>> {
    use std::sync::{Arc, Mutex};

    use js_sys::Reflect;
    use wasm_bindgen::JsValue;
    use wasm_bindgen_futures::JsFuture;

    let url = file_path.resolve();

    let streamer = match HttpStreamer::new(url).await {
        Ok(streamer) => streamer,
        Err(error) => {
            log::error!(
                "{:?}. Falling back to loading the entire resource into memory",
                error
            );
            return Ok(Box::new(std::io::Cursor::new(
                crate::file_manager::FileManager::read(file_path).await?,
            )));
        }
    };

    // pub trait MediaSource: io::Read + io::Seek + Send + Sync {
    //     /// Returns if the source is seekable. This may be an expensive operation.
    //     fn is_seekable(&self) -> bool;

    //     /// Returns the length in bytes, if available. This may be an expensive operation.
    //     fn byte_len(&self) -> Option<u64>;
    // }

    #[derive(Debug, Clone)]
    struct HttpStreamer {
        url: String,
        content_length: Option<u64>,
        current_file_position: usize,
        buffer: Arc<Mutex<Vec<u8>>>,
    }

    impl HttpStreamer {
        pub async fn new(url: String) -> Result<Self> {
            let head_response = gloo_net::http::RequestBuilder::new(&url)
                .method(gloo_net::http::Method::HEAD)
                .send()
                .await?;

            // TODO: do we need to check this pre-emptively? or can we wait for the first 'read'?
            if head_response.headers().get("Accept-Ranges") != Some(String::from("bytes")) {
                anyhow::bail!("Resource at URL {} does not support streaming", url);
            }

            let content_length = head_response
                .headers()
                .get("Content-Length")
                .and_then(|content_length_str| content_length_str.parse::<u64>().ok());

            log::info!(
                "Initted http streamer with content_length={:?}",
                content_length
            );

            let result = Self {
                url,
                content_length,
                current_file_position: 0,
                buffer: Arc::new(Mutex::new(vec![])),
            };

            Ok(result)
        }
    }

    impl std::io::Read for HttpStreamer {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let number_of_bytes_to_read = buf.len();

            log::info!(
                "Read called with number_of_bytes_to_read={:?}",
                number_of_bytes_to_read
            );

            let self_clone = self.clone();

            crate::block_on(async move {
                async fn get_bytes(
                    streamer: HttpStreamer,
                    number_of_bytes_to_read: usize,
                ) -> std::result::Result<(), JsValue> {
                    log::info!("get_bytes");
                    match gloo_net::http::RequestBuilder::new(&streamer.url)
                        .method(gloo_net::http::Method::GET)
                        .header(
                            "Range",
                            &format!(
                                "bytes={}-{}",
                                streamer.current_file_position,
                                streamer.current_file_position + number_of_bytes_to_read
                            ),
                        )
                        .send()
                        .await
                        .map_err(|err| JsValue::from_str(&format!("{:?}", err)))?
                        .body()
                    {
                        Some(body_stream) => {
                            log::info!("get_bytes 2");
                            let mut stream_reader =
                                web_sys::ReadableStreamDefaultReader::new(&body_stream)?;
                            log::info!("get_bytes 3");
                            loop {
                                let result = JsFuture::from(stream_reader.read()).await?;
                                log::info!("get_bytes 4");
                                let done = Reflect::get(&result, &JsValue::from_str("done"))?
                                    .as_bool()
                                    .ok_or_else(|| {
                                        JsValue::from_str(
                                            "Failed to read property 'done' from stream read promise result"
                                        )
                                    })?;
                                log::info!("get_bytes 5");
                                let mut chunk = js_sys::Uint8Array::new(&Reflect::get(
                                    &result,
                                    &JsValue::from_str("value"),
                                )?)
                                .to_vec();
                                log::info!("get_bytes 6");

                                streamer.buffer.lock().unwrap().append(&mut chunk);

                                log::info!(
                                    "get_bytes 7, done={}, chunk_length={}",
                                    done,
                                    chunk.len()
                                );

                                if done {
                                    break;
                                }
                            }

                            Ok(())
                        }
                        None => Ok(()),
                    }
                }

                get_bytes(self_clone, number_of_bytes_to_read).await;
            });

            log::info!("Reading from result buffer");

            let mut count = 0;
            for (i, byte) in self.buffer.lock().unwrap().iter().enumerate() {
                buf[i] = *byte;
                count += 1;
            }

            log::info!("Read {} bytes from result buffer", count);

            Ok(count)
        }
    }

    impl std::io::Seek for HttpStreamer {
        fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
            std::io::Result::Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "sorry bruh",
            ))
        }
    }

    impl MediaSource for HttpStreamer {
        fn is_seekable(&self) -> bool {
            true
        }

        fn byte_len(&self) -> Option<u64> {
            self.content_length
        }
    }

    Ok(Box::new(streamer))
}

#[cfg(target_arch = "wasm32")]
mod web {
    use wasm_bindgen::prelude::*;
    use web_sys::{AudioContext, HtmlMediaElement};

    pub struct AudioFileStreamer {
        file_path: GameFilePath,
    }

    impl AudioFileStreamer {
        pub async fn new(
            _device_sample_rate: u32,
            file_path: GameFilePath,
            _file_format: Option<AudioFileFormat>,
        ) -> anyhow::Result<Self> {
            let Some(document) = web_sys::window().and_then(|win| win.document()) else {
                anyhow::bail!("Couldn't get document");
            };

            Self::new_internal(document, file_path)
                .await
                .map_err(|err| anyhow::anyhow!("{:?}", err))
        }

        async fn new_internal(
            document: web_sys::Document,
            file_path: GameFilePath,
        ) -> Result<Self, JsValue> {
            let audio_context = AudioContext::new()?;

            let audio_element = document
                .create_element("audio")?
                .dyn_into::<HtmlMediaElement>()?;
            audio_element.set_attribute("src", file_path.resolve());

            let track = audio_context.create_media_element_source(&audio_element)?;

            // TODO: can we use the webcodecs API here?
            // https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API
            // bevy uses rodio... does it support streaming on the web?
            // https://github.com/RustAudio/rodio

            Ok(Self { file_path })
        }

        // TODO: does this need to be async?
        #[profiling::function]
        pub async fn read_chunk(&mut self, max_chunk_size: usize) -> Result<(SoundData, bool)> {
            let mut samples_interleaved: Vec<f32> = vec![];

            let sample_rate_ratio = self.device_sample_rate as f32
                / self.track_sample_rate.unwrap_or(self.device_sample_rate) as f32;

            let reached_end_of_stream = loop {
                let packet = {
                    profiling::scope!("read from disk");
                    match self.format_reader.next_packet() {
                        Ok(packet) => packet,
                        Err(symphonia::core::errors::Error::ResetRequired) => {
                            // The track list has been changed. Re-examine it and create a new set of decoders,
                            // then restart the decode loop. This is an advanced feature and it is not
                            // unreasonable to consider this "the end." As of v0.5.0, the only usage of this is
                            // for chained OGG physical streams.
                            anyhow::bail!("idk");
                        }
                        Err(symphonia::core::errors::Error::IoError(err)) => {
                            if err.kind() == std::io::ErrorKind::UnexpectedEof
                                && err.to_string() == "end of stream"
                            {
                                break true;
                            }
                            anyhow::bail!(err);
                        }
                        Err(err) => {
                            anyhow::bail!(err);
                        }
                    }
                };

                if packet.track_id() != self.track_id {
                    continue;
                }

                let decode_result = {
                    profiling::scope!("decode");
                    self.decoder.decode(&packet)
                };

                match decode_result {
                    Ok(audio_buf) => {
                        profiling::scope!("copy to buf");
                        let mut sample_buf = SampleBuffer::<f32>::new(
                            audio_buf.capacity() as u64,
                            *audio_buf.spec(),
                        );

                        sample_buf.copy_interleaved_ref(audio_buf);

                        let sample_count = sample_buf.samples().len();

                        for sample in sample_buf.samples() {
                            samples_interleaved.push(*sample);
                        }

                        if max_chunk_size != 0
                            && sample_rate_ratio
                                * ((samples_interleaved.len() + sample_count) / CHANNEL_COUNT)
                                    as f32
                                > max_chunk_size as f32
                        {
                            break false;
                        }
                    }
                    Err(symphonia::core::errors::Error::IoError(err)) => {
                        if err.kind() == std::io::ErrorKind::UnexpectedEof
                            && err.to_string() == "end of stream"
                        {
                            break true;
                        }
                        anyhow::bail!(err);
                    }
                    Err(err) => {
                        anyhow::bail!(err);
                    }
                }
            };

            let mut samples: Vec<[f32; CHANNEL_COUNT]> = samples_interleaved
                .chunks(CHANNEL_COUNT)
                .map(|chunk| [chunk[0], chunk[1]])
                .collect();

            if Some(self.device_sample_rate) != self.track_sample_rate {
                // resample the sound to the device sample rate using linear interpolation
                samples = resample_linear(
                    &samples,
                    self.track_sample_rate.unwrap(),
                    self.device_sample_rate,
                );
            }

            Ok((SoundData(samples), reached_end_of_stream))
        }

        pub fn track_length_seconds(&self) -> Option<f32> {
            self.track_length_seconds
        }

        pub fn file_path(&self) -> &GameFilePath {
            &self.file_path
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod web {
    pub struct AudioFileStreamer {
        device_sample_rate: u32,
        format_reader: Box<dyn symphonia::core::formats::FormatReader>,
        decoder: Box<dyn symphonia::core::codecs::Decoder>,
        track_id: u32,
        track_sample_rate: Option<u32>,
        track_length_seconds: Option<f32>,
        file_path: GameFilePath,
    }

    #[cfg(not(target_arch = "wasm32"))]
    impl AudioFileStreamer {
        pub async fn new(
            device_sample_rate: u32,
            file_path: GameFilePath,
            file_format: Option<AudioFileFormat>,
        ) -> anyhow::Result<Self> {
            use crate::file_manager::native_fs::File;
            let mss = MediaSourceStream::new(
                Box::new(File::open(file_path.resolve())?),
                Default::default(),
            );

            let mut hint = Hint::new();
            if let Some(file_format) = &file_format {
                hint.with_extension(match file_format {
                    AudioFileFormat::Mp3 => "mp3",
                    AudioFileFormat::Wav => "wav",
                });
            }

            let probed = symphonia::default::get_probe().format(
                &hint,
                mss,
                &Default::default(),
                &Default::default(),
            )?;

            let format_reader = probed.format;

            let track = format_reader
                .tracks()
                .iter()
                .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
                .ok_or(anyhow::anyhow!("no supported audio tracks"))?;

            let track_length_seconds =
                match (track.codec_params.time_base, track.codec_params.n_frames) {
                    (Some(time_base), Some(n_frames)) => {
                        let time = time_base.calc_time(n_frames);
                        Some(time.seconds as f32 + time.frac as f32)
                    }
                    _ => None,
                };

            let track_id = track.id;
            let track_sample_rate = track.codec_params.sample_rate;

            let decoder =
                symphonia::default::get_codecs().make(&track.codec_params, &Default::default())?;

            Ok(Self {
                device_sample_rate,
                format_reader,
                decoder,
                track_id,
                track_length_seconds,
                track_sample_rate,
                file_path,
            })
        }

        /// max_chunk_size=0 to read the whole stream at once
        /// The actual chunk size we end up getting will be a little lower than this value since we read
        /// the audio file in small 'packets' which might not fit evenly into max_chunk_size
        /// so we make sure not to overshoot
        #[profiling::function]
        pub async fn read_chunk(&mut self, max_chunk_size: usize) -> Result<(SoundData, bool)> {
            let mut samples_interleaved: Vec<f32> = vec![];

            let sample_rate_ratio = self.device_sample_rate as f32
                / self.track_sample_rate.unwrap_or(self.device_sample_rate) as f32;

            let reached_end_of_stream = loop {
                let packet = {
                    profiling::scope!("read from disk");
                    match self.format_reader.next_packet() {
                        Ok(packet) => packet,
                        Err(symphonia::core::errors::Error::ResetRequired) => {
                            // The track list has been changed. Re-examine it and create a new set of decoders,
                            // then restart the decode loop. This is an advanced feature and it is not
                            // unreasonable to consider this "the end." As of v0.5.0, the only usage of this is
                            // for chained OGG physical streams.
                            anyhow::bail!("idk");
                        }
                        Err(symphonia::core::errors::Error::IoError(err)) => {
                            if err.kind() == std::io::ErrorKind::UnexpectedEof
                                && err.to_string() == "end of stream"
                            {
                                break true;
                            }
                            anyhow::bail!(err);
                        }
                        Err(err) => {
                            anyhow::bail!(err);
                        }
                    }
                };

                if packet.track_id() != self.track_id {
                    continue;
                }

                let decode_result = {
                    profiling::scope!("decode");
                    self.decoder.decode(&packet)
                };

                match decode_result {
                    Ok(audio_buf) => {
                        profiling::scope!("copy to buf");
                        let mut sample_buf = SampleBuffer::<f32>::new(
                            audio_buf.capacity() as u64,
                            *audio_buf.spec(),
                        );

                        sample_buf.copy_interleaved_ref(audio_buf);

                        let sample_count = sample_buf.samples().len();

                        for sample in sample_buf.samples() {
                            samples_interleaved.push(*sample);
                        }

                        if max_chunk_size != 0
                            && sample_rate_ratio
                                * ((samples_interleaved.len() + sample_count) / CHANNEL_COUNT)
                                    as f32
                                > max_chunk_size as f32
                        {
                            break false;
                        }
                    }
                    Err(symphonia::core::errors::Error::IoError(err)) => {
                        if err.kind() == std::io::ErrorKind::UnexpectedEof
                            && err.to_string() == "end of stream"
                        {
                            break true;
                        }
                        anyhow::bail!(err);
                    }
                    Err(err) => {
                        anyhow::bail!(err);
                    }
                }
            };

            let mut samples: Vec<[f32; CHANNEL_COUNT]> = samples_interleaved
                .chunks(CHANNEL_COUNT)
                .map(|chunk| [chunk[0], chunk[1]])
                .collect();

            if Some(self.device_sample_rate) != self.track_sample_rate {
                // resample the sound to the device sample rate using linear interpolation
                samples = resample_linear(
                    &samples,
                    self.track_sample_rate.unwrap(),
                    self.device_sample_rate,
                );
            }

            Ok((SoundData(samples), reached_end_of_stream))
        }

        pub fn track_length_seconds(&self) -> Option<f32> {
            self.track_length_seconds
        }

        pub fn file_path(&self) -> &GameFilePath {
            &self.file_path
        }
    }
}

impl AudioManager {
    // TODO: should we really be returning a tuple here?
    pub fn new() -> Result<(AudioManager, AudioStreams)> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No output device found"))?;
        let device_sample_rate = device.default_output_config()?.sample_rate().0;

        let (spatial_scene_handle, spatial_scene) = oddio::split(oddio::SpatialScene::new());
        let (mixer_handle, mixer) = oddio::split(oddio::Mixer::new());

        let config = cpal::StreamConfig {
            channels: 2,
            sample_rate: cpal::SampleRate(device_sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let spatial_scene_output_stream = device.build_output_stream(
            &config,
            move |out_flat: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let out_stereo = oddio::frame_stereo(out_flat);
                oddio::run(&spatial_scene, device_sample_rate, out_stereo);
            },
            move |err| {
                log::error!("cpal audio output stream error: {err}");
            },
            None,
        )?;
        let mixer_output_stream = device.build_output_stream(
            &config,
            move |out_flat: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let out_stereo = oddio::frame_stereo(out_flat);
                oddio::run(&mixer, device_sample_rate, out_stereo);
            },
            move |err| {
                log::error!("cpal audio output stream error: {err}");
            },
            None,
        )?;
        spatial_scene_output_stream.play()?;
        mixer_output_stream.play()?;

        Ok((
            AudioManager {
                master_volume: 1.0,
                device_sample_rate,

                spatial_scene_handle,
                mixer_handle,
                sounds: vec![],
            },
            AudioStreams {
                _spatial_scene_output_stream: spatial_scene_output_stream,
                _mixer_output_stream: mixer_output_stream,
            },
        ))
    }

    pub fn get_signal(
        sound_data: &SoundData,
        params: SoundParams,
        device_sample_rate: u32,
    ) -> Option<SoundSignal> {
        let SoundParams {
            spacial_params,
            stream,
            ..
        } = params;

        let SoundData(samples) = sound_data;

        if stream {
            return None;
        }

        let channels = samples[0].len();

        Some(match spacial_params {
            Some(SpacialParams { .. }) => {
                let signal = FramesSignal::from(oddio::Frames::from_iter(
                    device_sample_rate,
                    samples.iter().map(|sample| sample[0]).collect::<Vec<_>>(),
                ));

                SoundSignal::Mono { signal }
            }
            None => {
                let signal = FramesSignal::from(oddio::Frames::from_iter(
                    device_sample_rate,
                    samples.iter().map(|sample| {
                        [sample[0], if channels > 1 { sample[1] } else { sample[0] }]
                    }),
                ));
                SoundSignal::Stereo { signal }
            }
        })
    }

    pub fn add_sound(
        &mut self,
        file_path: GameFilePath,
        sound_data: SoundData,
        length_seconds: Option<f32>,
        params: SoundParams,
        signal: Option<SoundSignal>,
    ) -> usize {
        let sound = Sound::new(self, file_path, sound_data, length_seconds, params, signal);
        self.sounds.push(Some(sound));
        self.sounds.len() - 1
    }

    pub fn write_stream_data(&mut self, sound_index: usize, sound_data: SoundData) {
        if let Some(sound) = self.sounds[sound_index].as_mut() {
            sound.write_stream_data(sound_data, self.device_sample_rate);
        }
    }

    pub fn play_sound(&mut self, sound_index: usize) {
        if let Some(sound) = self.sounds[sound_index].as_mut() {
            sound.resume();
        }
    }

    pub fn reload_sound(&mut self, sound_index: usize, params: SoundParams) {
        if let Some(sound) = self.sounds[sound_index].take() {
            let signal = Self::get_signal(&sound.data, params.clone(), self.device_sample_rate);
            self.sounds[sound_index] = Some(Sound::new(
                self,
                sound.file_path,
                sound.data,
                sound.length_seconds,
                params,
                signal,
            ));
        }
    }

    pub fn sound_is_playing(&self, sound_index: usize) -> bool {
        self.sounds[sound_index]
            .as_ref()
            .map(|sound| sound.is_playing)
            .unwrap_or(false)
    }

    pub fn get_sound_length_seconds(&self, sound_index: usize) -> Option<Option<f32>> {
        self.sounds[sound_index]
            .as_ref()
            .map(|sound| sound.length_seconds)
    }

    pub fn get_sound_pos_seconds(&self, sound_index: usize) -> Option<f32> {
        self.sounds[sound_index]
            .as_ref()
            .map(|sound| sound.pos_seconds())
    }

    pub fn get_sound_buffered_to_pos_seconds(&self, sound_index: usize) -> Option<f32> {
        self.sounds[sound_index]
            .as_ref()
            .map(|sound| sound.buffered_to_pos_seconds)
    }

    pub fn get_sound_file_path(&self, sound_index: usize) -> Option<&GameFilePath> {
        self.sounds[sound_index]
            .as_ref()
            .map(|sound| &sound.file_path)
    }

    pub fn _set_sound_volume(&mut self, sound_index: usize, volume: f32) {
        if let Some(sound) = self.sounds[sound_index].as_mut() {
            sound.set_volume(self.master_volume, volume)
        }
    }

    pub fn device_sample_rate(&self) -> u32 {
        self.device_sample_rate
    }

    pub fn sound_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.sounds
            .iter()
            .enumerate()
            .filter_map(|(sound_index, sound)| sound.as_ref().map(|_| sound_index))
    }
}

#[profiling::function]
fn resample_linear(
    samples: &Vec<[f32; CHANNEL_COUNT]>,
    from_hz: u32,
    to_hz: u32,
) -> Vec<[f32; CHANNEL_COUNT]> {
    let old_sample_count = samples.len();
    let length_seconds = old_sample_count as f32 / from_hz as f32;
    let new_sample_count = (length_seconds * to_hz as f32).ceil() as usize;
    let mut result: Vec<[f32; CHANNEL_COUNT]> = Vec::new();
    result.reserve(new_sample_count);
    for new_sample_number in 1..(new_sample_count + 1) {
        let old_sample_number = new_sample_number as f32 * (from_hz as f32 / to_hz as f32);

        // get the indices of the two samples that surround the old sample number
        let left_index = old_sample_number
            .clamp(1.0, old_sample_count as f32)
            .floor() as usize
            - 1;
        let right_index = (left_index + 1).min(old_sample_count - 1);

        let left_sample = samples[left_index];
        result.push(if left_index == right_index {
            left_sample
        } else {
            let right_sample = samples[right_index];
            let t = old_sample_number - old_sample_number.floor();
            [
                (1.0 - t) * left_sample[0] + t * right_sample[0],
                (1.0 - t) * left_sample[1] + t * right_sample[1],
            ]
        });
    }
    result
}

impl Sound {
    fn new(
        audio_manager: &mut AudioManager,
        file_path: GameFilePath,
        sound_data: SoundData,
        length_seconds: Option<f32>,
        params: SoundParams,
        signal: Option<SoundSignal>,
    ) -> Self {
        let SoundParams {
            initial_volume,
            fixed_volume,
            spacial_params,
            stream,
        } = params;

        let last_pause_pos_seconds = 0.0;
        let last_resume_time = None;
        let buffered_to_pos_seconds =
            sound_data.0.len() as f32 / audio_manager.device_sample_rate as f32;

        let signal_handle = match (spacial_params, signal, stream) {
            (
                Some(SpacialParams {
                    initial_position,
                    initial_velocity,
                }),
                Some(SoundSignal::Mono { signal }),
                false,
            ) => {
                let signal = Gain::new(signal);

                let signal_handle = audio_manager
                    .spatial_scene_handle
                    .control::<SpatialScene, _>()
                    .play_buffered(
                        signal,
                        SpatialOptions {
                            position: [initial_position.x, initial_position.y, initial_position.z]
                                .into(),
                            velocity: [initial_velocity.x, initial_velocity.y, initial_velocity.z]
                                .into(),
                            radius: 0.1,
                        },
                        1000.0,
                        audio_manager.device_sample_rate,
                        0.1,
                    );

                SoundSignalHandle::Spacial { signal_handle }
            }
            (None, Some(SoundSignal::Stereo { signal }), false) => {
                if fixed_volume {
                    let volume_amplitude_ratio =
                        (audio_manager.master_volume * initial_volume).powf(2.0);
                    let volume_db = 20.0 * volume_amplitude_ratio.log10();
                    let signal = FixedGain::new(signal, volume_db);
                    let signal_handle = audio_manager
                        .mixer_handle
                        .control::<Mixer<_>, _>()
                        .play(signal);
                    SoundSignalHandle::AmbientFixed { signal_handle }
                } else {
                    let signal = Gain::new(signal);
                    let signal_handle = audio_manager
                        .mixer_handle
                        .control::<Mixer<_>, _>()
                        .play(signal);
                    SoundSignalHandle::Ambient { signal_handle }
                }
            }
            (None, None, true) => {
                let signal = Gain::new(oddio::Stream::new(
                    audio_manager.device_sample_rate,
                    (audio_manager.device_sample_rate as f32 * AUDIO_STREAM_BUFFER_LENGTH_SECONDS)
                        as usize,
                ));
                let signal_handle = audio_manager
                    .mixer_handle
                    .control::<Mixer<_>, _>()
                    .play(signal);
                SoundSignalHandle::Streamed { signal_handle }
            }
            _ => {
                panic!("Signal didn't match spatial params");
            }
        };

        let mut sound = Sound {
            is_playing: false,
            volume: initial_volume,
            signal_handle,
            file_path,
            data: sound_data,
            length_seconds,
            last_pause_pos_seconds,
            last_resume_time,
            buffered_to_pos_seconds,
        };

        sound.set_volume(audio_manager.master_volume, initial_volume);
        sound.pause();
        sound
    }

    fn write_stream_data(&mut self, sound_data: SoundData, device_sample_rate: u32) {
        if let SoundSignalHandle::Streamed { signal_handle } = &mut self.signal_handle {
            self.buffered_to_pos_seconds += sound_data.0.len() as f32 / device_sample_rate as f32;
            let SoundData(samples) = sound_data;
            signal_handle
                .control::<oddio::Stream<_>, _>()
                .write(&samples);
        }
    }

    fn pause(&mut self) {
        self.is_playing = false;
        match &mut self.signal_handle {
            SoundSignalHandle::Spacial { signal_handle } => {
                signal_handle.control::<Stop<_>, _>().pause();
            }
            SoundSignalHandle::Ambient { signal_handle } => {
                signal_handle.control::<Stop<_>, _>().pause();
            }
            SoundSignalHandle::AmbientFixed { signal_handle } => {
                signal_handle.control::<Stop<_>, _>().pause();
            }
            SoundSignalHandle::Streamed { signal_handle } => {
                signal_handle.control::<Stop<_>, _>().pause();
            }
        }
        self.last_pause_pos_seconds = self.pos_seconds();
    }

    fn resume(&mut self) {
        self.is_playing = true;
        match &mut self.signal_handle {
            SoundSignalHandle::Spacial { signal_handle } => {
                signal_handle.control::<Stop<_>, _>().resume();
            }
            SoundSignalHandle::Ambient { signal_handle } => {
                signal_handle.control::<Stop<_>, _>().resume();
            }
            SoundSignalHandle::AmbientFixed { signal_handle } => {
                signal_handle.control::<Stop<_>, _>().resume();
            }
            SoundSignalHandle::Streamed { signal_handle } => {
                signal_handle.control::<Stop<_>, _>().resume();
            }
        }
        self.last_resume_time = Some(Instant::now());
    }

    fn set_volume(&mut self, master_volume: f32, volume: f32) {
        self.volume = volume;
        match &mut self.signal_handle {
            SoundSignalHandle::Spacial { signal_handle } => {
                signal_handle
                    .control::<Gain<_>, _>()
                    .set_amplitude_ratio((master_volume * self.volume).powf(2.0));
            }
            SoundSignalHandle::Ambient { signal_handle } => {
                signal_handle
                    .control::<Gain<_>, _>()
                    .set_amplitude_ratio((master_volume * self.volume).powf(2.0));
            }
            SoundSignalHandle::AmbientFixed { .. } => {}
            SoundSignalHandle::Streamed { signal_handle } => {
                signal_handle
                    .control::<Gain<_>, _>()
                    .set_amplitude_ratio((master_volume * self.volume).powf(2.0));
            }
        }
    }

    fn pos_seconds(&self) -> f32 {
        let pos = self.last_pause_pos_seconds
            + self
                .last_resume_time
                .map(|last_resume_time| last_resume_time.elapsed().as_secs_f32())
                .unwrap_or(0.0);
        self.length_seconds
            .map(|length_seconds| length_seconds.min(pos))
            .unwrap_or(pos)
    }

    fn _set_motion(&mut self, position: Vec3, velocity: Vec3, discontinuity: bool) {
        if let SoundSignalHandle::Spacial { signal_handle } = &mut self.signal_handle {
            signal_handle.control::<SpatialBuffered<_>, _>().set_motion(
                [position.x, position.y, position.z].into(),
                [velocity.x, velocity.y, velocity.z].into(),
                discontinuity,
            );
        }
    }
}
