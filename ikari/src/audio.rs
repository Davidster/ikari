use glam::Vec3;

use crate::file_manager::GameFilePath;

pub const AUDIO_STREAM_BUFFER_LENGTH_SECONDS: f32 = 2.5;

const CHANNEL_COUNT: usize = 2;

pub trait AudioManager: Send + Sync {
    fn add_sound(
        &mut self,
        file_path: GameFilePath,
        sound_data: SoundData,
        length_seconds: Option<f32>,
        params: SoundParams,
    ) -> usize;

    fn write_stream_data(&mut self, sound_index: usize, sound_data: SoundData);

    fn play_sound(&mut self, sound_index: usize);

    fn reload_sound(&mut self, sound_index: usize, params: SoundParams);

    fn sound_is_playing(&self, sound_index: usize) -> bool;

    fn get_sound_length_seconds(&self, sound_index: usize) -> Option<Option<f32>>;

    fn get_sound_pos_seconds(&self, sound_index: usize) -> Option<f32>;

    fn get_sound_buffered_to_pos_seconds(&self, sound_index: usize) -> Option<f32>;

    fn get_sound_file_path(&self, sound_index: usize) -> Option<&GameFilePath>;

    fn set_sound_volume(&mut self, sound_index: usize, volume: f32);

    fn device_sample_rate(&self) -> u32;

    fn sound_indices(&self) -> Vec<usize>;

    fn is_audio_enabled(&self) -> bool;
}

pub trait StreamAudio: Send + Sync {
    fn read_chunk(&mut self, _max_chunk_size: usize) -> anyhow::Result<(SoundData, bool)>;

    fn track_length_seconds(&self) -> Option<f32>;

    fn file_path(&self) -> &GameFilePath;
}

pub trait AudioOutputStreams {}

#[derive(Debug, Clone, Default)]
pub struct SoundData(Vec<[f32; CHANNEL_COUNT]>);

impl SoundData {
    pub fn sample_count(&self) -> usize {
        self.0.len()
    }
}

#[allow(dead_code)]
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

pub fn create_audio_manager(
    enable_audio: bool,
) -> (Box<dyn AudioManager>, Box<dyn AudioOutputStreams>) {
    #[cfg(feature = "audio")]
    if enable_audio {
        match ikari_audio::create_audio_manager() {
            Ok((audio_manager, audio_streams)) => {
                return (Box::new(audio_manager), Box::new(audio_streams));
            }
            Err(err) => log::error!(
                "Error initializing audio, falling back to null (silent) audio manager:\n{err:?}"
            ),
        }
    }

    #[cfg(not(feature = "audio"))]
    if enable_audio {
        log::warn!("Tried to enable audio but ikari was compiled without support for it. Did you forget to enable the audio feature in cargo?");
    }

    let (audio_manager, audio_streams) = create_null_audio_manager();
    (Box::new(audio_manager), Box::new(audio_streams))
}

#[allow(unused_variables)]
pub async fn create_audio_stream_from_file(
    is_audio_enabled: bool,
    device_sample_rate: u32,
    file_path: GameFilePath,
) -> anyhow::Result<Box<dyn StreamAudio>> {
    #[cfg(feature = "audio")]
    if is_audio_enabled {
        let audio_stream =
            ikari_audio::IkariStreamedAudioFile::new(device_sample_rate, file_path).await?;
        return Ok(Box::new(audio_stream));
    }

    Ok(Box::new(NullStreamedAudioFile { file_path }))
}

#[cfg(feature = "audio")]
mod ikari_audio {
    #[cfg(not(target_arch = "wasm32"))]
    use native::get_media_source;

    #[cfg(target_arch = "wasm32")]
    use web::get_media_source;

    use super::{
        AudioManager, AudioOutputStreams, SoundData, SoundParams, SpacialParams, StreamAudio,
        AUDIO_STREAM_BUFFER_LENGTH_SECONDS,
    };

    use crate::time::Instant;
    use crate::{audio::CHANNEL_COUNT, file_manager::GameFilePath};

    use anyhow::Result;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use glam::f32::Vec3;
    use oddio::{
        FixedGain, FramesSignal, Gain, Handle, Mixer, SpatialBuffered, SpatialOptions,
        SpatialScene, Stop,
    };
    use symphonia::core::{
        audio::SampleBuffer, codecs::CODEC_TYPE_NULL, io::MediaSourceStream, probe::Hint,
    };

    pub struct IkariAudioOutputStreams {
        _spatial_scene_output_stream: cpal::Stream,
        _mixer_output_stream: cpal::Stream,
    }

    impl AudioOutputStreams for IkariAudioOutputStreams {}

    pub struct IkariAudioManager {
        master_volume: f32,
        device_sample_rate: u32,

        spatial_scene_handle: Handle<SpatialScene>,
        mixer_handle: Handle<Mixer<[f32; 2]>>,
        sounds: Vec<Option<Sound>>,
    }

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

    pub struct SoundSignal {
        inner: SoundSignalInner,
    }

    pub enum SoundSignalInner {
        Mono { signal: FramesSignal<f32> },
        Stereo { signal: FramesSignal<[f32; 2]> },
    }

    impl From<SoundSignalInner> for SoundSignal {
        fn from(inner: SoundSignalInner) -> Self {
            Self { inner }
        }
    }

    enum SoundSignalHandle {
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

    #[cfg(not(target_arch = "wasm32"))]
    mod native {
        use crate::file_manager::native_fs::File;
        use crate::file_manager::GameFilePath;
        use symphonia::core::io::MediaSource;

        pub(super) async fn get_media_source(
            file_path: &GameFilePath,
        ) -> anyhow::Result<Box<dyn MediaSource>> {
            Ok(Box::new(File::open(file_path.resolve())?))
        }
    }

    #[cfg(target_arch = "wasm32")]
    mod web {
        use crate::file_manager::{GameFilePath, HttpFileStreamerSync};
        use symphonia::core::io::MediaSource;

        impl MediaSource for HttpFileStreamerSync {
            fn is_seekable(&self) -> bool {
                true
            }

            fn byte_len(&self) -> Option<u64> {
                Some(self.inner.lock().file_size)
            }
        }

        pub(super) async fn get_media_source(
            file_path: &GameFilePath,
        ) -> anyhow::Result<Box<dyn MediaSource>> {
            let url = file_path.resolve();
            Ok(Box::new(HttpFileStreamerSync::new(url).await?))
        }
    }

    pub struct IkariStreamedAudioFile {
        device_sample_rate: u32,
        format_reader: Box<dyn symphonia::core::formats::FormatReader>,
        decoder: Box<dyn symphonia::core::codecs::Decoder>,
        track_id: u32,
        track_sample_rate: Option<u32>,
        track_length_seconds: Option<f32>,
        file_path: GameFilePath,
    }

    /// streams are returned separately since they aren't Send/Sync
    pub fn create_audio_manager() -> Result<(IkariAudioManager, IkariAudioOutputStreams)> {
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
            IkariAudioManager {
                master_volume: 1.0,
                device_sample_rate,

                spatial_scene_handle,
                mixer_handle,
                sounds: vec![],
            },
            IkariAudioOutputStreams {
                _spatial_scene_output_stream: spatial_scene_output_stream,
                _mixer_output_stream: mixer_output_stream,
            },
        ))
    }

    impl StreamAudio for IkariStreamedAudioFile {
        /// max_chunk_size=0 to read the whole stream at once
        /// The actual chunk size we end up getting will be a little lower than this value since we read
        /// the audio file in small 'packets' which might not fit evenly into max_chunk_size
        /// so we make sure not to overshoot
        #[profiling::function]
        fn read_chunk(&mut self, max_chunk_size: usize) -> anyhow::Result<(SoundData, bool)> {
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

            if let Some(track_sample_rate) = self.track_sample_rate {
                if self.device_sample_rate != track_sample_rate {
                    // resample the sound to the device sample rate using linear interpolation
                    samples = resample_linear(&samples, track_sample_rate, self.device_sample_rate);
                }
            }

            Ok((SoundData(samples), reached_end_of_stream))
        }

        fn track_length_seconds(&self) -> Option<f32> {
            self.track_length_seconds
        }

        fn file_path(&self) -> &GameFilePath {
            &self.file_path
        }
    }

    impl IkariStreamedAudioFile {
        pub async fn new(device_sample_rate: u32, file_path: GameFilePath) -> anyhow::Result<Self> {
            let mss =
                MediaSourceStream::new(get_media_source(&file_path).await?, Default::default());

            let mut hint = Hint::new();
            if let Some(extension) = file_path
                .relative_path
                .extension()
                .and_then(|extension| extension.to_str())
            {
                hint.with_extension(extension);
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
    }

    impl AudioManager for IkariAudioManager {
        fn add_sound(
            &mut self,
            file_path: GameFilePath,
            sound_data: SoundData,
            length_seconds: Option<f32>,
            params: SoundParams,
        ) -> usize {
            let signal = self.make_signal(&sound_data, &params);
            let sound = Sound::new(self, file_path, sound_data, length_seconds, params, signal);
            self.sounds.push(Some(sound));
            self.sounds.len() - 1
        }

        fn write_stream_data(&mut self, sound_index: usize, sound_data: SoundData) {
            if let Some(sound) = self.sounds[sound_index].as_mut() {
                sound.write_stream_data(sound_data, self.device_sample_rate);
            }
        }

        fn play_sound(&mut self, sound_index: usize) {
            if let Some(sound) = self.sounds[sound_index].as_mut() {
                sound.resume();
            }
        }

        fn reload_sound(&mut self, sound_index: usize, params: SoundParams) {
            if let Some(sound) = self.sounds[sound_index].take() {
                let signal = self.make_signal(&sound.data, &params);
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

        fn sound_is_playing(&self, sound_index: usize) -> bool {
            self.sounds[sound_index]
                .as_ref()
                .map(|sound| sound.is_playing)
                .unwrap_or(false)
        }

        fn get_sound_length_seconds(&self, sound_index: usize) -> Option<Option<f32>> {
            self.sounds[sound_index]
                .as_ref()
                .map(|sound| sound.length_seconds)
        }

        fn get_sound_pos_seconds(&self, sound_index: usize) -> Option<f32> {
            self.sounds[sound_index]
                .as_ref()
                .map(|sound| sound.pos_seconds())
        }

        fn get_sound_buffered_to_pos_seconds(&self, sound_index: usize) -> Option<f32> {
            self.sounds[sound_index]
                .as_ref()
                .map(|sound| sound.buffered_to_pos_seconds)
        }

        fn get_sound_file_path(&self, sound_index: usize) -> Option<&GameFilePath> {
            self.sounds[sound_index]
                .as_ref()
                .map(|sound| &sound.file_path)
        }

        fn set_sound_volume(&mut self, sound_index: usize, volume: f32) {
            if let Some(sound) = self.sounds[sound_index].as_mut() {
                sound.set_volume(self.master_volume, volume)
            }
        }

        fn device_sample_rate(&self) -> u32 {
            self.device_sample_rate
        }

        fn sound_indices(&self) -> Vec<usize> {
            self.sounds
                .iter()
                .enumerate()
                .filter_map(|(sound_index, sound)| sound.as_ref().map(|_| sound_index))
                .collect()
        }

        fn is_audio_enabled(&self) -> bool {
            true
        }
    }

    impl IkariAudioManager {
        fn make_signal(&self, sound_data: &SoundData, params: &SoundParams) -> Option<SoundSignal> {
            let SoundParams {
                spacial_params,
                stream,
                ..
            } = params;

            let SoundData(samples) = sound_data;

            if *stream {
                return None;
            }

            let channels = samples[0].len();

            Some(match spacial_params {
                Some(SpacialParams { .. }) => {
                    let signal = FramesSignal::from(oddio::Frames::from_iter(
                        self.device_sample_rate,
                        samples.iter().map(|sample| sample[0]).collect::<Vec<_>>(),
                    ));

                    SoundSignalInner::Mono { signal }.into()
                }
                None => {
                    let signal = FramesSignal::from(oddio::Frames::from_iter(
                        self.device_sample_rate,
                        samples.iter().map(|sample| {
                            [sample[0], if channels > 1 { sample[1] } else { sample[0] }]
                        }),
                    ));
                    SoundSignalInner::Stereo { signal }.into()
                }
            })
        }
    }

    #[profiling::function]
    fn resample_linear(
        samples: &[[f32; CHANNEL_COUNT]],
        from_hz: u32,
        to_hz: u32,
    ) -> Vec<[f32; CHANNEL_COUNT]> {
        let old_sample_count = samples.len();
        let length_seconds = old_sample_count as f32 / from_hz as f32;
        let new_sample_count = (length_seconds * to_hz as f32).ceil() as usize;
        let mut result: Vec<[f32; CHANNEL_COUNT]> = Vec::with_capacity(new_sample_count);
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
            audio_manager: &mut IkariAudioManager,
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
                    Some(SoundSignal {
                        inner: SoundSignalInner::Mono { signal },
                    }),
                    false,
                ) => {
                    let signal = Gain::new(signal);

                    let signal_handle = audio_manager
                        .spatial_scene_handle
                        .control::<SpatialScene, _>()
                        .play_buffered(
                            signal,
                            SpatialOptions {
                                position: [
                                    initial_position.x,
                                    initial_position.y,
                                    initial_position.z,
                                ]
                                .into(),
                                velocity: [
                                    initial_velocity.x,
                                    initial_velocity.y,
                                    initial_velocity.z,
                                ]
                                .into(),
                                radius: 0.1,
                            },
                            1000.0,
                            audio_manager.device_sample_rate,
                            0.1,
                        );

                    SoundSignalHandle::Spacial { signal_handle }
                }
                (
                    None,
                    Some(SoundSignal {
                        inner: SoundSignalInner::Stereo { signal },
                    }),
                    false,
                ) => {
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
                        (audio_manager.device_sample_rate as f32
                            * AUDIO_STREAM_BUFFER_LENGTH_SECONDS) as usize,
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
                self.buffered_to_pos_seconds +=
                    sound_data.0.len() as f32 / device_sample_rate as f32;
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
}

/// Used to disable audio in ikari. Acts like a real audio manager but does nothing under the hood
pub struct NullAudioManager {}

pub struct NullAudioOutputStreams {}

impl AudioOutputStreams for NullAudioOutputStreams {}

pub struct NullStreamedAudioFile {
    file_path: GameFilePath,
}

impl StreamAudio for NullStreamedAudioFile {
    fn read_chunk(&mut self, _max_chunk_size: usize) -> anyhow::Result<(SoundData, bool)> {
        Ok((SoundData(vec![]), true))
    }

    fn track_length_seconds(&self) -> Option<f32> {
        None
    }

    fn file_path(&self) -> &GameFilePath {
        &self.file_path
    }
}

impl AudioManager for NullAudioManager {
    fn add_sound(
        &mut self,
        _file_path: GameFilePath,
        _sound_data: SoundData,
        _length_seconds: Option<f32>,
        _params: SoundParams,
    ) -> usize {
        0
    }

    fn write_stream_data(&mut self, _sound_index: usize, _sound_data: SoundData) {}

    fn play_sound(&mut self, _sound_index: usize) {}

    fn reload_sound(&mut self, _sound_index: usize, _params: SoundParams) {}

    fn sound_is_playing(&self, _sound_index: usize) -> bool {
        false
    }

    fn get_sound_length_seconds(&self, _sound_index: usize) -> Option<Option<f32>> {
        None
    }

    fn get_sound_pos_seconds(&self, _sound_index: usize) -> Option<f32> {
        None
    }

    fn get_sound_buffered_to_pos_seconds(&self, _sound_index: usize) -> Option<f32> {
        None
    }

    fn get_sound_file_path(&self, _sound_index: usize) -> Option<&GameFilePath> {
        None
    }

    fn set_sound_volume(&mut self, _sound_index: usize, _volume: f32) {}

    fn device_sample_rate(&self) -> u32 {
        1
    }

    fn sound_indices(&self) -> Vec<usize> {
        vec![]
    }

    fn is_audio_enabled(&self) -> bool {
        false
    }
}

pub fn create_null_audio_manager() -> (NullAudioManager, NullAudioOutputStreams) {
    (NullAudioManager {}, NullAudioOutputStreams {})
}
