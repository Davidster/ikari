use std::ops::Add;

use anyhow::Result;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, Stream,
};
use glam::f32::Vec3;
use hound::{WavReader, WavSpec};
use oddio::{
    FixedGain, FramesSignal, Gain, Handle, Mixer, SpatialBuffered, SpatialOptions, SpatialScene,
    Stop,
};

use super::*;

pub struct AudioStreams {
    _spatial_scene_output_stream: Stream,
    _mixer_output_stream: Stream,
}

pub struct AudioManager {
    master_volume: f32,
    device_sample_rate: u32,

    spatial_scene_handle: Handle<SpatialScene>,
    mixer_handle: Handle<Mixer<[f32; 2]>>,
    sounds: Vec<Sound>,
}

const CHANNEL_COUNT: usize = 2;

#[derive(Debug, Clone)]
pub struct SoundData(Vec<[f32; CHANNEL_COUNT]>);

pub struct Sound {
    volume: f32,
    is_playing: bool,
    signal_handle: SoundSignalHandle,
    data: SoundData,
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
}

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
}

impl AudioManager {
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
                eprintln!("{}", err);
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
                eprintln!("{}", err);
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

    pub fn decode_wav(sample_rate: u32, file_path: &str) -> Result<SoundData> {
        // get metadata from the WAV file
        let mut reader = WavReader::open(file_path)?;
        let WavSpec {
            sample_rate: source_sample_rate,
            sample_format,
            bits_per_sample,
            channels,
            ..
        } = reader.spec();

        if channels as usize != CHANNEL_COUNT {
            anyhow::bail!("Only dual-channel (stereo) wav files are supported");
        }

        // convert the WAV data to floating point samples
        // e.g. i8 data is converted from [-128, 127] to [-1.0, 1.0]
        let samples_result: Result<Vec<f32>, _> = match sample_format {
            hound::SampleFormat::Int => {
                let max_value = 2_u32.pow(bits_per_sample as u32 - 1) - 1;
                reader
                    .samples::<i32>()
                    .map(|sample| sample.map(|sample| sample as f32 / max_value as f32))
                    .collect()
            }
            hound::SampleFormat::Float => reader.samples::<f32>().collect(),
        };
        // channels are interleaved
        let samples_interleaved = samples_result.unwrap();
        let mut samples: Vec<[f32; CHANNEL_COUNT]> = samples_interleaved
            .chunks(channels.into())
            .map(|chunk| [chunk[0], chunk[1]])
            .collect();

        if sample_rate != source_sample_rate {
            // resample the sound to the device sample rate using linear interpolation
            samples = resample_linear(&samples, source_sample_rate, sample_rate);
        }

        Ok(SoundData(samples))
    }

    pub fn decode_mp3(sample_rate: u32, file_path: &str) -> Result<SoundData> {
        let file_bytes: &[u8] = &std::fs::read(file_path)?;
        let mut decoder = minimp3::Decoder::new(file_bytes);
        let mut mp3_frames: Vec<minimp3::Frame> = Vec::new();
        loop {
            match decoder.next_frame() {
                Ok(frame) => {
                    if frame.channels != CHANNEL_COUNT {
                        anyhow::bail!("Only dual-channel (stereo) mp3 files are supported");
                    }
                    mp3_frames.push(frame);
                }
                Err(minimp3::Error::Eof) => {
                    break;
                }
                Err(err) => anyhow::bail!(err),
            }
        }

        let mut samples: Vec<[f32; CHANNEL_COUNT]> = Vec::new();
        for mp3_frame in mp3_frames {
            let source_sample_rate = u32::try_from(mp3_frame.sample_rate).unwrap();
            let mut current_samples: Vec<[f32; CHANNEL_COUNT]> = Vec::new();
            current_samples.reserve(mp3_frame.data.len() / CHANNEL_COUNT);
            for sample in mp3_frame.data.chunks(CHANNEL_COUNT) {
                current_samples.push([
                    sample[0] as f32 / i16::MAX as f32,
                    sample[1] as f32 / i16::MAX as f32,
                ]);
            }
            if sample_rate != source_sample_rate {
                // resample the sound to the device sample rate using linear interpolation
                current_samples =
                    resample_linear(&current_samples, source_sample_rate, sample_rate);
            }
            samples.reserve(current_samples.len());
            for sample in current_samples {
                samples.push(sample);
            }
        }
        Ok(SoundData(samples))
    }

    pub fn get_signal(
        sound_data: &SoundData,
        params: SoundParams,
        device_sample_rate: u32,
    ) -> SoundSignal {
        let SoundParams { spacial_params, .. } = params;

        let SoundData(samples) = sound_data;

        let channels = samples[0].len();

        match spacial_params {
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
        }
    }

    pub fn add_sound(
        &mut self,
        sound_data: SoundData,
        params: SoundParams,
        signal: SoundSignal,
    ) -> usize {
        let sound = Sound::new(self, sound_data, params, signal);
        self.sounds.push(sound);
        self.sounds.len() - 1
    }

    pub fn play_sound(&mut self, sound_index: usize) {
        self.sounds[sound_index].resume();
    }

    pub fn reload_sound(&mut self, sound_index: usize, params: SoundParams) {
        // TODO: can avoid clone here by taking self.sounds[sound_index] out of the vec?
        let data_copy = self.sounds[sound_index].data.clone();
        let signal = Self::get_signal(&data_copy, params.clone(), self.device_sample_rate);
        self.sounds[sound_index] = Sound::new(self, data_copy, params, signal);
    }

    pub fn set_sound_volume(&mut self, sound_index: usize, volume: f32) {
        self.sounds[sound_index].set_volume(self.master_volume, volume);
    }

    pub fn device_sample_rate(&self) -> u32 {
        self.device_sample_rate
    }
}

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
        sound_data: SoundData,
        params: SoundParams,
        signal: SoundSignal,
    ) -> Self {
        let SoundParams {
            initial_volume,
            fixed_volume,
            spacial_params,
        } = params;

        let mut sound = match (spacial_params, signal) {
            (
                Some(SpacialParams {
                    initial_position,
                    initial_velocity,
                }),
                SoundSignal::Mono { signal },
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

                Sound {
                    is_playing: true,
                    volume: initial_volume,
                    signal_handle: SoundSignalHandle::Spacial { signal_handle },
                    data: sound_data,
                }
            }
            (None, SoundSignal::Stereo { signal }) => {
                if fixed_volume {
                    let volume_amplitude_ratio =
                        (audio_manager.master_volume * initial_volume).powf(2.0);
                    let volume_db = 20.0 * volume_amplitude_ratio.log10();
                    let signal = FixedGain::new(signal, volume_db);
                    let signal_handle = audio_manager
                        .mixer_handle
                        .control::<Mixer<_>, _>()
                        .play(signal);
                    Sound {
                        is_playing: true,
                        volume: initial_volume,
                        signal_handle: SoundSignalHandle::AmbientFixed { signal_handle },
                        data: sound_data,
                    }
                } else {
                    let signal = Gain::new(signal);
                    let signal_handle = audio_manager
                        .mixer_handle
                        .control::<Mixer<_>, _>()
                        .play(signal);
                    let mut sound = Sound {
                        is_playing: true,
                        volume: initial_volume,
                        signal_handle: SoundSignalHandle::Ambient { signal_handle },
                        data: sound_data,
                    };
                    sound.set_volume(audio_manager.master_volume, initial_volume);
                    sound
                }
            }
            _ => {
                panic!("Signal didn't match spatial params");
            }
        };

        sound.pause();
        sound
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
        }
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
        }
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
        }
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
