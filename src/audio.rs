use anyhow::Result;
use cgmath::Vector3;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Stream,
};
use hound::{WavReader, WavSpec};
use oddio::{
    FilterHaving, FramesSignal, Gain, Handle, Mixer, Seek, SpatialBuffered, SpatialOptions,
    SpatialScene, Stop,
};

pub struct AudioManager {
    master_volume: f32,
    device_sample_rate: u32,

    spatial_scene_handle: Handle<SpatialScene>,
    spatial_scene_output_stream: Stream,

    mixer_handle: Handle<Mixer<[f32; 2]>>,
    mixer_output_stream: Stream,

    sounds: Vec<Sound>,
}

pub struct SoundData(Vec<Vec<f32>>);

pub struct Sound {
    volume: f32,
    is_playing: bool,
    signal: SoundSignal,
}

pub enum SoundSignal {
    Spacial {
        signal_handle: Handle<SpatialBuffered<Stop<Gain<FramesSignal<f32>>>>>,
    },
    Ambient {
        signal_handle: Handle<Stop<Gain<FramesSignal<[f32; 2]>>>>,
    },
}

pub struct SpacialParams {
    initial_position: Vector3<f32>,
    initial_velocity: Vector3<f32>,
}

impl AudioManager {
    pub fn new() -> Result<Self> {
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
        )?;
        spatial_scene_output_stream.play()?;
        mixer_output_stream.play()?;

        Ok(Self {
            master_volume: 1.0,
            device_sample_rate,

            spatial_scene_handle,
            spatial_scene_output_stream,

            mixer_handle,
            mixer_output_stream,

            sounds: vec![],
        })
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
        let length_samples = reader.duration();
        let length_seconds = length_samples as f32 / source_sample_rate as f32;

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
        let mut samples: Vec<_> = samples_interleaved
            .chunks(channels.into())
            .map(|chunk| chunk.to_vec())
            .collect();

        if sample_rate != source_sample_rate {
            // resample the sound to the device sample rate using linear interpolation
            let old_sample_count = samples.len();
            let new_sample_count = (length_seconds * sample_rate as f32).ceil() as usize;
            // TODO: instead of vec of vecs, use one big vec with smart indexing
            let new_samples: Vec<_> = (1..(new_sample_count + 1))
                .map(|new_sample_number| {
                    let old_sample_number =
                        new_sample_number as f32 * (source_sample_rate as f32 / sample_rate as f32);

                    // get the indices of the two samples that surround the old sample number
                    let left_index = old_sample_number
                        .clamp(1.0, old_sample_count as f32)
                        .floor() as usize
                        - 1;
                    let right_index = (left_index + 1).min(old_sample_count - 1);

                    let left_sample = &samples[left_index];
                    if left_index == right_index {
                        left_sample.to_vec()
                    } else {
                        let right_sample = &samples[right_index];
                        let t = old_sample_number % 1.0;
                        (0..(channels as usize))
                            .map(|channel| {
                                (1.0 - t) * left_sample[channel] + t * right_sample[channel]
                            })
                            .collect::<Vec<_>>()
                    }
                })
                .collect();
            samples = new_samples;
        }

        Ok(SoundData(samples))
    }

    pub fn add_sound(
        &mut self,
        sound_data: &SoundData,
        spacial_params: Option<SpacialParams>,
    ) -> usize {
        let sound = Sound::new(self, sound_data, 1.0, spacial_params);
        self.sounds.push(sound);
        self.sounds.len() - 1
    }

    pub fn play_sound(&mut self, sound_index: usize) {
        self.sounds[sound_index].resume();
    }

    pub fn device_sample_rate(&self) -> u32 {
        self.device_sample_rate
    }
}

impl Sound {
    fn new(
        audio_manager: &mut AudioManager,
        sound_data: &SoundData,
        initial_volume: f32,
        spacial_params: Option<SpacialParams>,
    ) -> Self {
        let SoundData(samples) = sound_data;

        let channels = samples[0].len();

        let mut sound = match spacial_params {
            Some(SpacialParams {
                initial_position,
                initial_velocity,
            }) => {
                let signal = Gain::new(FramesSignal::from(oddio::Frames::from_iter(
                    audio_manager.device_sample_rate,
                    samples.iter().map(|sample| sample[0]).collect::<Vec<_>>(),
                )));

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
                    signal: SoundSignal::Spacial { signal_handle },
                }
            }
            None => {
                let signal = Gain::new(FramesSignal::from(oddio::Frames::from_iter(
                    audio_manager.device_sample_rate,
                    samples
                        .iter()
                        .map(|sample| [sample[0], if channels > 1 { sample[1] } else { sample[0] }])
                        .collect::<Vec<_>>(),
                )));
                let signal_handle = audio_manager
                    .mixer_handle
                    .control::<Mixer<_>, _>()
                    .play(signal);
                Sound {
                    is_playing: true,
                    volume: initial_volume,
                    signal: SoundSignal::Ambient { signal_handle },
                }
            }
        };

        sound.set_volume(audio_manager.master_volume, initial_volume);
        sound.pause();

        sound
    }

    fn pause(&mut self) {
        self.is_playing = false;
        match &mut self.signal {
            SoundSignal::Spacial { signal_handle } => {
                signal_handle.control::<Stop<_>, _>().pause();
            }
            SoundSignal::Ambient { signal_handle } => {
                signal_handle.control::<Stop<_>, _>().pause();
            }
        }
    }

    fn resume(&mut self) {
        self.is_playing = true;
        match &mut self.signal {
            SoundSignal::Spacial { signal_handle } => {
                signal_handle.control::<Stop<_>, _>().resume();
            }
            SoundSignal::Ambient { signal_handle } => {
                signal_handle.control::<Stop<_>, _>().resume();
            }
        }
    }

    fn set_volume(&mut self, master_volume: f32, volume: f32) {
        self.volume = volume;
        match &mut self.signal {
            SoundSignal::Spacial { signal_handle } => {
                signal_handle
                    .control::<Gain<_>, _>()
                    .set_gain(master_volume * self.volume);
            }
            SoundSignal::Ambient { signal_handle } => {
                signal_handle
                    .control::<Gain<_>, _>()
                    .set_gain(master_volume * self.volume);
            }
        }
    }

    fn set_motion(&mut self, position: Vector3<f32>, velocity: Vector3<f32>, discontinuity: bool) {
        if let SoundSignal::Spacial { signal_handle } = &mut self.signal {
            signal_handle.control::<SpatialBuffered<_>, _>().set_motion(
                [position.x, position.y, position.z].into(),
                [velocity.x, velocity.y, velocity.z].into(),
                discontinuity,
            );
        }
    }
}
