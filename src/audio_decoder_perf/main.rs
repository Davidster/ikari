use ikari::audio::{AudioFileFormat, AudioManager};

fn main() {
    let (audio_manager, _audio_streams) = AudioManager::new().unwrap();
    AudioManager::decode_audio_file(
        audio_manager.device_sample_rate(),
        "./src/sounds/bgm.mp3",
        Some(AudioFileFormat::Mp3),
    )
    .unwrap();
}
