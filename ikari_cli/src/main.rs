mod texture_compressor;

fn env_var_is_defined(var: &str) -> bool {
    match std::env::var(var) {
        Ok(val) => !val.is_empty(),
        Err(_) => false,
    }
}

fn main() {
    if !env_var_is_defined("RUST_BACKTRACE") {
        std::env::set_var("RUST_BACKTRACE", "1");
    }

    if env_var_is_defined("RUST_LOG") {
        env_logger::init();
    } else {
        env_logger::builder()
            .filter(Some(env!("CARGO_PKG_NAME")), log::LevelFilter::Info)
            .filter(Some(env!("CARGO_BIN_NAME")), log::LevelFilter::Info)
            .filter(Some("wgpu"), log::LevelFilter::Warn)
            .init();
    }

    texture_compressor::run();
}
