mod texture_compressor;

use texture_compressor::TextureCompressorArgs;

const HELP: &str = "\
ikari cli

Usage: clikari [COMMAND] [OPTIONS]

Commands:
  compress_textures

Options:
  --help  Optional  Display this help message
";

const TEXTURE_COMPRESSOR_HELP: &str = "\
Compress all textures found in a given folder by recursive search

Usage: clikari compress_textures --search_folder /path/to/folder [OPTIONS]

Options:
  --search_folder FOLDERNAME  Required  The folder to search in to find textures to compress
  --threads_per_texture VAL   Optional  The number of threads that will be used per texture.
                                        Textures will also be processed in parallel if possible,
                                        according to the formula: (cpu_count / threads_per_texture).ceil().
                                        This is done because individual texture processing is not fully parallel
                                        Defaults to 4 threads per texture
  --force                     Optional  Force re-compress all textures regardless of whether their
                                        _compressed.bin counterpart already exists
  --help                      Optional  Display this help message
";

enum Command {
    Help,
    CompressTextures(TextureCompressorArgs),
    CompressTexturesHelp,
}

enum ArgParseError {
    Root(String),
    CompressTextures(String),
}

impl Command {
    pub fn from_env() -> Result<Self, ArgParseError> {
        let mut args = pico_args::Arguments::from_env();

        if args.contains("compress_textures") {
            if args.contains("--help") {
                return Ok(Self::CompressTexturesHelp);
            }

            return Ok(Self::CompressTextures(TextureCompressorArgs {
                search_folder: args
                    .value_from_str("--search_folder")
                    .map_err(|err| ArgParseError::CompressTextures(format!("{err}")))?,
                threads_per_texture: args
                    .opt_value_from_str("--threads_per_texture")
                    .map_err(|err| ArgParseError::CompressTextures(format!("{err}")))?,
                force: args.contains("--force"),
            }));
        }

        if args.contains("--help") {
            return Ok(Self::Help);
        }

        Err(ArgParseError::Root(String::from("No command specified")))
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

    match Command::from_env() {
        Ok(Command::CompressTextures(args)) => {
            texture_compressor::run(args);
        }
        Ok(Command::Help) => {
            println!("{HELP}");
        }
        Ok(Command::CompressTexturesHelp) => {
            println!("{TEXTURE_COMPRESSOR_HELP}");
        }
        Err(err) => {
            let (err, helpmsg) = match err {
                ArgParseError::Root(err) => (err, HELP),
                ArgParseError::CompressTextures(err) => (err, TEXTURE_COMPRESSOR_HELP),
            };
            println!("Error: {err}\n\n{helpmsg}");
        }
    };
}

fn env_var_is_defined(var: &str) -> bool {
    match std::env::var(var) {
        Ok(val) => !val.is_empty(),
        Err(_) => false,
    }
}
