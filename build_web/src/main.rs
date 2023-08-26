use std::{ffi::OsStr, path::PathBuf, process::Command};

use arboard::Clipboard;
use warp::{http::HeaderValue, hyper::HeaderMap, Filter};

const HELP: &str = "\
cargo build_web

Hosts a binary or example of the local package as wasm in a local web server.

Usage: cargo build_web [OPTIONS]

Options:
  cargo build_web custom options:
    --build-only                 Only build the WASM artifacts, do not run the dev server
    --port <PORT>                Makes the dev server listen on port (default '8000')

  cargo run default options:
    -q, --quiet                     Do not print cargo log messages
        --bin [<NAME>]              Name of the bin target to run
        --example [<NAME>]          Name of the example target to run
    -p, --package [<SPEC>...]       Package with the target to run
    -v, --verbose                   Use verbose output (-vv very verbose/build.rs output)
    -j, --jobs <N>                  Number of parallel jobs, defaults to # of CPUs
        --color <WHEN>              Coloring: auto, always, never
        --keep-going                Do not abort the build as soon as there is an error (unstable)
        --frozen                    Require Cargo.lock and cache are up to date
    -r, --release                   Build artifacts in release mode, with optimizations
        --locked                    Require Cargo.lock is up to date
        --profile <PROFILE-NAME>    Build artifacts with the specified profile
    -F, --features <FEATURES>       Space or comma separated list of features to activate
        --offline                   Run without accessing the network
        --all-features              Activate all available features
        --config <KEY=VALUE>        Override a configuration value
        --no-default-features       Do not activate the `default` feature
    -Z <FLAG>                       Unstable (nightly-only) flags to Cargo, see 'cargo -Z help' for
                                    details
        --manifest-path <PATH>      Path to Cargo.toml
        --message-format <FMT>      Error format
        --unit-graph                Output build graph in JSON (unstable)
        --ignore-rust-version       Ignore `rust-version` specification in packages
        --timings[=<FMTS>...]       Timing output formats (unstable) (comma separated): html, json
    -h, --help                      Print help information

At least one of `--package`, `--bin` or `--example` must be used.

Normally you can run just `cargo run` to run the main binary of the current package.
The equivalent of that is `cargo build_web --package name_of_current_package`
";

struct Args {
    help: bool,
    profile: Option<String>,
    build_only: bool,
    port: Option<String>,
    build_args: Vec<String>,
    build_args_contains_wasm_alloc_feature: bool,
    package: Option<String>,
    example: Option<String>,
    bin: Option<String>,
    binary_name: String,
}

impl Args {
    pub fn from_env() -> Result<Self, String> {
        let mut args = pico_args::Arguments::from_env();

        let release_arg = args.contains("--release") || args.contains("-r");
        let profile_arg: Option<String> = args.opt_value_from_str("--profile").unwrap();
        if release_arg && profile_arg.is_some() {
            return Err(r#"conflicting usage of --profile and --release.
The `--release` flag is the same as `--profile=release`.
Remove one flag or the other to continue."#
                .to_owned());
        }
        let profile = profile_arg.or_else(|| {
            if release_arg {
                Some("release".to_owned())
            } else {
                None
            }
        });

        let build_only = args.contains("--build-only");
        let help = args.contains("--help") || args.contains("-h");

        let port: Option<String> = args.opt_value_from_str("--port").unwrap();

        let package: Option<String> = args
            .opt_value_from_str("--package")
            .unwrap()
            .or_else(|| args.opt_value_from_str("-p").unwrap());
        let example: Option<String> = args.opt_value_from_str("--example").unwrap();
        let bin: Option<String> = args.opt_value_from_str("--bin").unwrap();

        let banned_options = ["--target", "--target-dir"];
        for option in banned_options {
            if args
                .opt_value_from_str::<_, String>(option)
                .unwrap()
                .is_some()
            {
                return Err(format!("build_web does not support the {option} option"));
            }
        }

        let binary_name = match example.as_ref().or(bin.as_ref()).or(package.as_ref()) {
            Some(name) => name.clone(),
            None => {
                return Err("Need to use at least one of `--package NAME`, `--example NAME` `--bin NAME`.\nRun cargo build_web --help for more info.".to_owned());
            }
        };

        dbg!(&args.clone());

        let peeked_features: Option<String> = args
            .clone()
            .opt_value_from_str("--features")
            .unwrap()
            .or_else(|| args.opt_value_from_str("-F").unwrap());

        let build_args_contains_wasm_alloc_feature = peeked_features.is_some()
            && peeked_features.unwrap().contains("wasm-tracing-allocator");

        let build_args = args
            .finish()
            .into_iter()
            .map(|x| x.into_string().unwrap())
            .collect();

        Ok(Args {
            help,
            profile,
            build_only,
            build_args_contains_wasm_alloc_feature,
            port,
            build_args,
            package,
            example,
            bin,
            binary_name,
        })
    }
}

/// Adapted from cargo-run-wasm v0.3.2
/// TODO: exit with code 1 on failures?
fn main() {
    env_logger::init();

    let args = match Args::from_env() {
        Ok(args) => args,
        Err(err) => {
            println!("{err}\n\n{HELP}");
            return;
        }
    };
    if args.help {
        println!("{HELP}");
        return;
    }

    let profile_dir_name = match args.profile.as_deref() {
        Some("dev") => "debug",
        Some(profile) => profile,
        None => "debug",
    };

    let workspace_root = PathBuf::from("");
    let target_directory = workspace_root.join("target");
    let target_target = target_directory.join("wasm-examples-target");

    let mut cargo_args = vec![
        "build".as_ref(),
        "-Z".as_ref(),
        "build-std=panic_abort,std".as_ref(),
        "--target".as_ref(),
        "wasm32-unknown-unknown".as_ref(),
        // It is common to setup a faster linker such as mold or lld to run for just your native target.
        // It cant be set for wasm as wasm doesnt support building with these linkers.
        // This results in a separate rustflags value for native and wasm builds.
        // Currently rust triggers a full rebuild every time the rustflags value changes.
        //
        // Therefore we have this hack where we use a different target dir for wasm builds to avoid constantly triggering full rebuilds.
        // When this issue is resolved we might be able to remove this hack: https://github.com/rust-lang/cargo/issues/8716
        "--target-dir".as_ref(),
        target_target.as_os_str(),
    ];

    if let Some(package) = args.package.as_ref() {
        cargo_args.extend([OsStr::new("--package"), package.as_ref()]);
    }
    if let Some(example) = args.example.as_ref() {
        cargo_args.extend([OsStr::new("--example"), example.as_ref()]);
    }
    if let Some(bin) = args.bin.as_ref() {
        cargo_args.extend([OsStr::new("--bin"), bin.as_ref()]);
    }
    if let Some(profile) = &args.profile {
        cargo_args.extend([OsStr::new("--profile"), profile.as_ref()]);
    }

    cargo_args.extend(args.build_args.iter().map(OsStr::new));

    dbg!(&cargo_args);

    let status = Command::new("cargo")
        // .current_dir(&workspace_root)
        .env(
            "RUSTFLAGS",
            "--cfg=web_sys_unstable_apis -C target-feature=+atomics,+bulk-memory,+mutable-globals",
        )
        .args(&cargo_args)
        .status()
        .unwrap();
    if !status.success() {
        // We can return without printing anything because cargo will have already displayed an appropriate error.
        return;
    }

    let binary_name = args.binary_name;

    // run wasm-bindgen on wasm file output by cargo, write to the destination folder
    let target_profile = target_target
        .join("wasm32-unknown-unknown")
        .join(profile_dir_name);
    let wasm_source = if args.example.is_some() {
        target_profile.join("examples")
    } else {
        target_profile
    }
    .join(format!("{binary_name}.wasm"));

    if !wasm_source.exists() {
        println!("There is no binary at {wasm_source:?}, maybe you used `--package NAME` on a package that has no binary?");
        return;
    }

    let examples_dir_name = "wasm-examples";
    let example_dest = target_directory.join(examples_dir_name).join(&binary_name);
    std::fs::create_dir_all(&example_dest).unwrap();
    let mut bindgen = wasm_bindgen_cli_support::Bindgen::new();
    bindgen
        .web(true)
        .unwrap()
        .omit_default_module_path(false)
        .input_path(&wasm_source)
        .generate(&example_dest)
        .unwrap();

    dbg!(args.build_args_contains_wasm_alloc_feature);

    // process template html and write to the destination folder
    let index_template = include_str!("ikari-web.template.html");
    let index_processed = index_template
        .replace(
            "{{wasm_tracing_allocator}}",
            if args.build_args_contains_wasm_alloc_feature {
                "<script crossorigin=\"anonymous\" src=\"https://unpkg.com/wasm-tracing-allocator@0.1.1/js/hooks.js\"></script>"
            } else {
                ""
            }
        )
        .replace("{{name}}", &binary_name)
        .replace(
            "{{jspath}}",
            &format!("./target/{examples_dir_name}/{binary_name}/{binary_name}.js"),
        );

    let html_file_name = "ikari-web.html";

    std::fs::write(workspace_root.join(html_file_name), index_processed).unwrap();

    if !args.build_only {
        let port = args
            .port
            .unwrap_or_else(|| "8000".into())
            .parse()
            .expect("Port should be an integer");

        let url = format!("http://localhost:{port}/{html_file_name}");

        println!("\nServing `{binary_name}` on {url}");

        std::thread::spawn(move || {
            println!("Attempted to copy to clipboard");
            if let Err(err) = copy_text_to_clipboard(&url) {
                eprintln!("Failed to copy URL to clipboard: {err}");
            }
        });

        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let mut headers = HeaderMap::new();
                headers.insert(
                    "Cross-Origin-Opener-Policy",
                    HeaderValue::from_static("same-origin"),
                );
                headers.insert(
                    "Cross-Origin-Embedder-Policy",
                    HeaderValue::from_static("require-corp"),
                );
                warp::serve(
                    warp::fs::dir(workspace_root).with(warp::reply::with::headers(headers)),
                )
                .run(([127, 0, 0, 1], port))
                .await;
            });
    }
}

fn copy_text_to_clipboard(text: &str) -> Result<(), arboard::Error> {
    let mut clipboard = Clipboard::new()?;

    #[cfg(target_os = "linux")]
    {
        use arboard::SetExtLinux;
        clipboard.set().wait().text(text)?;
    }

    #[cfg(not(target_os = "linux"))]
    {
        clipboard.set().text(text)?;
    }

    Ok(())
}
