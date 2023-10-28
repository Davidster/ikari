use chrono::prelude::*;
use std::{
    io::{BufRead, BufReader},
    process::{Command, Stdio},
    sync::{Arc, Mutex},
};

pub type PendingPerfDump = Arc<Mutex<Option<anyhow::Result<String>>>>;

pub fn profiling_is_enabled() -> bool {
    cfg!(feature = "tracy-profile-dumps")
}

pub fn can_generate_profile_dump() -> bool {
    cfg!(not(target_arch = "wasm32"))
}

pub fn generate_profile_dump() -> PendingPerfDump {
    if !profiling_is_enabled() {
        let msg = "Warning: tried to capture a profile dump but profiling is not enabled. Please enable the tracy feature in cargo";
        log::warn!("{msg}");
        return Arc::new(Mutex::new(Some(Err(anyhow::anyhow!(msg)))));
    }

    let result: PendingPerfDump = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    crate::thread::spawn(move || {
        profiling::register_thread!("Generate profile dump");
        let dump_res = generate_profile_dump_internal();
        *result.lock().unwrap() = Some(dump_res);
    });

    result_clone
}

fn generate_profile_dump_internal() -> anyhow::Result<String> {
    let time_string = Utc::now().format("%Y-%m-%d_%H-%M-%S_utc").to_string();
    let dump_size_seconds = 5;
    let args = [
        "-o",
        &format!("ikari_perf_dump_{time_string}.tracy"),
        "-s",
        &format!("{dump_size_seconds}"),
    ];

    let tracy_capture_exe = if cfg!(target_os = "windows") {
        "./ikari/bin/tracy/win/capture.exe"
    } else if cfg!(target_os = "macos") {
        "./ikari/bin/tracy/macos/capture"
    } else {
        "./ikari/bin/tracy/unix/capture"
    };

    let mut child_process = Command::new(tracy_capture_exe)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let std_out_lines = {
        let mut lines: Vec<String> = vec![];
        let stdout = child_process.stdout.as_mut().unwrap();

        // why does this block?? the lines don't come in one at a time, they come all at once
        for line_result in BufReader::new(stdout).lines() {
            lines.push(line_result?);
        }

        lines
    };
    let exit_status = child_process.wait()?;

    let std_out_lines = std_out_lines.join("\n");
    log::info!("Profile capture stdout:");
    log::info!("{std_out_lines}");

    if !exit_status.success() {
        anyhow::bail!(std_out_lines);
    }

    Ok(std_out_lines)
}
