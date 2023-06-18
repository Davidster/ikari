use crate::logger::*;

use chrono::prelude::*;
use std::{
    io::{BufRead, BufReader},
    process::{Command, Stdio},
    sync::{Arc, Mutex},
};

pub type PendingPerfDump = Arc<Mutex<Option<anyhow::Result<String>>>>;

pub fn profiling_is_enabled() -> bool {
    cfg!(feature = "tracy")
}

pub fn generate_profile_dump() -> PendingPerfDump {
    if !profiling_is_enabled() {
        let msg = "Warning: tried to capture a profile dump but profiling is not enabled. Please enable the tracy feature in cargo";
        logger_log(msg);
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

// TODO: implement a timeout, could catch some failure cases
fn generate_profile_dump_internal() -> anyhow::Result<String> {
    let time_string = Utc::now().format("%Y-%m-%d_%H-%M-%S_utc").to_string();
    let dump_size_seconds = 5;
    let args = [
        "-o",
        &format!("ikari_perf_dump_{time_string}.tracy"),
        "-s",
        &format!("{dump_size_seconds}"),
    ];
    let mut child_process = if cfg!(target_os = "windows") {
        Command::new("./ikari/bin/tracy/win/capture.exe")
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?
    } else {
        Command::new("./ikari/bin/tracy/unix/capture")
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?
    };

    let std_out_lines = {
        let mut lines: Vec<String> = vec![];
        let stdout = child_process.stdout.as_mut().unwrap();

        // TODO: why does this block??
        for line_result in BufReader::new(stdout).lines() {
            lines.push(line_result?);
        }

        lines
    };
    let exit_status = child_process.wait()?;

    let std_out_lines = std_out_lines.join("\n");
    println!("Profile capture stdout:");
    println!("{std_out_lines}");

    if !exit_status.success() {
        anyhow::bail!(std_out_lines);
    }

    Ok(std_out_lines)
}
