#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;

#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;

#[cfg(target_arch = "wasm32")]
pub async fn read(path: &str) -> anyhow::Result<Vec<u8>> {
    const ASSET_SERVER: &str = "http://localhost:8000";

    use wasm_bindgen::prelude::*;
    use web_sys::{Blob, RequestInit, Response};

    let mut opts = RequestInit::new();
    opts.method("GET");
    let path = resolve_path(path);
    let url = format!("{ASSET_SERVER}/{path}");
    let request = map_js_err(web_sys::Request::new_with_str_and_init(&url, &opts))?;
    let window = web_sys::window().unwrap();
    let resp_value = run_promise(window.fetch_with_request(&request)).await?;
    assert!(resp_value.is_instance_of::<Response>());
    let resp: Response = resp_value.dyn_into().unwrap();

    if !resp.ok() {
        let status = resp.status();
        anyhow::bail!("Request to {url} responded with error status code: {status}")
    }

    let blob_promise = map_js_err(resp.blob())?;
    let blob = run_promise(blob_promise).await?;
    let array_buffer: JsValue = run_promise(Blob::from(blob).array_buffer()).await?;

    anyhow::Ok(js_sys::Uint8Array::new(&array_buffer).to_vec())
}

#[cfg(target_arch = "wasm32")]
pub async fn run_promise(promise: js_sys::Promise) -> anyhow::Result<JsValue> {
    map_js_err(wasm_bindgen_futures::JsFuture::from(promise).await)
}

#[cfg(target_arch = "wasm32")]
pub fn map_js_err<T>(result: std::result::Result<T, JsValue>) -> anyhow::Result<T> {
    result.map_err(|err| anyhow::anyhow!(err.as_string().unwrap_or_default()))
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn read(path: &str) -> anyhow::Result<Vec<u8>> {
    Ok(std::fs::read(resolve_path(path))?)
}

#[cfg(target_arch = "wasm32")]
pub async fn read_to_string(path: &str) -> anyhow::Result<String> {
    let bytes = read(path).await?;
    Ok(std::str::from_utf8(&bytes)?.to_string())
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn read_to_string(path: &str) -> anyhow::Result<String> {
    Ok(std::fs::read_to_string(resolve_path(path))?)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn open_file(path: &str) -> anyhow::Result<File> {
    Ok(std::fs::File::open(resolve_path(path))?)
}

fn resolve_path(path: &str) -> String {
    format!("ikari/{path}")
}
