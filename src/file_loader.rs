#[cfg(target_arch = "wasm32")]
pub async fn read(path: &str) -> anyhow::Result<Vec<u8>> {
    log::info!("read 1");
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Blob, RequestInit, Response};

    async fn blob_into_bytes(blob: Blob) -> Vec<u8> {
        let array_buffer_promise: JsFuture = blob.array_buffer().into();

        let array_buffer: JsValue = array_buffer_promise
            .await
            .expect("Could not get ArrayBuffer from file");

        js_sys::Uint8Array::new(&array_buffer).to_vec()
    }

    async fn smd(path: &str) -> Result<Vec<u8>, JsValue> {
        log::info!("smd 1");
        let mut opts = RequestInit::new();
        opts.method("GET");
        log::info!("smd 2");
        let request = web_sys::Request::new_with_str_and_init(&String::from(path), &opts)?;
        log::info!("smd 3");
        let window = web_sys::window().unwrap();
        log::info!("smd 4");
        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
        log::info!("smd 5");
        assert!(resp_value.is_instance_of::<Response>());
        log::info!("smd 6");
        let resp: Response = resp_value.dyn_into().unwrap();
        log::info!("smd 7");
        let blob = JsFuture::from(resp.blob()?).await?;
        log::info!("smd 8, blob={:?}", blob);
        Ok(blob_into_bytes(blob.into()).await)
    }
    log::info!("read 2");
    let yo = smd(path).await;
    log::info!("read 3");
    match yo {
        Ok(bytes) => anyhow::Ok(bytes),
        _ => anyhow::bail!("Idk how to use this JsValue thingy 🤷‍♂️"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn read(path: &str) -> anyhow::Result<Vec<u8>> {
    Ok(std::fs::read(path)?)
}

#[cfg(target_arch = "wasm32")]
pub async fn read_to_string(path: &str) -> anyhow::Result<String> {
    let bytes = read(path).await?;
    Ok(std::str::from_utf8(&bytes)?.to_string())
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn read_to_string(path: &str) -> anyhow::Result<String> {
    Ok(std::fs::read_to_string(path)?)
}
