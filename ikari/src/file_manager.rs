use std::path::PathBuf;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GameFilePath {
    pub root: PathBuf,
    pub relative_path: PathBuf,
    #[cfg(target_arch = "wasm32")]
    pub asset_server: String, // e.g. "http://localhost:8000"
}

pub struct GamePathMaker {
    root: PathBuf,
    #[cfg(target_arch = "wasm32")]
    pub asset_server: String, // e.g. "http://localhost:8000"
}

pub struct FileManager;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::path::PathBuf;

    use super::{FileManager, GameFilePath, GamePathMaker};

    pub mod native_fs {
        pub use std::fs::*;
    }

    impl GameFilePath {
        pub fn resolve(&self) -> PathBuf {
            self.root.join(&self.relative_path)
        }
    }

    impl GamePathMaker {
        pub fn new(root: Option<PathBuf>) -> Self {
            Self {
                root: root.unwrap_or_else(|| "".into()),
            }
        }

        pub fn make<T>(&self, relative_path: T) -> GameFilePath
        where
            T: Into<PathBuf>,
        {
            GameFilePath {
                root: self.root.clone(),
                relative_path: relative_path.into(),
            }
        }
    }

    impl FileManager {
        pub async fn read(path: &GameFilePath) -> anyhow::Result<Vec<u8>> {
            let path = path.resolve();
            std::fs::read(&path).map_err(|err| anyhow::anyhow!("{err} ({})", path.display()))
        }

        pub async fn read_to_string(path: &GameFilePath) -> anyhow::Result<String> {
            let path = path.resolve();
            std::fs::read_to_string(&path)
                .map_err(|err| anyhow::anyhow!("{err} ({})", path.display()))
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use js_sys::Reflect;
    use std::sync::Arc;
    use std::{path::PathBuf, sync::Mutex};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;

    use super::{FileManager, GameFilePath, GamePathMaker};
    use crate::thread::sleep_async;
    use crate::time::Duration;

    impl GameFilePath {
        pub fn resolve(&self) -> String {
            format!(
                "{}/{}",
                self.asset_server,
                self.root.join(&self.relative_path).display()
            )
        }
    }

    impl GamePathMaker {
        pub fn new(root: Option<PathBuf>, asset_server: String) -> Self {
            Self {
                root: root.unwrap_or_else(|| "".into()),
                asset_server,
            }
        }

        pub fn make<T>(&self, relative_path: T) -> GameFilePath
        where
            T: Into<PathBuf>,
        {
            GameFilePath {
                root: self.root.clone(),
                relative_path: relative_path.into(),
                asset_server: self.asset_server.clone(),
            }
        }
    }

    #[wasm_bindgen]
    extern "C" {
        type Global;

        #[wasm_bindgen(method, getter, js_name = Window)]
        fn window(this: &Global) -> JsValue;

        #[wasm_bindgen(method, getter, js_name = WorkerGlobalScope)]
        fn worker(this: &Global) -> JsValue;
    }

    impl FileManager {
        // TODO: implement read_to_end (https://doc.rust-lang.org/stable/std/io/trait.Read.html#method.read_to_end)
        //       on both native and web
        pub async fn read(path: &GameFilePath) -> anyhow::Result<Vec<u8>> {
            let url = path.resolve();
            let mut result = vec![];
            Self::read_whole_request_body(
                gloo_net::http::RequestBuilder::new(&url).method(gloo_net::http::Method::GET),
                &mut result,
            )
            .await?;
            Ok(result)
        }

        async fn read_whole_request_body(
            request: gloo_net::http::RequestBuilder,
            output_response_buffer: &mut Vec<u8>,
        ) -> anyhow::Result<u64> {
            Self::read_whole_request_body_internal(request, output_response_buffer)
                .await
                .map_err(|err| anyhow::anyhow!(err.as_string().unwrap_or_default()))
        }

        async fn read_whole_request_body_internal(
            request: gloo_net::http::RequestBuilder,
            output_response_buffer: &mut Vec<u8>,
        ) -> Result<u64, JsValue> {
            let response = request
                .send()
                .await
                .map_err(|err| JsValue::from_str(&format!("{:?}", err)))?;

            if let Some(content_length) = response
                .headers()
                .get("Content-Length")
                .and_then(|content_length_str| content_length_str.parse::<usize>().ok())
            {
                if (output_response_buffer.capacity() < content_length) {
                    output_response_buffer
                        .reserve_exact(content_length - output_response_buffer.capacity());
                }
            }

            let body_stream = response
                .body()
                .ok_or_else(|| JsValue::from_str("Request returned no body"))?;

            let mut bytes_read = 0;
            let mut body_reader = web_sys::ReadableStreamDefaultReader::new(&body_stream)?;
            loop {
                let body_chunk_result = JsFuture::from(body_reader.read()).await?;

                let done_reading_body =
                    Reflect::get(&body_chunk_result, &JsValue::from_str("done"))?
                        .as_bool()
                        .ok_or_else(|| {
                            JsValue::from_str(
                                "Failed to read property 'done' from stream read promise result",
                            )
                        })?;

                let mut body_chunk = js_sys::Uint8Array::new(&Reflect::get(
                    &body_chunk_result,
                    &JsValue::from_str("value"),
                )?)
                .to_vec();

                bytes_read += body_chunk.len() as u64;
                output_response_buffer.append(&mut body_chunk);

                if done_reading_body {
                    break;
                }
            }

            Ok(bytes_read)
        }

        pub async fn read_to_string(path: &GameFilePath) -> anyhow::Result<String> {
            let bytes = Self::read(path).await?;
            Ok(std::str::from_utf8(&bytes)?.to_string())
        }
    }

    /// Provides a File type that implements std::io::Read and std::io::Seek
    /// It cannot be used on the main thread since it can block, which produces a panic on the web
    #[derive(Debug, Clone)]
    pub struct HttpFileStreamerSync {
        pub(crate) inner: Arc<Mutex<HttpFileStreamerSyncInner>>,
    }

    #[derive(Debug, Clone)]
    pub(crate) struct HttpFileStreamerSyncInner {
        url: String,
        pub(crate) file_size: u64,
        current_file_position: u64,
        current_requested_chunk_size: u64,
        buffer: Vec<u8>,
        error: Option<String>,
    }

    impl HttpFileStreamerSync {
        pub async fn new(url: String) -> anyhow::Result<Self> {
            let head_response = gloo_net::http::RequestBuilder::new(&url)
                .method(gloo_net::http::Method::HEAD)
                .send()
                .await?;

            if head_response.headers().get("Accept-Ranges") != Some(String::from("bytes")) {
                anyhow::bail!("Resource at URL {} does not support streaming", url);
            }

            let file_size = head_response
                .headers()
                .get("Content-Length")
                .ok_or_else(|| anyhow::anyhow!("Didn't get content length header"))
                .and_then(|content_length_str| {
                    content_length_str
                        .parse::<u64>()
                        .map_err(|err| anyhow::anyhow!(err))
                })?;

            log::debug!("Initted http file streamer with file_size={:?}", file_size);

            let inner = Arc::new(Mutex::new(HttpFileStreamerSyncInner {
                url,
                file_size,
                current_file_position: 0,
                current_requested_chunk_size: 0,
                buffer: vec![],
                error: None,
            }));

            let inner_clone = inner.clone();

            crate::thread::spawn(move || {
                crate::block_on(async move { Self::run_producer_loop(inner_clone).await });
            });

            Ok(Self { inner })
        }

        async fn run_producer_loop(inner: Arc<Mutex<HttpFileStreamerSyncInner>>) {
            if let Err(err) = Self::run_producer_loop_inner(inner.clone()).await {
                inner.lock().unwrap().error = Some(err.as_string().unwrap_or_default());
            }
        }

        async fn run_producer_loop_inner(
            inner: Arc<Mutex<HttpFileStreamerSyncInner>>,
        ) -> Result<(), JsValue> {
            loop {
                // let this thread die if nobody else cares anymore (😥)
                if Arc::strong_count(&inner) == 1 {
                    log::debug!("Exiting http file streamer thread");
                    break Ok(());
                }

                let mut inner_guard = inner.lock().unwrap();

                // if there's no more work to do, we spin-wait for more
                if inner_guard.current_requested_chunk_size == 0 {
                    drop(inner_guard);

                    sleep_async(Duration::from_secs_f32(0.05)).await;
                    continue;
                }

                inner_guard.buffer.clear();
                let request = gloo_net::http::RequestBuilder::new(&inner_guard.url)
                    .method(gloo_net::http::Method::GET)
                    .header(
                        "Range",
                        &format!(
                            "bytes={}-{}",
                            inner_guard.current_file_position,
                            (inner_guard.current_file_position
                                + inner_guard.current_requested_chunk_size
                                - 1)
                            .min(inner_guard.file_size)
                        ),
                    );
                let bytes_read =
                    FileManager::read_whole_request_body_internal(request, &mut inner_guard.buffer)
                        .await?;
                inner_guard.current_file_position =
                    (inner_guard.current_file_position + bytes_read).min(inner_guard.file_size);
                inner_guard.current_requested_chunk_size = 0;
            }
        }
    }

    impl std::io::Read for HttpFileStreamerSync {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            if (buffer.len() == 0) {
                return Ok(0);
            }

            {
                let mut inner_guard = self.inner.lock().unwrap();

                if let Some(err) = inner_guard.error.take() {
                    return io_error(&err);
                }

                if inner_guard.current_file_position == inner_guard.file_size {
                    return Ok(0);
                }

                inner_guard.current_requested_chunk_size = buffer.len() as u64;
            }

            // wait for some data
            let mut inner_guard = loop {
                let inner_guard = self.inner.lock().unwrap();
                if inner_guard.current_requested_chunk_size == 0 {
                    break inner_guard;
                }
            };

            let bytes_read = inner_guard.buffer.len();

            // fill with zeros so copy_from_slice succeeds
            inner_guard.buffer.resize(buffer.len(), 0);
            buffer.copy_from_slice(&inner_guard.buffer);

            Ok(bytes_read)
        }
    }

    impl std::io::Seek for HttpFileStreamerSync {
        fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
            let mut inner_guard = self.inner.lock().unwrap();

            if let Some(err) = inner_guard.error.take() {
                return io_error(&err);
            }

            if inner_guard.current_requested_chunk_size != 0 {
                return io_error("Seeking in the middle of reading is not (yet) supported");
            }

            let new_file_position = match pos {
                std::io::SeekFrom::Start(start) => start as i64,
                std::io::SeekFrom::End(offset) => (inner_guard.file_size as i64 + offset),
                std::io::SeekFrom::Current(offset) => {
                    (inner_guard.current_file_position as i64 + offset)
                }
            };

            if new_file_position < 0 {
                return io_error(&format!(
                    "Seeking to an invalid position: {new_file_position}. content_length={}",
                    inner_guard.file_size
                ));
            }

            let new_file_position = new_file_position as u64;
            inner_guard.current_file_position = new_file_position;
            Ok(new_file_position)
        }
    }

    fn io_error<T>(err: &str) -> std::io::Result<T> {
        std::io::Result::Err(std::io::Error::new(std::io::ErrorKind::Other, err))
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(target_arch = "wasm32")]
pub use web::*;
