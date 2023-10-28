use std::path::{Path, PathBuf};

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
    use std::path::PathBuf;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;

    use super::{FileManager, GameFilePath, GamePathMaker};

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
        pub async fn read(path: &GameFilePath) -> anyhow::Result<Vec<u8>> {
            let resolved_path = path.resolve();
            Self::read_internal(&resolved_path)
                .await
                .map(|js_value| js_sys::Uint8Array::new(&js_value).to_vec())
                .map_err(|err| {
                    anyhow::anyhow!(
                        "Error reading from url {}:\n{}",
                        resolved_path,
                        err.as_string().unwrap_or_default()
                    )
                })
        }

        async fn read_internal(url: &str) -> std::result::Result<JsValue, JsValue> {
            use web_sys::{Blob, RequestInit, Response};

            let mut opts = RequestInit::new();
            opts.method("GET");
            let request = web_sys::Request::new_with_str_and_init(url, &opts)?;

            let global: Global = js_sys::global().unchecked_into();
            let resp_value = JsFuture::from(if !global.window().is_undefined() {
                global
                    .unchecked_into::<web_sys::Window>()
                    .fetch_with_request(&request)
            } else if !global.worker().is_undefined() {
                let global_scope = global.unchecked_into::<web_sys::WorkerGlobalScope>();
                global_scope.fetch_with_request(&request)
            } else {
                panic!(
                    "this function is only supported on the main thread or from a dedicated worker"
                );
            })
            .await?;

            let resp: Response = resp_value.dyn_into().unwrap();
            if !resp.ok() {
                let status = resp.status();
                return std::result::Result::Err(JsValue::from(format!(
                    "Request to {url} responded with error status code: {status}"
                )));
            }

            let blob = JsFuture::from(resp.blob()?).await?;
            let array_buffer: JsValue = JsFuture::from(Blob::from(blob).array_buffer()).await?;

            std::result::Result::Ok(array_buffer)
        }

        pub async fn read_to_string(path: &GameFilePath) -> anyhow::Result<String> {
            let bytes = Self::read(path).await?;
            Ok(std::str::from_utf8(&bytes)?.to_string())
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(target_arch = "wasm32")]
pub use web::*;
