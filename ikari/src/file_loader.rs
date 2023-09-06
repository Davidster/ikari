use std::path::{Path, PathBuf};

lazy_static::lazy_static! {
    pub static ref IKARI_PATH_MAKER: GamePathMaker = GamePathMaker::new(Some(PathBuf::from("ikari")));
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GameFilePath {
    root: PathBuf,
    pub relative_path: PathBuf,
}

impl GameFilePath {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn resolve(&self) -> PathBuf {
        self.root.join(&self.relative_path)
    }
}

pub struct GamePathMaker {
    root: PathBuf,
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

pub struct FileLoader;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use crate::file_loader::{FileLoader, GameFilePath};

    impl FileLoader {
        pub async fn read(path: &GameFilePath) -> anyhow::Result<Vec<u8>> {
            let path = path.resolve();
            std::fs::read(&path).map_err(|err| anyhow::anyhow!("{err} ({})", path.display()))
        }

        pub async fn read_to_string(path: &GameFilePath) -> anyhow::Result<String> {
            let path = path.resolve();
            std::fs::read_to_string(&path)
                .map_err(|err| anyhow::anyhow!("{err} ({})", path.display()))
        }

        pub fn open_file(path: &GameFilePath) -> anyhow::Result<std::fs::File> {
            let path = path.resolve();
            std::fs::File::open(&path).map_err(|err| anyhow::anyhow!("{err} ({})", path.display()))
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use wasm_bindgen::prelude::*;

    use crate::file_loader::{FileLoader, GameFilePath};

    #[wasm_bindgen]
    extern "C" {
        type Global;

        #[wasm_bindgen(method, getter, js_name = Window)]
        fn window(this: &Global) -> JsValue;

        #[wasm_bindgen(method, getter, js_name = WorkerGlobalScope)]
        fn worker(this: &Global) -> JsValue;
    }

    async fn run_promise(promise: js_sys::Promise) -> anyhow::Result<JsValue> {
        map_js_err(wasm_bindgen_futures::JsFuture::from(promise).await)
    }

    fn map_js_err<T>(result: std::result::Result<T, JsValue>) -> anyhow::Result<T> {
        result.map_err(|err| anyhow::anyhow!(err.as_string().unwrap_or_default()))
    }

    // TODO: add the path to all error messages here?
    impl FileLoader {
        pub async fn read(path: &GameFilePath) -> anyhow::Result<Vec<u8>> {
            const ASSET_SERVER: &str = "http://localhost:8000";

            use web_sys::{Blob, RequestInit, Response};

            let mut opts = RequestInit::new();
            opts.method("GET");
            let path = path.resolve();
            let url = format!("{ASSET_SERVER}/{}", path.display());
            let request = map_js_err(web_sys::Request::new_with_str_and_init(&url, &opts))?;

            let global: Global = js_sys::global().unchecked_into();
            let resp_value = run_promise(if !global.window().is_undefined() {
                global
                    .unchecked_into::<web_sys::Window>()
                    .fetch_with_request(&request)
            } else if !global.worker().is_undefined() {
                let global_scope = global.unchecked_into::<web_sys::WorkerGlobalScope>();
                global_scope.fetch_with_request(&request)
            } else {
                panic!(
                    "read function is only supported on the main thread or from a dedicated worker"
                );
            })
            .await?;

            let resp: Response = resp_value.dyn_into().unwrap();
            if !resp.ok() {
                let status = resp.status();
                anyhow::bail!("Request to {url} responded with error status code: {status}")
            }

            let blob = run_promise(map_js_err(resp.blob())?).await?;
            let array_buffer: JsValue = run_promise(Blob::from(blob).array_buffer()).await?;

            anyhow::Ok(js_sys::Uint8Array::new(&array_buffer).to_vec())
        }

        pub async fn read_to_string(path: &GameFilePath) -> anyhow::Result<String> {
            let bytes = FileLoader::read(path).await?;
            Ok(std::str::from_utf8(&bytes)?.to_string())
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(target_arch = "wasm32")]
pub use web::*;
