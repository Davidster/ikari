use crate::time::*;

#[cfg(target_arch = "wasm32")]
pub fn spawn<F, T>(f: F)
where
    F: FnOnce() -> T,
    F: Send + 'static,
    T: Send + 'static,
{
    wasm_thread::spawn(|| {
        f();
        wasm_bindgen::throw_str("Cursed hack to keep workers alive. See https://github.com/rustwasm/wasm-bindgen/issues/2945");
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub fn spawn<F, T>(f: F)
where
    F: FnOnce() -> T,
    F: Send + 'static,
    T: Send + 'static,
{
    std::thread::spawn(f);
}

#[cfg(not(target_arch = "wasm32"))]
pub fn sleep(dur: Duration) {
    std::thread::sleep(dur);
}

pub async fn sleep_async(dur: Duration) {
    async_std::task::sleep(dur).await;
}
