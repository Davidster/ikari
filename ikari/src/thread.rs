#[cfg(target_arch = "wasm32")]
pub fn spawn<F, T>(f: F)
where
    F: FnOnce() -> T,
    F: Send + 'static,
    T: Send + 'static,
{
    f();
}

#[cfg(target_arch = "wasm32")]
pub fn sleep(_dur: std::time::Duration) {
    // TODO: log a warning here, no-op
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
pub fn sleep(dur: std::time::Duration) {
    std::thread::sleep(dur);
}
