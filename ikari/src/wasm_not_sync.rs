/// inspired by https://crates.io/crates/maybe-sync

#[cfg(not(target_arch = "wasm32"))]
mod sync {
    pub trait WasmNotSend: Send {}

    impl<T: Send> WasmNotSend for T {}

    pub trait WasmNotSync: Sync {}

    impl<T: Sync> WasmNotSync for T {}
}

#[cfg(target_arch = "wasm32")]
mod unsync {
    pub trait WasmNotSend {}

    impl<T> WasmNotSend for T {}

    pub trait WasmNotSync {}

    impl<T> WasmNotSync for T {}
}

#[cfg(not(target_arch = "wasm32"))]
pub use sync::*;

#[cfg(target_arch = "wasm32")]
pub use unsync::*;
