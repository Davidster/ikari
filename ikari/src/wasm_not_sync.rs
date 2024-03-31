/// inspired by https://crates.io/crates/maybe-sync
/// uses Arc/Mutex/Send/Sync on all targets except uses Rc/RefCell on wasm
/// used for types that are not Send/Sync on wasm, such as many of the wgpu types

#[cfg(not(target_arch = "wasm32"))]
mod sync {
    pub trait WasmNotSend: Send {}

    impl<T: Send> WasmNotSend for T {}

    pub trait WasmNotSync: Sync {}

    impl<T: Sync> WasmNotSync for T {}

    pub type WasmNotArc<T> = std::sync::Arc<T>;

    pub type WasmNotMutex<T> = crate::mutex::Mutex<T>;
}

#[cfg(target_arch = "wasm32")]
mod unsync {
    use core::cell::{RefCell, RefMut};

    pub trait WasmNotSend {}

    impl<T> WasmNotSend for T {}

    pub trait WasmNotSync {}

    impl<T> WasmNotSync for T {}

    pub struct WasmNotMutex<T: ?Sized> {
        cell: RefCell<T>,
    }

    pub type WasmNotArc<T> = std::rc::Rc<T>;

    impl<T> WasmNotMutex<T> {
        pub fn new(value: T) -> Self {
            Self {
                cell: RefCell::new(value),
            }
        }
    }

    impl<T> WasmNotMutex<T> {
        #[allow(clippy::result_unit_err)]
        pub fn lock(&self) -> RefMut<T> {
            self.cell.borrow_mut()
        }

        pub fn try_lock(&self) -> Option<RefMut<T>> {
            self.cell.try_borrow_mut().ok()
        }

        pub fn get_mut(&mut self) -> &mut T {
            self.cell.get_mut()
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use sync::*;

#[cfg(target_arch = "wasm32")]
pub use unsync::*;
