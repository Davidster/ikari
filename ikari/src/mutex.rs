mod std {
    use std::sync::Mutex as StdMutex;
    use std::sync::MutexGuard as StdMutexGuard;

    #[derive(Debug, Default)]
    pub struct Mutex<T: ?Sized>(StdMutex<T>);

    #[allow(dead_code)]
    impl<T> Mutex<T> {
        pub fn new(val: T) -> Mutex<T> {
            Mutex(StdMutex::new(val))
        }

        pub fn lock(&self) -> StdMutexGuard<T> {
            self.0
                .lock()
                .expect("Mutex poisoning is a fatal error in ikari")
        }

        pub fn try_lock(&self) -> Option<StdMutexGuard<T>> {
            self.0.try_lock().ok()
        }
    }

    impl<T> From<T> for Mutex<T> {
        fn from(value: T) -> Self {
            Mutex::new(value)
        }
    }
}

mod spin {
    use spin::Mutex as SpinMutex;
    use spin::MutexGuard as SpinMutexGuard;

    #[derive(Debug, Default)]
    pub struct Mutex<T: ?Sized>(SpinMutex<T>);

    #[allow(dead_code)]
    impl<T> Mutex<T> {
        pub fn new(val: T) -> Mutex<T> {
            Mutex(SpinMutex::new(val))
        }

        pub fn lock(&self) -> SpinMutexGuard<T> {
            self.0.lock()
        }

        pub fn try_lock(&self) -> Option<SpinMutexGuard<T>> {
            self.0.try_lock()
        }
    }

    impl<T> From<T> for Mutex<T> {
        fn from(value: T) -> Self {
            Mutex::new(value)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use std::*;

#[cfg(target_arch = "wasm32")]
pub use spin::*;
