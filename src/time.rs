use std::ops::Deref;

#[derive(Debug, Copy, Clone)]
pub struct Instant(instant::Instant);

impl Deref for Instant {
    type Target = instant::Instant;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn now() -> Instant {
    Instant(instant::Instant::now())
}
