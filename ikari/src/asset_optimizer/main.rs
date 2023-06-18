#[cfg(not(target_arch = "wasm32"))]
mod asset_optimizer;

#[cfg(target_arch = "wasm32")]
fn main() {
    todo!("texture compression not yet supported in wasm");
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    asset_optimizer::run();
}
