#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::sync::Arc;

    /// does nothing on native
    pub struct WebCanvasManager;

    impl WebCanvasManager {
        pub fn new(_window: Arc<winit::window::Window>) -> Self {
            Self
        }

        pub fn on_update(&self, size: winit::dpi::PhysicalSize<u32>) {}
        // pub fn on_update(&self, _event: &winit::event::Event<()>) {}
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use std::sync::Arc;
    use wasm_bindgen::prelude::*;
    use winit::event::Event;
    use winit::event::WindowEvent;
    use winit::window::Window;

    pub struct WebCanvasManager {
        pub window: Arc<Window>,
        pub canvas_container: web_sys::HtmlElement,
    }

    impl WebCanvasManager {
        pub fn new(window: Arc<Window>) -> Self {
            let dom_window = web_sys::window().unwrap();
            let document = dom_window.document().unwrap();
            let canvas_container = document
                .get_element_by_id("canvas_container")
                .unwrap()
                .dyn_into::<web_sys::HtmlElement>()
                .unwrap();
            Self {
                window: window.clone(),
                canvas_container,
            }
        }

        pub fn on_update(&self, size: winit::dpi::PhysicalSize<u32>) {
            // let new_size = winit::dpi::PhysicalSize::new(
            //     (size.width as f64 * self.window.scale_factor()) as u32,
            //     (size.height as f64 * self.window.scale_factor()) as u32,
            // );
            // self.window.request_inner_size(size);
            let new_size = winit::dpi::PhysicalSize::new(
                (self.canvas_container.offset_width() as f64 * self.window.scale_factor()) as u32,
                (self.canvas_container.offset_height() as f64 * self.window.scale_factor()) as u32,
            );
            if self.window.inner_size() != new_size {
                let _resized_immediately = self.window.request_inner_size(new_size);
            }
        }

        // pub fn on_update(&self, event: &Event<()>) {
        //     match event {
        //         Event::WindowEvent {
        //             event: WindowEvent::RedrawRequested,
        //             ..
        //         } => {
        //             self.resize();
        //         }
        //         Event::LoopExiting => {
        //             self.canvas_container.remove();
        //         }
        //         Event::WindowEvent {
        //             event, window_id, ..
        //         } if *window_id == self.window.id() => {
        //             match &event {
        //                 WindowEvent::Resized(_) => {
        //                     self.resize();
        //                 }
        //                 WindowEvent::ScaleFactorChanged { .. } => {
        //                     self.resize();
        //                 }
        //                 _ => {}
        //             };
        //         }
        //         _ => {}
        //     };
        // }

        // fn resize(&self) {
        //     let new_size = winit::dpi::PhysicalSize::new(
        //         (self.canvas_container.offset_width() as f64 * self.window.scale_factor()) as u32,
        //         (self.canvas_container.offset_height() as f64 * self.window.scale_factor()) as u32,
        //     );
        //     if self.window.inner_size() != new_size {
        //         let _resized_immediately = self.window.request_inner_size(new_size);
        //     }
        // }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(target_arch = "wasm32")]
pub use web::*;
