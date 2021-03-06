use crate::{
    graphics::{GraphicsDevice, TexturedQuad},
    particle_system::ParticleSystem,
};
use std::time::{Duration, Instant};
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

mod graphics;
mod particle_system;

const TARGET_FPS: usize = 60;
const FRAME_DT: Duration = Duration::from_micros((1000000.0 / TARGET_FPS as f64) as u64);

async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title("Strange Attractors").build(&event_loop).unwrap();

    let mut graphics_device = GraphicsDevice::new(&window).await;
    let textured_quad = TexturedQuad::new(&graphics_device);
    let mut particle_system = ParticleSystem::new(&graphics_device);

    let mut last_frame_time = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::MainEventsCleared => {
                if last_frame_time.elapsed() >= FRAME_DT {
                    let now = Instant::now();
                    last_frame_time = now;

                    window.request_redraw();
                }
            },
            Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
                println!("Resizing to {}x{}", new_size.width, new_size.height);
                graphics_device.resize(new_size);

                window.request_redraw();
            },
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                },
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(virtual_code),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    if let VirtualKeyCode::Escape = virtual_code {
                        *control_flow = ControlFlow::Exit;
                    }
                },
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(virtual_code),
                            state: ElementState::Released,
                            ..
                        },
                    ..
                } => {
                    if let VirtualKeyCode::Escape = virtual_code {
                        *control_flow = ControlFlow::Exit;
                    }
                },
                _ => (),
            },
            Event::RedrawRequested(_window_id) => {
                // Draw the scene
                let mut frame_encoder = graphics_device.begin_frame();
                textured_quad.render(&mut frame_encoder);
                particle_system.render(&mut frame_encoder);
                frame_encoder.finish();
            },
            _ => (),
        }
    });
}

fn main() {
    pollster::block_on(run());
}
