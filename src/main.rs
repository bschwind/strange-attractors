use crate::particle_system::ParticleSystem;
use simple_game::{
    graphics::{
        text::{AxisAlign, StyledText, TextAlignment, TextSystem},
        FrameEncoder, GraphicsDevice,
    },
    util::FPSCounter,
    GameApp, WindowDimensions,
};
use winit::window::Window;

mod particle_system;

struct StrangeAttractorSim {
    particle_system: ParticleSystem,
    text_system: TextSystem,
    fps_counter: FPSCounter,
}

impl GameApp for StrangeAttractorSim {
    fn init(graphics_device: &mut GraphicsDevice) -> Self {
        let (screen_width, screen_height) = graphics_device.surface_dimensions();
        let device = graphics_device.device();
        let surface_texture_format = graphics_device.surface_texture_format();

        Self {
            particle_system: ParticleSystem::new(
                device,
                surface_texture_format,
                screen_width,
                screen_height,
            ),
            text_system: TextSystem::new(
                device,
                surface_texture_format,
                screen_width,
                screen_height,
            ),
            fps_counter: FPSCounter::new(),
        }
    }

    fn resize(&mut self, _graphics_device: &mut GraphicsDevice, width: u32, height: u32) {
        self.text_system.resize(width, height);
        self.particle_system.resize(width, height);
    }

    fn window_title() -> &'static str {
        "Strange Attractors"
    }

    fn window_dimensions() -> WindowDimensions {
        WindowDimensions::Windowed(1280, 720)
    }

    fn desired_fps() -> usize {
        60
    }

    fn tick(&mut self, _dt: f32) {}

    fn render(&mut self, frame_encoder: &mut FrameEncoder, _window: &Window) {
        self.particle_system.render(frame_encoder);

        self.text_system.render_horizontal(
            TextAlignment {
                x: AxisAlign::Start(10),
                y: AxisAlign::Start(10),
                max_width: None,
                max_height: None,
            },
            &[StyledText::default_styling(&format!("FPS: {}", self.fps_counter.fps()))],
            &mut frame_encoder.encoder,
            &frame_encoder.backbuffer_view,
            frame_encoder.queue,
        );

        self.fps_counter.tick();
    }
}

fn main() {
    simple_game::run_game_app::<StrangeAttractorSim>();
}
