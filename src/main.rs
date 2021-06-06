use crate::particle_system::ParticleSystem;
use simple_game::{
    graphics::{
        text::{AxisAlign, StyledText, TextAlignment, TextSystem},
        FrameEncoder, GraphicsDevice,
    },
    util::FPSCounter,
    winit::window::Window,
    GameApp, WindowDimensions,
};

mod particle_system;

struct StrangeAttractorSim {
    particle_system: ParticleSystem,
    text_system: TextSystem,
    fps_counter: FPSCounter,
}

impl GameApp for StrangeAttractorSim {
    fn init(graphics_device: &mut GraphicsDevice) -> Self {
        Self {
            particle_system: ParticleSystem::new(graphics_device),
            text_system: TextSystem::new(graphics_device),
            fps_counter: FPSCounter::new(),
        }
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
            frame_encoder,
        );

        self.fps_counter.tick();
    }
}

fn main() {
    simple_game::run_game_app::<StrangeAttractorSim>();
}
