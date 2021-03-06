use bytemuck::{Pod, Zeroable};
use wgpu::{
    util::DeviceExt, Adapter, BackendBit, BindGroup, Buffer, CommandEncoder, Device, Instance,
    Queue, RenderPipeline, Surface, SwapChain, SwapChainDescriptor, SwapChainTexture,
};
use winit::{dpi::PhysicalSize, window::Window};

// pub mod text;

const CORNFLOWER_BLUE: wgpu::Color =
    wgpu::Color { r: 100.0 / 255.0, g: 149.0 / 255.0, b: 237.0 / 255.0, a: 1.0 };

pub struct GraphicsDevice {
    adapter: Adapter,
    device: Device,
    queue: Queue,
    surface: Surface,
    swap_chain_descriptor: SwapChainDescriptor,
    swap_chain: SwapChain,
}

impl GraphicsDevice {
    pub async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // PRIMARY: All the apis that wgpu offers first tier of support for (Vulkan + Metal + DX12 + Browser WebGPU).
        let instance = Instance::new(BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let swapchain_format = wgpu::TextureFormat::Bgra8Unorm;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                // Prefer low power when on battery, high performance when on mains.
                power_preference: wgpu::PowerPreference::default(),
                // Request an adapter which can render to our surface
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an appropiate adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let swap_chain_descriptor = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Mailbox,
        };

        let swap_chain = device.create_swap_chain(&surface, &swap_chain_descriptor);

        Self { adapter, device, queue, surface, swap_chain_descriptor, swap_chain }
    }

    pub fn load_shader(&self, shader_src: &'static str) -> wgpu::ShaderModule {
        let mut flags = wgpu::ShaderFlags::VALIDATION;
        match self.adapter().get_info().backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Metal => {
                flags |= wgpu::ShaderFlags::EXPERIMENTAL_TRANSLATION;
            },
            _ => {},
        }

        self.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_src)),
            flags,
        })
    }

    pub fn begin_frame(&mut self) -> FrameEncoder {
        let frame = self
            .swap_chain
            .get_current_frame()
            .expect("Failed to acquire next swap chain texture")
            .output;

        let encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        FrameEncoder { queue: &mut self.queue, frame, encoder }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.swap_chain_descriptor.width = new_size.width;
        self.swap_chain_descriptor.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.swap_chain_descriptor);
    }

    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn swap_chain_descriptor(&self) -> &SwapChainDescriptor {
        &self.swap_chain_descriptor
    }
}

pub struct FrameEncoder<'a> {
    queue: &'a mut Queue,
    pub frame: SwapChainTexture,
    pub encoder: CommandEncoder,
}

impl<'a> FrameEncoder<'a> {
    pub fn queue(&mut self) -> &mut Queue {
        &mut self.queue
    }

    // TODO(bschwind) - Maybe do this in a Drop impl
    pub fn finish(self) {
        self.queue.submit(Some(self.encoder.finish()));
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TexturedQuadVertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

pub struct TexturedQuad {
    vertex_buf: Buffer,
    index_buf: Buffer,
    bind_group: BindGroup,
    pipeline: RenderPipeline,
}

impl TexturedQuad {
    pub fn new(graphics_device: &GraphicsDevice) -> Self {
        let vertex_data = vec![
            TexturedQuadVertex { pos: [-1.0, -1.0], uv: [0.0, 1.0] },
            TexturedQuadVertex { pos: [-1.0, 1.0], uv: [0.0, 0.0] },
            TexturedQuadVertex { pos: [1.0, 1.0], uv: [1.0, 0.0] },
            TexturedQuadVertex { pos: [1.0, -1.0], uv: [1.0, 1.0] },
        ];

        let index_data = vec![0u16, 1, 3, 2];

        let device = graphics_device.device();

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsage::VERTEX,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsage::INDEX,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[],
            label: None,
        });

        let draw_shader =
            graphics_device.load_shader(include_str!("../../resources/shaders/draw.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),

            vertex: wgpu::VertexState {
                module: &draw_shader,
                entry_point: "main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: (std::mem::size_of::<TexturedQuadVertex>())
                        as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        // Pos (vec2)
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float2,
                            offset: 0,
                            shader_location: 0,
                        },
                        // UV (vec2)
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float2,
                            offset: 2 * 4,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &draw_shader,
                entry_point: "main",
                targets: &[wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8Unorm,
                    alpha_blend: wgpu::BlendState::REPLACE,
                    color_blend: wgpu::BlendState::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint16),
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Front,
                polygon_mode: wgpu::PolygonMode::Fill,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        });

        Self { vertex_buf, index_buf, pipeline, bind_group }
    }

    pub fn render(&self, frame_encoder: &mut FrameEncoder) {
        let frame = &frame_encoder.frame;
        let encoder = &mut frame_encoder.encoder;

        encoder.push_debug_group("TexturedQuad::render");
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CORNFLOWER_BLUE),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
            rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            rpass.draw_indexed(0..4 as u32, 0, 0..1);
        }
        encoder.pop_debug_group();
    }
}
