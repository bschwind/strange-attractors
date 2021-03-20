use crate::{graphics::FrameEncoder, GraphicsDevice};
use bytemuck::{Pod, Zeroable};
use glam::{vec3, Mat4};
use std::{mem, slice};
use wgpu::{util::DeviceExt, ComputePipeline, RenderPipeline};

const NUM_PARTICLES: usize = 1000000;
const PARTICLES_PER_GROUP: u32 = 512;

pub struct ParticleSystem {
    compute_pipeline: ComputePipeline,
    render_pipeline: RenderPipeline,
    particle_buffers: Vec<wgpu::Buffer>,
    triangle_vertex_buffer: wgpu::Buffer,
    particle_bind_groups: Vec<wgpu::BindGroup>,
    vertex_shader_bind_group: wgpu::BindGroup,
    uniform_bind_group: wgpu::BindGroup,
    consts: Consts,
    uniform_buffer: wgpu::Buffer,
    frame_counter: usize,
    work_group_count: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct Particle {
    /// XYZ position of the particle (w unused)
    pos: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct Consts {
    algo: u32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    _align: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct TriangleVertex {
    /// XYZ position of the triangle vertex
    pos: [f32; 3],
}

impl ParticleSystem {
    pub fn new(graphics_device: &GraphicsDevice) -> Self {
        let compute_pipeline = Self::build_compute_pipeline(graphics_device);
        let (particle_buffers, particle_bind_groups, consts, uniform_buffer, uniform_bind_group) =
            Self::build_particle_buffers(
                graphics_device,
                &compute_pipeline.get_bind_group_layout(0),
                &compute_pipeline.get_bind_group_layout(1),
            );
        let (render_pipeline, vertex_bind_group_layout) =
            Self::build_render_pipeline(graphics_device);
        let vertex_shader_bind_group =
            Self::build_vertex_shader_bind_group(graphics_device, &vertex_bind_group_layout);

        let triangle_vertex_buffer = Self::build_triangle_vertex_buffer(graphics_device);

        let frame_counter = 0;
        let work_group_count =
            ((NUM_PARTICLES as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;

        Self {
            compute_pipeline,
            render_pipeline,
            particle_buffers,
            triangle_vertex_buffer,
            particle_bind_groups,
            vertex_shader_bind_group,
            frame_counter,
            work_group_count,
            consts,
            uniform_buffer,
            uniform_bind_group,
        }
    }

    pub fn render(&mut self, frame_encoder: &mut FrameEncoder) {
        self.run_compute(frame_encoder);
        self.run_render(frame_encoder);
        self.frame_counter += 1;
    }

    fn run_compute(&self, frame_encoder: &mut FrameEncoder) {
        let encoder = &mut frame_encoder.encoder;

        encoder.push_debug_group("Particle System Compute");
        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.particle_bind_groups[self.frame_counter % 2], &[]);
            compute_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
            compute_pass.dispatch(self.work_group_count, 1, 1);
        }
        encoder.pop_debug_group();
    }

    fn run_render(&self, frame_encoder: &mut FrameEncoder) {
        let frame = &frame_encoder.frame;
        let encoder = &mut frame_encoder.encoder;

        encoder.push_debug_group("Particle System Render");
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            // render particles from the dst buffer
            render_pass.set_vertex_buffer(
                0,
                self.particle_buffers[(self.frame_counter + 1) % 2].slice(..),
            );
            render_pass.set_bind_group(0, &self.vertex_shader_bind_group, &[]);
            // the three instance-local vertices
            render_pass.set_vertex_buffer(1, self.triangle_vertex_buffer.slice(..));
            render_pass.draw(0..3, 0..NUM_PARTICLES as u32);
        }
        encoder.pop_debug_group();
    }

    fn build_compute_pipeline(graphics_device: &GraphicsDevice) -> ComputePipeline {
        let device = graphics_device.device();

        let compute_shader = graphics_device
            .load_shader(include_str!("../resources/shaders/strange_attractor_compute.wgsl"));

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (NUM_PARTICLES * mem::size_of::<Particle>()) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (NUM_PARTICLES * mem::size_of::<Particle>()) as _,
                            ),
                        },
                        count: None,
                    },
                ],
                label: None,
            });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(mem::size_of::<Consts>() as _),
                    },
                    count: None,
                }],
                label: None,
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle system compute"),
                bind_group_layouts: &[&compute_bind_group_layout, &uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        })
    }

    fn build_particle_buffers(
        graphics_device: &GraphicsDevice,
        compute_bind_group_layout: &wgpu::BindGroupLayout,
        uniform_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> (Vec<wgpu::Buffer>, Vec<wgpu::BindGroup>, Consts, wgpu::Buffer, wgpu::BindGroup) {
        let device = graphics_device.device();

        let mut particles = vec![Particle { pos: [0.0, 0.0, 0.0, 0.0] }; NUM_PARTICLES];

        for particle in &mut particles {
            particle.pos[0] = 2.0 * (rand::random::<f32>() - 0.5); // posx
            particle.pos[1] = 2.0 * (rand::random::<f32>() - 0.5); // posy
            particle.pos[2] = 2.0 * (rand::random::<f32>() - 0.5); // posz
            particle.pos[3] = 1.0;
        }

        let mut particle_buffers = vec![];
        let mut bind_groups = vec![];

        let consts = Consts { algo: 0, a: 0.0, b: 0.19, c: 0.0, d: 0.0, _align: [0f32; 3] };
        let p: *const Consts = &consts; // the same operator is used as with references
        let s: &[u8] = unsafe { slice::from_raw_parts(p as *const u8, mem::size_of::<Consts>()) };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle system compute shader uniform buffer"),
            contents: s,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: None,
        });

        // Create "ping-pong" buffers so the compute shader can alternate
        // between reading from a source buffer and writing to a destination buffer.
        for i in 0..2 {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Particle Buffer {}", i)),
                contents: bytemuck::cast_slice(&particles),
                usage: wgpu::BufferUsage::VERTEX
                    | wgpu::BufferUsage::STORAGE
                    | wgpu::BufferUsage::COPY_DST,
            });

            particle_buffers.push(buffer);
        }

        for i in 0..2 {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particle_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_buffers[(i + 1) % 2].as_entire_binding(), // bind to opposite buffer
                    },
                ],
                label: None,
            });

            bind_groups.push(bind_group);
        }

        (particle_buffers, bind_groups, consts, uniform_buffer, uniform_bind_group)
    }

    fn build_render_pipeline(
        graphics_device: &GraphicsDevice,
    ) -> (RenderPipeline, wgpu::BindGroupLayout) {
        let device = graphics_device.device();

        let draw_shader = graphics_device
            .load_shader(include_str!("../resources/shaders/strange_attractor_render.wgsl"));

        let vertex_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            mem::size_of::<[[f32; 4]; 4]>() as u64
                        ),
                    },
                    count: None,
                }],
                label: None,
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle system render"),
                bind_group_layouts: &[&vertex_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &draw_shader,
                entry_point: "main",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: mem::size_of::<Particle>() as u64,
                        step_mode: wgpu::InputStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float4],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: mem::size_of::<TriangleVertex>() as u64,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![1 => Float3],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &draw_shader,
                entry_point: "main",
                targets: &[graphics_device.swap_chain_descriptor().format.into()],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
        });

        (render_pipeline, vertex_bind_group_layout)
    }

    fn build_camera_matrix() -> Mat4 {
        let aspect_ratio = 1.0;
        let proj = Mat4::perspective_rh(std::f32::consts::PI / 2.0, aspect_ratio, 0.01, 1000.0);

        let view = Mat4::look_at_rh(
            vec3(1.0, 1.0, -1.0) * 4.0, // Eye position
            vec3(0.0, 0.0, 0.0),        // Look-at target
            vec3(0.0, 1.0, 0.0),        // Up vector of the camera
        );

        proj * view
    }

    fn build_vertex_shader_bind_group(
        graphics_device: &GraphicsDevice,
        vertex_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        let device = graphics_device.device();

        let camera_matrix = Self::build_camera_matrix();

        let vertex_shader_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Particle system vertex shader uniform buffer"),
                contents: bytemuck::cast_slice(camera_matrix.as_ref()),
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            });

        let vertex_shader_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &vertex_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: vertex_shader_uniform_buffer.as_entire_binding(),
            }],
            label: None,
        });

        vertex_shader_bind_group
    }

    fn build_triangle_vertex_buffer(graphics_device: &GraphicsDevice) -> wgpu::Buffer {
        let device = graphics_device.device();
        let vertex_buffer_data = [
            TriangleVertex { pos: [-0.005, -0.005, 0.0] },
            TriangleVertex { pos: [0.005, -0.005, 0.0] },
            TriangleVertex { pos: [0.00, 0.005, 0.0] },
        ];

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle system triangle vertex buffer"),
            contents: bytemuck::bytes_of(&vertex_buffer_data),
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        })
    }
}
