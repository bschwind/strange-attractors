use bytemuck::{Pod, Zeroable};
use glam::{vec3, Mat4};
use rand::Rng;
use simple_game::{
    glam,
    graphics::{FrameEncoder, GraphicsDevice},
    wgpu,
};
use std::{convert::TryInto, mem};
use wgpu::{util::DeviceExt, ComputePipeline, RenderPipeline};

const NUM_PARTICLES: usize = 1200000;
const PARTICLES_PER_GROUP: u32 = 256;

struct Buffers {
    particles: [wgpu::Buffer; 2],
    triangle_vertex: wgpu::Buffer,
    compute_uniform: wgpu::Buffer,
    vertex_uniform: wgpu::Buffer,
}

struct BindGroups {
    particles: [wgpu::BindGroup; 2],
    compute_uniform: wgpu::BindGroup,
    vertex_uniform: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, Pod, Zeroable)]
struct Consts {
    algo: u32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct Particle {
    /// XYZ position of the particle (w unused)
    pos: [f32; 4],
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, Pod, Zeroable)]
struct VertexUniforms {
    proj: Mat4,
    consts: Consts,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct TriangleVertex {
    /// XYZ position of the triangle vertex
    pos: [f32; 3],
}

pub struct ParticleSystem {
    compute_pipeline: ComputePipeline,
    render_pipeline: RenderPipeline,
    buffers: Buffers,
    bind_groups: BindGroups,
    consts: Consts,
    frame_counter: usize,
    work_group_count: u32,
}

impl ParticleSystem {
    pub fn new(graphics_device: &GraphicsDevice) -> Self {
        let compute_pipeline = Self::build_compute_pipeline(graphics_device);
        let render_pipeline = Self::build_render_pipeline(graphics_device);
        let buffers = Self::build_buffers(graphics_device);
        let bind_groups =
            Self::build_bind_groups(graphics_device, &compute_pipeline, &render_pipeline, &buffers);

        let consts = Consts::default();
        let frame_counter = 0;
        let work_group_count =
            ((NUM_PARTICLES as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;

        Self {
            compute_pipeline,
            render_pipeline,
            buffers,
            bind_groups,
            consts,
            frame_counter,
            work_group_count,
        }
    }

    fn update_compute_uniforms(&mut self, frame_encoder: &mut FrameEncoder) {
        // TODO - Update state from a MIDI controller.
        // let (algo, [a, b, c, d, e, f, g, _]) = self.midi_state.read().unwrap().clone();
        self.consts = Consts { a: 0.0, b: 0.1, ..Consts::default() };

        frame_encoder.queue().write_buffer(
            &self.buffers.compute_uniform,
            0,
            bytemuck::bytes_of(&self.consts),
        )
    }

    fn update_vertex_uniforms(&mut self, frame_encoder: &mut FrameEncoder) {
        let (width, height) = frame_encoder.surface_dimensions();
        let uniforms =
            VertexUniforms { proj: Self::build_camera_matrix(width, height), consts: self.consts };

        frame_encoder.queue().write_buffer(
            &self.buffers.vertex_uniform,
            0,
            bytemuck::bytes_of(&uniforms),
        )
    }

    pub fn render(&mut self, frame_encoder: &mut FrameEncoder) {
        self.update_compute_uniforms(frame_encoder);
        self.update_vertex_uniforms(frame_encoder);
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
            compute_pass.set_bind_group(
                0,
                &self.bind_groups.particles[self.frame_counter % 2],
                &[],
            );
            compute_pass.set_bind_group(1, &self.bind_groups.compute_uniform, &[]);
            compute_pass.dispatch(self.work_group_count, 1, 1);
        }
        encoder.pop_debug_group();
    }

    fn run_render(&self, frame_encoder: &mut FrameEncoder) {
        let encoder = &mut frame_encoder.encoder;

        encoder.push_debug_group("Particle System Render");
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &frame_encoder.backbuffer_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: true },
                }],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            // render particles from the dst buffer
            render_pass.set_vertex_buffer(
                0,
                self.buffers.particles[(self.frame_counter + 1) % 2].slice(..),
            );
            render_pass.set_bind_group(0, &self.bind_groups.vertex_uniform, &[]);
            // The three instance-local vertices
            render_pass.set_vertex_buffer(1, self.buffers.triangle_vertex.slice(..));
            render_pass.draw(0..3, 0..NUM_PARTICLES as u32);
        }
        encoder.pop_debug_group();
    }

    fn build_compute_pipeline(graphics_device: &GraphicsDevice) -> ComputePipeline {
        let device = graphics_device.device();

        let compute_shader = graphics_device
            .load_wgsl_shader(include_str!("../resources/shaders/strange_attractor_compute.wgsl"));

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
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
                        visibility: wgpu::ShaderStages::COMPUTE,
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
                    visibility: wgpu::ShaderStages::COMPUTE,
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

    fn build_render_pipeline(graphics_device: &GraphicsDevice) -> RenderPipeline {
        let device = graphics_device.device();

        let draw_shader = graphics_device
            .load_wgsl_shader(include_str!("../resources/shaders/strange_attractor_render.wgsl"));

        let vertex_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            mem::size_of::<VertexUniforms>() as u64
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
                entry_point: "main_vs",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: mem::size_of::<Particle>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x4],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: mem::size_of::<TriangleVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x3],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &draw_shader,
                entry_point: "main_fs",
                targets: &[graphics_device.surface_config().format.into()],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        render_pipeline
    }

    fn build_buffers(graphics_device: &GraphicsDevice) -> Buffers {
        Buffers {
            particles: Self::build_particle_buffers(graphics_device),
            triangle_vertex: Self::build_triangle_vertex_buffer(graphics_device),
            compute_uniform: Self::build_compute_uniform_buffer(graphics_device),
            vertex_uniform: Self::build_vertex_uniform_buffer(graphics_device),
        }
    }

    fn build_compute_uniform_buffer(graphics_device: &GraphicsDevice) -> wgpu::Buffer {
        let device = graphics_device.device();
        let consts = Consts::default();

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle system compute shader uniform buffer"),
            contents: bytemuck::bytes_of(&consts),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn build_vertex_uniform_buffer(graphics_device: &GraphicsDevice) -> wgpu::Buffer {
        let (width, height) = graphics_device.surface_dimensions();
        let device = graphics_device.device();

        let uniforms = VertexUniforms {
            proj: Self::build_camera_matrix(width, height),
            consts: Consts::default(),
        };

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle system vertex shader uniform buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn build_bind_groups(
        graphics_device: &GraphicsDevice,
        compute_pipeline: &ComputePipeline,
        render_pipeline: &RenderPipeline,
        buffers: &Buffers,
    ) -> BindGroups {
        let device = graphics_device.device();
        let mut particle_bind_groups = Vec::with_capacity(2);

        for i in 0..2 {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &compute_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.particles[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.particles[(i + 1) % 2].as_entire_binding(), // bind to opposite buffer
                    },
                ],
                label: None,
            });

            particle_bind_groups.push(bind_group);
        }

        let compute_uniform = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_pipeline.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.compute_uniform.as_entire_binding(),
            }],
            label: None,
        });

        let vertex_uniform = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &render_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.vertex_uniform.as_entire_binding(),
            }],
            label: None,
        });

        BindGroups {
            particles: particle_bind_groups.try_into().unwrap(),
            compute_uniform,
            vertex_uniform,
        }
    }

    fn build_particle_buffers(graphics_device: &GraphicsDevice) -> [wgpu::Buffer; 2] {
        let device = graphics_device.device();
        let mut particles = vec![Particle { pos: [0.0, 0.0, 0.0, 0.0] }; NUM_PARTICLES];

        let mut rng = rand::thread_rng();
        let now = std::time::Instant::now();
        for particle in &mut particles {
            particle.pos = rng.gen();
            particle.pos[0] = (particle.pos[0] - 0.5) * 2.0;
            particle.pos[1] = (particle.pos[1] - 0.5) * 2.0;
            particle.pos[2] = (particle.pos[2] - 0.5) * 2.0;
            particle.pos[3] = 1.0;
        }

        println!("generated particles in {}ms", now.elapsed().as_millis());

        let mut particle_buffers = vec![];

        // Create "ping-pong" buffers so the compute shader can alternate
        // between reading from a source buffer and writing to a destination buffer.
        for i in 0..2 {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Particle Buffer {}", i)),
                contents: bytemuck::cast_slice(&particles),
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            });

            particle_buffers.push(buffer);
        }

        particle_buffers.try_into().unwrap()
    }

    fn build_camera_matrix(width: u32, height: u32) -> Mat4 {
        let aspect_ratio = width as f32 / height as f32;
        let proj = Mat4::perspective_rh(std::f32::consts::PI / 2.0, aspect_ratio, 0.01, 1000.0);

        let view = Mat4::look_at_rh(
            vec3(1.0, 1.0, -1.0) * 4.0, // Eye position
            vec3(0.0, 0.0, 0.0),        // Look-at target
            vec3(0.0, 1.0, 0.0),        // Up vector of the camera
        );

        proj * view
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
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        })
    }
}
