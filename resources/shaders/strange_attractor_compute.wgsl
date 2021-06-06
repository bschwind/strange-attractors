let NUM_PARTICLES: u32 = 1000000u32;
let B: f32 = 0.19;

let DT: f32 = 0.033333333;

struct Particle {
    pos : vec4<f32>;
};

[[block]]
struct Particles {
    particles: [[stride(16)]] array<Particle>;
};

[[group(0), binding(0)]] var<storage> particles_src : [[access(read)]] Particles;
[[group(0), binding(1)]] var<storage> particles_dst : [[access(read_write)]] Particles;

[[stage(compute), workgroup_size(512)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    let index: u32 = global_invocation_id.x;

    // TODO - Wait until wgpu-rs uses a more recent version of naga which supports arrayLength.
    // let total = arrayLength(&particles_src.particles);

    if (index >= NUM_PARTICLES) {
        return;
    }

    let vPos : vec4<f32> = particles_src.particles[index].pos;
    let x0: f32 = vPos.x;
    let y0: f32 = vPos.y;
    let z0: f32 = vPos.z;

    let dx: f32 = (-B * x0 + sin(y0)) * DT;
    let dy: f32 = (-B * y0 + sin(z0)) * DT;
    let dz: f32 = (-B * z0 + sin(x0)) * DT;

    let new_x: f32 = vPos.x + dx;
    let new_y: f32 = vPos.y + dy;
    let new_z: f32 = vPos.z + dz;

    // Write back
    particles_dst.particles[index].pos = vec4<f32>(new_x, new_y, new_z, 1.0);
}
