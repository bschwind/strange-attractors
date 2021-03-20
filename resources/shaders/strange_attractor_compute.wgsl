// This should match `NUM_PARTICLES` on the Rust side.
const NUM_PARTICLES: u32 = 1000000;

const B: f32 = 0.19;

const DT: f32 = 0.033333333;

[[block]]
struct Particle {
    pos : vec4<f32>;
};

[[block]]
struct Particles {
    particles: [[stride(16)]] array<Particle>;
};

[[block]]
struct Consts {
    algo : u32;
    a : f32;
    b : f32;
    c : f32;
    d : f32;
    align : vec3<f32>;
};

[[group(0), binding(0)]] var<storage> particles_src : [[access(read)]] Particles;
[[group(0), binding(1)]] var<storage> particles_dst : [[access(read_write)]] Particles;
[[group(1), binding(0)]] var<uniform> consts : Consts;

[[builtin(global_invocation_id)]] var gl_GlobalInvocationID : vec3<u32>;

[[stage(compute), workgroup_size(512)]]
fn main() {
    const index : u32 = gl_GlobalInvocationID.x;
    if (index >= NUM_PARTICLES) {
        return;
    }

    var vPos : vec4<f32> = particles_src.particles[index].pos;
    var x0: f32 = vPos.x;
    var y0: f32 = vPos.y;
    var z0: f32 = vPos.z;

    var dx: f32 = (-B * x0 + sin(y0)) * DT;
    var dy: f32 = (-B * y0 + sin(z0)) * DT;
    var dz: f32 = (-B * z0 + sin(x0)) * DT;

    var new_x: f32 = vPos.x + dx;
    var new_y: f32 = vPos.y + dy;
    var new_z: f32 = vPos.z + dz;

    // Write back
    particles_dst.particles[index].pos = vec4<f32>(new_x, new_y, new_z, 1.0);
}
