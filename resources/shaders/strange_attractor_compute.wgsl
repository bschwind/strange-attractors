let B: f32 = 0.19;

let DT: f32 = 0.033333333;

struct Particle {
    pos : vec4<f32>;
};

struct Particles {
    particles: array<Particle>;
};

struct Consts {
    algo : u32;
    a : f32;
    b : f32;
    c : f32;
    d : f32;
    e : f32;
    f : f32;
    g : f32;
};

[[group(0), binding(0)]] var<storage> particles_src : Particles;
[[group(0), binding(1)]] var<storage, read_write> particles_dst : Particles;
[[group(1), binding(0)]] var<uniform> consts : Consts;

[[stage(compute), workgroup_size(256)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    let index: u32 = global_invocation_id.x;

    let total = arrayLength(&particles_src.particles);
    if (index >= total) {
        return;
    }

    let vPos : vec4<f32> = particles_src.particles[index].pos;
    let x0: f32 = vPos.x;
    let y0: f32 = vPos.y;
    let z0: f32 = vPos.z;
    var dx: f32;
    var dy: f32;
    var dz: f32;

    if (consts.algo == 0u) {
        dx = (-consts.b * x0 + sin(y0)) * DT;
        dy = (-consts.b * y0 + sin(z0)) * DT;
        dz = (-consts.b * z0 + sin(x0)) * DT;
    } else {
        if (consts.algo == 1u) {
            dx = ((z0 - consts.b) * x0 - consts.d*y0) * DT;
            dy = (consts.d * x0 + (z0-consts.b) * y0) * DT;
            dz = (consts.c + consts.a*z0 - ((z0*z0*z0) / 3.0) - (x0*x0) + consts.f * z0 * (x0*x0*x0)) * DT;
        }
    }

    let new_x: f32 = vPos.x + dx;
    let new_y: f32 = vPos.y + dy;
    let new_z: f32 = vPos.z + dz;

    var vel: f32 = sqrt(dx*dx + dy*dy + dz*dz);

    // Write back
    particles_dst.particles[index].pos = vec4<f32>(new_x, new_y, new_z, vel);
}
