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

struct Globals {
    proj: mat4x4<f32>;
    consts: Consts;
};

// Uniforms
[[group(0), binding(0)]]
var<uniform> globals: Globals;

struct VertexInput {
    // Per-instance data
    [[location(0)]] in_particle_pos: vec4<f32>;

    // Per mesh vertex data
    [[location(1)]] in_position: vec3<f32>;
};

struct VertexOutput {
    [[builtin(position)]] out_position: vec4<f32>;
    [[location(0)]] pos: vec4<f32>;
};

[[stage(vertex)]]
fn main_vs(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.out_position = vec4<f32>(input.in_position + input.in_particle_pos.xyz, 1.0);
    out.out_position = globals.proj * out.out_position;

    let vel: f32 = input.in_particle_pos.w;
    out.pos = vec4<f32>(out.out_position.xyz, vel);

    return out;
}

[[stage(fragment)]]
fn main_fs(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let r: f32 = in.pos.w / (globals.consts.g * 0.5);
    let g: f32 = 0.1;
    let b: f32 = 1.0 - r;

    return vec4<f32>(r, g, b, 1.0);
}
