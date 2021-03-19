[[block]]
struct Globals {
    proj: mat4x4<f32>;
};

// Uniforms
[[group(0), binding(0)]]
var<uniform> globals: Globals;

// Per instance data
[[location(0)]]
var<in> in_particle_pos: vec4<f32>;

// Per mesh vertex data
[[location(1)]]
var<in> in_position: vec3<f32>;

// Vertex shader output
[[builtin(position)]]
var<out> out_position: vec4<f32>;

[[location(1)]]
var<out> pos: vec4<f32>;

[[stage(vertex)]]
fn main() {
    out_position = vec4<f32>(in_position + in_particle_pos.xyz, 1.0);
    out_position = globals.proj * out_position;

    pos = vec4<f32>(out_position.xyz, in_particle_pos.w);
}

[[location(0)]]
var<out> out_color: vec4<f32>;

[[location(1)]]
var<in> pos: vec4<f32>;

[[stage(fragment)]]
fn main() {
    var val: f32 = pos.w;
    var r: f32 = val;
    var g: f32 = val;
    var b: f32 = val;

    out_color = vec4<f32>(r, g, b, 1.0);
}
