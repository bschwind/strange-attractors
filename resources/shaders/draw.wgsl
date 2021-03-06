// Vertex shader input
[[location(0)]]
var<in> pos: vec2<f32>;

[[location(1)]]
var<in> uv: vec2<f32>;

// Vertex shader output
[[location(0)]]
var<out> vert_uv: vec2<f32>;

[[builtin(position)]]
var<out> out_pos: vec4<f32>;

[[stage(vertex)]]
fn main() {
    vert_uv = uv;
    out_pos = vec4<f32>(pos, 0.0, 1.0);
}

// Fragment shader input
[[location(0)]]
var<in> vert_uv: vec2<f32>;

// Fragment shader output
[[location(0)]]
var<out> out_color: vec4<f32>;

[[stage(fragment)]]
fn main() {
    out_color = vec4<f32>(vert_uv.x, vert_uv.y, 1.0, 1.0);
}
