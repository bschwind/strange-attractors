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

fn hue2rgb(f1: f32, f2: f32, hue: f32) -> f32 {
    if ((6.0 * hue) < 1.0) {
        return f1 + (f2 - f1) * 6.0 * hue;
    }
    if ((2.0 * hue) < 1.0) {
        return f2;
    }
    if ((3.0 * hue) < 2.0) {
        return f1 + (f2 - f1) * ((2.0 / 3.0) - hue) * 6.0;
    }
    return f1;
}

fn hsl2rgb(hsl: vec3<f32>) -> vec3<f32> {
    var rgb: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    if (hsl.y == 0.0) {
        rgb = vec3<f32>(hsl.z);
    } else {
        var f2: f32 = 0.0;
        if (hsl.z < 0.5) {
            f2 = hsl.z * (1.0 + hsl.y);
        } else {
            f2 = hsl.z + hsl.y - hsl.y * hsl.z;
        }
        var f1: f32 = 2.0 * hsl.z - f2;
        rgb.r = hue2rgb(f1, f2, hsl.x + (1.0/3.0));
        rgb.g = hue2rgb(f1, f2, hsl.x);
        rgb.b = hue2rgb(f1, f2, hsl.x - (1.0/3.0));
    }   
    return rgb;
}

[[stage(fragment)]]
fn main() {
    var hsl: vec3<f32> = vec3<f32>(1.0 - clamp(pos.w, 0.0, 1.0), 1.0, 0.4 + pos.w / 10.0);

    out_color = vec4<f32>(hsl2rgb(hsl), 1.0);
    // out_color = vec4<f32>(hsl, 1.0);
}
