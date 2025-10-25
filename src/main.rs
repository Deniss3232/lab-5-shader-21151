use minifb::{Key, Window, WindowOptions};
use std::time::{Instant, Duration};

// Vec3 y utilidades

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}
impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    fn add(self, o: Vec3) -> Vec3 { Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z) }
    fn sub(self, o: Vec3) -> Vec3 { Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z) }
    fn mul(self, k: f32) -> Vec3 { Vec3::new(self.x * k, self.y * k, self.z * k) }
    fn dot(self, o: Vec3) -> f32 { self.x*o.x + self.y*o.y + self.z*o.z }
    fn length(self) -> f32 { self.dot(self).sqrt() }
    fn normalize(self) -> Vec3 {
        let len = self.length();
        if len == 0.0 { self } else { self.mul(1.0/len) }
    }
}

// rotación alrededor del eje Y 
fn rotate_y(v: Vec3, angle: f32) -> Vec3 {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    Vec3::new(
        v.x * cos_a + v.z * sin_a,
        v.y,
        -v.x * sin_a + v.z * cos_a,
    )
}

fn clamp01(v: f32) -> f32 {
    if v < 0.0 { 0.0 } else if v > 1.0 { 1.0 } else { v }
}

// Color

#[derive(Clone, Copy, Debug)]
struct Color {
    r: f32,
    g: f32,
    b: f32,
}
impl Color {
    fn new(r: f32, g: f32, b: f32) -> Self { Self { r, g, b } }

    fn mul(self, k: f32) -> Self { Color::new(self.r * k, self.g * k, self.b * k) }
    fn add(self, o: Color) -> Self { Color::new(self.r + o.r, self.g + o.g, self.b + o.b) }

    fn to_u32(self) -> u32 {
        // formato 0x00RRGGBB
        let rr = (clamp01(self.r) * 255.0) as u32;
        let gg = (clamp01(self.g) * 255.0) as u32;
        let bb = (clamp01(self.b) * 255.0) as u32;
        (rr << 16) | (gg << 8) | bb
    }
}

// Ray, Sphere

#[derive(Clone, Copy, Debug)]
struct Ray {
    origin: Vec3,
    dir: Vec3,
}

#[derive(Clone, Copy, Debug)]
struct Sphere {
    center: Vec3,
    radius: f32,
    kind: PlanetType,
}

#[derive(Clone, Copy, Debug)]
enum PlanetType {
    Star,
    Rocky,
    GasGiant,
}

// intersección rayo-esfera
fn ray_sphere_intersect(ray: Ray, s: &Sphere) -> Option<f32> {
    let oc = ray.origin.sub(s.center);
    let a = ray.dir.dot(ray.dir);
    let b = 2.0 * oc.dot(ray.dir);
    let c = oc.dot(oc) - s.radius * s.radius;
    let disc = b*b - 4.0*a*c;
    if disc < 0.0 {
        None
    } else {
        let sq = disc.sqrt();
        let t1 = (-b - sq) / (2.0*a);
        let t2 = (-b + sq) / (2.0*a);
        let t_min = 0.001;
        if t1 > t_min {
            Some(t1)
        } else if t2 > t_min {
            Some(t2)
        } else {
            None
        }
    }
}

// "Iluminación" simple

fn simple_lighting(normal: Vec3, light_dir: Vec3) -> f32 {
    let n_dot_l = normal.normalize().dot(light_dir.normalize());
    clamp01(n_dot_l)
}

// Shaders con animación

// Estrella zol
fn shader_star(surface_normal: Vec3, time: f32) -> Color {

    let wobble = rotate_y(surface_normal, time * 0.3); // 0.3 rad/s

    let base = Color::new(1.0, 0.9, 0.4);
    let hot_core = Color::new(1.0, 1.0, 0.8);

  
    let mix_factor = (wobble.y * 0.5 + 0.5).powf(2.0); // 0..1
    let core_color = base.mul(1.0 - mix_factor).add(hot_core.mul(mix_factor));

    let light = Vec3::new(-1.0, 1.0, -0.5);
    let diff = 0.5 + 0.5 * simple_lighting(surface_normal, light);

    core_color.mul(diff * 1.3)
}

// Planeta rocoso: gira en Y
fn shader_rocky(surface_normal: Vec3, time: f32) -> Color {
    let rot_normal = rotate_y(surface_normal, time * 0.5); 

    let nx = rot_normal.x;
    let nz = rot_normal.z;

    let continents = ((nx * 12.3).sin() * (nz * 7.7).sin()).abs();

    let land_color = Color::new(0.4, 0.3, 0.15);
    let veg_color  = Color::new(0.1, 0.35, 0.1);
    let mix_land = if continents > 0.3 { veg_color } else { land_color };

    let light = Vec3::new(-1.0, 1.0, -0.5);
    let diff = simple_lighting(surface_normal, light);

    let shadow_tint = Color::new(0.05, 0.07, 0.1);
    let lit_color = mix_land.mul(0.4 + 0.6 * diff);
    lit_color.add(shadow_tint.mul(0.3 * (1.0 - diff)))
}

fn shader_gas_giant(surface_normal: Vec3, time: f32) -> Color {
    let rot_normal = rotate_y(surface_normal, time * 0.2);

    let ny = rot_normal.y;
    let bands = ((ny * 20.0) + time * 2.0).sin() * 0.5 + 0.5;

    let band_a = Color::new(0.8, 0.7, 0.5);
    let band_b = Color::new(0.9, 0.6, 0.4);
    let base_color = band_a.mul(bands).add(band_b.mul(1.0 - bands));

    let light = Vec3::new(-1.0, 1.0, -0.5);
    let diff = 0.3 + 0.7 * simple_lighting(surface_normal, light);

    base_color.mul(diff)
}

//  shader que se usan
fn shade(kind: PlanetType, normal: Vec3, time: f32) -> Color {
    match kind {
        PlanetType::Star => shader_star(normal, time),
        PlanetType::Rocky => shader_rocky(normal, time),
        PlanetType::GasGiant => shader_gas_giant(normal, time),
    }
}

// Render de la escena

fn render_scene(
    buffer: &mut [u32],
    w: usize,
    h: usize,
    time: f32,
) {
    // cámara
    let cam_pos = Vec3::new(0.0, 0.0, 0.0);
    let fov: f32 = 60.0_f32.to_radians();
    let aspect = w as f32 / h as f32;
    let scale = (fov * 0.5).tan();

    // escena con 3 esferas
    let spheres = [
        Sphere {
            center: Vec3::new(-1.5, 0.5, -5.0),
            radius: 1.0,
            kind: PlanetType::Star,
        },
        Sphere {
            center: Vec3::new(0.5, -0.2, -4.0),
            radius: 0.8,
            kind: PlanetType::Rocky,
        },
        Sphere {
            center: Vec3::new(2.0, 0.3, -6.0),
            radius: 1.2,
            kind: PlanetType::GasGiant,
        },
    ];

    // barrer cada pixel y hacer raycasting
    for y in 0..h {
        for x in 0..w {
            // Ray direction desde cámara
            let px = (((x as f32 + 0.5) / w as f32) * 2.0 - 1.0) * aspect * scale;
            let py = (1.0 - ((y as f32 + 0.5) / h as f32) * 2.0) * scale;

            let ray_dir = Vec3::new(px, py, -1.0).normalize();
            let ray = Ray { origin: cam_pos, dir: ray_dir };

            let mut closest_t = f32::INFINITY;
            let mut final_color = Color::new(0.0, 0.0, 0.0); // fondo negro

            for s in &spheres {
                if let Some(t_hit) = ray_sphere_intersect(ray, s) {
                    if t_hit < closest_t {
                        closest_t = t_hit;
                        let hit_pos = ray.origin.add(ray.dir.mul(t_hit));
                        let normal = hit_pos.sub(s.center).normalize();
                        final_color = shade(s.kind, normal, time);
                    }
                }
            }

            buffer[y * w + x] = final_color.to_u32();
        }
    }
}


// main (loop interactivo)

fn main() {
    let width = 800usize;
    let height = 600usize;

    let mut window = Window::new(
        "Lab 5 - Shaders (Rust software renderer)  |  ESC para salir",
        width,
        height,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .expect("No pude abrir la ventana");

    let mut buffer: Vec<u32> = vec![0; width * height];

    let start_time = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let elapsed = start_time.elapsed();
        let t = elapsed.as_secs_f32();

        render_scene(&mut buffer, width, height, t);

        window
            .update_with_buffer(&buffer, width, height)
            .expect("error al actualizar la ventana");

        // pausa pequeña
        std::thread::sleep(Duration::from_millis(10));
    }
}
