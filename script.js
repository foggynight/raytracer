// 3D Ray Tracer
// Copyright (C) 2026 Robert Coffey
// Released under the MIT license.

// canvas
const CANV_W = 640;
const CANV_H = 640;

// viewport
const VIEW_W = 1;
const VIEW_H = 1;
const VIEW_D = 1;

// coordinates -----------------------------------------------------------------
// "Canvas" coordinates are centered such that center of canvas is (0, 0).
// Viewport coordinates have units (vx, vy, vz).

// Convert centered canvas coordinates to viewport coordinates.
function canv_to_view(cx, cy, dist = VIEW_D) {
  const vx = cx * VIEW_W / CANV_W;
  const vy = cy * VIEW_H / CANV_H;
  return [vx, vy, dist];
}

// util ------------------------------------------------------------------------
// Vec3 represented by array [x, y, z].

function range_contains(x_min, x_max, x) { return x_min <= x && x <= x_max; }

function Vec3_len(v) { return Math.sqrt(Vec3_dot(v, v)); }
function Vec3_invert(v) { return v.map(a => -a); }

function Vec3_mulc(v, c) { return v.map(a => a * c); }
function Vec3_divc(v, c) { return v.map(a => a / c); }

function Vec3_add(v1, v2) { return v1.map((a, i) => a + v2[i]); }
function Vec3_sub(v1, v2) { return v1.map((a, i) => a - v2[i]); }
function Vec3_dot(v1, v2) { return v1.map((a, i) => a * v2[i])
                                     .reduce((acc, a) => acc + a); }

function Vec3_normalize(v) {
  const len = Vec3_len(v);
  v.forEach(function(a, i) { v[i] /= len; });
}

// Clamp all 3 values of vec between lo/hi.
function Vec3_clamp(v, lo, hi) {
  return v.map(a => Math.max(lo, Math.min(a, hi))); }

function Vec3_color(v) { return `rgb(${v[0]}, ${v[1]}, ${v[2]})`; }

function Matrix_mulv(m, v) {
  const out = [];
  for (const row of m) {
    let x = 0;
    for (const col_i in row) {
      x += row[col_i] * v[col_i];
    }
    out.push(x);
  }
  return out;
}

// engine ----------------------------------------------------------------------
// Light convention: light vector points toward source.

const EPSILON = 0.001; // used as shadow and reflection t_min

class Camera {
  constructor(position) {
    this.position = position;
    this.angle_y = 0;
    this.rotation = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
  }

  rotate_y(delta_ay) {
    this.angle_y += delta_ay;
    this.rotation = [
      [Math.cos(this.angle_y), 0, Math.sin(this.angle_y)],
      [0, 1, 0],
      [-Math.sin(this.angle_y), 0, Math.cos(this.angle_y)],
    ];
  }
}

class Sphere {
  constructor(pos, radius, color, specular, reflective) {
    this.pos = pos;
    this.radius = radius;
    this.color = color;
    this.specular = specular;
    this.reflective = reflective;
  }
}

// 3 types of light: ambient, directional, point.
const [L_AMB, L_DIR, L_PNT] = [0, 1, 2];
class Light {
  constructor(type, intensity, position, direction) {
    this.type = type;
    this.intensity = intensity;
    this.position = position;
    this.direction = direction;
  }
}

// Calculate intersection distances of sphere's shell and a ray represented by
// origin and direction. Returns [inf, inf] if no intersection found.
function sphere_ray_inters(sphere, ray_origin, ray_direction) {
  const RO_SC = Vec3_sub(ray_origin, sphere.pos); // ray origin to sphere center

  const a = Vec3_dot(ray_direction, ray_direction);
  const b = 2 * Vec3_dot(RO_SC, ray_direction);
  const c = Vec3_dot(RO_SC, RO_SC) - (sphere.radius * sphere.radius);

  const discrim = b*b - 4*a*c;
  if (discrim < 0) {
    return [Infinity, Infinity];
  }

  const t1 = (-b + Math.sqrt(discrim)) / (2*a);
  const t2 = (-b - Math.sqrt(discrim)) / (2*a);
  return [t1, t2];
}

function closest_intersection(scene, pos, dir, t_min, t_max) {
  let closest_sphere = null;
  let closest_t = Infinity;
  for (const sphere of scene.spheres) {
    const [t1, t2] = sphere_ray_inters(sphere, pos, dir);
    if (range_contains(t_min, t_max, t1) && t1 < closest_t) {
      closest_sphere = sphere;
      closest_t = t1;
    }
    if (range_contains(t_min, t_max, t2) && t2 < closest_t) {
      closest_sphere = sphere;
      closest_t = t2;
    }
  }
  return [closest_sphere, closest_t];
}

function reflect_ray(ray, normal) {
  return Vec3_sub(
    Vec3_mulc(normal, 2 * Vec3_dot(normal, ray)),
    ray);
}

function compute_lighting(scene, point, normal, view_ray, specular) {
  let intens = 0;
  for (const light of scene.lights) {
    if (light.type == L_AMB) {
      intens += light.intensity;
    } else {
      let light_ray, t_max;
      if (light.type == L_PNT) {
        light_ray = Vec3_sub(light.position, point);
        t_max = 1;
      } else {
        light_ray = light.direction;
        t_max = Infinity;
      }

      // shadow check
      const [shad_sphere, shad_t] =
        closest_intersection(scene, point, light_ray, EPSILON, t_max);
      if (shad_sphere != null)
        continue;

      // diffuse
      const n_dot_l = Vec3_dot(normal, light_ray);
      if (n_dot_l > 0) {
        intens += light.intensity
          * n_dot_l / (Vec3_len(normal) * Vec3_len(light_ray));
      }

      // specular
      if (specular >= 0) {
        const reflected_ray = reflect_ray(light_ray, normal);
        const r_dot_v = Vec3_dot(reflected_ray, view_ray);
        if (r_dot_v > 0) {
          const rl = Vec3_len(reflected_ray);
          const vl = Vec3_len(view_ray);
          intens += light.intensity * Math.pow(r_dot_v / (rl * vl), specular);
        }
      }
    }
  }
  return intens;
}

function trace_ray(scene, pos, dir_ray, t_min, t_max, recursion_depth) {
  // Determine closest intersection with ray in scene.
  const [closest_sphere, closest_t] =
    closest_intersection(scene, pos, dir_ray, t_min, t_max);
  if (closest_sphere == null)
    return null;

  // Compute lighting at intersection point.
  const point = Vec3_add(pos, Vec3_mulc(dir_ray, closest_t));
  const normal = Vec3_sub(point, closest_sphere.pos);
  Vec3_normalize(normal);
  const light_intens = compute_lighting(
    scene, point, normal,
    Vec3_invert(dir_ray), closest_sphere.specular);

  const local_color = Vec3_clamp(
    Vec3_mulc(closest_sphere.color, light_intens),
    0, 255);

  const r = closest_sphere.reflective;
  if (recursion_depth <= 0 || r <= 0) {
    return local_color;
  }

  const reflected_ray = reflect_ray(Vec3_invert(dir_ray), normal);
  const reflected_color = trace_ray(
    scene, point, reflected_ray,
    EPSILON, Infinity, recursion_depth - 1);
  if (reflected_color == null) {
    return local_color;
  }

  return Vec3_add(
    Vec3_mulc(local_color, 1 - r),
    Vec3_mulc(reflected_color, r));
}

// draw ------------------------------------------------------------------------

const canvas = document.getElementById("canvas");
canvas.width = CANV_W;
canvas.height = CANV_H;

const ctx = canvas.getContext("2d");

function draw_clear(color) {
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, CANV_W, CANV_H);
}

function draw_pixel(x, y, color) {
  ctx.fillStyle = color;
  ctx.fillRect(x, y, 1, 1);
}

// main ------------------------------------------------------------------------

function render(camera, scene) {
  draw_clear("black");
  for (let cy = -CANV_H/2; cy <= CANV_H/2; ++cy) {
    for (let cx = -CANV_W/2; cx <= CANV_W/2; ++cx) {
      const view_pos = Vec3_add(camera.position, canv_to_view(cx, cy));
      const dir_ray = Matrix_mulv(
        camera.rotation,
        Vec3_sub(view_pos, camera.position));
      const color = trace_ray(scene, camera.position, dir_ray, 1, Infinity, 3);
      if (color != null) {
        draw_pixel(cx + CANV_W/2, -cy + CANV_H/2, Vec3_color(color));
      }
    }
  }
}

async function main() {
  const camera = new Camera([0, 0, 0]);
  const scene = {
    "spheres": [
      new Sphere([ 0, -1,  3], 1, [255, 0, 0], 500, 0.1),
      new Sphere([-2,  0,  4], 1, [0, 255, 0], 10, 0.3),
      new Sphere([ 2,  0,  4], 1, [0, 0, 255], 100, 0.2),
      new Sphere([0, -5001, 0], 5000, [255, 255, 0], 1000, 0.5),
    ],
    "lights": [
      new Light(L_AMB, 0.2, null, null),
      new Light(L_DIR, 0.2, null, [1, 4, 4]),
      new Light(L_PNT, 0.6, [10, 1, 0], null),
    ],
  };

  render(camera, scene);

  // Move the camera.
  //for (let i = 0; i < 20; ++i) {
  //  camera.position[0] += 0.3;
  //  camera.rotate_y(-Math.PI / 60);
  //  render(camera, scene);
  //  await new Promise(resolve => setTimeout(resolve, 10));
  //}
}

main();
