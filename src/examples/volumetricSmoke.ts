// src/examples/volumetricSmoke.ts
import GUI from "lil-gui";

// =========================================================================
// 1. 标准 3D 数学库 (针对 WebGPU 定制)
// =========================================================================
function mat4Perspective(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const out = new Float32Array(16);
  const f = 1.0 / Math.tan(fovRad / 2);
  out[0] = f / aspect; out[5] = f;
  out[10] = far / (near - far); out[11] = -1; out[14] = (near * far) / (near - far);
  return out;
}

function mat4LookAt(eye: number[], center: number[], up: number[]): Float32Array {
  const out = new Float32Array(16);
  let z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
  let len = 1 / Math.hypot(z0, z1, z2); z0 *= len; z1 *= len; z2 *= len;
  let x0 = up[1] * z2 - up[2] * z1, x1 = up[2] * z0 - up[0] * z2, x2 = up[0] * z1 - up[1] * z0;
  len = 1 / Math.hypot(x0, x1, x2); x0 *= len; x1 *= len; x2 *= len;
  let y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;
  out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
  out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
  out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
  out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
  out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
  out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]); out[15] = 1;
  return out;
}

function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] = a[0 * 4 + r] * b[c * 4 + 0] + a[1 * 4 + r] * b[c * 4 + 1] + 
                       a[2 * 4 + r] * b[c * 4 + 2] + a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function mat4Inverse(m: Float32Array): Float32Array {
  const out = new Float32Array(16);
  const m00 = m[0], m01 = m[1], m02 = m[2], m03 = m[3];
  const m10 = m[4], m11 = m[5], m12 = m[6], m13 = m[7];
  const m20 = m[8], m21 = m[9], m22 = m[10], m23 = m[11];
  const m30 = m[12], m31 = m[13], m32 = m[14], m33 = m[15];

  const b00 = m00 * m11 - m01 * m10, b01 = m00 * m12 - m02 * m10, b02 = m00 * m13 - m03 * m10;
  const b03 = m01 * m12 - m02 * m11, b04 = m01 * m13 - m03 * m11, b05 = m02 * m13 - m03 * m12;
  const b06 = m20 * m31 - m21 * m30, b07 = m20 * m32 - m22 * m30, b08 = m20 * m33 - m23 * m30;
  const b09 = m21 * m32 - m22 * m31, b10 = m21 * m33 - m23 * m31, b11 = m22 * m33 - m23 * m32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return out;
  det = 1.0 / det;

  out[0] = (m11 * b11 - m12 * b10 + m13 * b09) * det;
  out[1] = (m02 * b10 - m01 * b11 - m03 * b09) * det;
  out[2] = (m31 * b05 - m32 * b04 + m33 * b03) * det;
  out[3] = (m22 * b04 - m21 * b05 - m23 * b03) * det;
  out[4] = (m12 * b08 - m10 * b11 - m13 * b07) * det;
  out[5] = (m00 * b11 - m02 * b08 + m03 * b07) * det;
  out[6] = (m32 * b02 - m30 * b05 - m33 * b01) * det;
  out[7] = (m20 * b05 - m22 * b02 + m23 * b01) * det;
  out[8] = (m10 * b10 - m11 * b08 + m13 * b06) * det;
  out[9] = (m01 * b08 - m00 * b10 - m03 * b06) * det;
  out[10] = (m30 * b04 - m31 * b02 + m33 * b00) * det;
  out[11] = (m21 * b02 - m20 * b04 - m23 * b00) * det;
  out[12] = (m11 * b07 - m10 * b09 - m12 * b06) * det;
  out[13] = (m00 * b09 - m01 * b07 + m02 * b06) * det;
  out[14] = (m31 * b01 - m30 * b03 - m32 * b00) * det;
  out[15] = (m20 * b03 - m21 * b01 + m22 * b00) * det;
  return out;
}

function transformPoint(p: number[], m: Float32Array): number[] {
  const x = p[0], y = p[1], z = p[2];
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
  ];
}

function hexToRgb(hex: string): [number, number, number] {
  const bigint = parseInt(hex.replace("#", ""), 16);
  return [
    ((bigint >> 16) & 255) / 255,
    ((bigint >> 8) & 255) / 255,
    (bigint & 255) / 255,
  ];
}

// =========================================================================
// 2. 主程序入口
// =========================================================================
export function runVolumetricSmoke(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: GUI
) {
  // 1. 实体方块几何数据
  const cubeData = new Float32Array([
    // Front (Z+)
    -1,-1, 1, 0,0,1,   1,-1, 1, 0,0,1,   1, 1, 1, 0,0,1,
    -1,-1, 1, 0,0,1,   1, 1, 1, 0,0,1,  -1, 1, 1, 0,0,1,
    // Back (Z-)
     1,-1,-1, 0,0,-1, -1,-1,-1, 0,0,-1, -1, 1,-1, 0,0,-1,
     1,-1,-1, 0,0,-1, -1, 1,-1, 0,0,-1,  1, 1,-1, 0,0,-1,
    // Top (Y+)
    -1, 1, 1, 0,1,0,   1, 1, 1, 0,1,0,   1, 1,-1, 0,1,0,
    -1, 1, 1, 0,1,0,   1, 1,-1, 0,1,0,  -1, 1,-1, 0,1,0,
    // Bottom (Y-)
    -1,-1,-1, 0,-1,0,  1,-1,-1, 0,-1,0,  1,-1, 1, 0,-1,0,
    -1,-1,-1, 0,-1,0,  1,-1, 1, 0,-1,0, -1,-1, 1, 0,-1,0,
    // Right (X+)
     1,-1, 1, 1,0,0,   1,-1,-1, 1,0,0,   1, 1,-1, 1,0,0,
     1,-1, 1, 1,0,0,   1, 1,-1, 1,0,0,   1, 1, 1, 1,0,0,
    // Left (X-)
    -1,-1,-1,-1,0,0,  -1,-1, 1,-1,0,0,  -1, 1, 1,-1,0,0,
    -1,-1,-1,-1,0,0,  -1, 1, 1,-1,0,0,  -1, 1,-1,-1,0,0,
  ]);
  const cubeVertexBuffer = device.createBuffer({
    size: cubeData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(cubeVertexBuffer, 0, cubeData);

  // 2. 场景障碍建筑物设置 (完全按照参考图位置精确安放)
  const buildingDefs = [
    // 中央主建筑 (高耸长方体)
    { pos: [1.2, 3.8, -1.0], scale: [2.6, 3.8, 2.4], color: [0.18, 0.24, 0.29] },
    // 前方左中立方体
    { pos: [-0.6, 1.8, 3.4], scale: [2.4, 1.8, 2.2], color: [0.16, 0.21, 0.26] },
    // 右前矮方块
    { pos: [3.4, 1.1, 3.2], scale: [1.3, 1.1, 1.3], color: [0.17, 0.22, 0.27] },
    // 右后长条箱体
    { pos: [5.8, 1.5, 1.4], scale: [1.8, 1.5, 3.2], color: [0.15, 0.20, 0.25] },
  ];

  const buildingInstanceData = new Float32Array(buildingDefs.length * 16);
  buildingDefs.forEach((b, i) => {
    buildingInstanceData[i * 16 + 0] = b.pos[0];
    buildingInstanceData[i * 16 + 1] = b.pos[1];
    buildingInstanceData[i * 16 + 2] = b.pos[2];

    buildingInstanceData[i * 16 + 4] = b.scale[0];
    buildingInstanceData[i * 16 + 5] = b.scale[1];
    buildingInstanceData[i * 16 + 6] = b.scale[2];

    buildingInstanceData[i * 16 + 8] = b.color[0];
    buildingInstanceData[i * 16 + 9] = b.color[1];
    buildingInstanceData[i * 16 + 10] = b.color[2];
    buildingInstanceData[i * 16 + 11] = 1.0;
  });

  const buildingBuffer = device.createBuffer({
    size: buildingInstanceData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buildingBuffer, 0, buildingInstanceData);

  // 3. 常量缓冲区
  const uniformBufferSize = 256;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 4. 着色器实现
  // =========================================================================
  const sceneShaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
      invViewProj: mat4x4f,
      camPos: vec4f,
      lightDir: vec4f,
      shockwave: vec4f,
      params: vec4f,       // x: time, y: density, z: absorption, w: steps
      smokeColor: vec4f,   // rgb: 气体散射色, a: 蓬松尺寸
      lightColor: vec4f,   // rgb: 日光颜色
      ambientColor: vec4f, // rgb: 天空天光
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct Instance {
      pos: vec4f,
      scale: vec4f,
      color: vec4f,
      pad: vec4f,
    };
    @group(0) @binding(1) var<storage, read> instances: array<Instance>;

    struct VertexOut {
      @builtin(position) clipPos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) color: vec3f,
      @location(3) isGround: f32,
    };

    @vertex
    fn vs_building(@location(0) pos: vec3f, @location(1) normal: vec3f, @builtin(instance_index) instIdx: u32) -> VertexOut {
      var out: VertexOut;
      let inst = instances[instIdx];
      let worldPos = pos * inst.scale.xyz + inst.pos.xyz;
      out.clipPos = u.viewProj * vec4f(worldPos, 1.0);
      out.worldPos = worldPos;
      out.normal = normal;
      out.color = inst.color.rgb;
      out.isGround = 0.0;
      return out;
    }

    @vertex
    fn vs_ground(@builtin(vertex_index) vid: u32) -> VertexOut {
      var out: VertexOut;
      let sz = 90.0;
      var pos = vec3f(0.0);
      if (vid == 0u) { pos = vec3f(-sz, 0.0, -sz); }
      else if (vid == 1u) { pos = vec3f(sz, 0.0, -sz); }
      else if (vid == 2u) { pos = vec3f(-sz, 0.0, sz); }
      else if (vid == 3u) { pos = vec3f(-sz, 0.0, sz); }
      else if (vid == 4u) { pos = vec3f(sz, 0.0, -sz); }
      else { pos = vec3f(sz, 0.0, sz); }

      out.clipPos = u.viewProj * vec4f(pos, 1.0);
      out.worldPos = pos;
      out.normal = vec3f(0.0, 1.0, 0.0);
      out.color = vec3f(0.14, 0.20, 0.24);
      out.isGround = 1.0;
      return out;
    }

    @fragment
    fn fs_scene(in: VertexOut) -> @location(0) vec4f {
      let N = normalize(in.normal);
      let L = normalize(u.lightDir.xyz);
      let diff = max(dot(N, L), 0.0);

      var baseColor = in.color;
      let coord = in.worldPos.xz * 0.40;
      let fw = fwidth(coord);

      // 参考图的大格线青色地面
      if (in.isGround > 0.5) {
        let grid = abs(fract(coord - 0.5) - 0.5) / max(fw, vec2f(0.001));
        let line = min(grid.x, grid.y);
        let gridMask = 1.0 - min(line, 1.0);
        let lineColor = vec3f(0.22, 0.35, 0.42);
        baseColor = mix(vec3f(0.11, 0.16, 0.20), lineColor, gridMask * 0.85);
      }

      // 充足的环境漫反射补光，杜绝背光死黑
      let skyAmbient = max(N.y * 0.4 + 0.6, 0.0) * 0.40;
      let finalColor = baseColor * (diff * 0.65 + skyAmbient);
      return vec4f(finalColor, 1.0);
    }
  `;

  // 核心流体仿真与体积渲染着色器
  const smokeShaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
      invViewProj: mat4x4f,
      camPos: vec4f,
      lightDir: vec4f,
      shockwave: vec4f,
      params: vec4f,       // x: time, y: density, z: absorption, w: steps
      smokeColor: vec4f,   // rgb: 烟雾颜色, a: 蓬松尺度
      lightColor: vec4f,   // rgb: 直射光颜色
      ambientColor: vec4f, // rgb: 天空环境光
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var depthTex: texture_depth_2d;

    struct FullscreenOut {
      @builtin(position) clipPos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_post(@builtin(vertex_index) vid: u32) -> FullscreenOut {
      var out: FullscreenOut;
      var uv = vec2f(0.0);
      if (vid == 0u) { uv = vec2f(-1.0, -1.0); }
      else if (vid == 1u) { uv = vec2f(3.0, -1.0); }
      else { uv = vec2f(-1.0, 3.0); }
      out.clipPos = vec4f(uv, 0.0, 1.0);
      out.uv = uv * 0.5 + 0.5;
      return out;
    }

    fn bayerDither(coord: vec2f) -> f32 {
      let bayer = mat4x4f(
         0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,
        12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0,
         3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,
        15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0
      );
      let p = vec2u(coord) % 4u;
      return bayer[p.x][p.y];
    }

    fn hash3(p: vec3f) -> vec3f {
      var q = vec3f(
        dot(p, vec3f(127.1, 311.7, 74.7)),
        dot(p, vec3f(269.5, 183.3, 246.1)),
        dot(p, vec3f(113.5, 271.9, 124.6))
      );
      return fract(sin(q) * 43758.5453123);
    }

    // 五阶 Perlin 平滑连续噪声 (C2 连续)
    fn smoothNoise(p: vec3f) -> f32 {
      let i = floor(p);
      let f = fract(p);
      let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

      let n000 = hash3(i + vec3f(0.0, 0.0, 0.0)).x;
      let n100 = hash3(i + vec3f(1.0, 0.0, 0.0)).x;
      let n010 = hash3(i + vec3f(0.0, 1.0, 0.0)).x;
      let n110 = hash3(i + vec3f(1.0, 1.0, 0.0)).x;
      let n001 = hash3(i + vec3f(0.0, 0.0, 1.0)).x;
      let n101 = hash3(i + vec3f(1.0, 0.0, 1.0)).x;
      let n011 = hash3(i + vec3f(0.0, 1.0, 1.0)).x;
      let n111 = hash3(i + vec3f(1.0, 1.0, 1.0)).x;

      return mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z
      );
    }

    fn billowFbm(p: vec3f) -> f32 {
      var v = 0.0;
      var a = 0.50;
      var pos = p;
      for (var i = 0; i < 4; i++) {
        let n = smoothNoise(pos);
        let b = abs(n * 2.0 - 1.0);
        v += a * (1.0 - b);
        pos = pos * 2.15 + vec3f(2.4, 4.1, 1.7);
        a *= 0.48;
      }
      return v;
    }

    // 建筑盒子符号距离函数 (Box SDF)
    fn sdBox(p: vec3f, center: vec3f, b: vec3f) -> f32 {
      let q = abs(p - center) - b;
      return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
    }

    // 场景固体障碍物整体 SDF 与流体推斥法线
    fn sceneObstacleField(p: vec3f) -> vec4f {
      // 场景中 4 栋主要建筑物
      let d1 = sdBox(p, vec3f(1.2, 3.8, -1.0), vec3f(2.6, 3.8, 2.4));
      let d2 = sdBox(p, vec3f(-0.6, 1.8, 3.4), vec3f(2.4, 1.8, 2.2));
      let d3 = sdBox(p, vec3f(3.4, 1.1, 3.2), vec3f(1.3, 1.1, 1.3));
      let d4 = sdBox(p, vec3f(5.8, 1.5, 1.4), vec3f(1.8, 1.5, 3.2));

      var minDist = min(min(d1, d2), min(d3, d4));
      return vec4f(minDist, 0.0, 0.0, 0.0);
    }

    // 真实的物理流体烟羽建模 (结合高斯烟羽模型 + 固体绕流边界层)
    fn getSmokeDensity(worldP: vec3f) -> f32 {
      if (worldP.y < 0.05 || worldP.y > 12.0) { return 0.0; }

      var p = worldP;

      // 1. 固体障碍物物理防穿透与表面滑移绕流 (Solid Obstacle Deflection)
      let obs = sceneObstacleField(p);
      let distToSolid = obs.x;
      if (distToSolid < 0.0) {
        // 在固体建筑内部，密度严格为 0，杜绝内部穿模
        return 0.0;
      }

      // 靠近建筑物表面时产生流体贴壁爬升与绕流（阻挡偏转）
      var boundaryFade = 1.0;
      if (distToSolid < 0.95) {
        boundaryFade = smoothstep(0.0, 0.95, distToSolid);
        // 将流体采样点沿固体表面法线推移，实现顺畅绕流
        let eps = 0.08;
        let grad = vec3f(
          sceneObstacleField(p + vec3f(eps, 0.0, 0.0)).x - sceneObstacleField(p - vec3f(eps, 0.0, 0.0)).x,
          sceneObstacleField(p + vec3f(0.0, eps, 0.0)).x - sceneObstacleField(p - vec3f(0.0, eps, 0.0)).x,
          sceneObstacleField(p + vec3f(0.0, 0.0, eps)).x - sceneObstacleField(p - vec3f(0.0, 0.0, eps)).x
        );
        let normal = normalize(grad + vec3f(0.0001));
        p += normal * ((0.95 - distToSolid) * 0.7);
      }

      // 2. 鼠标点击物理冲击驱散
      var shockAttenuation = 1.0;
      let shockAge = u.shockwave.z;
      if (shockAge >= 0.0 && shockAge < 3.2) {
        let shockCenter = vec3f(u.shockwave.x, 0.2, u.shockwave.y);
        let diff = p - shockCenter;
        let dist = length(diff);
        let waveRadius = shockAge * 5.6;
        let wavePower = exp(-shockAge * 1.35) * u.shockwave.w;

        let clearHole = smoothstep(0.0, waveRadius * 0.9, dist);
        shockAttenuation = mix(clearHole, 1.0, clamp(shockAge / 2.8, 0.0, 1.0));

        let ringDelta = dist - waveRadius;
        let pushWave = exp(-pow(ringDelta / 1.8, 2.0)) * wavePower;
        let pushDir = normalize(diff + vec3f(0.001));
        p -= pushDir * (pushWave * 3.8);
      }

      // 3. 参考图真实烟羽流体力学方程 (Gaussian Plume & Buoyancy)
      // 发射源：左侧地面开阔处 [-6.2, 0.15, 3.8]
      let emitter = vec3f(-6.2, 0.15, 3.8);
      // 主风向朝右后方，热浮力使得气流自然上扬
      let windDir = vec3f(1.15, 0.0, -0.85);
      
      let delta = p - emitter;
      // 沿水平主风向的行进距离 x
      let distDownwind = dot(delta.xz, normalize(windDir.xz));
      if (distDownwind < -0.3) { return 0.0; }

      // 热浮力导致的上升轴线方程 (物理抛物线抬升)
      let centerlineY = 0.15 + pow(distDownwind * 0.45, 1.15) * 0.85;
      
      // 沿水平风向中心轴线的横向漂移距离 y
      let centerXZ = emitter.xz + normalize(windDir.xz) * distDownwind;
      let lateralDist = length(p.xz - centerXZ);
      let verticalDist = abs(p.y - centerlineY);

      // 高斯烟羽扩散半径随下风向距离自然变粗 (Gaussian Plume Expansion)
      let plumeWidth = 0.85 + pow(distDownwind * 0.35, 0.82) * 1.15;
      let plumeHeight = 0.75 + pow(distDownwind * 0.35, 0.85) * 1.35;

      // 椭圆高斯截面
      let rNorm = pow(lateralDist / plumeWidth, 2.0) + pow(verticalDist / plumeHeight, 2.0);
      let plumeEnvelope = exp(-rNorm * 1.6);
      if (plumeEnvelope < 0.002) { return 0.0; }

      // 沿主轴前进的平流翻滚
      let t = u.params.x * 0.85;
      let flowVel = vec3f(windDir.x, 0.45, windDir.z);
      let plumeScale = u.smokeColor.a;
      let advectPos = p * (0.38 * plumeScale) - flowVel * t * 1.15;

      // 翻滚卷云噪波 (Billowy Puff Vortices)
      let f1 = billowFbm(advectPos);
      let f2 = smoothNoise(advectPos * 1.8 + vec3f(t * 0.35));
      let densityTurb = f1 * 0.75 + f2 * 0.25;

      // 随着距离拉远，烟羽逐渐自然弥散变薄
      let distanceFade = smoothstep(0.0, 1.0, distDownwind) * smoothstep(22.0, 6.0, distDownwind);
      let heightFade = smoothstep(0.05, 0.4, p.y) * smoothstep(12.0, 5.0, p.y);

      // 结合流体附面层衰减、冲击波与湍流
      let d = smoothstep(0.20, 0.70, densityTurb) * plumeEnvelope * distanceFade * heightFade * boundaryFade;
      return d * 3.8 * shockAttenuation * u.params.y;
    }

    // 双叶 Henyey-Greenstein 相函数
    fn hgPhase(cosTheta: f32, g: f32) -> f32 {
      let g2 = g * g;
      return (1.0 / (4.0 * 3.14159265)) * ((1.0 - g2) / pow(max(0.001, 1.0 + g2 - 2.0 * g * cosTheta), 1.5));
    }

    // AABB 紧密包围盒求交算法
    fn intersectAABB(rayOrigin: vec3f, rayDir: vec3f, boxMin: vec3f, boxMax: vec3f) -> vec2f {
      let invD = 1.0 / rayDir;
      let t0 = (boxMin - rayOrigin) * invD;
      let t1 = (boxMax - rayOrigin) * invD;
      let tmin = min(t0, t1);
      let tmax = max(t0, t1);
      let enter = max(max(tmin.x, tmin.y), tmin.z);
      let exit  = min(min(tmax.x, tmax.y), tmax.z);
      return vec2f(enter, exit);
    }

    @fragment
    fn fs_smoke(in: FullscreenOut) -> @location(0) vec4f {
      let rawDepth = textureLoad(depthTex, vec2u(in.clipPos.xy), 0);
      let ndc = vec4f(in.uv * 2.0 - 1.0, rawDepth, 1.0);
      let worldNear = u.invViewProj * vec4f(in.uv * 2.0 - 1.0, 0.0, 1.0);
      let worldFar  = u.invViewProj * vec4f(in.uv * 2.0 - 1.0, 1.0, 1.0);
      let pNear = worldNear.xyz / worldNear.w;
      let pFar  = worldFar.xyz / worldFar.w;

      let rayDir = normalize(pFar - pNear);
      let sceneWorld = u.invViewProj * ndc;
      let scenePos = sceneWorld.xyz / sceneWorld.w;
      let sceneDist = length(scenePos - u.camPos.xyz);

      // 将步进限制在紧凑的烟羽空间区域，消除混叠噪点
      let boxMin = vec3f(-8.5, 0.05, -10.0);
      let boxMax = vec3f(12.0, 12.0, 8.5);
      let aabbHit = intersectAABB(u.camPos.xyz, rayDir, boxMin, boxMax);

      if (aabbHit.x > aabbHit.y || aabbHit.y < 0.0) {
        return vec4f(0.0);
      }

      let tNear = max(0.0, aabbHit.x);
      let tFar  = min(sceneDist, aabbHit.y);

      if (tNear >= tFar) { return vec4f(0.0); }

      let steps = i32(u.params.w);
      let dt = (tFar - tNear) / f32(steps);
      
      // Bayer 抖动消除颗粒感
      let dither = (bayerDither(in.clipPos.xy) - 0.5) * 0.8 * dt;
      var curT = tNear + max(0.0, dither);

      var transmittance = 1.0;
      var accumulatedColor = vec3f(0.0);

      let L = normalize(u.lightDir.xyz);
      let V = -rayDir;
      let cosTheta = dot(V, -L);
      let phase = mix(hgPhase(cosTheta, -0.15), hgPhase(cosTheta, 0.56), 0.70);

      let smokeAlbedo     = u.smokeColor.rgb;   // 面板实时拾色
      let sunLightColor   = u.lightColor.rgb;   
      let ambientSkyColor = u.ambientColor.rgb; 
      let absorptionCoeff = u.params.z * 1.15;

      for (var i = 0; i < steps; i++) {
        if (curT >= tFar || transmittance < 0.01) { break; }

        let curPos = u.camPos.xyz + rayDir * curT;
        let density = getSmokeDensity(curPos);

        if (density > 0.001) {
          // 深度接触软化 (Soft Intersection)，消除建筑相交硬边
          let distToScene = sceneDist - curT;
          let softFade = smoothstep(0.0, 0.5, distToScene);

          // 两次采样柔和自阴影，立体感极佳
          let shadowStep = 0.65;
          let shadow1 = getSmokeDensity(curPos + L * shadowStep);
          let shadow2 = getSmokeDensity(curPos + L * (shadowStep * 2.0));
          let shadowAtten = exp(-(shadow1 * 0.7 + shadow2 * 0.3) * absorptionCoeff * 2.2);

          // 多重散射提亮 (Powder Sugar)，杜绝死黑
          let powder = 1.0 - exp(-density * dt * 4.2);
          let multipleScatter = ambientSkyColor * (1.2 + powder * 0.8);

          let directScatter = sunLightColor * (shadowAtten * phase * 2.4 + powder * 0.35);
          let S = (directScatter + multipleScatter) * smokeAlbedo;

          // 能量守恒单步消光
          let stepExtinction = exp(-density * softFade * absorptionCoeff * dt);
          let stepScattering = S * (1.0 - stepExtinction);

          accumulatedColor += transmittance * stepScattering;
          transmittance *= stepExtinction;
        }

        curT += dt;
      }

      let alpha = 1.0 - transmittance;
      return vec4f(accumulatedColor, alpha);
    }
  `;

  // =========================================================================
  // 5. 显式创建 BindGroupLayout 与 管线配置
  // =========================================================================
  const sceneModule = device.createShaderModule({ code: sceneShaderCode });
  const smokeModule = device.createShaderModule({ code: smokeShaderCode });

  const sceneBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });

  const scenePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [sceneBindGroupLayout],
  });

  const scenePipeline = device.createRenderPipeline({
    layout: scenePipelineLayout,
    vertex: {
      module: sceneModule,
      entryPoint: "vs_building",
      buffers: [{
        arrayStride: 6 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" },
        ],
      }],
    },
    fragment: { module: sceneModule, entryPoint: "fs_scene", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const groundPipeline = device.createRenderPipeline({
    layout: scenePipelineLayout,
    vertex: { module: sceneModule, entryPoint: "vs_ground" },
    fragment: { module: sceneModule, entryPoint: "fs_scene", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // 烟雾后处理体积管线
  const smokePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: smokeModule, entryPoint: "vs_post" },
    fragment: {
      module: smokeModule,
      entryPoint: "fs_smoke",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        },
      }],
    },
    primitive: { topology: "triangle-list" },
  });

  const sceneBindGroup = device.createBindGroup({
    layout: sceneBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: buildingBuffer } },
    ],
  });

  // =========================================================================
  // 6. GUI 控制面板 (气体颜色、流动参数与动态仿真)
  // =========================================================================
  const settings = {
    // 默认完美对应参考图中的柔和青白冷烟羽
    smokeColor: "#c8dce6",
    lightColor: "#d2e4f0",
    ambientColor: "#425968",

    // 仿真物理流动控制
    density: 1.8,
    absorption: 0.85,
    plumeScale: 1.0,
    smokeSpeed: 1.0,
    stepCount: 72,
    shockStrength: 2.2,
  };

  gui.title("3D 体积流体烟雾仿真");

  const colorFolder = gui.addFolder("气体色彩配置");
  colorFolder.addColor(settings, "smokeColor").name("气体散射体色");
  colorFolder.addColor(settings, "lightColor").name("直射光色调");
  colorFolder.addColor(settings, "ambientColor").name("天光补光色");
  colorFolder.open();

  const fluidFolder = gui.addFolder("流体流动仿真");
  fluidFolder.add(settings, "density", 0.5, 4.0, 0.1).name("气体厚度 (Density)");
  fluidFolder.add(settings, "absorption", 0.2, 2.5, 0.05).name("光吸收衰减");
  fluidFolder.add(settings, "smokeSpeed", 0.0, 3.0, 0.1).name("升腾风速");
  fluidFolder.add(settings, "plumeScale", 0.5, 2.0, 0.05).name("卷云蓬松尺度");
  fluidFolder.add(settings, "shockStrength", 0.5, 5.0, 0.1).name("气浪驱散强度");
  fluidFolder.add(settings, "stepCount", 32, 96, 4).name("步进采样精度");
  fluidFolder.open();

  // 严密贴合参考图构图视角
  const camera = { target: [0.6, 2.2, 0.8], radius: 23.5, theta: 26.0, phi: 21.0 };
  let isDragging = false, dragButton = 0, lastX = 0, lastY = 0;
  let clickDownPos = [0, 0];

  const shockwave = { x: 0.0, z: 0.0, age: -1.0, strength: 2.2 };
  const cachedInvViewProj = new Float32Array(16);

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    isDragging = true;
    dragButton = e.shiftKey ? 2 : e.button;
    lastX = e.clientX;
    lastY = e.clientY;
    clickDownPos = [e.clientX, e.clientY];
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!isDragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    if (dragButton === 0) {
      camera.theta -= dx * 0.35;
      camera.phi = Math.max(5, Math.min(85, camera.phi + dy * 0.35));
    } else if (dragButton === 2) {
      const radTheta = (camera.theta * Math.PI) / 180;
      const pan = camera.radius * 0.0018;
      camera.target[0] -= Math.cos(radTheta) * dx * pan;
      camera.target[2] -= -Math.sin(radTheta) * dx * pan;
      camera.target[1] += dy * pan;
    }
  });

  canvas.addEventListener("pointerup", (e) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}

    const moveDist = Math.hypot(e.clientX - clickDownPos[0], e.clientY - clickDownPos[1]);
    if (moveDist < 6 && e.button === 0) {
      triggerShockwaveAtMouse(e.clientX, e.clientY);
    }
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.radius = Math.max(5, Math.min(60, camera.radius * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });

  // 鼠标点击求地面交点并产生物理气浪推移
  function triggerShockwaveAtMouse(clientX: number, clientY: number) {
    const rect = canvas.getBoundingClientRect();
    const ndcX = ((clientX - rect.left) / rect.width) * 2.0 - 1.0;
    const ndcY = -(((clientY - rect.top) / rect.height) * 2.0 - 1.0);

    const pNear = transformPoint([ndcX, ndcY, 0.0], cachedInvViewProj);
    const pFar  = transformPoint([ndcX, ndcY, 1.0], cachedInvViewProj);

    const dy = pFar[1] - pNear[1];
    if (Math.abs(dy) > 1e-4) {
      const t = (0.25 - pNear[1]) / dy;
      if (t > 0) {
        shockwave.x = pNear[0] + (pFar[0] - pNear[0]) * t;
        shockwave.z = pNear[2] + (pFar[2] - pNear[2]) * t;
        shockwave.age = 0.0;
        shockwave.strength = settings.shockStrength;
      }
    }
  }

  // =========================================================================
  // 7. 渲染帧循环
  // =========================================================================
  let depthTexture: GPUTexture | null = null;
  let smokeBindGroup: GPUBindGroup | null = null;
  let animId: number;
  let startTime = performance.now();
  let lastFrameTime = performance.now();

  function frame() {
    const now = performance.now();
    const dt = (now - lastFrameTime) * 0.001;
    lastFrameTime = now;
    const elapsed = (now - startTime) * 0.001 * settings.smokeSpeed;

    if (shockwave.age >= 0.0) {
      shockwave.age += dt;
      if (shockwave.age > 3.2) { shockwave.age = -1.0; }
    }

    const dpr = window.devicePixelRatio || 1;
    const renderWidth = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const renderHeight = Math.max(1, Math.floor(canvas.clientHeight * dpr));

    if (!depthTexture || depthTexture.width !== renderWidth || depthTexture.height !== renderHeight) {
      canvas.width = renderWidth;
      canvas.height = renderHeight;
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [renderWidth, renderHeight],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });

      smokeBindGroup = device.createBindGroup({
        layout: smokePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: depthTexture.createView() },
        ],
      });
    }

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.target[0] + camera.radius * Math.cos(radPhi) * Math.sin(radTheta),
      camera.target[1] + camera.radius * Math.sin(radPhi),
      camera.target[2] + camera.radius * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const proj = mat4Perspective((42 * Math.PI) / 180, renderWidth / renderHeight, 0.5, 120.0);
    const view = mat4LookAt(eye, camera.target, [0, 1, 0]);
    const viewProj = mat4Multiply(proj, view);
    const invViewProj = mat4Inverse(viewProj);
    cachedInvViewProj.set(invViewProj);

    const smokeRGB = hexToRgb(settings.smokeColor);
    const lightRGB = hexToRgb(settings.lightColor);
    const ambientRGB = hexToRgb(settings.ambientColor);

    const uniformData = new Float32Array(uniformBufferSize / 4);
    uniformData.set(viewProj, 0);
    uniformData.set(invViewProj, 16);
    uniformData[32] = eye[0]; uniformData[33] = eye[1]; uniformData[34] = eye[2]; uniformData[35] = 1.0;
    
    // 主光方向
    uniformData[36] = 0.45; uniformData[37] = 0.85; uniformData[38] = 0.32; uniformData[39] = 0.0;
    
    // 冲击波
    uniformData[40] = shockwave.x;
    uniformData[41] = shockwave.z;
    uniformData[42] = shockwave.age;
    uniformData[43] = shockwave.strength;
    
    // 核心物理参数
    uniformData[44] = elapsed;
    uniformData[45] = settings.density;
    uniformData[46] = settings.absorption;
    uniformData[47] = settings.stepCount;

    // 动态色彩设置
    uniformData[48] = smokeRGB[0]; uniformData[49] = smokeRGB[1]; uniformData[50] = smokeRGB[2]; uniformData[51] = settings.plumeScale;
    uniformData[52] = lightRGB[0]; uniformData[53] = lightRGB[1]; uniformData[54] = lightRGB[2]; uniformData[55] = 1.0;
    uniformData[56] = ambientRGB[0]; uniformData[57] = ambientRGB[1]; uniformData[58] = ambientRGB[2]; uniformData[59] = 1.0;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const currentTextureView = context.getCurrentTexture().createView();
    const depthView = depthTexture.createView();

    // 阶段 1: 渲染实体与地面
    const scenePass = encoder.beginRenderPass({
      colorAttachments: [{
        view: currentTextureView,
        clearValue: { r: 0.12, g: 0.18, b: 0.22, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    scenePass.setPipeline(groundPipeline);
    scenePass.setBindGroup(0, sceneBindGroup);
    scenePass.draw(6, 1, 0, 0);

    scenePass.setPipeline(scenePipeline);
    scenePass.setBindGroup(0, sceneBindGroup);
    scenePass.setVertexBuffer(0, cubeVertexBuffer);
    scenePass.draw(36, buildingDefs.length, 0, 0);
    scenePass.end();

    // 阶段 2: 渲染 3D 体积烟羽
    if (smokeBindGroup) {
      const smokePass = encoder.beginRenderPass({
        colorAttachments: [{
          view: currentTextureView,
          loadOp: "load",
          storeOp: "store",
        }],
      });

      smokePass.setPipeline(smokePipeline);
      smokePass.setBindGroup(0, smokeBindGroup);
      smokePass.draw(3, 1, 0, 0);
      smokePass.end();
    }

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    cubeVertexBuffer.destroy();
    buildingBuffer.destroy();
    uniformBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}