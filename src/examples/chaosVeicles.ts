// src/examples/chaosVehicles.ts
import GUI from "lil-gui";

// =========================================================================
// 1. 标准 3D 矩阵与四元数数学库 (针对 WebGPU 定制)
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
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function mat4FromRotationTranslationScale(q: number[], v: number[], s: number[]): Float32Array {
  const out = new Float32Array(16);
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const sx = s[0], sy = s[1], sz = s[2];

  out[0] = (1 - (yy + zz)) * sx;
  out[1] = (xy + wz) * sx;
  out[2] = (xz - wy) * sx;
  out[3] = 0;

  out[4] = (xy - wz) * sy;
  out[5] = (1 - (xx + zz)) * sy;
  out[6] = (yz + wx) * sy;
  out[7] = 0;

  out[8] = (xz + wy) * sz;
  out[9] = (yz - wx) * sz;
  out[10] = (1 - (xx + yy)) * sz;
  out[11] = 0;

  out[12] = v[0];
  out[13] = v[1];
  out[14] = v[2];
  out[15] = 1;
  return out;
}

function quatMultiply(a: number[], b: number[]): number[] {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

function quatRotateVec3(q: number[], v: number[]): number[] {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const vx = v[0], vy = v[1], vz = v[2];
  const ix = w * vx + y * vz - z * vy;
  const iy = w * vy + z * vx - x * vz;
  const iz = w * vz + x * vy - y * vx;
  const iw = -x * vx - y * vy - z * vz;
  return [
    ix * w + iw * -x + iy * -z - iz * -y,
    iy * w + iw * -y + iz * -x - ix * -z,
    iz * w + iw * -z + ix * -y - iy * -x,
  ];
}

function quatFromAxisAngle(axis: number[], rad: number): number[] {
  const half = rad * 0.5;
  const s = Math.sin(half);
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(half)];
}

function quatNormalize(q: number[]): number[] {
  let len = Math.hypot(q[0], q[1], q[2], q[3]);
  if (len === 0) return [0, 0, 0, 1];
  len = 1 / len;
  return [q[0] * len, q[1] * len, q[2] * len, q[3] * len];
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
// 2. 几何网格构造器
// =========================================================================
function createBoxMesh(hx: number, hy: number, hz: number) {
  // prettier-ignore
  return new Float32Array([
    // Front
    -hx,-hy, hz, 0,0,1,   hx,-hy, hz, 0,0,1,   hx, hy, hz, 0,0,1,
    -hx,-hy, hz, 0,0,1,   hx, hy, hz, 0,0,1,  -hx, hy, hz, 0,0,1,
    // Back
     hx,-hy,-hz, 0,0,-1, -hx,-hy,-hz, 0,0,-1, -hx, hy,-hz, 0,0,-1,
     hx,-hy,-hz, 0,0,-1, -hx, hy,-hz, 0,0,-1,  hx, hy,-hz, 0,0,-1,
    // Top
    -hx, hy, hz, 0,1,0,   hx, hy, hz, 0,1,0,   hx, hy,-hz, 0,1,0,
    -hx, hy, hz, 0,1,0,   hx, hy,-hz, 0,1,0,  -hx, hy,-hz, 0,1,0,
    // Bottom
    -hx,-hy,-hz, 0,-1,0,  hx,-hy,-hz, 0,-1,0,  hx,-hy, hz, 0,-1,0,
    -hx,-hy,-hz, 0,-1,0,  hx,-hy, hz, 0,-1,0, -hx,-hy, hz, 0,-1,0,
    // Right
     hx,-hy, hz, 1,0,0,   hx,-hy,-hz, 1,0,0,   hx, hy,-hz, 1,0,0,
     hx,-hy, hz, 1,0,0,   hx, hy,-hz, 1,0,0,   hx, hy, hz, 1,0,0,
    // Left
    -hx,-hy,-hz,-1,0,0,  -hx,-hy, hz,-1,0,0,  -hx, hy, hz,-1,0,0,
    -hx,-hy,-hz,-1,0,0,  -hx, hy, hz,-1,0,0,  -hx, hy,-hz,-1,0,0,
  ]);
}

function createCylinderMesh(radius: number, halfWidth: number, segments = 24) {
  const verts: number[] = [];
  for (let i = 0; i < segments; i++) {
    const a0 = (i / segments) * Math.PI * 2;
    const a1 = ((i + 1) / segments) * Math.PI * 2;
    const cos0 = Math.cos(a0), sin0 = Math.sin(a0);
    const cos1 = Math.cos(a1), sin1 = Math.sin(a1);

    // 圆柱轮面
    verts.push(-halfWidth, radius * cos0, radius * sin0, 0, cos0, sin0);
    verts.push( halfWidth, radius * cos0, radius * sin0, 0, cos0, sin0);
    verts.push( halfWidth, radius * cos1, radius * sin1, 0, cos1, sin1);

    verts.push(-halfWidth, radius * cos0, radius * sin0, 0, cos0, sin0);
    verts.push( halfWidth, radius * cos1, radius * sin1, 0, cos1, sin1);
    verts.push(-halfWidth, radius * cos1, radius * sin1, 0, cos1, sin1);

    // 轮毂面
    verts.push( halfWidth, 0, 0, 1, 0, 0);
    verts.push( halfWidth, radius * cos0, radius * sin0, 1, 0, 0);
    verts.push( halfWidth, radius * cos1, radius * sin1, 1, 0, 0);

    verts.push(-halfWidth, 0, 0, -1, 0, 0);
    verts.push(-halfWidth, radius * cos1, radius * sin1, -1, 0, 0);
    verts.push(-halfWidth, radius * cos0, radius * sin0, -1, 0, 0);
  }
  return new Float32Array(verts);
}

function getTerrainHeight(x: number, z: number): number {
  var h = 0.0;
  // 特技跳台 (位于 X: [-4.5, 4.5], Z: [16, 27])
  if (x >= -4.5 && x <= 4.5 && z >= 16.0 && z <= 27.0) {
    const rampT = (z - 16.0) / 11.0;
    h += Math.sin(rampT * Math.PI * 0.5) * 3.8;
  }
  // 远景轻微起伏
  h += Math.sin(x * 0.05) * Math.cos(z * 0.05) * 0.8;
  return h;
}

function getTerrainNormal(x: number, z: number): number[] {
  const eps = 0.15;
  const hL = getTerrainHeight(x - eps, z);
  const hR = getTerrainHeight(x + eps, z);
  const hD = getTerrainHeight(x, z - eps);
  const hU = getTerrainHeight(x, z + eps);
  const n = [-((hR - hL) / (2 * eps)), 1.0, -((hU - hD) / (2 * eps))];
  const len = 1 / Math.hypot(n[0], n[1], n[2]);
  return [n[0] * len, n[1] * len, n[2] * len];
}

function createTerrainMesh(size = 140, grid = 70) {
  const verts: number[] = [];
  const half = size / 2;
  const step = size / grid;

  for (let z = 0; z < grid; z++) {
    for (let x = 0; x < grid; x++) {
      const x0 = -half + x * step;
      const z0 = -half + z * step;
      const x1 = x0 + step;
      const z1 = z0 + step;

      const y00 = getTerrainHeight(x0, z0);
      const y10 = getTerrainHeight(x1, z0);
      const y01 = getTerrainHeight(x0, z1);
      const y11 = getTerrainHeight(x1, z1);

      const n00 = getTerrainNormal(x0, z0);
      const n10 = getTerrainNormal(x1, z0);
      const n01 = getTerrainNormal(x0, z1);
      const n11 = getTerrainNormal(x1, z1);

      verts.push(x0, y00, z0, n00[0], n00[1], n00[2]);
      verts.push(x0, y01, z1, n01[0], n01[1], n01[2]);
      verts.push(x1, y00, z0, n10[0], n10[1], n10[2]);

      verts.push(x1, y00, z0, n10[0], n10[1], n10[2]);
      verts.push(x0, y01, z1, n01[0], n01[1], n01[2]);
      verts.push(x1, y11, z1, n11[0], n11[1], n11[2]);
    }
  }
  return new Float32Array(verts);
}

// =========================================================================
// 3. 主程序入口
// =========================================================================
export function runChaosVehicles(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: GUI
) {
  const chassisBox = createBoxMesh(1.15, 0.35, 2.35);
  const cabinBox = createBoxMesh(0.85, 0.30, 1.25);
  const spoilerBox = createBoxMesh(1.05, 0.05, 0.25);
  const wheelCylinder = createCylinderMesh(0.44, 0.18, 28);
  const terrainMesh = createTerrainMesh(160, 80);

  function createGpuBuffer(data: Float32Array, usage: GPUBufferUsageFlags) {
    const buf = device.createBuffer({
      size: Math.max(data.byteLength, 16),
      usage: usage | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, data as any);
    return buf;
  }

  const chassisVertexBuffer = createGpuBuffer(chassisBox, GPUBufferUsage.VERTEX);
  const cabinVertexBuffer = createGpuBuffer(cabinBox, GPUBufferUsage.VERTEX);
  const spoilerVertexBuffer = createGpuBuffer(spoilerBox, GPUBufferUsage.VERTEX);
  const wheelVertexBuffer = createGpuBuffer(wheelCylinder, GPUBufferUsage.VERTEX);
  const terrainVertexBuffer = createGpuBuffer(terrainMesh, GPUBufferUsage.VERTEX);

  const uniformBufferSize = 1024;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 4. 着色器实现 (修复内存对齐与平坦离散属性)
  // =========================================================================
  const sceneShaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,     // 0..15  (0 bytes)
      camPos: vec4f,         // 16..19 (64 bytes)
      lightDir: vec4f,       // 20..23 (80 bytes)
      carColor: vec4f,       // 24..27 (96 bytes)
      pad: vec4f,            // 28..31 (112 bytes) -> 【关键修复】：填充保证下方的矩阵严格以 128 bytes (32 floats) 对齐
      chassisMat: mat4x4f,   // 32..47 (128 bytes)
      cabinMat: mat4x4f,     // 48..63 (192 bytes)
      spoilerMat: mat4x4f,   // 64..79 (256 bytes)
      wheelMat0: mat4x4f,    // 80..95 (320 bytes)
      wheelMat1: mat4x4f,    // 96..111 (384 bytes)
      wheelMat2: mat4x4f,    // 112..127 (448 bytes)
      wheelMat3: mat4x4f,    // 128..143 (512 bytes)
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexOut {
      @builtin(position) clipPos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) @interpolate(flat) partType: u32, // 【关键修复】：采用 flat 整数，禁止在三角形间线性插值导致误入地砖着色
    };

    @vertex
    fn vs_chassis(@location(0) pos: vec3f, @location(1) normal: vec3f) -> VertexOut {
      var out: VertexOut;
      let wPos = u.chassisMat * vec4f(pos, 1.0);
      out.clipPos = u.viewProj * wPos;
      out.worldPos = wPos.xyz;
      out.normal = (u.chassisMat * vec4f(normal, 0.0)).xyz;
      out.partType = 0u;
      return out;
    }

    @vertex
    fn vs_cabin(@location(0) pos: vec3f, @location(1) normal: vec3f) -> VertexOut {
      var out: VertexOut;
      let wPos = u.cabinMat * vec4f(pos, 1.0);
      out.clipPos = u.viewProj * wPos;
      out.worldPos = wPos.xyz;
      out.normal = (u.cabinMat * vec4f(normal, 0.0)).xyz;
      out.partType = 1u;
      return out;
    }

    @vertex
    fn vs_spoiler(@location(0) pos: vec3f, @location(1) normal: vec3f) -> VertexOut {
      var out: VertexOut;
      let wPos = u.spoilerMat * vec4f(pos, 1.0);
      out.clipPos = u.viewProj * wPos;
      out.worldPos = wPos.xyz;
      out.normal = (u.spoilerMat * vec4f(normal, 0.0)).xyz;
      out.partType = 4u;
      return out;
    }

    @vertex
    fn vs_wheel(@location(0) pos: vec3f, @location(1) normal: vec3f, @builtin(instance_index) instIdx: u32) -> VertexOut {
      var out: VertexOut;
      var m = u.wheelMat0;
      if (instIdx == 1u) { m = u.wheelMat1; }
      else if (instIdx == 2u) { m = u.wheelMat2; }
      else if (instIdx == 3u) { m = u.wheelMat3; }

      let wPos = m * vec4f(pos, 1.0);
      out.clipPos = u.viewProj * wPos;
      out.worldPos = wPos.xyz;
      out.normal = (m * vec4f(normal, 0.0)).xyz;
      out.partType = 2u;
      return out;
    }

    @vertex
    fn vs_terrain(@location(0) pos: vec3f, @location(1) normal: vec3f) -> VertexOut {
      var out: VertexOut;
      out.clipPos = u.viewProj * vec4f(pos, 1.0);
      out.worldPos = pos;
      out.normal = normal;
      out.partType = 3u;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4f {
      let N = normalize(in.normal);
      let L = normalize(u.lightDir.xyz);
      let V = normalize(u.camPos.xyz - in.worldPos);
      let H = normalize(L + V);

      // 在均一流中提前求导
      let terrainCoord = in.worldPos.xz * 0.45;
      let terrainFw = fwidth(terrainCoord);

      let NdotL = max(dot(N, L), 0.0);
      let NdotV = max(dot(N, V), 0.0);
      let fresnel = pow(1.0 - NdotV, 4.0);

      var baseColor = vec3f(0.8);
      var roughness = 0.3;
      var specularPower = 64.0;
      var emissive = vec3f(0.0);

      if (in.partType == 0u) {
        // 主车身亮漆
        baseColor = u.carColor.rgb;
        roughness = 0.15;
        specularPower = 120.0;

        // 车尾刹车灯
        if (in.normal.z < -0.7 && in.worldPos.y > 0.4) {
          if (u.carColor.a > 0.5) {
            emissive = vec3f(4.0, 0.05, 0.05); // 刹车爆亮红光
          } else {
            emissive = vec3f(0.5, 0.02, 0.02);
          }
        }
      } else if (in.partType == 1u) {
        // 驾驶舱有色黑晶玻璃
        baseColor = vec3f(0.06, 0.08, 0.12);
        roughness = 0.05;
        specularPower = 180.0;
      } else if (in.partType == 2u) {
        // 轮胎橡胶与金属轮毂
        if (abs(in.normal.x) > 0.8) {
          baseColor = vec3f(0.78, 0.80, 0.85); // 轮毂银色
          roughness = 0.2;
          specularPower = 90.0;
        } else {
          baseColor = vec3f(0.12, 0.12, 0.13); // 胎面哑光黑
          roughness = 0.85;
          specularPower = 16.0;
        }
      } else if (in.partType == 4u) {
        // 碳纤维尾翼
        baseColor = vec3f(0.15, 0.15, 0.16);
        roughness = 0.25;
      } else {
        // 赛道地面与特技跳台
        let grid = abs(fract(terrainCoord - 0.5) - 0.5) / max(terrainFw, vec2f(0.001));
        let line = min(grid.x, grid.y);
        let gridMask = 1.0 - min(line, 1.0);

        let asphalt = vec3f(0.14, 0.17, 0.21);
        let gridLineCol = vec3f(0.24, 0.38, 0.48);
        baseColor = mix(asphalt, gridLineCol, gridMask * 0.75);

        // 特技跳台黄色警示条纹
        if (in.worldPos.z >= 16.0 && in.worldPos.z <= 27.0 && abs(in.worldPos.x) <= 4.5) {
          let stripe = step(0.5, fract(in.worldPos.z * 0.5));
          baseColor = mix(vec3f(0.85, 0.70, 0.10), vec3f(0.15), stripe);
        }
        roughness = 0.8;
      }

      let spec = pow(max(dot(N, H), 0.0), specularPower) * (1.0 - roughness);
      let ambient = vec3f(0.14, 0.18, 0.24) * baseColor;
      let diffuse = baseColor * NdotL * 0.8;
      let coat = vec3f(1.0) * fresnel * (1.0 - roughness);

      let finalCol = ambient + diffuse + vec3f(spec) + coat + emissive;
      return vec4f(finalCol, 1.0);
    }
  `;

  // =========================================================================
  // 5. 创建渲染管线
  // =========================================================================
  const sceneModule = device.createShaderModule({ code: sceneShaderCode });

  const sceneBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ],
  });

  const scenePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [sceneBindGroupLayout],
  });

  const vertexLayout: GPUVertexBufferLayout = {
    arrayStride: 6 * 4,
    attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" },
      { shaderLocation: 1, offset: 12, format: "float32x3" },
    ],
  };

  function createPipeline(entryPoint: string, topology: GPUPrimitiveTopology = "triangle-list") {
    return device.createRenderPipeline({
      layout: scenePipelineLayout,
      vertex: { module: sceneModule, entryPoint, buffers: [vertexLayout] },
      fragment: { module: sceneModule, entryPoint: "fs_main", targets: [{ format }] },
      primitive: { topology, cullMode: "back" },
      depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    });
  }

  const chassisPipeline = createPipeline("vs_chassis");
  const cabinPipeline = createPipeline("vs_cabin");
  const spoilerPipeline = createPipeline("vs_spoiler");
  const wheelPipeline = createPipeline("vs_wheel");
  const terrainPipeline = createPipeline("vs_terrain");

  const sceneBindGroup = device.createBindGroup({
    layout: sceneBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // =========================================================================
  // 6. 物理引擎与控制
  // =========================================================================
  const settings = {
    engineHorsepower: 420,
    topSpeedKmh: 180,
    brakeTorque: 60,

    suspensionRestLength: 0.75,
    springStiffness: 52.0,
    damperRate: 5.2,
    antiRollBarRate: 20.0,

    tireGripFront: 3.4,
    tireGripRear: 2.8,
    handbrakeDriftSlip: 0.35,

    carColor: "#e63946",
    resetCar: () => resetVehicle(),
  };

  const vehicle = {
    pos: [0, 2.5, 0],
    quat: [0, 0, 0, 1],
    linearVelocity: [0, 0, 0],
    angularVelocity: [0, 0, 0],
    mass: 1450,

    wheelMounts: [
      [-1.15, -0.15,  1.45],
      [ 1.15, -0.15,  1.45],
      [-1.15, -0.15, -1.45],
      [ 1.15, -0.15, -1.45],
    ],
    steeringAngle: 0,
    wheelRotations: [0, 0, 0, 0],
    suspensionCompressions: [0, 0, 0, 0],
    isGrounded: [false, false, false, false],
  };

  function resetVehicle() {
    vehicle.pos = [0, 2.8, 0];
    vehicle.quat = [0, 0, 0, 1];
    vehicle.linearVelocity = [0, 0, 0];
    vehicle.angularVelocity = [0, 0, 0];
    vehicle.steeringAngle = 0;
  }

  const keys = { KeyW: false, KeyS: false, KeyA: false, KeyD: false, Space: false };
  window.addEventListener("keydown", (e) => {
    if (e.code in keys) {
      keys[e.code as keyof typeof keys] = true;
      e.preventDefault();
    }
  });
  window.addEventListener("keyup", (e) => {
    if (e.code in keys) {
      keys[e.code as keyof typeof keys] = false;
      e.preventDefault();
    }
  });

  const camera = {
    pos: [0, 5.0, -8.5],
    distance: 8.5,
    height: 3.2,
  };

  // GUI
  gui.title("WebGPU Chaos Vehicles 载具动力学");
  const driveFolder = gui.addFolder("动力与转向");
  driveFolder.add(settings, "engineHorsepower", 150, 800, 10).name("引擎马力");
  driveFolder.add(settings, "brakeTorque", 20, 100, 2).name("刹车制动力");
  driveFolder.open();

  const suspFolder = gui.addFolder("悬挂与轮胎力学");
  suspFolder.add(settings, "springStiffness", 20, 90, 1).name("弹簧刚度 (K)");
  suspFolder.add(settings, "damperRate", 1.0, 10.0, 0.2).name("减震阻尼 (C)");
  suspFolder.add(settings, "tireGripRear", 1.0, 5.0, 0.1).name("后轮抓地力");
  suspFolder.open();

  const appFolder = gui.addFolder("涂装外观");
  appFolder.addColor(settings, "carColor").name("车身漆面");
  gui.add(settings, "resetCar").name("重置车辆 (Reset)");

  // HUD
  const hud = document.createElement("div");
  hud.style.position = "absolute";
  hud.style.left = "20px";
  hud.style.bottom = "20px";
  hud.style.fontFamily = "monospace";
  hud.style.color = "#4cc9f0";
  hud.style.fontSize = "16px";
  hud.style.background = "rgba(10, 18, 26, 0.75)";
  hud.style.padding = "14px 20px";
  hud.style.borderRadius = "8px";
  hud.style.pointerEvents = "none";
  hud.style.boxShadow = "0 4px 16px rgba(0,0,0,0.5)";
  hud.style.border = "1px solid rgba(76, 201, 240, 0.3)";
  hud.innerHTML = `
    <div style="font-weight: bold; font-size: 20px; color: #fff; margin-bottom: 6px;">CHAOS VEHICLE TELEMETRY</div>
    <div>操作: [W]油门加速  [S]刹车/倒车  [A/D]转向  [空格]手刹漂移</div>
    <div id="hud-speed" style="font-size: 28px; color: #f72585; font-weight: bold; margin: 8px 0;">0 KM/H</div>
    <div id="hud-susp">悬挂行程: FL: 0% | FR: 0% | RL: 0% | RR: 0%</div>
  `;
  canvas.parentElement?.appendChild(hud);

  // =========================================================================
  // 7. 物理计算与渲染循环
  // =========================================================================
  let depthTexture: GPUTexture | null = null;
  let animId: number;
  let lastTime = performance.now();

  function updatePhysics(dt: number) {
    const targetSteer = (keys.KeyA ? 1 : 0) - (keys.KeyD ? 1 : 0);
    vehicle.steeringAngle += (targetSteer * 0.55 - vehicle.steeringAngle) * Math.min(1.0, dt * 10.0);

    const fwd = quatRotateVec3(vehicle.quat, [0, 0, 1]);
    const up = quatRotateVec3(vehicle.quat, [0, 1, 0]);
    const right = quatRotateVec3(vehicle.quat, [1, 0, 0]);

    const totalForce = [0, -9.8 * vehicle.mass, 0];
    const totalTorque = [0, 0, 0];
    const currentSpeed = (vehicle.linearVelocity[0] * fwd[0] + vehicle.linearVelocity[1] * fwd[1] + vehicle.linearVelocity[2] * fwd[2]);

    for (let i = 0; i < 4; i++) {
      const mountLocal = vehicle.wheelMounts[i];
      const mountWorld = quatRotateVec3(vehicle.quat, mountLocal);
      const rayOrigin = [
        vehicle.pos[0] + mountWorld[0],
        vehicle.pos[1] + mountWorld[1],
        vehicle.pos[2] + mountWorld[2],
      ];

      const groundH = getTerrainHeight(rayOrigin[0], rayOrigin[2]);
      const rayLen = settings.suspensionRestLength + 0.44;
      const distToGround = rayOrigin[1] - groundH;

      if (distToGround < rayLen) {
        vehicle.isGrounded[i] = true;
        const compression = rayLen - distToGround;
        vehicle.suspensionCompressions[i] = compression;

        const springForce = compression * (settings.springStiffness * 1000);
        const pointVelY = vehicle.linearVelocity[1] + (vehicle.angularVelocity[0] * mountWorld[2] - vehicle.angularVelocity[2] * mountWorld[0]);
        const damperForce = -pointVelY * (settings.damperRate * 900);

        const normalForceMag = Math.max(0, springForce + damperForce);

        totalForce[0] += up[0] * normalForceMag;
        totalForce[1] += up[1] * normalForceMag;
        totalForce[2] += up[2] * normalForceMag;

        totalTorque[0] += mountWorld[1] * (up[2] * normalForceMag) - mountWorld[2] * (up[1] * normalForceMag);
        totalTorque[1] += mountWorld[2] * (up[0] * normalForceMag) - mountWorld[0] * (up[2] * normalForceMag);
        totalTorque[2] += mountWorld[0] * (up[1] * normalForceMag) - mountWorld[1] * (up[0] * normalForceMag);

        const isFront = i < 2;
        const steer = isFront ? vehicle.steeringAngle : 0;
        const wheelFwd = [
          fwd[0] * Math.cos(steer) + right[0] * Math.sin(steer),
          fwd[1],
          fwd[2] * Math.cos(steer) + right[2] * Math.sin(steer),
        ];
        const wheelRight = [-wheelFwd[2], 0, wheelFwd[0]];

        var driveMag = 0;
        if (keys.KeyW) { driveMag = settings.engineHorsepower * 35; }
        if (keys.KeyS) { driveMag = -settings.engineHorsepower * 25; }
        if (keys.Space) { driveMag = 0; }

        totalForce[0] += wheelFwd[0] * driveMag;
        totalForce[2] += wheelFwd[2] * driveMag;

        const sideSlipVel = vehicle.linearVelocity[0] * wheelRight[0] + vehicle.linearVelocity[2] * wheelRight[2];
        var gripFactor = isFront ? settings.tireGripFront : settings.tireGripRear;
        if (keys.Space && !isFront) {
          gripFactor *= settings.handbrakeDriftSlip;
        }

        const lateralForce = -sideSlipVel * (gripFactor * 1800);
        totalForce[0] += wheelRight[0] * lateralForce;
        totalForce[2] += wheelRight[2] * lateralForce;

        totalTorque[1] += (mountWorld[0] * wheelRight[2] - mountWorld[2] * wheelRight[0]) * lateralForce * 0.15;
        vehicle.wheelRotations[i] += (currentSpeed / 0.44) * dt;
      } else {
        vehicle.isGrounded[i] = false;
        vehicle.suspensionCompressions[i] = 0;
      }
    }

    totalForce[0] -= vehicle.linearVelocity[0] * 65.0;
    totalForce[2] -= vehicle.linearVelocity[2] * 65.0;
    vehicle.angularVelocity[0] *= 0.96;
    vehicle.angularVelocity[1] *= 0.95;
    vehicle.angularVelocity[2] *= 0.96;

    const invMass = 1.0 / vehicle.mass;
    vehicle.linearVelocity[0] += totalForce[0] * invMass * dt;
    vehicle.linearVelocity[1] += totalForce[1] * invMass * dt;
    vehicle.linearVelocity[2] += totalForce[2] * invMass * dt;

    vehicle.pos[0] += vehicle.linearVelocity[0] * dt;
    vehicle.pos[1] += vehicle.linearVelocity[1] * dt;
    vehicle.pos[2] += vehicle.linearVelocity[2] * dt;

    const invInertia = invMass * 1.5;
    vehicle.angularVelocity[0] += totalTorque[0] * invInertia * dt;
    vehicle.angularVelocity[1] += totalTorque[1] * invInertia * dt;
    vehicle.angularVelocity[2] += totalTorque[2] * invInertia * dt;

    const wLen = Math.hypot(vehicle.angularVelocity[0], vehicle.angularVelocity[1], vehicle.angularVelocity[2]);
    if (wLen > 1e-5) {
      const dq = quatFromAxisAngle(
        [vehicle.angularVelocity[0] / wLen, vehicle.angularVelocity[1] / wLen, vehicle.angularVelocity[2] / wLen],
        wLen * dt
      );
      vehicle.quat = quatNormalize(quatMultiply(dq, vehicle.quat));
    }
  }

  function frame() {
    const now = performance.now();
    const rawDt = Math.min((now - lastTime) * 0.001, 0.05);
    lastTime = now;

    const subSteps = 4;
    const subDt = rawDt / subSteps;
    for (let s = 0; s < subSteps; s++) {
      updatePhysics(subDt);
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
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    // 追随相机
    const fwd = quatRotateVec3(vehicle.quat, [0, 0, 1]);
    const targetCamPos = [
      vehicle.pos[0] - fwd[0] * camera.distance,
      vehicle.pos[1] + camera.height,
      vehicle.pos[2] - fwd[2] * camera.distance,
    ];
    camera.pos[0] += (targetCamPos[0] - camera.pos[0]) * 0.12;
    camera.pos[1] += (targetCamPos[1] - camera.pos[1]) * 0.12;
    camera.pos[2] += (targetCamPos[2] - camera.pos[2]) * 0.12;

    const proj = mat4Perspective((48 * Math.PI) / 180, renderWidth / renderHeight, 0.2, 300.0);
    const view = mat4LookAt(camera.pos, [vehicle.pos[0], vehicle.pos[1] + 0.8, vehicle.pos[2]], [0, 1, 0]);
    const viewProj = mat4Multiply(proj, view);

    // 计算各部件矩阵
    const chassisMat = mat4FromRotationTranslationScale(vehicle.quat, vehicle.pos, [1, 1, 1]);

    const cabinOffset = quatRotateVec3(vehicle.quat, [0, 0.45, -0.15]);
    const cabinPos = [vehicle.pos[0] + cabinOffset[0], vehicle.pos[1] + cabinOffset[1], vehicle.pos[2] + cabinOffset[2]];
    const cabinMat = mat4FromRotationTranslationScale(vehicle.quat, cabinPos, [1, 1, 1]);

    const spoilerOffset = quatRotateVec3(vehicle.quat, [0, 0.65, -2.15]);
    const spoilerPos = [vehicle.pos[0] + spoilerOffset[0], vehicle.pos[1] + spoilerOffset[1], vehicle.pos[2] + spoilerOffset[2]];
    const spoilerMat = mat4FromRotationTranslationScale(vehicle.quat, spoilerPos, [1, 1, 1]);

    const wheelMats: Float32Array[] = [];
    for (let i = 0; i < 4; i++) {
      const isFront = i < 2;
      const steer = isFront ? vehicle.steeringAngle : 0;
      const steerQ = quatFromAxisAngle([0, 1, 0], steer);
      const rollQ = quatFromAxisAngle([1, 0, 0], vehicle.wheelRotations[i]);
      const wheelLocalQ = quatMultiply(steerQ, rollQ);
      const wheelWorldQ = quatMultiply(vehicle.quat, wheelLocalQ);

      const mount = vehicle.wheelMounts[i];
      const suspDisplacement = (0.75 - vehicle.suspensionCompressions[i]);
      const wOffset = quatRotateVec3(vehicle.quat, [mount[0], mount[1] - suspDisplacement + 0.44, mount[2]]);
      const wPos = [vehicle.pos[0] + wOffset[0], vehicle.pos[1] + wOffset[1], vehicle.pos[2] + wOffset[2]];

      wheelMats.push(mat4FromRotationTranslationScale(wheelWorldQ, wPos, [1, 1, 1]));
    }

    // 写入精确对齐的常量缓冲区
    const carRGB = hexToRgb(settings.carColor);
    const uniformData = new Float32Array(uniformBufferSize / 4);
    uniformData.set(viewProj, 0); // 0..15
    uniformData[16] = camera.pos[0]; uniformData[17] = camera.pos[1]; uniformData[18] = camera.pos[2]; uniformData[19] = 1.0;
    uniformData[20] = 0.45; uniformData[21] = 0.85; uniformData[22] = 0.35; uniformData[23] = 0.0;
    uniformData[24] = carRGB[0]; uniformData[25] = carRGB[1]; uniformData[26] = carRGB[2];
    uniformData[27] = (keys.KeyS || keys.Space) ? 1.0 : 0.0; // 刹车灯

    // 28..31 为填充 pad
    uniformData.set(chassisMat, 32);
    uniformData.set(cabinMat, 48);
    uniformData.set(spoilerMat, 64);
    uniformData.set(wheelMats[0], 80);
    uniformData.set(wheelMats[1], 96);
    uniformData.set(wheelMats[2], 112);
    uniformData.set(wheelMats[3], 128);

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const currentView = context.getCurrentTexture().createView();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: currentView,
        clearValue: { r: 0.08, g: 0.11, b: 0.15, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    renderPass.setPipeline(terrainPipeline);
    renderPass.setBindGroup(0, sceneBindGroup);
    renderPass.setVertexBuffer(0, terrainVertexBuffer);
    renderPass.draw(terrainMesh.length / 6);

    renderPass.setPipeline(chassisPipeline);
    renderPass.setBindGroup(0, sceneBindGroup);
    renderPass.setVertexBuffer(0, chassisVertexBuffer);
    renderPass.draw(36);

    renderPass.setPipeline(cabinPipeline);
    renderPass.setBindGroup(0, sceneBindGroup);
    renderPass.setVertexBuffer(0, cabinVertexBuffer);
    renderPass.draw(36);

    renderPass.setPipeline(spoilerPipeline);
    renderPass.setBindGroup(0, sceneBindGroup);
    renderPass.setVertexBuffer(0, spoilerVertexBuffer);
    renderPass.draw(36);

    renderPass.setPipeline(wheelPipeline);
    renderPass.setBindGroup(0, sceneBindGroup);
    renderPass.setVertexBuffer(0, wheelVertexBuffer);
    renderPass.draw(wheelCylinder.length / 6, 4, 0, 0);

    renderPass.end();
    device.queue.submit([encoder.finish()]);

    const speedKmh = Math.floor(Math.hypot(vehicle.linearVelocity[0], vehicle.linearVelocity[2]) * 3.6);
    const speedEl = document.getElementById("hud-speed");
    if (speedEl) speedEl.innerText = `${speedKmh} KM/H`;

    const suspEl = document.getElementById("hud-susp");
    if (suspEl) {
      const p = vehicle.suspensionCompressions.map((c) => Math.min(100, Math.floor((c / 0.75) * 100)));
      suspEl.innerText = `悬挂行程: FL:${p[0]}% | FR:${p[1]}% | RL:${p[2]}% | RR:${p[3]}%`;
    }

    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    hud.remove();
    chassisVertexBuffer.destroy();
    cabinVertexBuffer.destroy();
    spoilerVertexBuffer.destroy();
    wheelVertexBuffer.destroy();
    terrainVertexBuffer.destroy();
    uniformBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}