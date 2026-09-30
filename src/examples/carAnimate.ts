import type { SimpleGUI } from "../utils/gui";

// ======================== 1. 矩阵与三维数学函数 ========================
function createPerspectiveMatrix(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(fovRad / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect; out[5] = f;
  out[10] = far / (near - far); out[11] = -1;
  out[14] = (near * far) / (near - far);
  return out;
}

function createLookAtMatrix(eye: number[], center: number[], up: number[]): Float32Array {
  const z = [eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]];
  const lenZ = 1 / (Math.hypot(z[0], z[1], z[2]) || 1);
  z[0] *= lenZ; z[1] *= lenZ; z[2] *= lenZ;
  const x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]];
  const lenX = 1 / (Math.hypot(x[0], x[1], x[2]) || 1);
  x[0] *= lenX; x[1] *= lenX; x[2] *= lenX;
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const out = new Float32Array(16);
  out[0] = x[0]; out[1] = y[0]; out[2] = z[0]; out[3] = 0;
  out[4] = x[1]; out[5] = y[1]; out[6] = z[1]; out[7] = 0;
  out[8] = x[2]; out[9] = y[2]; out[10] = z[2]; out[11] = 0;
  out[12] = -(x[0]*eye[0] + x[1]*eye[1] + x[2]*eye[2]);
  out[13] = -(y[0]*eye[0] + y[1]*eye[1] + y[2]*eye[2]);
  out[14] = -(z[0]*eye[0] + z[1]*eye[1] + z[2]*eye[2]);
  out[15] = 1;
  return out;
}

function multiplyMat4(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      out[i * 4 + j] =
        a[j] * b[i * 4] + a[4 + j] * b[i * 4 + 1] + a[8 + j] * b[i * 4 + 2] + a[12 + j] * b[i * 4 + 3];
    }
  }
  return out;
}

// ======================== 2. 闭合样条路线库 (多路线系统) ========================
type Waypoint = [number, number, number];

const ROUTE_0: Waypoint[] = [
  [-24.0, 0.0,  18.0],
  [ -8.0, 0.0,  24.0],
  [ 15.0, 0.0,  20.0],
  [ 25.0, 0.0,   7.0],
  [ 22.0, 0.0, -16.0],
  [  7.0, 0.0, -24.0],
  [-16.0, 0.0, -20.0],
  [-25.0, 0.0,  -2.0],
];

const ROUTE_1: Waypoint[] = [
  [  0.0, 0.0,   0.0],
  [ 11.0, 0.0,  13.0],
  [ 17.0, 1.4,   0.0],
  [ 11.0, 2.8, -13.0],
  [  0.0, 2.2,   0.0],
  [-11.0, 1.1,  13.0],
  [-17.0, 0.2,   0.0],
  [-11.0, 0.0, -13.0],
];

const ROUTE_2: Waypoint[] = [
  [-8.0, 0.0,   8.0],
  [ 7.0, 0.0,   9.0],
  [13.0, 0.0,   3.0],
  [ 9.0, 0.0,  -7.0],
  [-1.0, 0.0, -10.0],
  [-10.0, 0.0, -6.0],
  [-12.0, 0.0,   2.0],
];

const ALL_ROUTES = [ROUTE_0, ROUTE_1, ROUTE_2];

function getRoutePoint(waypoints: Waypoint[], progress: number): [number, number, number] {
  const count = waypoints.length;
  const p = ((progress % 1.0) + 1.0) % 1.0;
  const floatIdx = p * count;
  const i = Math.floor(floatIdx);
  const u = floatIdx - i;

  const p0 = waypoints[(i - 1 + count) % count];
  const p1 = waypoints[i % count];
  const p2 = waypoints[(i + 1) % count];
  const p3 = waypoints[(i + 2) % count];

  const u2 = u * u;
  const u3 = u2 * u;

  const f0 = -0.5 * u3 + u2 - 0.5 * u;
  const f1 =  1.5 * u3 - 2.5 * u2 + 1.0;
  const f2 = -1.5 * u3 + 2.0 * u2 + 0.5 * u;
  const f3 =  0.5 * u3 - 0.5 * u2;

  return [
    p0[0] * f0 + p1[0] * f1 + p2[0] * f2 + p3[0] * f3,
    p0[1] * f0 + p1[1] * f1 + p2[1] * f2 + p3[1] * f3,
    p0[2] * f0 + p1[2] * f1 + p2[2] * f2 + p3[2] * f3,
  ];
}

function getVehiclePoseMatrix(waypoints: Waypoint[], progress: number): Float32Array {
  const cur = getRoutePoint(waypoints, progress);
  const next = getRoutePoint(waypoints, progress + 0.002);
  const prev = getRoutePoint(waypoints, progress - 0.002);

  let fx = next[0] - cur[0];
  let fy = next[1] - cur[1];
  let fz = next[2] - cur[2];
  const flen = Math.hypot(fx, fy, fz) || 1;
  fx /= flen; fy /= flen; fz /= flen;

  const ax = (next[0] - 2 * cur[0] + prev[0]) * 400;
  const az = (next[2] - 2 * cur[2] + prev[2]) * 400;

  let ux = -az * 0.15;
  let uy = 1.0;
  let uz = ax * 0.15;

  let rx = uy * fz - uz * fy;
  let ry = uz * fx - ux * fz;
  let rz = ux * fy - uy * fx;
  const rlen = Math.hypot(rx, ry, rz) || 1;
  rx /= rlen; ry /= rlen; rz /= rlen;

  ux = fy * rz - fz * ry;
  uy = fz * rx - fx * rz;
  uz = fx * ry - fy * rx;

  const m = new Float32Array(16);
  m[0] = rx; m[1] = ry; m[2] = rz; m[3] = 0;
  m[4] = ux; m[5] = uy; m[6] = uz; m[7] = 0;
  m[8] = fx; m[9] = fy; m[10]= fz; m[11]= 0;
  m[12]= cur[0]; m[13]= cur[1]; m[14]= cur[2]; m[15]= 1;
  return m;
}

// ======================== 3. 精致跑车几何构建 ========================
function createSportsCarMesh() {
  const v: number[] = [];

  function addBox(min: number[], max: number[], matId: number) {
    const p = [
      [min[0], min[1], max[2]], [max[0], min[1], max[2]], [max[0], max[1], max[2]], [min[0], max[1], max[2]],
      [max[0], min[1], min[2]], [min[0], min[1], min[2]], [min[0], max[1], min[2]], [max[0], max[1], min[2]],
      [min[0], max[1], max[2]], [max[0], max[1], max[2]], [max[0], max[1], min[2]], [min[0], max[1], min[2]],
      [min[0], min[1], min[2]], [max[0], min[1], min[2]], [max[0], min[1], max[2]], [min[0], min[1], max[2]],
      [max[0], min[1], max[2]], [max[0], min[1], min[2]], [max[0], max[1], min[2]], [max[0], max[1], max[2]],
      [min[0], min[1], min[2]], [min[0], min[1], max[2]], [min[0], max[1], max[2]], [min[0], max[1], min[2]],
    ];
    const norms = [
      [0,0,1], [0,0,-1], [0,1,0], [0,-1,0], [1,0,0], [-1,0,0]
    ];
    for (let f = 0; f < 6; f++) {
      const n = norms[f];
      const i0 = f * 4;
      const quad = [0, 1, 2, 0, 2, 3];
      for (const q of quad) {
        const pt = p[i0 + q];
        v.push(pt[0], pt[1], pt[2], n[0], n[1], n[2], matId);
      }
    }
  }

  addBox([-0.65, 0.12, -1.3], [0.65, 0.42, 1.3], 0);
  addBox([-0.60, 0.08,  1.2], [0.60, 0.22, 1.45], 0);
  addBox([-0.45, 0.42, -0.45], [0.45, 0.78, 0.45], 1);
  addBox([-0.42, 0.78, -0.4], [0.42, 0.82, 0.4], 0);
  addBox([-0.60, 0.65, -1.35], [0.60, 0.70, -1.15], 0);
  addBox([-0.45, 0.42, -1.25], [-0.38, 0.65, -1.20], 0);
  addBox([ 0.38, 0.42, -1.25], [ 0.45, 0.65, -1.20], 0);
  addBox([-0.76, 0.0,  0.55], [-0.58, 0.42,  0.95], 2);
  addBox([ 0.58, 0.0,  0.55], [ 0.76, 0.42,  0.95], 2);
  addBox([-0.76, 0.0, -0.95], [-0.58, 0.42, -0.55], 2);
  addBox([ 0.58, 0.0, -0.95], [ 0.76, 0.42, -0.55], 2);
  addBox([-0.55, 0.26, 1.32], [-0.25, 0.36, 1.35], 3);
  addBox([ 0.25, 0.26, 1.32], [ 0.55, 0.36, 1.35], 3);
  addBox([-0.58, 0.30, -1.33], [0.58, 0.38, -1.30], 4);

  return new Float32Array(v);
}

function createTrackLineMesh(waypoints: Waypoint[], color: number[], segments = 180) {
  const v: number[] = [];
  for (let i = 0; i <= segments; i++) {
    const pt = getRoutePoint(waypoints, i / segments);
    v.push(pt[0], pt[1] + 0.05, pt[2], color[0], color[1], color[2]);
  }
  return new Float32Array(v);
}

function createGroundGridMesh(size = 70.0) {
  const half = size * 0.5;
  return new Float32Array([
    -half, 0, -half,   half, 0, -half,   half, 0,  half,
    -half, 0, -half,   half, 0,  half,  -half, 0,  half
  ]);
}

// ======================== 4. WebGPU 核心管线 ========================
export function runCarAnimate(
  device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat, canvas: HTMLCanvasElement, gui: SimpleGUI
) {
  const carMesh = createSportsCarMesh();
  const carVBuffer = device.createBuffer({ size: carMesh.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(carVBuffer, 0, carMesh);
  const carVertexCount = carMesh.length / 7;

  const groundMesh = createGroundGridMesh();
  const groundVBuffer = device.createBuffer({ size: groundMesh.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(groundVBuffer, 0, groundMesh);

  const trackColors = [
    [1.0, 0.35, 0.2],
    [0.1, 0.8, 1.0],
    [1.0, 0.85, 0.1],
  ];
  const trackBuffers = ALL_ROUTES.map((route, i) => {
    const trackData = createTrackLineMesh(route, trackColors[i], 200);
    const buf = device.createBuffer({ size: trackData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, trackData);
    return { buffer: buf, count: trackData.length / 6 };
  });

  const sceneUBO = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const CARS_CONFIG = [
    { name: "车 1 (超跑红)", route: 0, lapTime: 14.0, offset: 0.00, color: [1.0, 0.15, 0.12, 1.0] },
    { name: "车 2 (极光青)", route: 1, lapTime: 18.0, offset: 0.25, color: [0.0, 0.85, 1.00, 1.0] },
    { name: "车 3 (竞速金)", route: 2, lapTime: 11.0, offset: 0.50, color: [1.0, 0.80, 0.05, 1.0] },
    { name: "车 4 (荧光绿)", route: 0, lapTime: 14.0, offset: 0.50, color: [0.2, 1.00, 0.25, 1.0] },
  ];
  const CAR_COUNT = CARS_CONFIG.length;

  const carInstanceStorage = device.createBuffer({
    size: 256 * CAR_COUNT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  const carInstanceCpuData = new Float32Array((256 / 4) * CAR_COUNT);

  // 【核心修复】显式定义共用的场景 Uniform 布局，避免 auto 模式下的 BindGroup 冲突
  const sceneBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" }
      }
    ]
  });

  const sceneBindGroup = device.createBindGroup({
    layout: sceneBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: sceneUBO } }]
  });

  const scenePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [sceneBindGroupLayout]
  });

  // Ground Pipeline
  const groundShader = `
    struct Scene { viewProj: mat4x4f, eyePos: vec4f };
    @group(0) @binding(0) var<uniform> scene: Scene;

    struct VOut {
      @builtin(position) pos: vec4f,
      @location(0) wPos: vec3f,
    };

    @vertex fn vs(@location(0) pos: vec3f) -> VOut {
      var o: VOut;
      o.pos = scene.viewProj * vec4f(pos, 1.0);
      o.wPos = pos;
      return o;
    }

    @fragment fn fs(in: VOut) -> @location(0) vec4f {
      let coord = in.wPos.xz * 0.5;
      let grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
      let line = min(grid.x, grid.y);
      let gridAlpha = 1.0 - min(line, 1.0);

      let dist = length(in.wPos.xz);
      let fade = clamp(1.0 - dist / 32.0, 0.0, 1.0);

      let baseColor = vec3f(0.08, 0.09, 0.12);
      let gridColor = vec3f(0.20, 0.25, 0.35);
      let color = mix(baseColor, gridColor, gridAlpha * 0.7);

      return vec4f(color * fade, 1.0);
    }
  `;
  const groundPipeline = device.createRenderPipeline({
    layout: scenePipelineLayout, // 使用公共 Layout
    vertex: {
      module: device.createShaderModule({ code: groundShader }), entryPoint: "vs",
      buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }]
    },
    fragment: { module: device.createShaderModule({ code: groundShader }), entryPoint: "fs", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    primitive: { topology: "triangle-list" }
  });

  // Track Pipeline
  const trackShader = `
    struct Scene { viewProj: mat4x4f, eyePos: vec4f };
    @group(0) @binding(0) var<uniform> scene: Scene;
    struct VOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec3f,
    };
    @vertex fn vs(@location(0) pos: vec3f, @location(1) col: vec3f) -> VOut {
      var o: VOut;
      o.pos = scene.viewProj * vec4f(pos, 1.0);
      o.color = col;
      return o;
    }
    @fragment fn fs(in: VOut) -> @location(0) vec4f {
      return vec4f(in.color, 0.85);
    }
  `;
  const trackPipeline = device.createRenderPipeline({
    layout: scenePipelineLayout, // 使用公共 Layout，完全兼容 sceneBindGroup
    vertex: {
      module: device.createShaderModule({ code: trackShader }), entryPoint: "vs",
      buffers: [{ arrayStride: 24, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" }
      ]}]
    },
    fragment: { module: device.createShaderModule({ code: trackShader }), entryPoint: "fs", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
    primitive: { topology: "line-strip" }
  });

  // Car Pipeline
  const carShader = `
    struct Scene { viewProj: mat4x4f, eyePos: vec4f };
    struct CarInstance {
      model: mat4x4f,
      color: vec4f,
    };

    @group(0) @binding(0) var<uniform> scene: Scene;
    @group(0) @binding(1) var<storage, read> cars: array<CarInstance>;

    struct VOut {
      @builtin(position) pos: vec4f,
      @location(0) wPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) @interpolate(flat) instanceIdx: u32,
      @location(3) @interpolate(flat) matId: u32,
    };

    @vertex fn vs(
      @location(0) pos: vec3f,
      @location(1) norm: vec3f,
      @location(2) matId: f32,
      @builtin(instance_index) instIdx: u32
    ) -> VOut {
      var o: VOut;
      let car = cars[instIdx];
      let worldPos = car.model * vec4f(pos, 1.0);
      o.pos = scene.viewProj * worldPos;
      o.wPos = worldPos.xyz;
      o.normal = normalize((car.model * vec4f(norm, 0.0)).xyz);
      o.instanceIdx = instIdx;
      o.matId = u32(matId + 0.1);
      return o;
    }

    @fragment fn fs(in: VOut) -> @location(0) vec4f {
      let car = cars[in.instanceIdx];
      let sunDir = normalize(vec3f(0.5, 0.8, -0.4));
      let diff = max(dot(in.normal, sunDir), 0.25);
      let V = normalize(scene.eyePos.xyz - in.wPos);
      let H = normalize(sunDir + V);

      var baseColor = car.color.rgb;
      var emissive = vec3f(0.0);
      var specPow = 32.0;
      var specStrength = 0.5;

      switch(in.matId) {
        case 0u: {
          baseColor = car.color.rgb;
          specPow = 64.0;
          specStrength = 0.9;
        }
        case 1u: {
          baseColor = vec3f(0.04, 0.08, 0.12);
          specPow = 128.0;
          specStrength = 1.2;
        }
        case 2u: {
          baseColor = vec3f(0.12, 0.12, 0.14);
          specPow = 8.0;
          specStrength = 0.1;
        }
        case 3u: {
          baseColor = vec3f(1.0, 0.98, 0.85);
          emissive = vec3f(2.5, 2.4, 2.0);
        }
        case 4u: {
          baseColor = vec3f(1.0, 0.05, 0.02);
          emissive = vec3f(2.6, 0.1, 0.05);
        }
        default: {}
      }

      let spec = pow(max(dot(in.normal, H), 0.0), specPow) * specStrength;
      let finalColor = baseColor * diff + vec3f(1.0, 0.95, 0.9) * spec + emissive;
      return vec4f(finalColor, 1.0);
    }
  `;
  const carPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: carShader }), entryPoint: "vs",
      buffers: [{ arrayStride: 28, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32" }
      ]}]
    },
    fragment: { module: device.createShaderModule({ code: carShader }), entryPoint: "fs", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    primitive: { topology: "triangle-list" }
  });

  const carBindGroup = device.createBindGroup({
    layout: carPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sceneUBO } },
      { binding: 1, resource: { buffer: carInstanceStorage } }
    ]
  });

  let curW = 0, curH = 0;
  let depthTexture: GPUTexture;
  function ensureDepthTexture() {
    const w = canvas.width || 800;
    const h = canvas.height || 600;
    if (w === curW && h === curH && depthTexture) return;
    curW = w; curH = h;
    if (depthTexture) depthTexture.destroy();
    depthTexture = device.createTexture({
      size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
  }

  // ======================== 5. 时间轴与多车控制 ========================
  const timeline = {
    time: 0.0,
    timeScale: 1.0,
    isPlaying: true,
    duration: 60.0,
    step: 0.1,
  };

  const cameraState = {
    distance: 28.0, theta: 40, phi: 32, panY: 1.5,
    cameraMode: "自由全景视角",
  };

  const appSettings = {
    showTracks: true,
    speedMultiplier: 1.0,
  };

  const actions = {
    togglePlay: () => { timeline.isPlaying = !timeline.isPlaying; gui.updateDisplay(); },
    rewind: () => { timeline.timeScale = -2.0; timeline.isPlaying = true; gui.updateDisplay(); },
    normal: () => { timeline.timeScale = 1.0; timeline.isPlaying = true; gui.updateDisplay(); },
    fastForward: () => { timeline.timeScale = 2.5; timeline.isPlaying = true; gui.updateDisplay(); },
    stepBack: () => { timeline.isPlaying = false; timeline.time = Math.max(0, timeline.time - timeline.step); gui.updateDisplay(); },
    stepNext: () => { timeline.isPlaying = false; timeline.time = Math.min(timeline.duration, timeline.time + timeline.step); gui.updateDisplay(); },
    resetTime: () => { timeline.time = 0.0; gui.updateDisplay(); },
  };

  // ======================== 6. 构建 GUI 控制面板 ========================
  gui.addTextInfo("<b>🏎️ 多车多赛道动力学系统</b><br>基于时间轴驱动的并发多车沿闭合路径巡航，支持加速、回退与追车视角");

  gui.add(timeline, "time", 0.0, timeline.duration, 0.05).name("⏱️ 时间轴进度(秒)");
  gui.add(timeline, "isPlaying").name("▶️ 播放 / 暂停");
  gui.add(timeline, "timeScale", -4.0, 4.0, 0.1).name("⏩ 流速倍率 (负为倒退)");

  gui.add(actions, "togglePlay").name("⏯ 播放/暂停切换");
  gui.add(actions, "rewind").name("⏪ 倒退回放 (-2.0x)");
  gui.add(actions, "normal").name("▶ 正常播放 (1.0x)");
  gui.add(actions, "fastForward").name("⏩ 加速狂飙 (2.5x)");
  gui.add(actions, "stepBack").name("⏮ 单帧微调 (-0.1s)");
  gui.add(actions, "stepNext").name("⏭ 单帧微调 (+0.1s)");
  gui.add(actions, "resetTime").name("⏹ 时间轴归零");

  gui.add(cameraState, "cameraMode", ["自由全景视角", "跟随 车 1 (超跑红)", "跟随 车 2 (极光青)", "跟随 车 3 (竞速金)", "跟随 车 4 (荧光绿)"] as any).name("🎥 摄像机模式");
  gui.add(appSettings, "showTracks").name("显示赛道引导线");
  gui.add(cameraState, "distance", 10.0, 60.0, 1.0).name("视角距离");
  gui.add(cameraState, "theta", -180, 180, 1).name("相机偏航角");
  gui.add(cameraState, "phi", 5, 85, 1).name("相机俯仰角");

  let isDragging = false, lastX = 0, lastY = 0;
  canvas.onpointerdown = (e) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); };
  canvas.onpointermove = (e) => {
    if (!isDragging) return;
    cameraState.theta -= (e.clientX - lastX) * 0.4;
    cameraState.phi = Math.max(5, Math.min(85, cameraState.phi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY;
    gui.updateDisplay();
  };
  canvas.onpointerup = (e) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} };

  // ======================== 7. 主渲染帧循环 ========================
  const sceneData = new Float32Array(128 / 4);
  let animId: number;
  let lastTime = performance.now();
  const carMatrices: Float32Array[] = [];

  function frame() {
    ensureDepthTexture();

    const now = performance.now();
    const dt = (now - lastTime) * 0.001;
    lastTime = now;

    if (timeline.isPlaying) {
      timeline.time += dt * timeline.timeScale;
      if (timeline.time > timeline.duration) timeline.time = 0.0;
      else if (timeline.time < 0.0) timeline.time = timeline.duration;
    }

    carMatrices.length = 0;
    for (let i = 0; i < CAR_COUNT; i++) {
      const cfg = CARS_CONFIG[i];
      const routeWaypoints = ALL_ROUTES[cfg.route];
      const progress = (timeline.time / cfg.lapTime) + cfg.offset;
      const m = getVehiclePoseMatrix(routeWaypoints, progress);
      carMatrices.push(m);

      const baseOffset = (256 / 4) * i;
      carInstanceCpuData.set(m, baseOffset);
      carInstanceCpuData.set(cfg.color, baseOffset + 16);
    }
    device.queue.writeBuffer(carInstanceStorage, 0, carInstanceCpuData);

    const aspect = curW / curH;
    let eye: number[], center: number[];

    const followIdx = ["跟随 车 1 (超跑红)", "跟随 车 2 (极光青)", "跟随 车 3 (竞速金)", "跟随 车 4 (荧光绿)"].indexOf(cameraState.cameraMode);
    if (followIdx >= 0) {
      const carMat = carMatrices[followIdx];
      const carPos = [carMat[12], carMat[13], carMat[14]];
      const fwd = [carMat[8], carMat[9], carMat[10]];
      eye = [
        carPos[0] - fwd[0] * 6.5,
        carPos[1] + 3.2,
        carPos[2] - fwd[2] * 6.5
      ];
      center = [carPos[0] + fwd[0] * 2.0, carPos[1] + 0.8, carPos[2] + fwd[2] * 2.0];
    } else {
      const radTheta = (cameraState.theta * Math.PI) / 180;
      const radPhi = (cameraState.phi * Math.PI) / 180;
      eye = [
        cameraState.distance * Math.cos(radPhi) * Math.sin(radTheta),
        cameraState.panY + cameraState.distance * Math.sin(radPhi),
        cameraState.distance * Math.cos(radPhi) * Math.cos(radTheta)
      ];
      center = [0, 0.5, 0];
    }

    const view = createLookAtMatrix(eye, center, [0, 1, 0]);
    const proj = createPerspectiveMatrix((50 * Math.PI) / 180, aspect, 0.1, 150);
    const viewProj = multiplyMat4(proj, view);

    sceneData.set(viewProj, 0);
    sceneData.set([eye[0], eye[1], eye[2], 1.0], 16);
    device.queue.writeBuffer(sceneUBO, 0, sceneData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.05, g: 0.06, b: 0.08, a: 1.0 },
        loadOp: "clear", storeOp: "store"
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store"
      }
    });

    // 地面
    pass.setPipeline(groundPipeline);
    pass.setBindGroup(0, sceneBindGroup);
    pass.setVertexBuffer(0, groundVBuffer);
    pass.draw(6);

    // 赛道轨迹
    if (appSettings.showTracks) {
      pass.setPipeline(trackPipeline);
      pass.setBindGroup(0, sceneBindGroup); // 完全兼容！
      for (const track of trackBuffers) {
        pass.setVertexBuffer(0, track.buffer);
        pass.draw(track.count);
      }
    }

    // 赛车实例化渲染
    pass.setPipeline(carPipeline);
    pass.setBindGroup(0, carBindGroup);
    pass.setVertexBuffer(0, carVBuffer);
    pass.draw(carVertexCount, CAR_COUNT);

    pass.end();
    device.queue.submit([encoder.finish()]);

    animId = requestAnimationFrame(frame);
  }
  frame();

  return () => {
    cancelAnimationFrame(animId);
    carVBuffer.destroy();
    groundVBuffer.destroy();
    for (const t of trackBuffers) t.buffer.destroy();
    sceneUBO.destroy();
    carInstanceStorage.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}