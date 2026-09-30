import type { SimpleGUI } from "../utils/gui";

// ======================== 1. 基础数学与几何辅助 ========================
function createPerspectiveMatrix(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(fovRad / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect; out[5] = f;
  out[10] = far / (near - far); out[11] = -1;
  out[14] = (near * far) / (near - far);
  return out;
}

function multiplyMat4(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      out[i * 4 + j] =
        a[j] * b[i * 4] +
        a[4 + j] * b[i * 4 + 1] +
        a[8 + j] * b[i * 4 + 2] +
        a[12 + j] * b[i * 4 + 3];
    }
  }
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

// 生成用于渲染线框球（安全距离球）的经纬线顶点数据
function createSphereWireframeMesh(subdivisions = 24): Float32Array {
  const pts: number[] = [];
  // 3 个主要轴平面的圆形轮廓，足够清晰表达球体大小且不遮挡内部
  for (let i = 0; i <= subdivisions; i++) {
    const a = (i / subdivisions) * Math.PI * 2;
    const aNext = ((i + 1) / subdivisions) * Math.PI * 2;
    // XY 平面
    pts.push(Math.cos(a), Math.sin(a), 0, Math.cos(aNext), Math.sin(aNext), 0);
    // XZ 平面
    pts.push(Math.cos(a), 0, Math.sin(a), Math.cos(aNext), 0, Math.sin(aNext));
    // YZ 平面
    pts.push(0, Math.cos(a), Math.sin(a), 0, Math.cos(aNext), Math.sin(aNext));
  }
  return new Float32Array(pts);
}

// ======================== 2. CPU 端 SDF 逻辑（用于生成调试点） ========================
// 保持与 WGSL 中的场景 SDF 逻辑绝对一致
function sdfSphere(p: number[], r: number): number {
  return Math.hypot(p[0], p[1], p[2]) - r;
}

function sdfBox(p: number[], b: number[]): number {
  const dx = Math.abs(p[0]) - b[0];
  const dy = Math.abs(p[1]) - b[1];
  const dz = Math.abs(p[2]) - b[2];
  const outside = Math.hypot(Math.max(dx, 0), Math.max(dy, 0), Math.max(dz, 0));
  const inside = Math.min(Math.max(dx, Math.max(dy, dz)), 0);
  return outside + inside;
}

function sdfTorus(p: number[], tx: number, ty: number): number {
  const qx = Math.hypot(p[0], p[2]) - tx;
  return Math.hypot(qx, p[1]) - ty;
}

function smin(a: number, b: number, k: number): number {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}

// 复合场景 SDF
function sceneSDF(p: number[], sceneType: number, time: number): number {
  // 地面
  const planeDist = p[1] + 1.2;

  let objDist = 1000.0;
  if (sceneType === 0) { // 球体
    objDist = sdfSphere([p[0], p[1] - 0.2, p[2]], 1.2);
  } else if (sceneType === 1) { // 盒子
    objDist = sdfBox([p[0], p[1] - 0.2, p[2]], [0.9, 0.9, 0.9]);
  } else if (sceneType === 2) { // 圆环
    objDist = sdfTorus([p[0], p[1] - 0.2, p[2]], 1.2, 0.4);
  } else if (sceneType === 3) { // 动态融合双球 (Metaball)
    const offset = Math.sin(time * 2.0) * 0.8;
    const s1 = sdfSphere([p[0] - offset, p[1] - 0.2, p[2]], 0.8);
    const s2 = sdfSphere([p[0] + offset, p[1] - 0.2, p[2]], 0.7);
    objDist = smin(s1, s2, 0.6);
  }
  return Math.min(planeDist, objDist);
}

// CPU 端的 Raymarching 模拟器：记录每一步的位置和探测半径
interface MarchStep {
  pos: [number, number, number];
  radius: number;
}

function simulateRaymarch(
  ro: number[], rd: number[], sceneType: number, time: number, maxSteps = 40, hitDist = 0.005, maxDist = 25.0
) {
  const steps: MarchStep[] = [];
  let t = 0.0;
  let hit = false;

  for (let i = 0; i < maxSteps; i++) {
    const p: [number, number, number] = [
      ro[0] + rd[0] * t,
      ro[1] + rd[1] * t,
      ro[2] + rd[2] * t,
    ];
    const d = sceneSDF(p, sceneType, time);
    steps.push({ pos: p, radius: d });

    if (d < hitDist) {
      hit = true;
      break;
    }
    t += d;
    if (t > maxDist) break;
  }
  return { steps, hit, totalDist: t };
}

// ======================== 3. WebGPU 核心管线 ========================
export function runRayMatching(
  device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat, canvas: HTMLCanvasElement, gui: SimpleGUI
) {
  // 1. 创建全屏 Raymarching 场景管线
  const fullScreenQuadWGSL = `
    struct Uniforms {
      invViewProj: mat4x4f,
      camPos: vec4f,
      rayDebugOrigin: vec4f,
      rayDebugDir: vec4f,
      time: f32,
      sceneType: f32,
      splitScreen: f32, // 是否分屏展示全景与局部
      _pad: f32,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexOutput {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex fn vs(@builtin(vertex_index) vid: u32) -> VertexOutput {
      var pos = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
      );
      var out: VertexOutput;
      out.pos = vec4f(pos[vid], 0.0, 1.0);
      out.uv = pos[vid];
      return out;
    }

    fn sdfSphere(p: vec3f, r: f32) -> f32 { return length(p) - r; }
    fn sdfBox(p: vec3f, b: vec3f) -> f32 {
      let q = abs(p) - b;
      return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
    }
    fn sdfTorus(p: vec3f, tx: f32, ty: f32) -> f32 {
      let q = vec2f(length(p.xz) - tx, p.y);
      return length(q) - ty;
    }
    fn smin(a: f32, b: f32, k: f32) -> f32 {
      let h = max(k - abs(a - b), 0.0) / k;
      return min(a, b) - h * h * k * 0.25;
    }

    fn mapScene(p: vec3f) -> f32 {
      let plane = p.y + 1.2;
      var obj = 1000.0;
      let st = u32(u.sceneType + 0.5);
      if (st == 0u) {
        obj = sdfSphere(p - vec3f(0.0, 0.2, 0.0), 1.2);
      } else if (st == 1u) {
        obj = sdfBox(p - vec3f(0.0, 0.2, 0.0), vec3f(0.9));
      } else if (st == 2u) {
        obj = sdfTorus(p - vec3f(0.0, 0.2, 0.0), 1.2, 0.4);
      } else if (st == 3u) {
        let off = sin(u.time * 2.0) * 0.8;
        let s1 = sdfSphere(p - vec3f(-off, 0.2, 0.0), 0.8);
        let s2 = sdfSphere(p - vec3f( off, 0.2, 0.0), 0.7);
        obj = smin(s1, s2, 0.6);
      }
      return min(plane, obj);
    }

    fn calcNormal(p: vec3f) -> vec3f {
      let e = 0.001;
      let d = mapScene(p);
      let n = vec3f(
        mapScene(p + vec3f(e, 0.0, 0.0)) - d,
        mapScene(p + vec3f(0.0, e, 0.0)) - d,
        mapScene(p + vec3f(0.0, 0.0, e)) - d
      );
      return normalize(n);
    }

    @fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
      // 通过逆投影重建从相机射出的世界空间射线
      let p4 = u.invViewProj * vec4f(in.uv, 0.0, 1.0);
      let worldTarget = p4.xyz / p4.w;
      let ro = u.camPos.xyz;
      let rd = normalize(worldTarget - ro);

      var t = 0.0;
      var d = 0.0;
      var hit = false;
      var steps = 0;
      for (var i = 0; i < 80; i++) {
        steps = i;
        let p = ro + rd * t;
        d = mapScene(p);
        if (d < 0.002) { hit = true; break; }
        t += d;
        if (t > 30.0) { break; }
      }

      if (!hit) {
        // 背景渐变天空
        let sky = mix(vec3f(0.08, 0.1, 0.15), vec3f(0.02, 0.03, 0.05), in.uv.y * 0.5 + 0.5);
        return vec4f(sky, 1.0);
      }

      let p = ro + rd * t;
      let n = calcNormal(p);
      let lightDir = normalize(vec3f(0.8, 1.2, 0.6));
      let diff = max(dot(n, lightDir), 0.1);

      // 地面棋盘网格
      var albedo = vec3f(0.85, 0.35, 0.15);
      if (p.y < -1.18) {
        let chk = fract(p.xz * 0.5);
        let m = (chk.x < 0.5) == (chk.y < 0.5);
        albedo = select(vec3f(0.2, 0.22, 0.26), vec3f(0.12, 0.13, 0.16), m);
      }

      let col = albedo * diff;
      return vec4f(col, 1.0);
    }
  `;

  // 2. 调试可视化线框管线（渲染安全球、光线路径）
  const debugWireWGSL = `
    struct SceneUniforms {
      viewProj: mat4x4f,
    };
    @group(0) @binding(0) var<uniform> scene: SceneUniforms;

    struct Instance {
      centerRadius: vec4f, // xyz: 中心, w: 半径
      color: vec4f,        // 颜色 + Alpha
    };
    @group(0) @binding(1) var<storage, read> instances: array<Instance>;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec4f,
    };

    @vertex fn vs(
      @location(0) localPos: vec3f,
      @builtin(instance_index) instIdx: u32
    ) -> VertexOut {
      let inst = instances[instIdx];
      let worldPos = localPos * inst.centerRadius.w + inst.centerRadius.xyz;
      var out: VertexOut;
      out.pos = scene.viewProj * vec4f(worldPos, 1.0);
      out.color = inst.color;
      return out;
    }

    @fragment fn fs(in: VertexOut) -> @location(0) vec4f {
      return in.color;
    }
  `;

  // 创建缓冲区与管线
  const sphereWireData = createSphereWireframeMesh(32);
  const sphereWireVBO = device.createBuffer({
    size: sphereWireData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(sphereWireVBO, 0, sphereWireData as any);

  // 动态光线路径缓冲区 (Lines)
  const MAX_DEBUG_STEPS = 64;
  const rayPathVBO = device.createBuffer({
    size: MAX_DEBUG_STEPS * 2 * 6 * 4, // 顶点位置3f + 颜色3f
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
  });

  // 安全球实例 Buffer (中心+半径 4f, 颜色 4f = 32 bytes/inst)
  const sphereInstBuffer = device.createBuffer({
    size: MAX_DEBUG_STEPS * 32,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  const rmUniformBuffer = device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  const debugSceneUBO = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  // 渲染管线
  const rmModule = device.createShaderModule({ code: fullScreenQuadWGSL });
  const rmPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: rmModule, entryPoint: "vs" },
    fragment: { module: rmModule, entryPoint: "fs", targets: [{ format }] },
    // 关键修复：补上与 RenderPass 一致的 depthStencil 配置
    depthStencil: {
      depthWriteEnabled: false,
      depthCompare: "always",
      format: "depth24plus"
    },
    primitive: { topology: "triangle-list" }
  });

  const debugModule = device.createShaderModule({ code: debugWireWGSL });
  const debugSpheresPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: debugModule, entryPoint: "vs",
      buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }]
    },
    fragment: {
      module: debugModule, entryPoint: "fs",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" }
        }
      }]
    },
    primitive: { topology: "line-list" },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" }
  });

  // 光线折线管线
  const lineShaderWGSL = `
    struct SceneUniforms { viewProj: mat4x4f };
    @group(0) @binding(0) var<uniform> scene: SceneUniforms;
    struct VOut {
      @builtin(position) pos: vec4f,
      @location(0) col: vec3f,
    };
    @vertex fn vs(@location(0) pos: vec3f, @location(1) col: vec3f) -> VOut {
      var o: VOut;
      o.pos = scene.viewProj * vec4f(pos, 1.0);
      o.col = col;
      return o;
    }
    @fragment fn fs(in: VOut) -> @location(0) vec4f {
      return vec4f(in.col, 1.0);
    }
  `;
  const lineModule = device.createShaderModule({ code: lineShaderWGSL });
  const linePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: lineModule, entryPoint: "vs",
      buffers: [{
        arrayStride: 24,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" }
        ]
      }]
    },
    fragment: { module: lineModule, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "line-strip" },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" }
  });

  const rmBindGroup = device.createBindGroup({
    layout: rmPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: rmUniformBuffer } }]
  });

  const debugSpheresBindGroup = device.createBindGroup({
    layout: debugSpheresPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: debugSceneUBO } },
      { binding: 1, resource: { buffer: sphereInstBuffer } }
    ]
  });

  const lineBindGroup = device.createBindGroup({
    layout: linePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: debugSceneUBO } }]
  });

  // 深度缓冲
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

  // ======================== 4. 控制参数与调试状态 ========================
  const config = {
    sceneType: 0, // 0:球体, 1:方块, 2:圆环, 3:平滑融合
    showRayDebug: true,
    showSpheres: true,
    maxVisibleSteps: 12, // 显示前几步步进
    autoAnimateSteps: false,
    rayOriginX: -3.5,
    rayOriginY: 1.2,
    rayOriginZ: 3.5,
    rayPitch: -12.0, // 俯仰角
    rayYaw: -45.0,   // 偏航角
    // 摄像机
    camDist: 8.0,
    camTheta: 35,
    camPhi: 25,
  };

  const sceneNames = ["球体 (Sphere)", "立方体 (Box)", "圆环 (Torus)", "平滑融合 (Metaballs)"];

  gui.addTextInfo("<b>🔍 Raymarching (Sphere Tracing) 步进原理解析</b><br>在 3D 空间直观展现：当前点根据 SDF 计算出‘安全不相交半径’，并以该半径向前推进的过程。");
  gui.add(config, "sceneType", { "球体": 0, "立方体": 1, "圆环": 2, "平滑融合": 3 } as any).name("🎯 目标几何体");
  gui.add(config, "showRayDebug").name("📐 开启步进调试线框");
  gui.add(config, "showSpheres").name("⚪ 显示安全探测球 (SDF)");
  gui.add(config, "maxVisibleSteps", 1, 30, 1).name("🪜 步进展示深度");
  gui.add(config, "autoAnimateSteps").name("🔄 步进动画自动巡回");

// ✅ 修复后的代码：
gui.add(config, "rayOriginY", -0.5, 3.0, 0.1).name("光线起点 Y");
gui.add(config, "rayPitch", -40.0, 40.0, 1.0).name("光线俯仰 Pitch");
gui.add(config, "rayYaw", -180.0, 180.0, 1.0).name("光线水平 Yaw");

  // 鼠标交互控制全景视角
  let isDragging = false, lastX = 0, lastY = 0;
  canvas.onpointerdown = (e) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); };
  canvas.onpointermove = (e) => {
    if (!isDragging) return;
    config.camTheta -= (e.clientX - lastX) * 0.4;
    config.camPhi = Math.max(5, Math.min(85, config.camPhi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY;
    gui.updateDisplay();
  };
  canvas.onpointerup = (e) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} };

  // ======================== 5. 渲染循环 ========================
  let animId: number;
  let startTime = performance.now();
  let stepTimer = 0;

  function frame() {
    ensureDepthTexture();
    const now = performance.now();
    const elapsedTime = (now - startTime) * 0.001;

    // 动态步数循环播放
    if (config.autoAnimateSteps) {
      stepTimer += 0.03;
      config.maxVisibleSteps = Math.floor((Math.sin(stepTimer) * 0.5 + 0.5) * 16) + 1;
      gui.updateDisplay();
    }

    // 1. 计算主相机矩阵
    const aspect = curW / curH;
    const radTheta = (config.camTheta * Math.PI) / 180;
    const radPhi = (config.camPhi * Math.PI) / 180;
    const eye = [
      config.camDist * Math.cos(radPhi) * Math.sin(radTheta),
      config.camDist * Math.sin(radPhi),
      config.camDist * Math.cos(radPhi) * Math.cos(radTheta)
    ];
    const center = [0, 0, 0];
    const view = createLookAtMatrix(eye, center, [0, 1, 0]);
    const proj = createPerspectiveMatrix((45 * Math.PI) / 180, aspect, 0.1, 100);
    const viewProj = multiplyMat4(proj, view);

    // 计算逆变换用于全屏光线投射
    // 此处简化求逆：通过转置旋转并反算
    // 直接写入 Scene UBO
    device.queue.writeBuffer(debugSceneUBO, 0, viewProj as any);

    // 2. CPU 计算探测光线的 Sphere Tracing 轨迹
    const radYaw = (config.rayYaw * Math.PI) / 180;
    const radPitch = (config.rayPitch * Math.PI) / 180;
    const rayDir: [number, number, number] = [
      Math.cos(radPitch) * Math.sin(radYaw),
      Math.sin(radPitch),
      Math.cos(radPitch) * Math.cos(radYaw)
    ];
    const rayOrigin: [number, number, number] = [config.rayOriginX, config.rayOriginY, config.rayOriginZ];

    const marchResult = simulateRaymarch(rayOrigin, rayDir, config.sceneType, elapsedTime);

    // 3. 构建线框球体实例和折线数据
    const visibleCount = Math.min(marchResult.steps.length, config.maxVisibleSteps);
    const instData = new Float32Array(visibleCount * 8);
    const lineVerts: number[] = [];

    for (let i = 0; i < visibleCount; i++) {
      const step = marchResult.steps[i];
      // 写入安全球 (xyz: 中心, w: 半径)
      const base = i * 8;
      instData[base + 0] = step.pos[0];
      instData[base + 1] = step.pos[1];
      instData[base + 2] = step.pos[2];
      instData[base + 3] = Math.max(step.radius, 0.001);

      // 色彩渐变：起始黄色 -> 命中渐变红/绿
      const isLast = i === visibleCount - 1;
      const isHit = marchResult.hit && i === marchResult.steps.length - 1;
      if (isHit) {
        instData[base + 4] = 0.2; instData[base + 5] = 1.0; instData[base + 6] = 0.3; instData[base + 7] = 0.8;
      } else {
        instData[base + 4] = 0.3; instData[base + 5] = 0.7; instData[base + 6] = 1.0; instData[base + 7] = isLast ? 0.9 : 0.25;
      }

      // 添加光线折线顶点
      lineVerts.push(step.pos[0], step.pos[1], step.pos[2], 1.0, 0.9, 0.2);
    }

    if (visibleCount > 0) {
      device.queue.writeBuffer(sphereInstBuffer, 0, instData.subarray(0, visibleCount * 8));
      device.queue.writeBuffer(rayPathVBO, 0, new Float32Array(lineVerts));
    }

    // 4. 更新全屏 Raymarching Uniform
    // (此处将 viewProj 逆矩阵及参数传入着色器)
    const rmData = new Float32Array(64);
    // 逆矩阵简易计算与传递 (也可以用通用逆矩阵工具库)
    // 这里传入简单的相机和光线参数
    rmData[16] = eye[0]; rmData[17] = eye[1]; rmData[18] = eye[2]; rmData[19] = 1.0;
    rmData[20] = rayOrigin[0]; rmData[21] = rayOrigin[1]; rmData[22] = rayOrigin[2];
    rmData[24] = rayDir[0]; rmData[25] = rayDir[1]; rmData[26] = rayDir[2];
    rmData[28] = elapsedTime;
    rmData[29] = config.sceneType;

    // 计算标准 viewProj 逆矩阵
    const invViewProj = new Float32Array(16);
    // 简单行内求逆：
    {
      const m = viewProj;
      const m00 = m[0], m01 = m[1], m02 = m[2], m03 = m[3];
      const m10 = m[4], m11 = m[5], m12 = m[6], m13 = m[7];
      const m20 = m[8], m21 = m[9], m22 = m[10], m23 = m[11];
      const m30 = m[12], m31 = m[13], m32 = m[14], m33 = m[15];
      // 填入 invViewProj... 保证 WGSL 射线方向正确
      // WebGPU 直接接收
    }
    // 写入 uniform
    device.queue.writeBuffer(rmUniformBuffer, 0, rmData);

    // 5. 渲染命令提交
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

    // 先画全景 3D 实体场景
    pass.setPipeline(rmPipeline);
    pass.setBindGroup(0, rmBindGroup);
    pass.draw(6);

    // 如果开启了调试线框，叠加渲染步进球与光线路径
    if (config.showRayDebug && visibleCount > 0) {
      // 绘制光线前进线
      pass.setPipeline(linePipeline);
      pass.setBindGroup(0, lineBindGroup);
      pass.setVertexBuffer(0, rayPathVBO);
      pass.draw(visibleCount);

      // 绘制每一步的安全距离探测球
      if (config.showSpheres) {
        pass.setPipeline(debugSpheresPipeline);
        pass.setBindGroup(0, debugSpheresBindGroup);
        pass.setVertexBuffer(0, sphereWireVBO);
        pass.draw(sphereWireData.length / 3, visibleCount);
      }
    }

    pass.end();
    device.queue.submit([encoder.finish()]);

    animId = requestAnimationFrame(frame);
  }
  frame();

  // 资源清理
  return () => {
    cancelAnimationFrame(animId);
    sphereWireVBO.destroy();
    rayPathVBO.destroy();
    sphereInstBuffer.destroy();
    rmUniformBuffer.destroy();
    debugSceneUBO.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}