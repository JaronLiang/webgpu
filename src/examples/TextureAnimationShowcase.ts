import type { SimpleGUI } from "../utils/gui";

// ==========================================
// 1. 基础 3D 矩阵数学 (免第三方依赖)
// ==========================================
namespace Math3D {
  export function lookAt(eye: number[], center: number[], up: number[]): Float32Array {
    const z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
    const lenZ = 1 / (Math.hypot(z0, z1, z2) || 1);
    const zx = z0 * lenZ, zy = z1 * lenZ, zz = z2 * lenZ;

    const x0 = up[1] * zz - up[2] * zy, x1 = up[2] * zx - up[0] * zz, x2 = up[0] * zy - up[1] * zx;
    const lenX = 1 / (Math.hypot(x0, x1, x2) || 1);
    const xx = x0 * lenX, xy = x1 * lenX, xz = x2 * lenX;

    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;

    const out = new Float32Array(16);
    out[0] = xx;  out[1] = yx;  out[2] = zx;  out[3] = 0;
    out[4] = xy;  out[5] = yy;  out[6] = zy;  out[7] = 0;
    out[8] = xz;  out[9] = yz;  out[10] = zz; out[11] = 0;
    out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    out[15] = 1;
    return out;
  }

  export function perspective(fovRad: number, aspect: number, near: number, far: number): Float32Array {
    const f = 1.0 / Math.tan(fovRad / 2.0);
    const nf = 1 / (near - far);
    const out = new Float32Array(16);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = far * nf;
    out[11] = -1.0;
    out[14] = near * far * nf;
    return out;
  }

  export function multiply(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      const ai0 = a[i], ai1 = a[i + 4], ai2 = a[i + 8], ai3 = a[i + 12];
      out[i]      = ai0 * b[0]  + ai1 * b[1]  + ai2 * b[2]  + ai3 * b[3];
      out[i + 4]  = ai0 * b[4]  + ai1 * b[5]  + ai2 * b[6]  + ai3 * b[7];
      out[i + 8]  = ai0 * b[8]  + ai1 * b[9]  + ai2 * b[10] + ai3 * b[11];
      out[i + 12] = ai0 * b[12] + ai1 * b[13] + ai2 * b[14] + ai3 * b[15];
    }
    return out;
  }
}

// ==========================================
// 2. 烘焙 4x4 (16帧) 全息科技 Sprite Sheet 纹理
// ==========================================
function createHoloSpriteSheetTexture(device: GPUDevice): GPUTexture {
  const size = 512;
  const cols = 4;
  const rows = 4;
  const tileSize = size / cols; // 128x128 每帧

  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d")!;

  ctx.clearRect(0, 0, size, size);

  for (let frame = 0; frame < 16; frame++) {
    const cx = (frame % cols) * tileSize + tileSize / 2;
    const cy = Math.floor(frame / cols) * tileSize + tileSize / 2;
    const radius = tileSize * 0.38;

    ctx.save();
    ctx.translate(cx, cy);

    // 旋转角与脉冲大小
    const angle = (frame / 16) * Math.PI * 2;
    const pulse = 0.85 + 0.15 * Math.sin(angle * 2);

    // 绘制外圈刻度环
    ctx.strokeStyle = "rgba(0, 240, 255, 0.8)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(0, 0, radius * pulse, 0, Math.PI * 2);
    ctx.stroke();

    // 绘制十字旋转瞄准标
    ctx.rotate(angle);
    ctx.strokeStyle = "rgba(100, 255, 218, 0.9)";
    ctx.lineWidth = 2;
    for (let j = 0; j < 4; j++) {
      ctx.beginPath();
      ctx.moveTo(radius * 0.45, 0);
      ctx.lineTo(radius * 1.05, 0);
      ctx.stroke();
      ctx.rotate(Math.PI / 2);
    }

    // 内部发光能量圆核
    const radGrad = ctx.createRadialGradient(0, 0, 2, 0, 0, radius * 0.5);
    radGrad.addColorStop(0, "rgba(255, 255, 255, 1.0)");
    radGrad.addColorStop(0.4, "rgba(0, 220, 255, 0.8)");
    radGrad.addColorStop(1, "rgba(0, 100, 255, 0.0)");
    ctx.fillStyle = radGrad;
    ctx.beginPath();
    ctx.arc(0, 0, radius * 0.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  const texture = device.createTexture({
    size: [size, size, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  device.queue.copyExternalImageToTexture(
    { source: c },
    { texture },
    [size, size]
  );

  return texture;
}

// ==========================================
// 3. 纹理动画材质展示主入口
// ==========================================
export function runTextureAnimationShowcase(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // ----------------------------------------------------
  // 1. 几何体生成：地面、建筑、流线、能量墙、序列帧标牌
  // ----------------------------------------------------
  // 格式统一为: Pos(3), Normal(3), UV(2) -> Stride: 8 Floats (32 Bytes)
  const baseVerts: number[] = [];
  const flowVerts: number[] = [];
  const wallVerts: number[] = [];
  const spriteVerts: number[] = [];

  function pushQuad(arr: number[], p1: number[], p2: number[], p3: number[], p4: number[], n: number[]) {
    arr.push(
      ...p1, ...n, 0, 0,
      ...p2, ...n, 1, 0,
      ...p3, ...n, 1, 1,
      ...p1, ...n, 0, 0,
      ...p3, ...n, 1, 1,
      ...p4, ...n, 0, 1
    );
  }

  // A. 大地基与中心城市建筑群 (Base)
  const gh = 25;
  pushQuad(baseVerts, [-gh, 0,  gh], [ gh, 0,  gh], [ gh, 0, -gh], [-gh, 0, -gh], [0, 1, 0]);

  function addBuilding(cx: number, cy: number, cz: number, sx: number, sy: number, sz: number) {
    const x0 = cx - sx / 2, x1 = cx + sx / 2, y0 = cy - sy / 2, y1 = cy + sy / 2, z0 = cz - sz / 2, z1 = cz + sz / 2;
    pushQuad(baseVerts, [x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], [ 0,  1,  0]);
    pushQuad(baseVerts, [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [ 0,  0,  1]);
    pushQuad(baseVerts, [x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [ 0,  0, -1]);
    pushQuad(baseVerts, [x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [-1,  0,  0]);
    pushQuad(baseVerts, [x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [ 1,  0,  0]);
  }
  addBuilding(0, 1.5, 0, 2.8, 3.0, 2.8);
  addBuilding(-5, 1.0, -4, 2.0, 2.0, 2.0);
  addBuilding( 5, 1.2, -3, 1.8, 2.4, 1.8);
  addBuilding( 4, 0.8,  5, 2.2, 1.6, 2.2);

  // B. 平滑贝塞尔能量流动管道
  function addBezierFlowRibbon(p0: number[], p1: number[], p2: number[], width: number, segs = 40) {
    const halfW = width * 0.5;
    for (let i = 0; i < segs; i++) {
      const tA = i / segs;
      const tB = (i + 1) / segs;

      const calcPt = (t: number) => [
        (1 - t) * (1 - t) * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
        (1 - t) * (1 - t) * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1],
        (1 - t) * (1 - t) * p0[2] + 2 * (1 - t) * t * p2[2] + t * t * p2[2],
      ];

      const calcTangent = (t: number) => {
        const dx = 2 * (1 - t) * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0]);
        const dz = 2 * (1 - t) * (p1[2] - p0[2]) + 2 * t * (p2[2] - p1[2]);
        const len = Math.hypot(dx, dz) || 1;
        return [-dz / len, dx / len];
      };

      const posA = calcPt(tA);
      const posB = calcPt(tB);
      const normA = calcTangent(tA);
      const normB = calcTangent(tB);

      const vA_left  = [posA[0] - normA[0] * halfW, posA[1] + 0.05, posA[2] - normA[1] * halfW];
      const vA_right = [posA[0] + normA[0] * halfW, posA[1] + 0.05, posA[2] + normA[1] * halfW];
      const vB_left  = [posB[0] - normB[0] * halfW, posB[1] + 0.05, posB[2] - normB[1] * halfW];
      const vB_right = [posB[0] + normB[0] * halfW, posB[1] + 0.05, posB[2] + normB[1] * halfW];

      flowVerts.push(
        ...vA_left,  0, 1, 0,  tA, 0.0,
        ...vA_right, 0, 1, 0,  tA, 1.0,
        ...vB_right, 0, 1, 0,  tB, 1.0,

        ...vA_left,  0, 1, 0,  tA, 0.0,
        ...vB_right, 0, 1, 0,  tB, 1.0,
        ...vB_left,  0, 1, 0,  tB, 0.0
      );
    }
  }
  addBezierFlowRibbon([-18, 0, -18], [-2, 0, -12], [-1.6, 0, -1.0], 0.65);
  addBezierFlowRibbon([ 18, 0, -16], [ 12, 0,  2], [ 1.8, 0, -0.8], 0.65);
  addBezierFlowRibbon([ 16, 0,  18], [-2, 0,  14], [ 1.2, 0,  1.5], 0.65);
  addBezierFlowRibbon([-16, 0,  16], [-12, 0, -2], [-1.5, 0,  1.2], 0.65);

  // C. 环形立体能量围墙网格 (Cylinder Strip)
  const wallRadius = 11.5;
  const wallHeight = 4.2;
  const wallSegments = 96;
  for (let i = 0; i < wallSegments; i++) {
    const a0 = (i / wallSegments) * Math.PI * 2;
    const a1 = ((i + 1) / wallSegments) * Math.PI * 2;
    const u0 = i / wallSegments;
    const u1 = (i + 1) / wallSegments;

    const x0 = Math.cos(a0) * wallRadius, z0 = Math.sin(a0) * wallRadius;
    const x1 = Math.cos(a1) * wallRadius, z1 = Math.sin(a1) * wallRadius;
    const n0 = [Math.cos(a0), 0, Math.sin(a0)];
    const n1 = [Math.cos(a1), 0, Math.sin(a1)];

    wallVerts.push(
      x0, 0, z0, ...n0, u0 * 8.0, 0.0,
      x1, 0, z1, ...n1, u1 * 8.0, 0.0,
      x1, wallHeight, z1, ...n1, u1 * 8.0, 1.0,

      x0, 0, z0, ...n0, u0 * 8.0, 0.0,
      x1, wallHeight, z1, ...n1, u1 * 8.0, 1.0,
      x0, wallHeight, z0, ...n0, u0 * 8.0, 1.0
    );
  }

  // D. 序列帧全息展示板 (立于中心建筑上方双向交叉 billboard)
  const sw = 1.8, sh = 1.8, sy = 4.6;
  // 两个互相垂直的十字交错面
  pushQuad(spriteVerts, [-sw, sy - sh, 0], [sw, sy - sh, 0], [sw, sy + sh, 0], [-sw, sy + sh, 0], [0, 0, 1]);
  pushQuad(spriteVerts, [0, sy - sh, -sw], [0, sy - sh, sw], [0, sy + sh, sw], [0, sy + sh, -sw], [1, 0, 0]);

  // 创建 VBO
  function createVBO(data: number[]) {
    const buf = device.createBuffer({ size: data.length * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, new Float32Array(data));
    return buf;
  }
  const baseVBO = createVBO(baseVerts);
  const flowVBO = createVBO(flowVerts);
  const wallVBO = createVBO(wallVerts);
  const spriteVBO = createVBO(spriteVerts);

  // 烘焙全息 4x4 Sprite 纹理与采样器
  const spriteTexture = createHoloSpriteSheetTexture(device);
  const spriteSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  // ----------------------------------------------------
  // 2. 着色器实现
  // ----------------------------------------------------
  const shaderWGSL = `
    struct Uniforms {
      viewProj: mat4x4f,
      camPos: vec4f,
      params: vec4f, // x: time, y: flowSpeed, z: wallDisplace, w: scanSpeed
      flipbookParams: vec4f, // x: fps, y: cols(4), z: rows(4), w: totalFrames(16)
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    // 序列帧纹理与采样器
    @group(1) @binding(0) var spriteTex: texture_2d<f32>;
    @group(1) @binding(1) var spriteSmp: sampler;

    struct VertexInput {
      @location(0) pos: vec3f,
      @location(1) normal: vec3f,
      @location(2) uv: vec2f,
    };

    struct VertexOutput {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
      @location(1) normal: vec3f,
      @location(2) worldPos: vec3f,
    };

    // --- 1. 基础场景管线 ---
    @vertex fn vs_base(in: VertexInput) -> VertexOutput {
      var o: VertexOutput;
      o.pos = u.viewProj * vec4f(in.pos, 1.0);
      o.uv = in.uv;
      o.normal = in.normal;
      o.worldPos = in.pos;
      return o;
    }
    @fragment fn fs_base(in: VertexOutput) -> @location(0) vec4f {
      let grid = abs(fract(in.worldPos.xz * 0.5) - 0.5);
      let line = smoothstep(0.46, 0.49, max(grid.x, grid.y));
      let baseCol = vec3f(0.04, 0.06, 0.09) + vec3f(0.08, 0.12, 0.18) * line;
      let diff = max(dot(normalize(in.normal), normalize(vec3f(1.0, 2.0, 1.0))), 0.2);
      return vec4f(baseCol * diff, 1.0);
    }

    // --- 2. 动态流线管线 ---
    @vertex fn vs_flow(in: VertexInput) -> VertexOutput {
      var o: VertexOutput;
      o.pos = u.viewProj * vec4f(in.pos, 1.0);
      o.uv = in.uv;
      o.normal = in.normal;
      o.worldPos = in.pos;
      return o;
    }
    @fragment fn fs_flow(in: VertexOutput) -> @location(0) vec4f {
      let t = u.params.x * u.params.y;
      let progress = fract(in.uv.x * 3.0 - t);
      let tail = pow(progress, 3.5);
      let edge = smoothstep(0.0, 0.25, in.uv.y) * smoothstep(1.0, 0.75, in.uv.y);
      let headGlow = smoothstep(0.92, 1.0, progress) * 2.0;
      let flowColor = mix(vec3f(1.0, 0.35, 0.05), vec3f(1.0, 0.95, 0.3), progress) * (tail + headGlow);

      let alpha = clamp(tail * edge * 1.8, 0.0, 1.0);
      if (alpha < 0.01) { discard; }
      return vec4f(flowColor, alpha);
    }

    // --- 3. 动态立体能量墙 ---
    @vertex fn vs_wall(in: VertexInput) -> VertexOutput {
      var o: VertexOutput;
      let wave = sin(in.uv.x * 3.0 + u.params.x * 3.0) * cos(in.uv.y * 3.14 + u.params.x * 2.0);
      let displacement = in.normal * (wave * u.params.z * in.uv.y);
      let displacedWorldPos = in.pos + displacement;

      o.pos = u.viewProj * vec4f(displacedWorldPos, 1.0);
      o.uv = in.uv;
      o.normal = in.normal;
      o.worldPos = displacedWorldPos;
      return o;
    }
    @fragment fn fs_wall(in: VertexOutput) -> @location(0) vec4f {
      let gridUV = vec2f(in.uv.x, in.uv.y * 5.0 - u.params.x * 0.4);
      let g = abs(fract(gridUV) - 0.5);
      let gridLine = smoothstep(0.42, 0.48, max(g.x, g.y));

      let scanY = (sin(u.params.x * u.params.w) * 0.5 + 0.5);
      let scanBeam = exp(-pow((in.uv.y - scanY) * 12.0, 2.0)) * 2.5;
      let groundGlow = exp(-in.uv.y * 8.0) * 1.2;
      let topFade = smoothstep(1.0, 0.5, in.uv.y);

      let neonCyan = vec3f(0.0, 0.9, 1.0);
      let neonBlue = vec3f(0.08, 0.25, 0.85);
      let wallColor = mix(neonBlue, neonCyan, gridLine * 0.6 + scanBeam * 0.8) + vec3f(1.0) * (scanBeam * 0.5);

      let alpha = (0.2 + gridLine * 0.4 + scanBeam * 0.8 + groundGlow) * topFade;
      if (alpha < 0.02) { discard; }
      return vec4f(wallColor, clamp(alpha, 0.0, 0.95));
    }

    // --- 4. 序列帧全息 HUD 动画材质 (Flipbook Sprite Animation) ---
    @vertex fn vs_sprite(in: VertexInput) -> VertexOutput {
      var o: VertexOutput;
      o.pos = u.viewProj * vec4f(in.pos, 1.0);
      o.uv = in.uv;
      o.normal = in.normal;
      o.worldPos = in.pos;
      return o;
    }
    @fragment fn fs_sprite(in: VertexOutput) -> @location(0) vec4f {
      let fps = u.flipbookParams.x;
      let cols = u.flipbookParams.y;
      let rows = u.flipbookParams.z;
      let total = u.flipbookParams.w;

      // 计算当前帧
      let currentFrame = floor(u.params.x * fps) % total;
      let colIdx = currentFrame % cols;
      let rowIdx = floor(currentFrame / cols);

      // 计算映射到 Sprite Sheet 上的瓦片 UV 坐标
      let tileUV = (in.uv + vec2f(colIdx, rowIdx)) / vec2f(cols, rows);

      let texColor = textureSample(spriteTex, spriteSmp, tileUV);
      if (texColor.a < 0.05) { discard; }

      // 强化发光强度
      return vec4f(texColor.rgb * 1.5, texColor.a);
    }
  `;

  const shaderModule = device.createShaderModule({ code: shaderWGSL });

  // ----------------------------------------------------
  // 3. 显式创建 PipelineLayout 和 BindGroupLayout (解决报错核心)
  // ----------------------------------------------------
  const uniformBuffer = device.createBuffer({
    size: 160,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Group 0: 全局 Uniform (场景矩阵 + 动画控制参数)
  const sceneBindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      buffer: { type: "uniform" },
    }],
  });

  const sceneBindGroup = device.createBindGroup({
    layout: sceneBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // Group 1: 序列帧纹理与采样器
  const spriteBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
    ],
  });

  const spriteBindGroup = device.createBindGroup({
    layout: spriteBindGroupLayout,
    entries: [
      { binding: 0, resource: spriteTexture.createView() },
      { binding: 1, resource: spriteSampler },
    ],
  });

  // 显式 Pipeline Layout
  const standardPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [sceneBindGroupLayout],
  });

  const spritePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [sceneBindGroupLayout, spriteBindGroupLayout],
  });

  // 顶点布局
  const vLayout: GPUVertexBufferLayout = {
    arrayStride: 8 * 4,
    attributes: [
      { shaderLocation: 0, offset: 0,  format: "float32x3" },
      { shaderLocation: 1, offset: 12, format: "float32x3" },
      { shaderLocation: 2, offset: 24, format: "float32x2" },
    ],
  };

  const blendState: GPUBlendState = {
    color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  };

  // 1. 实体建筑地面管线
  const basePipeline = device.createRenderPipeline({
    layout: standardPipelineLayout,
    vertex: { module: shaderModule, entryPoint: "vs_base", buffers: [vLayout] },
    fragment: { module: shaderModule, entryPoint: "fs_base", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // 2. 能量流线管线 (复用 standardPipelineLayout)
  const flowPipeline = device.createRenderPipeline({
    layout: standardPipelineLayout,
    vertex: { module: shaderModule, entryPoint: "vs_flow", buffers: [vLayout] },
    fragment: { module: shaderModule, entryPoint: "fs_flow", targets: [{ format, blend: blendState }] },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less", format: "depth24plus" },
  });

  // 3. 能量立体围墙管线 (复用 standardPipelineLayout)
  const wallPipeline = device.createRenderPipeline({
    layout: standardPipelineLayout,
    vertex: { module: shaderModule, entryPoint: "vs_wall", buffers: [vLayout] },
    fragment: { module: shaderModule, entryPoint: "fs_wall", targets: [{ format, blend: blendState }] },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
    primitive: { cullMode: "none" },
  });

  // 4. 序列帧全息 HUD 管线
  const spritePipeline = device.createRenderPipeline({
    layout: spritePipelineLayout,
    vertex: { module: shaderModule, entryPoint: "vs_sprite", buffers: [vLayout] },
    fragment: { module: shaderModule, entryPoint: "fs_sprite", targets: [{ format, blend: blendState }] },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
    primitive: { cullMode: "none" },
  });

  // ----------------------------------------------------
  // 4. GUI 参数配置与交互
  // ----------------------------------------------------
  const state = {
    flowSpeed: 1.2,
    wallDisplacement: 0.28,
    scanSpeed: 1.5,
    flipbookFps: 18,
    autoRotate: 1,
  };

  gui.addTextInfo("✨ <b>科技感纹理动画特性:</b><br>• <b>流线材质</b>: UV 滚动彗星光梭脉冲。<br>• <b>立体围墙</b>: 顶点法线真位移 + 激光扫描。<br>• <b>序列帧动画</b>: 4x4 烘焙 Sprite Sheet 全息 HUD。");
  gui.add(state, "flowSpeed", 0.2, 3.0, 0.1).name("流线流动速度");
  gui.add(state, "wallDisplacement", 0.0, 0.6, 0.02).name("围墙立体波动");
  gui.add(state, "scanSpeed", 0.2, 3.0, 0.1).name("光墙扫描频率");
  gui.add(state, "flipbookFps", 1, 60, 1).name("序列帧 FPS");
  gui.add(state, "autoRotate", 0, 1, 1).name("相机自动旋转");

  const camera = { distance: 26.0, phi: 30, theta: 45 };
  let isDragging = false, lastX = 0, lastY = 0;

  canvas.addEventListener("pointerdown", (e) => {
    isDragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.4;
    camera.phi = Math.max(-85, Math.min(85, camera.phi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY;
  });
  canvas.addEventListener("pointerup", (e) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.distance = Math.max(8.0, Math.min(60.0, camera.distance + e.deltaY * 0.02));
  }, { passive: false });

  // ----------------------------------------------------
  // 5. 渲染循环
  // ----------------------------------------------------
  let animId: number;
  let depthTexture: GPUTexture | null = null;
  const uniformData = new Float32Array(40);

  function frame(timestamp: number) {
    if (!depthTexture || depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    if (state.autoRotate) camera.theta += 0.25;

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.distance * Math.cos(radPhi) * Math.sin(radTheta),
      camera.distance * Math.sin(radPhi),
      camera.distance * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const camView = Math3D.lookAt(eye, [0, 1.8, 0], [0, 1, 0]);
    const camProj = Math3D.perspective((45 * Math.PI) / 180, (canvas.width || 800) / (canvas.height || 600), 0.1, 150.0);
    const camViewProj = Math3D.multiply(camProj, camView);

    // 填充全局 Uniform 数据
    uniformData.set(camViewProj, 0);
    uniformData.set([eye[0], eye[1], eye[2], 1.0], 16);
    uniformData[20] = timestamp * 0.001;        // time (s)
    uniformData[21] = state.flowSpeed;          // 流线速度
    uniformData[22] = state.wallDisplacement;   // 围墙波形幅度
    uniformData[23] = state.scanSpeed;          // 激光扫描频率
    // 序列帧配置：fps, cols, rows, totalFrames
    uniformData[24] = state.flipbookFps;
    uniformData[25] = 4.0;
    uniformData[26] = 4.0;
    uniformData[27] = 16.0;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.02, g: 0.03, b: 0.05, a: 1.0 },
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

    // 1. 实体地面与建筑 (写入深度)
    pass.setPipeline(basePipeline);
    pass.setBindGroup(0, sceneBindGroup);
    pass.setVertexBuffer(0, baseVBO);
    pass.draw(baseVerts.length / 8);

    // 2. 金橙色能量流动管道
    pass.setPipeline(flowPipeline);
    pass.setBindGroup(0, sceneBindGroup);
    pass.setVertexBuffer(0, flowVBO);
    pass.draw(flowVerts.length / 8);

    // 3. 3D 立体波动青蓝能量围墙
    pass.setPipeline(wallPipeline);
    pass.setBindGroup(0, sceneBindGroup);
    pass.setVertexBuffer(0, wallVBO);
    pass.draw(wallVerts.length / 8);

    // 4. 全息 HUD 序列帧动画标牌
    pass.setPipeline(spritePipeline);
    pass.setBindGroup(0, sceneBindGroup);
    pass.setBindGroup(1, spriteBindGroup);
    pass.setVertexBuffer(0, spriteVBO);
    pass.draw(spriteVerts.length / 8);

    pass.end();
    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  // ----------------------------------------------------
  // 6. 销毁与资源清理
  // ----------------------------------------------------
  return () => {
    cancelAnimationFrame(animId);
    baseVBO.destroy();
    flowVBO.destroy();
    wallVBO.destroy();
    spriteVBO.destroy();
    spriteTexture.destroy();
    uniformBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}