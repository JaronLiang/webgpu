// src/examples/chaosDestruction.ts
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
// 2. 几何网格构造器 (倒角断裂砖石几何体)
// =========================================================================
function createBeveledChunkMesh(): Float32Array {
  // prettier-ignore
  return new Float32Array([
    // Front (+Z)
    -0.5,-0.5, 0.5, 0,0,1,   0.5,-0.5, 0.5, 0,0,1,   0.5, 0.5, 0.5, 0,0,1,
    -0.5,-0.5, 0.5, 0,0,1,   0.5, 0.5, 0.5, 0,0,1,  -0.5, 0.5, 0.5, 0,0,1,
    // Back (-Z)
     0.5,-0.5,-0.5, 0,0,-1, -0.5,-0.5,-0.5, 0,0,-1, -0.5, 0.5,-0.5, 0,0,-1,
     0.5,-0.5,-0.5, 0,0,-1, -0.5, 0.5,-0.5, 0,0,-1,  0.5, 0.5,-0.5, 0,0,-1,
    // Top (+Y)
    -0.5, 0.5, 0.5, 0,1,0,   0.5, 0.5, 0.5, 0,1,0,   0.5, 0.5,-0.5, 0,1,0,
    -0.5, 0.5, 0.5, 0,1,0,   0.5, 0.5,-0.5, 0,1,0,  -0.5, 0.5,-0.5, 0,1,0,
    // Bottom (-Y)
    -0.5,-0.5,-0.5, 0,-1,0,  0.5,-0.5,-0.5, 0,-1,0,  0.5,-0.5, 0.5, 0,-1,0,
    -0.5,-0.5,-0.5, 0,-1,0,  0.5,-0.5, 0.5, 0,-1,0, -0.5,-0.5, 0.5, 0,-1,0,
    // Right (+X)
     0.5,-0.5, 0.5, 1,0,0,   0.5,-0.5,-0.5, 1,0,0,   0.5, 0.5,-0.5, 1,0,0,
     0.5,-0.5, 0.5, 1,0,0,   0.5, 0.5,-0.5, 1,0,0,   0.5, 0.5, 0.5, 1,0,0,
    // Left (-X)
    -0.5,-0.5,-0.5,-1,0,0,  -0.5,-0.5, 0.5,-1,0,0,  -0.5, 0.5, 0.5,-1,0,0,
    -0.5,-0.5,-0.5,-1,0,0,  -0.5, 0.5, 0.5,-1,0,0,  -0.5, 0.5,-0.5,-1,0,0,
  ]);
}

// =========================================================================
// 3. Chaos Geometry Collection 核心结构与连接图体系
// =========================================================================
interface ChaosChunk {
  id: number;
  initialPos: number[];
  pos: number[];
  quat: number[];
  scale: number[];
  linearVel: number[];
  angularVel: number[];
  mass: number;
  isAnchor: boolean;      // 地基锚定节点（永远不会因重力下坠）
  isDynamic: boolean;     // 是否已脱离结构成为活动刚体
  stress: number;         // 当前承受的冲击应力
  neighbors: number[];    // 连接图边 (Connectivity Graph Edges)
}

function generateTowerCollection(): { chunks: ChaosChunk[]; edges: [number, number][] } {
  const chunks: ChaosChunk[] = [];
  const edges: [number, number][] = [];

  const LAYERS = 14;      // 楼层高度
  const COLS = 5;         // X轴柱数
  const ROWS = 5;         // Z轴柱数
  const BLOCK_W = 1.1;
  const BLOCK_H = 0.75;
  const BLOCK_D = 1.1;

  let currentId = 0;

  for (let y = 0; y < LAYERS; y++) {
    const isGroundAnchor = (y === 0);
    const isCorePillarLayer = (y % 3 !== 0);

    for (let z = 0; z < ROWS; z++) {
      for (let x = 0; x < COLS; x++) {
        // 空心走廊结构：中间空出天井，强化真实建筑剪力墙与立柱效果
        const isCore = (x >= 1 && x <= 3 && z >= 1 && z <= 3);
        if (isCore && isCorePillarLayer && !isGroundAnchor) {
          continue; // 中空天井
        }

        const posX = (x - (COLS - 1) / 2) * (BLOCK_W + 0.05);
        const posY = 0.4 + y * (BLOCK_H + 0.02);
        const posZ = (z - (ROWS - 1) / 2) * (BLOCK_D + 0.05);

        chunks.push({
          id: currentId++,
          initialPos: [posX, posY, posZ],
          pos: [posX, posY, posZ],
          quat: [0, 0, 0, 1],
          scale: [BLOCK_W, BLOCK_H, BLOCK_D],
          linearVel: [0, 0, 0],
          angularVel: [0, 0, 0],
          mass: isGroundAnchor ? 0 : 25.0,
          isAnchor: isGroundAnchor,
          isDynamic: false,
          stress: 0.0,
          neighbors: [],
        });
      }
    }
  }

  // 构建拓扑连接图 (Connectivity Graph Construction)
  // 如果两个块之间的曼哈顿空间距离接近，则建立物理连接约束边
  for (let i = 0; i < chunks.length; i++) {
    for (let j = i + 1; j < chunks.length; j++) {
      const a = chunks[i];
      const b = chunks[j];
      const dx = Math.abs(a.pos[0] - b.pos[0]);
      const dy = Math.abs(a.pos[1] - b.pos[1]);
      const dz = Math.abs(a.pos[2] - b.pos[2]);

      const isConnected =
        (dx < BLOCK_W * 1.08 && dy < 0.2 && dz < 0.2) || // 水平 X 相邻
        (dz < BLOCK_D * 1.08 && dy < 0.2 && dx < 0.2) || // 水平 Z 相邻
        (dy < BLOCK_H * 1.08 && dx < 0.2 && dz < 0.2);   // 垂直 Y 叠压

      if (isConnected) {
        a.neighbors.push(b.id);
        b.neighbors.push(a.id);
        edges.push([a.id, b.id]);
      }
    }
  }

  return { chunks, edges };
}

// =========================================================================
// 4. 主程序入口
// =========================================================================
export function runChaosDestruction(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: GUI
) {
  const { chunks, edges } = generateTowerCollection();
  const NUM_CHUNKS = chunks.length;

  // 1. 创建网格几何缓冲
  const chunkMesh = createBeveledChunkMesh();
  const chunkVertexBuffer = device.createBuffer({
    size: chunkMesh.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(chunkVertexBuffer, 0, chunkMesh as any);

  // 2. GPU 实例缓冲区 (每实例包含: 16 float 矩阵 + 4 float 状态属性 = 80 字节)
  const INSTANCE_STRIDE = 20 * 4; // 80 bytes
  const instanceData = new Float32Array(NUM_CHUNKS * 20);
  const instanceBuffer = device.createBuffer({
    size: instanceData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // 3. 常量缓冲区
  const uniformBufferSize = 256;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 5. 着色器实现 (双层材质：外部建筑外墙 + 内部粗糙破碎石料断裂面)
  // =========================================================================
  const destructionShaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
      camPos: vec4f,
      lightDir: vec4f,
      params: vec4f, // x: time, y: shockwaveAge, z: shockX, w: shockZ
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexInput {
      @location(0) pos: vec3f,
      @location(1) normal: vec3f,
      // 实例属性输入
      @location(2) matCol0: vec4f,
      @location(3) matCol1: vec4f,
      @location(4) matCol2: vec4f,
      @location(5) matCol3: vec4f,
      @location(6) extraData: vec4f, // x: isDynamic, y: isAnchor, z: stress, w: id
    };

    struct VertexOut {
      @builtin(position) clipPos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) extraData: vec4f,
    };

    @vertex
    fn vs_chunk(in: VertexInput) -> VertexOut {
      var out: VertexOut;
      let modelMat = mat4x4f(in.matCol0, in.matCol1, in.matCol2, in.matCol3);
      let wPos = modelMat * vec4f(in.pos, 1.0);
      out.clipPos = u.viewProj * wPos;
      out.worldPos = wPos.xyz;
      out.normal = (modelMat * vec4f(in.normal, 0.0)).xyz;
      out.extraData = in.extraData;
      return out;
    }

    @fragment
    fn fs_chunk(in: VertexOut) -> @location(0) vec4f {
      let N = normalize(in.normal);
      let L = normalize(u.lightDir.xyz);
      let V = normalize(u.camPos.xyz - in.worldPos);
      let H = normalize(L + V);

      let diff = max(dot(N, L), 0.0);
      let spec = pow(max(dot(N, H), 0.0), 32.0);

      let isDynamic = in.extraData.x;
      let isAnchor = in.extraData.y;
      let stress = in.extraData.z;

      // 建筑外立面 vs 内部石材断裂面
      var baseColor = vec3f(0.82, 0.85, 0.88); // 混凝土浅灰白外墙

      if (isAnchor > 0.5) {
        baseColor = vec3f(0.24, 0.32, 0.38); // 坚固深石墨青地基
      } else if (isDynamic > 0.5) {
        // 断裂飞溅的碎片展示粗糙断口与红热碰撞余烬
        let fractureNoise = fract(sin(dot(in.worldPos.xyz, vec3f(12.9898, 78.233, 45.164))) * 43758.5453);
        baseColor = mix(vec3f(0.65, 0.60, 0.55), vec3f(0.45, 0.40, 0.38), fractureNoise);
      }

      // 受力应力红色预警发光 (Chaos Stress Field Visualization)
      if (stress > 0.05) {
        let stressGlow = vec3f(1.0, 0.15, 0.05) * min(stress, 1.5);
        baseColor = mix(baseColor, stressGlow, min(stress * 0.7, 0.85));
      }

      let ambient = vec3f(0.12, 0.16, 0.22) * baseColor;
      let diffuse = baseColor * (diff * 0.75);
      let specular = vec3f(0.9) * (spec * 0.35);

      return vec4f(ambient + diffuse + specular, 1.0);
    }

    // 地面网格着色
    @vertex
    fn vs_ground(@builtin(vertex_index) vid: u32) -> VertexOut {
      var out: VertexOut;
      let sz = 120.0;
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
      out.extraData = vec4f(0.0);
      return out;
    }

    @fragment
    fn fs_ground(in: VertexOut) -> @location(0) vec4f {
      let coord = in.worldPos.xz * 0.4;
      let fw = fwidth(coord);
      let grid = abs(fract(coord - 0.5) - 0.5) / max(fw, vec2f(0.001));
      let line = min(grid.x, grid.y);
      let gridMask = 1.0 - min(line, 1.0);

      let gridCol = vec3f(0.18, 0.28, 0.35);
      let groundCol = mix(vec3f(0.08, 0.12, 0.16), gridCol, gridMask * 0.8);
      return vec4f(groundCol, 1.0);
    }
  `;

  // =========================================================================
  // 6. 创建渲染管线与绑定
  // =========================================================================
  const shaderModule = device.createShaderModule({ code: destructionShaderCode });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  // 碎块管线 (Mesh 顶点属性 + 实例矩阵属性)
  const chunkPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_chunk",
      buffers: [
        // Buffer 0: 几何网格
        {
          arrayStride: 6 * 4,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
          ],
        },
        // Buffer 1: 实例矩阵与属性 (Instance Data)
        {
          arrayStride: INSTANCE_STRIDE,
          stepMode: "instance",
          attributes: [
            { shaderLocation: 2, offset: 0, format: "float32x4" },  // mat col 0
            { shaderLocation: 3, offset: 16, format: "float32x4" }, // mat col 1
            { shaderLocation: 4, offset: 32, format: "float32x4" }, // mat col 2
            { shaderLocation: 5, offset: 48, format: "float32x4" }, // mat col 3
            { shaderLocation: 6, offset: 64, format: "float32x4" }, // extraData
          ],
        },
      ],
    },
    fragment: { module: shaderModule, entryPoint: "fs_chunk", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const groundPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: shaderModule, entryPoint: "vs_ground" },
    fragment: { module: shaderModule, entryPoint: "fs_ground", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const mainBindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // =========================================================================
  // 7. Chaos Destruction 拓扑崩塌算法与 Chaos Cache 缓存系统
  // =========================================================================
  const settings = {
    // 破坏力学
    damageThreshold: 0.85,    // 连接断裂阈值
    impactForce: 35.0,        // 炮弹冲击动量
    blastRadius: 4.8,         // 爆炸杀伤半径
    gravity: -18.0,           // 崩塌重力加速度

    // Chaos Cache 缓存与回放系统
    isRecording: true,
    isPlayback: false,
    playbackFrame: 0,
    playbackSpeed: 1.0,

    fireCannonball: () => fireImpactAtTarget(),
    resetStructure: () => resetDestruction(),
  };

  // Chaos Cache 内存轨道系统 (记录每个物理帧的所有 Chunks 姿态)
  interface CacheFrame {
    positions: Float32Array; // N * 3
    quats: Float32Array;     // N * 4
    extras: Float32Array;    // N * 4
  }
  const chaosCache: CacheFrame[] = [];
  const MAX_CACHE_FRAMES = 900; // 最多缓存 15 秒 60FPS 破坏动画

  function recordCurrentFrame() {
    if (!settings.isRecording || settings.isPlayback) return;

    const posTrack = new Float32Array(NUM_CHUNKS * 3);
    const quatTrack = new Float32Array(NUM_CHUNKS * 4);
    const extraTrack = new Float32Array(NUM_CHUNKS * 4);

    for (let i = 0; i < NUM_CHUNKS; i++) {
      const c = chunks[i];
      posTrack[i * 3 + 0] = c.pos[0];
      posTrack[i * 3 + 1] = c.pos[1];
      posTrack[i * 3 + 2] = c.pos[2];

      quatTrack[i * 4 + 0] = c.quat[0];
      quatTrack[i * 4 + 1] = c.quat[1];
      quatTrack[i * 4 + 2] = c.quat[2];
      quatTrack[i * 4 + 3] = c.quat[3];

      extraTrack[i * 4 + 0] = c.isDynamic ? 1.0 : 0.0;
      extraTrack[i * 4 + 1] = c.isAnchor ? 1.0 : 0.0;
      extraTrack[i * 4 + 2] = c.stress;
      extraTrack[i * 4 + 3] = c.id;
    }

    chaosCache.push({ positions: posTrack, quats: quatTrack, extras: extraTrack });
    if (chaosCache.length > MAX_CACHE_FRAMES) {
      chaosCache.shift();
    }
  }

  function applyCacheFrame(frameIndex: number) {
    if (chaosCache.length === 0) return;
    const idx = Math.max(0, Math.min(chaosCache.length - 1, Math.floor(frameIndex)));
    const frame = chaosCache[idx];

    for (let i = 0; i < NUM_CHUNKS; i++) {
      chunks[i].pos[0] = frame.positions[i * 3 + 0];
      chunks[i].pos[1] = frame.positions[i * 3 + 1];
      chunks[i].pos[2] = frame.positions[i * 3 + 2];

      chunks[i].quat[0] = frame.quats[i * 4 + 0];
      chunks[i].quat[1] = frame.quats[i * 4 + 1];
      chunks[i].quat[2] = frame.quats[i * 4 + 2];
      chunks[i].quat[3] = frame.quats[i * 4 + 3];

      chunks[i].isDynamic = frame.extras[i * 4 + 0] > 0.5;
      chunks[i].stress = frame.extras[i * 4 + 2];
    }
  }

  // 触发高能穿透炮击
  function fireImpactAtTarget(targetPos?: number[]) {
    // 默认射向建筑中下部核心承重立柱
    const hit = targetPos || [0.4, 3.2, 0.2];

    for (let i = 0; i < NUM_CHUNKS; i++) {
      const c = chunks[i];
      if (c.isAnchor) continue;

      const dx = c.pos[0] - hit[0];
      const dy = c.pos[1] - hit[1];
      const dz = c.pos[2] - hit[2];
      const dist = Math.hypot(dx, dy, dz);

      if (dist < settings.blastRadius) {
        const falloff = 1.0 - dist / settings.blastRadius;
        const impulse = falloff * settings.impactForce;

        // 斩断连接图边
        c.neighbors = [];
        c.isDynamic = true;
        c.stress = 2.0;

        // 爆炸扩散动量
        const nx = (dx + (Math.random() - 0.5) * 0.4) / (dist + 0.1);
        const ny = (dy + 0.5) / (dist + 0.1);
        const nz = (dz + (Math.random() - 0.5) * 0.4) / (dist + 0.1);

        c.linearVel[0] += nx * impulse;
        c.linearVel[1] += ny * impulse * 1.2;
        c.linearVel[2] += nz * impulse;

        c.angularVel[0] += (Math.random() - 0.5) * impulse * 0.4;
        c.angularVel[1] += (Math.random() - 0.5) * impulse * 0.4;
        c.angularVel[2] += (Math.random() - 0.5) * impulse * 0.4;
      }
    }

    // 结构断裂后立即执行拓扑连通性崩塌解算
    resolveStructuralConnectivity();
  }

  // 【核心机制】：广度优先搜索（BFS）解算整体结构崩塌
  // 任何无法连通回地基锚点的悬空碎块全部失去支撑，发生连锁坍塌！
  function resolveStructuralConnectivity() {
    const isSupported = new Uint8Array(NUM_CHUNKS);
    const queue: number[] = [];

    // 1. 将所有锚定地基加入种子队列
    for (let i = 0; i < NUM_CHUNKS; i++) {
      if (chunks[i].isAnchor) {
        isSupported[i] = 1;
        queue.push(i);
      }
    }

    // 2. 沿连接图向上传递支撑力
    let head = 0;
    while (head < queue.length) {
      const currId = queue[head++];
      const currChunk = chunks[currId];

      for (let k = 0; k < currChunk.neighbors.length; k++) {
        const nId = currChunk.neighbors[k];
        if (!isSupported[nId]) {
          isSupported[nId] = 1;
          queue.push(nId);
        }
      }
    }

    // 3. 未被支撑的悬空结构，全部转为崩塌动态刚体 (Structural Disintegration)
    for (let i = 0; i < NUM_CHUNKS; i++) {
      const c = chunks[i];
      if (!isSupported[i] && !c.isDynamic) {
        c.isDynamic = true;
        c.stress = 0.8;
        // 赋予轻微初始自旋与外翻速度
        c.angularVel[0] = (Math.random() - 0.5) * 1.8;
        c.angularVel[2] = (Math.random() - 0.5) * 1.8;
      }
    }
  }

  function resetDestruction() {
    chaosCache.length = 0;
    settings.isPlayback = false;
    settings.playbackFrame = 0;

    const { chunks: freshChunks, edges: freshEdges } = generateTowerCollection();
    for (let i = 0; i < NUM_CHUNKS; i++) {
      Object.assign(chunks[i], freshChunks[i]);
    }
  }

  // =========================================================================
  // 8. GUI 面板与相机控制
  // =========================================================================
  gui.title("WebGPU Chaos Destruction 破坏模拟");

  const blastFolder = gui.addFolder("破坏与应力系统");
  blastFolder.add(settings, "fireCannonball").name("发射高能破坏弹");
  blastFolder.add(settings, "impactForce", 10, 80, 2).name("冲击动量 (Force)");
  blastFolder.add(settings, "blastRadius", 2.0, 10.0, 0.5).name("爆炸半径 (Radius)");
  blastFolder.add(settings, "gravity", -35.0, -5.0, 1.0).name("坍塌重力加速度");
  blastFolder.open();

  const cacheFolder = gui.addFolder("Chaos Cache 缓存与影视级回放");
  cacheFolder.add(settings, "isRecording").name("录制物理轨道 (Record)");
  cacheFolder.add(settings, "isPlayback").name("开启时间轴回放").listen().onChange((val: boolean) => {
    if (val && chaosCache.length > 0) {
      settings.playbackFrame = chaosCache.length - 1;
    }
  });

  const scrubberController = cacheFolder.add(settings, "playbackFrame", 0, 100, 1).name("时间轴拖拽 (Scrub)").listen().onChange((f: number) => {
    if (settings.isPlayback) {
      applyCacheFrame(f);
    }
  });

  cacheFolder.add(settings, "playbackSpeed", -2.0, 2.0, 0.1).name("回放速率 (倒放/慢动作)");
  cacheFolder.open();

  gui.add(settings, "resetStructure").name("复原建筑物 (Reset)");

  // 轨道交互相机
  const camera = { target: [0, 4.5, 0], radius: 24.0, theta: 45.0, phi: 26.0 };
  let isDragging = false, dragButton = 0, lastX = 0, lastY = 0;

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    isDragging = true;
    dragButton = e.button;
    lastX = e.clientX;
    lastY = e.clientY;
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
      const pan = camera.radius * 0.0016;
      camera.target[0] -= Math.cos(radTheta) * dx * pan;
      camera.target[2] -= -Math.sin(radTheta) * dx * pan;
      camera.target[1] += dy * pan;
    }
  });

  canvas.addEventListener("pointerup", (e) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.radius = Math.max(6, Math.min(60, camera.radius * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });

  // 鼠标双击直接发射定向破墙弹
  canvas.addEventListener("dblclick", () => {
    fireImpactAtTarget([
      (Math.random() - 0.5) * 3.5,
      1.5 + Math.random() * 5.5,
      (Math.random() - 0.5) * 3.5,
    ]);
  });

  // =========================================================================
  // 9. 物理与渲染主循环
  // =========================================================================
  let depthTexture: GPUTexture | null = null;
  let animId: number;
  let lastTime = performance.now();

  function updatePhysics(dt: number) {
    if (settings.isPlayback) {
      // 回放模式：直接沿时间轴读取 Chaos Cache
      if (chaosCache.length > 0) {
        settings.playbackFrame += dt * 60.0 * settings.playbackSpeed;
        if (settings.playbackFrame >= chaosCache.length) {
          settings.playbackFrame = 0; // 循环循环
        } else if (settings.playbackFrame < 0) {
          settings.playbackFrame = chaosCache.length - 1;
        }
        applyCacheFrame(settings.playbackFrame);
      }
      return;
    }

    // 实时物理仿真模式：解算崩塌与碎片动力学
    for (let i = 0; i < NUM_CHUNKS; i++) {
      const c = chunks[i];
      if (c.isAnchor || !c.isDynamic) continue;

      // 重力加速度
      c.linearVel[1] += settings.gravity * dt;

      // 简单空气阻力
      c.linearVel[0] *= 0.99;
      c.linearVel[2] *= 0.99;
      c.angularVel[0] *= 0.98;
      c.angularVel[1] *= 0.98;
      c.angularVel[2] *= 0.98;

      // 位移更新
      c.pos[0] += c.linearVel[0] * dt;
      c.pos[1] += c.linearVel[1] * dt;
      c.pos[2] += c.linearVel[2] * dt;

      // 地面接触反弹与滑动摩擦
      const groundContactY = c.scale[1] * 0.5;
      if (c.pos[1] < groundContactY) {
        c.pos[1] = groundContactY;
        c.linearVel[1] = -c.linearVel[1] * 0.28; // 弹跳恢复系数
        c.linearVel[0] *= 0.72; // 地面摩擦力
        c.linearVel[2] *= 0.72;
      }

      // 旋转角位移积分 (四元数积分)
      const wLen = Math.hypot(c.angularVel[0], c.angularVel[1], c.angularVel[2]);
      if (wLen > 1e-4) {
        const dq = quatFromAxisAngle(
          [c.angularVel[0] / wLen, c.angularVel[1] / wLen, c.angularVel[2] / wLen],
          wLen * dt
        );
        c.quat = quatNormalize(quatMultiply(dq, c.quat));
      }

      // 应力自然消退
      if (c.stress > 0.0) {
        c.stress = Math.max(0.0, c.stress - dt * 0.6);
      }
    }

    // 录制当前物理步
    recordCurrentFrame();
  }

  function frame() {
    const now = performance.now();
    const dt = Math.min((now - lastTime) * 0.001, 0.033);
    lastTime = now;

    // 更新物理步进
    updatePhysics(dt);

    // 动态同步时间轴滑块上限
    if (chaosCache.length > 0) {
      scrubberController.max(chaosCache.length - 1);
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

    // 相机与投影矩阵
    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.target[0] + camera.radius * Math.cos(radPhi) * Math.sin(radTheta),
      camera.target[1] + camera.radius * Math.sin(radPhi),
      camera.target[2] + camera.radius * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const proj = mat4Perspective((45 * Math.PI) / 180, renderWidth / renderHeight, 0.2, 200.0);
    const view = mat4LookAt(eye, camera.target, [0, 1, 0]);
    const viewProj = mat4Multiply(proj, view);

    // 更新 Uniform 常量
    const uniformData = new Float32Array(uniformBufferSize / 4);
    uniformData.set(viewProj, 0); // 0..15
    uniformData[16] = eye[0]; uniformData[17] = eye[1]; uniformData[18] = eye[2]; uniformData[19] = 1.0;
    uniformData[20] = 0.55; uniformData[21] = 0.85; uniformData[22] = 0.40; uniformData[23] = 0.0; // 主光源
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // 批量同步所有断裂块实例矩阵到 GPU 缓冲 (Batch Instance Transform Update)
    for (let i = 0; i < NUM_CHUNKS; i++) {
      const c = chunks[i];
      const modelMat = mat4FromRotationTranslationScale(c.quat, c.pos, c.scale);
      const offset = i * 20;

      // 写入 4x4 模型矩阵
      instanceData.set(modelMat, offset);
      // 写入状态元数据
      instanceData[offset + 16] = c.isDynamic ? 1.0 : 0.0;
      instanceData[offset + 17] = c.isAnchor ? 1.0 : 0.0;
      instanceData[offset + 18] = c.stress;
      instanceData[offset + 19] = c.id;
    }
    device.queue.writeBuffer(instanceBuffer, 0, instanceData);

    // 绘制通道
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

    // 1. 绘制网格地面
    renderPass.setPipeline(groundPipeline);
    renderPass.setBindGroup(0, mainBindGroup);
    renderPass.draw(6, 1, 0, 0);

    // 2. 批量实例化绘制 Chaos 几何体断裂合集 (Instanced Geometry Collection)
    renderPass.setPipeline(chunkPipeline);
    renderPass.setBindGroup(0, mainBindGroup);
    renderPass.setVertexBuffer(0, chunkVertexBuffer);
    renderPass.setVertexBuffer(1, instanceBuffer);
    renderPass.draw(36, NUM_CHUNKS, 0, 0);

    renderPass.end();
    device.queue.submit([encoder.finish()]);

    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    chunkVertexBuffer.destroy();
    instanceBuffer.destroy();
    uniformBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}