// src/examples/tileBased.ts
import GUI from "lil-gui";

// =========================================================================
// 1. 标准 3D 数学库 (包含逆矩阵计算，用于屏幕空间反投影构建 Tile 视锥体)
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

function mat4Invert(m: Float32Array): Float32Array {
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

// =========================================================================
// 2. 主程序
// =========================================================================
export function runTileBased(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: any
) {
  const TILE_SIZE = 16;              // 每个 Tile 为 16x16 像素
  const MAX_LIGHTS_PER_TILE = 256;  // 每个 Tile 允许存储的最大光源数
  const LIGHT_COUNT = 512;          // 动态光源总数
  const PILLAR_COUNT = 256;         // 场景柱体数量

  // 1. 创建场景几何体 (立方体柱子网格)
  const cubeData = new Float32Array([
    // X, Y, Z,  NX, NY, NZ
    -1,-1, 1, 0,0,1,   1,-1, 1, 0,0,1,   1, 1, 1, 0,0,1,
    -1,-1, 1, 0,0,1,   1, 1, 1, 0,0,1,  -1, 1, 1, 0,0,1,
     1,-1,-1, 0,0,-1, -1,-1,-1, 0,0,-1, -1, 1,-1, 0,0,-1,
     1,-1,-1, 0,0,-1, -1, 1,-1, 0,0,-1,  1, 1,-1, 0,0,-1,
    -1, 1, 1, 0,1,0,   1, 1, 1, 0,1,0,   1, 1,-1, 0,1,0,
    -1, 1, 1, 0,1,0,   1, 1,-1, 0,1,0,  -1, 1,-1, 0,1,0,
    -1,-1,-1, 0,-1,0,  1,-1,-1, 0,-1,0,  1,-1, 1, 0,-1,0,
    -1,-1,-1, 0,-1,0,  1,-1, 1, 0,-1,0, -1,-1, 1, 0,-1,0,
     1,-1, 1, 1,0,0,   1,-1,-1, 1,0,0,   1, 1,-1, 1,0,0,
     1,-1, 1, 1,0,0,   1, 1,-1, 1,0,0,   1, 1, 1, 1,0,0,
    -1,-1,-1,-1,0,0,  -1,-1, 1,-1,0,0,  -1, 1, 1,-1,0,0,
    -1,-1,-1,-1,0,0,  -1, 1, 1,-1,0,0,  -1, 1,-1,-1,0,0,
  ]);
  const vertexBuffer = device.createBuffer({ size: cubeData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vertexBuffer, 0, cubeData);

  // 场景实例数据 (构建一个地砖和多根立柱)
  const instancesData = new Float32Array(PILLAR_COUNT * 4);
  const side = Math.sqrt(PILLAR_COUNT);
  for (let i = 0; i < PILLAR_COUNT; i++) {
    const gx = (i % side) - side * 0.5;
    const gz = Math.floor(i / side) - side * 0.5;
    instancesData[i * 4 + 0] = gx * 4.5;                     // PosX
    instancesData[i * 4 + 1] = 0.0;                          // PosY
    instancesData[i * 4 + 2] = gz * 4.5;                     // PosZ
    instancesData[i * 4 + 3] = 1.0 + Math.random() * 2.5;    // ScaleY (高矮不齐的柱子)
  }
  const instanceStorage = device.createBuffer({ size: instancesData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(instanceStorage, 0, instancesData);

  // 2. 光源数据 [pos.xyz, radius, color.rgb, padding]
  const rawLightData = new Float32Array(LIGHT_COUNT * 8);
  const lightVelocities: { x: number; y: number; z: number }[] = [];
  for (let i = 0; i < LIGHT_COUNT; i++) {
    rawLightData[i * 8 + 0] = (Math.random() - 0.5) * 60;   // X
    rawLightData[i * 8 + 1] = 0.5 + Math.random() * 5.0;    // Y
    rawLightData[i * 8 + 2] = (Math.random() - 0.5) * 60;   // Z
    rawLightData[i * 8 + 3] = 4.0 + Math.random() * 3.0;    // 影响半径 Radius

    rawLightData[i * 8 + 4] = 0.2 + Math.random() * 0.8;    // R
    rawLightData[i * 8 + 5] = 0.2 + Math.random() * 0.8;    // G
    rawLightData[i * 8 + 6] = 0.2 + Math.random() * 0.8;    // B
    rawLightData[i * 8 + 7] = 1.0;

    lightVelocities.push({
      x: (Math.random() - 0.5) * 0.1,
      y: (Math.random() - 0.5) * 0.05,
      z: (Math.random() - 0.5) * 0.1
    });
  }
  const lightStorage = device.createBuffer({ size: rawLightData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  // 3. Uniform 缓冲区
  // 包含: ViewProj(64), View(64), InvProj(64), ScreenSize(8), TileGrid(8), Params(16) -> 总共 224 bytes
  const uniformBuffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // =========================================================================
  // 3. WGSL 核心着色器：Compute 视锥分块剔除 + Tile-Based 渲染
  // =========================================================================
  const shaderCode = `
    struct Light {
      pos: vec3f,
      radius: f32,
      color: vec3f,
      padding: f32,
    };

    struct CameraUniforms {
      viewProj: mat4x4f,
      view: mat4x4f,
      invProj: mat4x4f,
      screenSize: vec2f,
      tileCount: vec2u,
      mode: u32,             // 0: Tiled Forward+, 1: 全光源暴力循环(对照组), 2: Tile 热力图
      lightCount: u32,
      showGrid: u32,
      pad: u32,
    };

    @group(0) @binding(0) var<uniform> u: CameraUniforms;
    @group(0) @binding(1) var<storage, read> lights: array<Light>;
    // 每个 Tile 存储格式: [0] = 实际相交的光源数，[1..N] = 光源索引
    @group(0) @binding(2) var<storage, read_write> tileLightGrid: array<u32>;

    // -------------------------------------------------------------
    // COMPUTE SHADER: Tile 视锥体构建与多线程光源求交剔除
    // -------------------------------------------------------------
    var<workgroup> tileVisibleCount: atomic<u32>;
    var<workgroup> tileLightIndices: array<u32, ${MAX_LIGHTS_PER_TILE}>;

    fn getTileFrustumPlane(p0: vec3f, p1: vec3f) -> vec4f {
      let n = normalize(cross(p0, p1));
      return vec4f(n, 0.0); // 视点为 (0,0,0) 的视锥体侧面平面
    }

    fn unproject(ndc: vec2f) -> vec3f {
      let clip = vec4f(ndc, 1.0, 1.0);
      let v = u.invProj * clip;
      return v.xyz / v.w;
    }

    @compute @workgroup_size(16, 16, 1)
    fn cs_tile_cull(
      @builtin(workgroup_id) workgroupId: vec3u,
      @builtin(local_invocation_index) localIdx: u32
    ) {
      let tileX = workgroupId.x;
      let tileY = workgroupId.y;
      if (tileX >= u.tileCount.x || tileY >= u.tileCount.y) { return; }

      if (localIdx == 0u) {
        atomicStore(&tileVisibleCount, 0u);
      }
      workgroupBarrier();

      // 1. 还原 Tile 在 NDC 空间的四角坐标
      let tileStep = 2.0 / vec2f(f32(u.tileCount.x), f32(u.tileCount.y));
      let minNdc = vec2f(-1.0, -1.0) + vec2f(f32(tileX), f32(tileY)) * tileStep;
      let maxNdc = minNdc + tileStep;

      // 2. 反投影至观察空间 (View Space)
      let p00 = unproject(vec2f(minNdc.x, minNdc.y));
      let p10 = unproject(vec2f(maxNdc.x, minNdc.y));
      let p01 = unproject(vec2f(minNdc.x, maxNdc.y));
      let p11 = unproject(vec2f(maxNdc.x, maxNdc.y));

      // 3. 构建 4 个侧剪裁面 (法线指向视锥体内侧)
      let planeLeft   = getTileFrustumPlane(p00, p01);
      let planeRight  = getTileFrustumPlane(p11, p10);
      let planeBottom = getTileFrustumPlane(p10, p00);
      let planeTop    = getTileFrustumPlane(p01, p11);

      // 4. 并行求交：当前工作组内的 256 个线程分批测试全场景光源
      let totalLights = u.lightCount;
      for (var i = localIdx; i < totalLights; i += 256u) {
        let light = lights[i];
        let lightViewPos = (u.view * vec4f(light.pos, 1.0)).xyz;

        // 测试光源球体是否在视锥体内
        let r = light.radius;
        let inside = dot(planeLeft.xyz, lightViewPos) > -r &&
                     dot(planeRight.xyz, lightViewPos) > -r &&
                     dot(planeBottom.xyz, lightViewPos) > -r &&
                     dot(planeTop.xyz, lightViewPos) > -r &&
                     (-lightViewPos.z > 0.1 - r); // 近平面剔除

        if (inside) {
          let slot = atomicAdd(&tileVisibleCount, 1u);
          if (slot < ${MAX_LIGHTS_PER_TILE}u) {
            tileLightIndices[slot] = i;
          }
        }
      }

      workgroupBarrier();

      // 5. 将当前 Tile 的裁剪结果写入全局 Storage Buffer
      let tileIndex = tileY * u.tileCount.x + tileX;
      let baseOffset = tileIndex * (${MAX_LIGHTS_PER_TILE}u + 1u);
      let finalCount = min(atomicLoad(&tileVisibleCount), ${MAX_LIGHTS_PER_TILE}u);

      if (localIdx == 0u) {
        tileLightGrid[baseOffset] = finalCount;
      }
      for (var i = localIdx; i < finalCount; i += 256u) {
        tileLightGrid[baseOffset + 1u + i] = tileLightIndices[i];
      }
    }

    // -------------------------------------------------------------
    // RENDER SHADER: 读取 Tile 列表进行分块着色
    // -------------------------------------------------------------
    struct VertexInput {
      @location(0) pos: vec3f,
      @location(1) normal: vec3f,
      @builtin(instance_index) instIdx: u32,
    };

    struct VertexOutput {
      @builtin(position) clipPos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
    };

    @group(0) @binding(3) var<storage, read> instances: array<vec4f>;

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
      var out: VertexOutput;
      let inst = instances[in.instIdx];
      // inst: x, y, z, scaleY
      var localPos = in.pos;
      localPos.y = (localPos.y + 1.0) * inst.w; // 缩放柱体高度
      let worldPos = localPos + vec3f(inst.x, inst.y, inst.z);

      out.clipPos = u.viewProj * vec4f(worldPos, 1.0);
      out.worldPos = worldPos;
      out.normal = in.normal;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let N = normalize(in.normal);
      var accumColor = vec3f(0.04); // 微弱环境光

      // 确定像素归属于哪个 Tile
      let screenPixel = in.clipPos.xy;
      let tileCoord = vec2u(screenPixel) / ${TILE_SIZE}u;
      let tileIdx = tileCoord.y * u.tileCount.x + tileCoord.x;
      let baseOffset = tileIdx * (${MAX_LIGHTS_PER_TILE}u + 1u);

      // 读取由 Compute Shader 裁剪后留存在当前 Tile 内的光源数量
      let tileLightCount = tileLightGrid[baseOffset];

      // 模式 2: 热力图可视化模式 (Tile 内光源越多，颜色越向红黄色渐变)
      if (u.mode == 2u) {
        let heat = clamp(f32(tileLightCount) / 30.0, 0.0, 1.0);
        var heatColor = vec3f(heat, 1.0 - heat, 0.1);
        if (heat > 0.5) { heatColor = vec3f(1.0, 1.0 - (heat - 0.5) * 2.0, 0.0); }
        // 渲染 Tile 物理网格线
        if (u.showGrid == 1u) {
          let modP = vec2u(screenPixel) % ${TILE_SIZE}u;
          if (modP.x == 0u || modP.y == 0u) { return vec4f(1.0, 1.0, 1.0, 1.0); }
        }
        return vec4f(heatColor, 1.0);
      }

      // 模式 0: Tile-Based 分块光照 (仅遍历当前 Tile 的光源)
      if (u.mode == 0u) {
        for (var i = 0u; i < tileLightCount; i++) {
          let lightIdx = tileLightGrid[baseOffset + 1u + i];
          let l = lights[lightIdx];
          let toL = l.pos - in.worldPos;
          let dist = length(toL);
          if (dist < l.radius) {
            let atten = clamp(1.0 - (dist / l.radius), 0.0, 1.0);
            let diff = max(dot(N, toL / dist), 0.0);
            accumColor += l.color * diff * atten * atten * 2.0;
          }
        }
      } 
      // 模式 1: 暴力遍历所有光源 (O(N) 灾难模式)
      else if (u.mode == 1u) {
        for (var i = 0u; i < u.lightCount; i++) {
          let l = lights[i];
          let toL = l.pos - in.worldPos;
          let dist = length(toL);
          if (dist < l.radius) {
            let atten = clamp(1.0 - (dist / l.radius), 0.0, 1.0);
            let diff = max(dot(N, toL / dist), 0.0);
            accumColor += l.color * diff * atten * atten * 2.0;
          }
        }
      }

      // 显示 Tile 网格线以便直观观察划分
      if (u.showGrid == 1u) {
        let modP = vec2u(screenPixel) % ${TILE_SIZE}u;
        if (modP.x == 0u || modP.y == 0u) {
          accumColor += vec3f(0.2);
        }
      }

      return vec4f(accumColor, 1.0);
    }
  `;

  const shaderModule = device.createShaderModule({ code: shaderCode });

  // =========================================================================
  // 4. 管线与绑定组设置
  // =========================================================================
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ]
  });

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  // Compute Pipeline
  const computePipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "cs_tile_cull" }
  });

  // Render Pipeline
  const renderPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule, entryPoint: "vs_main",
      buffers: [{
        arrayStride: 6 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" }
        ]
      }]
    },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" }
  });

  // =========================================================================
  // 5. 交互与状态控制
  // =========================================================================
  const settings = {
    mode: "Tile-Based Forward+ (推荐, 极快)",
    activeLights: LIGHT_COUNT,
    showGrid: true,
    autoRotate: true,
  };

  gui.title("WebGPU Tile-Based 分块光照");
  gui.add(settings, "mode", [
    "Tile-Based Forward+ (推荐, 极快)",
    "暴力全光源循环 (无Tile, 极卡)",
    "Tile 光源热力图可视化"
  ]).name("渲染方案");
  gui.add(settings, "activeLights", 16, LIGHT_COUNT, 1).name("活动光源数");
  gui.add(settings, "showGrid").name("显示 Tile 网格 (16x16)");
  gui.add(settings, "autoRotate").name("相机旋转");

  const camera = { target: [0, 2, 0], radius: 45.0, theta: 45.0, phi: 30.0 };
  let isDragging = false, dragButton = 0, lastX = 0, lastY = 0;

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    isDragging = true; dragButton = e.shiftKey ? 2 : e.button;
    lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!isDragging) return;
    const dx = e.clientX - lastX; const dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (dragButton === 0) {
      camera.theta -= dx * 0.3; camera.phi = Math.max(-85, Math.min(85, camera.phi + dy * 0.3));
    } else if (dragButton === 2) {
      const radTheta = (camera.theta * Math.PI) / 180; const pan = camera.radius * 0.0015;
      camera.target[0] -= Math.cos(radTheta) * dx * pan; camera.target[2] -= -Math.sin(radTheta) * dx * pan; camera.target[1] += dy * pan;
    }
  });
  canvas.addEventListener("pointerup", (e) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); camera.radius = Math.max(5, camera.radius * Math.exp(e.deltaY * 0.001)); }, { passive: false });

  // 动态创建深度纹理与 Tile 缓冲区
  let depthTexture: GPUTexture | null = null;
  let tileGridBuffer: GPUBuffer | null = null;
  let bindGroup: GPUBindGroup | null = null;
  let animId: number;

  function frame() {
    if (settings.autoRotate && !isDragging) camera.theta += 0.15;

    // 动态模拟光源运动漂浮
    for (let i = 0; i < settings.activeLights; i++) {
      rawLightData[i * 8 + 0] += lightVelocities[i].x;
      rawLightData[i * 8 + 1] += lightVelocities[i].y;
      rawLightData[i * 8 + 2] += lightVelocities[i].z;
      if (Math.abs(rawLightData[i * 8 + 0]) > 35) lightVelocities[i].x *= -1;
      if (rawLightData[i * 8 + 1] < 0.5 || rawLightData[i * 8 + 1] > 6.0) lightVelocities[i].y *= -1;
      if (Math.abs(rawLightData[i * 8 + 2]) > 35) lightVelocities[i].z *= -1;
    }
    device.queue.writeBuffer(lightStorage, 0, rawLightData);

    const dpr = window.devicePixelRatio || 1;
    const renderWidth = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const renderHeight = Math.max(1, Math.floor(canvas.clientHeight * dpr));

    const tileCountX = Math.ceil(renderWidth / TILE_SIZE);
    const tileCountY = Math.ceil(renderHeight / TILE_SIZE);
    const totalTiles = tileCountX * tileCountY;

    // 屏幕尺寸变化时重建深度图与 Tile 缓冲区
    if (!depthTexture || depthTexture.width !== renderWidth || depthTexture.height !== renderHeight) {
      canvas.width = renderWidth; canvas.height = renderHeight;
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({ size: [renderWidth, renderHeight], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });

      // 每个 Tile: 1 个 count + MAX_LIGHTS_PER_TILE 个 uint32
      const tileBufferSize = totalTiles * (MAX_LIGHTS_PER_TILE + 1) * 4;
      if (tileGridBuffer) tileGridBuffer.destroy();
      tileGridBuffer = device.createBuffer({
        size: tileBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });

      bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: lightStorage } },
          { binding: 2, resource: { buffer: tileGridBuffer } },
          { binding: 3, resource: { buffer: instanceStorage } },
        ]
      });
    }

    // 更新相机与变换矩阵
    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.target[0] + camera.radius * Math.cos(radPhi) * Math.sin(radTheta),
      camera.target[1] + camera.radius * Math.sin(radPhi),
      camera.target[2] + camera.radius * Math.cos(radPhi) * Math.cos(radTheta),
    ];

    const proj = mat4Perspective((45 * Math.PI) / 180, renderWidth / renderHeight, 0.1, 500.0);
    const view = mat4LookAt(eye, camera.target, [0, 1, 0]);
    const viewProj = mat4Multiply(proj, view);
    const invProj = mat4Invert(proj);

    // 上传 Uniform 数据
    const uniformData = new ArrayBuffer(256);
    const f32View = new Float32Array(uniformData);
    const u32View = new Uint32Array(uniformData);

    f32View.set(viewProj, 0);     // 0..15 (64B)
    f32View.set(view, 16);        // 16..31 (64B)
    f32View.set(invProj, 32);     // 32..47 (64B)
    f32View[48] = renderWidth;    // 48 (ScreenW)
    f32View[49] = renderHeight;   // 49 (ScreenH)
    u32View[50] = tileCountX;     // 50 (TilesX)
    u32View[51] = tileCountY;     // 51 (TilesY)

    let modeVal = 0;
    if (settings.mode.includes("暴力")) modeVal = 1;
    if (settings.mode.includes("热力图")) modeVal = 2;
    u32View[52] = modeVal;
    u32View[53] = settings.activeLights;
    u32View[54] = settings.showGrid ? 1 : 0;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // =========================================================================
    // 渲染指令记录
    // =========================================================================
    const encoder = device.createCommandEncoder();

    // 阶段 1: Compute Pass 进行 Tile 视锥体构建与多线程光源剔除
    const cpass = encoder.beginComputePass();
    cpass.setPipeline(computePipeline);
    cpass.setBindGroup(0, bindGroup!);
    // 调度 Workgroup，每个 Tile 分配一个 (16x16) 的 Workgroup
    cpass.dispatchWorkgroups(tileCountX, tileCountY, 1);
    cpass.end();

    // 阶段 2: Render Pass 进行 Tile-Based Forward 着色绘制
    const rpass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear", clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1.0 }, storeOp: "store"
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthLoadOp: "clear", depthClearValue: 1.0, depthStoreOp: "store"
      }
    });

    rpass.setPipeline(renderPipeline);
    rpass.setBindGroup(0, bindGroup!);
    rpass.setVertexBuffer(0, vertexBuffer);
    rpass.draw(36, PILLAR_COUNT, 0, 0);
    rpass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  // 资源清理
  return () => {
    cancelAnimationFrame(animId);
    vertexBuffer.destroy();
    instanceStorage.destroy();
    lightStorage.destroy();
    uniformBuffer.destroy();
    if (tileGridBuffer) tileGridBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}