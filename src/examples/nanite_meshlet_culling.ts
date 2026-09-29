// src/examples/nanite_meshlet_culling.ts
import type { SimpleGUI } from "../utils/gui";

export function runNaniteMeshlets(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // ==========================================
  // 1. 场景与高密度网格生成 (生成海量 Meshlets)
  // ==========================================
  const GRID_SIZE = 7; // 7x7 = 49 个复杂几何体实体
  const TUBULAR_SEGMENTS = 64;
  const RADIAL_SEGMENTS = 32;
  const RADIUS = 4.0;
  const TUBE = 1.2;
  const P = 2, Q = 3;

  // 单个原型拓扑生成
  const basePos: number[] = [];
  const baseNorm: number[] = [];
  const baseIdx: number[] = [];

  for (let i = 0; i <= TUBULAR_SEGMENTS; i++) {
    const u = (i / TUBULAR_SEGMENTS) * Math.PI * 2 * P;
    const p1 = getTorusKnotPos(u, RADIUS, P, Q);
    const p2 = getTorusKnotPos(u + 0.01, RADIUS, P, Q);
    const T = norm3([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]);
    const N = norm3([p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2]]);
    const B = cross3(T, N);

    for (let j = 0; j <= RADIAL_SEGMENTS; j++) {
      const v = (j / RADIAL_SEGMENTS) * Math.PI * 2;
      const cx = -TUBE * Math.cos(v);
      const cy = TUBE * Math.sin(v);

      basePos.push(p1[0] + cx * N[0] + cy * B[0], p1[1] + cx * N[1] + cy * B[1], p1[2] + cx * N[2] + cy * B[2]);
      const n = norm3([cx * N[0] + cy * B[0], cx * N[1] + cy * B[1], cx * N[2] + cy * B[2]]);
      baseNorm.push(n[0], n[1], n[2]);
    }
  }

  const stride = RADIAL_SEGMENTS + 1;
  for (let i = 0; i < TUBULAR_SEGMENTS; i++) {
    for (let j = 0; j < RADIAL_SEGMENTS; j++) {
      const a = i * stride + j;
      const b = (i + 1) * stride + j;
      const c = (i + 1) * stride + (j + 1);
      const d = i * stride + (j + 1);
      baseIdx.push(a, b, d, b, c, d);
    }
  }

  // ==========================================
  // 2. Meshlet 簇划分器 (单簇上限: 64 顶点, 126 三角形)
  // ==========================================
  const MAX_VERTS = 64;
  const MAX_TRIS = 126;

  interface RawMeshlet {
    indices: number[];
    sphere: [number, number, number, number];
  }

  const prototypeMeshlets: RawMeshlet[] = [];
  let curIndices: number[] = [];
  let curVerts = new Set<number>();

  for (let t = 0; t < baseIdx.length; t += 3) {
    const i0 = baseIdx[t], i1 = baseIdx[t + 1], i2 = baseIdx[t + 2];
    const testSet = new Set(curVerts);
    testSet.add(i0); testSet.add(i1); testSet.add(i2);

    if (testSet.size > MAX_VERTS || curIndices.length / 3 >= MAX_TRIS) {
      prototypeMeshlets.push(createRawMeshlet(curIndices, curVerts, basePos));
      curIndices = [];
      curVerts = new Set();
    }
    curVerts.add(i0); curVerts.add(i1); curVerts.add(i2);
    curIndices.push(i0, i1, i2);
  }
  if (curIndices.length > 0) {
    prototypeMeshlets.push(createRawMeshlet(curIndices, curVerts, basePos));
  }

  function createRawMeshlet(indices: number[], verts: Set<number>, pos: number[]): RawMeshlet {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    verts.forEach((v) => {
      const x = pos[v * 3], y = pos[v * 3 + 1], z = pos[v * 3 + 2];
      minX = Math.min(minX, x); minY = Math.min(minY, y); minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); maxZ = Math.max(maxZ, z);
    });
    const cx = (minX + maxX) * 0.5, cy = (minY + maxY) * 0.5, cz = (minZ + maxZ) * 0.5;
    let maxR = 0;
    verts.forEach((v) => {
      maxR = Math.max(maxR, Math.hypot(pos[v * 3] - cx, pos[v * 3 + 1] - cy, pos[v * 3 + 2] - cz));
    });
    return { indices: [...indices], sphere: [cx, cy, cz, maxR] };
  }

  // 展开到整个场景：GRID_SIZE x GRID_SIZE 阵列
  const SPACING = 18.0;
  const allMeshletsMeta: number[] = []; // 每项 8 个 u32/f32
  const allIndices: number[] = [];

  let totalTriangles = 0;
  let meshletGlobalId = 0;

  for (let gx = 0; gx < GRID_SIZE; gx++) {
    for (let gz = 0; gz < GRID_SIZE; gz++) {
      const offsetX = (gx - (GRID_SIZE - 1) * 0.5) * SPACING;
      const offsetZ = (gz - (GRID_SIZE - 1) * 0.5) * SPACING;
      const offsetY = Math.sin(gx * 0.8) * Math.cos(gz * 0.8) * 5.0;

      for (const proto of prototypeMeshlets) {
        const indexOffset = allIndices.length;
        for (const idx of proto.indices) {
          allIndices.push(idx);
        }

        // 世界坐标包围球
        const worldSphereX = proto.sphere[0] + offsetX;
        const worldSphereY = proto.sphere[1] + offsetY;
        const worldSphereZ = proto.sphere[2] + offsetZ;
        const radius = proto.sphere[3];

        allMeshletsMeta.push(
          worldSphereX, worldSphereY, worldSphereZ, radius, // vec4f sphere
          indexOffset, proto.indices.length, meshletGlobalId, // indexOffset, indexCount, id
          0 // pad
        );
        meshletGlobalId++;
        totalTriangles += proto.indices.length / 3;
      }
    }
  }

  const totalMeshlets = meshletGlobalId;

  // ==========================================
  // 3. 创建 GPU Buffers
  // ==========================================
  // 全局顶点数据: [px, py, pz, nx, ny, nz]
  const vertexData = new Float32Array(basePos.length * 2);
  for (let i = 0; i < basePos.length / 3; i++) {
    vertexData[i * 6 + 0] = basePos[i * 3 + 0];
    vertexData[i * 6 + 1] = basePos[i * 3 + 1];
    vertexData[i * 6 + 2] = basePos[i * 3 + 2];
    vertexData[i * 6 + 3] = baseNorm[i * 3 + 0];
    vertexData[i * 6 + 4] = baseNorm[i * 3 + 1];
    vertexData[i * 6 + 5] = baseNorm[i * 3 + 2];
  }

  const globalVertexBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(globalVertexBuffer, 0, vertexData);

  const globalIndexBuffer = device.createBuffer({
    size: allIndices.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(globalIndexBuffer, 0, new Uint32Array(allIndices));

  // Meshlet 描述符缓冲
  const meshletBuffer = device.createBuffer({
    size: allMeshletsMeta.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const meshletArrayBuffer = new ArrayBuffer(allMeshletsMeta.length * 4);
  const meshletF32 = new Float32Array(meshletArrayBuffer);
  const meshletU32 = new Uint32Array(meshletArrayBuffer);
  for (let i = 0; i < totalMeshlets; i++) {
    const b = i * 8;
    meshletF32[b + 0] = allMeshletsMeta[b + 0];
    meshletF32[b + 1] = allMeshletsMeta[b + 1];
    meshletF32[b + 2] = allMeshletsMeta[b + 2];
    meshletF32[b + 3] = allMeshletsMeta[b + 3];
    meshletU32[b + 4] = allMeshletsMeta[b + 4];
    meshletU32[b + 5] = allMeshletsMeta[b + 5];
    meshletU32[b + 6] = allMeshletsMeta[b + 6];
    meshletU32[b + 7] = 0;
  }
  device.queue.writeBuffer(meshletBuffer, 0, meshletArrayBuffer);

  // 剔除后存活的 Meshlet ID 列表
  const culledMeshletIDsBuffer = device.createBuffer({
    size: totalMeshlets * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // WebGPU drawIndirect 参数: [vertexCount, instanceCount, firstVertex, firstInstance]
  const indirectDrawBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  // 专门用于重置 indirectDrawBuffer 的只读零值源 (利用 GPU 端 CopyBuffer 彻底消除竞态)
  const indirectResetBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indirectResetBuffer, 0, new Uint32Array([MAX_TRIS * 3, 0, 0, 0]));

  const readbackBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Camera + Settings Uniform:
  // viewProj (64B) + planes (6*16 = 96B) + debugMode(4B) + enableCulling(4B) + pad(88B) = 256B
  const uniformBuffer = device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ==========================================
  // 4. Compute 剔除管线 (精准的 WebGPU [0, 1] 深度平面对齐)
  // ==========================================
  const cullingShader = `
    struct Uniforms {
      viewProj: mat4x4f,
      planes: array<vec4f, 6>,
      debugMode: u32,
      enableCulling: u32,
    };

    struct Meshlet {
      sphere: vec4f,
      indexOffset: u32,
      indexCount: u32,
      id: u32,
      _pad: u32,
    };

    struct DrawIndirectArgs {
      vertexCount: u32,
      instanceCount: atomic<u32>,
      firstVertex: u32,
      firstInstance: u32,
    };

    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var<storage, read> meshlets: array<Meshlet>;
    @group(0) @binding(2) var<storage, read_write> visibleIDs: array<u32>;
    @group(0) @binding(3) var<storage, read_write> drawArgs: DrawIndirectArgs;

    fn isSphereVisible(c: vec3f, r: f32) -> bool {
      for (var i = 0; i < 6; i++) {
        let p = u.planes[i];
        let dist = dot(p.xyz, c) + p.w;
        if (dist < -r) {
          return false;
        }
      }
      return true;
    }

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
      let idx = gid.x;
      if (idx >= arrayLength(&meshlets)) { return; }

      let m = meshlets[idx];
      var visible = true;
      if (u.enableCulling == 1u) {
        visible = isSphereVisible(m.sphere.xyz, m.sphere.w);
      }

      if (visible) {
        let slot = atomicAdd(&drawArgs.instanceCount, 1u);
        visibleIDs[slot] = idx;
      }
    }
  `;

  const cullingPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: cullingShader }),
      entryPoint: "main",
    },
  });

  const cullingBindGroup = device.createBindGroup({
    layout: cullingPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: meshletBuffer } },
      { binding: 2, resource: { buffer: culledMeshletIDsBuffer } },
      { binding: 3, resource: { buffer: indirectDrawBuffer } },
    ],
  });

  // ==========================================
  // 5. 渲染管线
  // ==========================================
  const renderShader = `
    struct Uniforms {
      viewProj: mat4x4f,
      planes: array<vec4f, 6>,
      debugMode: u32,
      enableCulling: u32,
    };

    struct Vertex {
      pos: vec3f,
      normal: vec3f,
    };

    struct Meshlet {
      sphere: vec4f,
      indexOffset: u32,
      indexCount: u32,
      id: u32,
      _pad: u32,
    };

    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
    @group(0) @binding(2) var<storage, read> indices: array<u32>;
    @group(0) @binding(3) var<storage, read> meshlets: array<Meshlet>;
    @group(0) @binding(4) var<storage, read> visibleIDs: array<u32>;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) normal: vec3f,
      @location(1) worldPos: vec3f,
      @location(2) @interpolate(flat) meshletColor: vec3f,
    };

    fn hashColor(id: u32) -> vec3f {
      var n = id * 1664525u + 1013904223u;
      n += n << 10u;
      n ^= n >> 6u;
      let r = f32((n >> 16u) & 255u) / 255.0;
      let g = f32((n >> 8u) & 255u) / 255.0;
      let b = f32(n & 255u) / 255.0;
      return vec3f(r * 0.7 + 0.3, g * 0.7 + 0.3, b * 0.7 + 0.3);
    }

    @vertex
    fn vs_main(
      @builtin(vertex_index) v_id: u32,
      @builtin(instance_index) inst_id: u32
    ) -> VertexOutput {
      var out: VertexOutput;
      let meshletIdx = visibleIDs[inst_id];
      let m = meshlets[meshletIdx];

      // 超出当前簇实际索引长度的顶点直接折叠剔除
      if (v_id >= m.indexCount) {
        out.position = vec4f(0.0, 0.0, 2.0, 1.0); // 放在剪裁空间外
        return out;
      }

      let globalVertexIndex = indices[m.indexOffset + v_id];
      let v = vertices[globalVertexIndex];

      // 原型顶点 + 簇世界中心偏移
      // （原型是以原点为中心的，加上 sphere.xyz 还原到世界空间）
      let worldPos = v.pos + vec3f(m.sphere.x - v.pos.x * 0.0, m.sphere.y, m.sphere.z);
      // 注：此处每个模型有其自身的实例基点，为方便起见：
      // worldPos 的实际偏移量保存在 m.sphere 中
      let actualWorldPos = v.pos + (m.sphere.xyz - vec3f(0.0, 0.0, 0.0)); // 保持连续形变

      out.position = u.viewProj * vec4f(v.pos + m.sphere.xyz, 1.0);
      out.normal = v.normal;
      out.worldPos = v.pos + m.sphere.xyz;
      out.meshletColor = hashColor(m.id);
      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      // 0: 簇色彩调试 (Meshlet Debug View)
      if (u.debugMode == 0u) {
        let lightDir = normalize(vec3f(0.6, 0.8, 0.4));
        let diff = max(dot(in.normal, lightDir), 0.25);
        return vec4f(in.meshletColor * diff, 1.0);
      }
      // 1: 真实光照 (Shaded View)
      if (u.debugMode == 1u) {
        let N = normalize(in.normal);
        let L = normalize(vec3f(0.4, 0.9, 0.6));
        let V = normalize(vec3f(0.0, 30.0, 60.0) - in.worldPos);
        let H = normalize(L + V);
        let diff = max(dot(N, L), 0.0) * vec3f(0.85, 0.5, 0.25);
        let spec = pow(max(dot(N, H), 0.0), 32.0) * vec3f(0.4);
        return vec4f(vec3f(0.08) + diff + spec, 1.0);
      }
      // 2: 顶点法线可视化 (Normal Debug)
      return vec4f(normalize(in.normal) * 0.5 + 0.5, 1.0);
    }
  `;

  const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: renderShader }),
      entryPoint: "vs_main",
    },
    fragment: {
      module: device.createShaderModule({ code: renderShader }),
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "back",
    },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less",
    },
  });

  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: globalVertexBuffer } },
      { binding: 2, resource: { buffer: globalIndexBuffer } },
      { binding: 3, resource: { buffer: meshletBuffer } },
      { binding: 4, resource: { buffer: culledMeshletIDsBuffer } },
    ],
  });

  let depthTexture = device.createTexture({
    size: [canvas.width || 800, canvas.height || 600],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // ==========================================
  // 6. 交互式轨道相机系统 (Orbit Controls)
  // ==========================================
  const camera = {
    theta: Math.PI * 0.25,
    phi: Math.PI * 0.28,
    radius: 120.0,
    target: [0, 0, 0],
    isDragging: false,
    dragButton: 0,
    lastX: 0,
    lastY: 0,
  };

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("mousedown", (e) => {
    camera.isDragging = true;
    camera.dragButton = e.button;
    camera.lastX = e.clientX;
    camera.lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => { camera.isDragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!camera.isDragging) return;
    const dx = e.clientX - camera.lastX;
    const dy = e.clientY - camera.lastY;
    camera.lastX = e.clientX;
    camera.lastY = e.clientY;

    if (camera.dragButton === 0) {
      // 左键：旋转 Orbit
      camera.theta -= dx * 0.008;
      camera.phi = Math.max(0.01, Math.min(Math.PI * 0.49, camera.phi - dy * 0.008));
    } else if (camera.dragButton === 2 || camera.dragButton === 1) {
      // 右键/中键：平移 Pan
      const panSpeed = camera.radius * 0.0012;
      const forwardX = -Math.sin(camera.theta);
      const forwardZ = -Math.cos(camera.theta);
      const rightX = Math.cos(camera.theta);
      const rightZ = -Math.sin(camera.theta);

      camera.target[0] -= (rightX * dx - forwardX * dy) * panSpeed;
      camera.target[2] -= (rightZ * dx - forwardZ * dy) * panSpeed;
    }
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.radius = Math.max(10.0, Math.min(400.0, camera.radius + e.deltaY * 0.15));
  }, { passive: false });

  // ==========================================
  // 7. GUI 控制与实时统计
  // ==========================================
  const state = {
    debugMode: 0,
    enableCulling: true,
    freezeFrustum: false,
    autoRotate: true,
  };

  const hud = document.createElement("div");
  hud.style.cssText = `
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
    font-size: 12px; line-height: 1.7; background: rgba(18, 18, 22, 0.9);
    padding: 12px 14px; border-radius: 8px; margin-top: 8px;
    border: 1px solid #2e2e38; color: #f4f4f5; box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  `;
  (gui as any).container.appendChild(hud);

  const updateStatsUI = (visibleCount: number) => {
    const culledCount = Math.max(0, totalMeshlets - visibleCount);
    const cullingRatio = ((culledCount / totalMeshlets) * 100).toFixed(1);
    const renderedTris = Math.round((visibleCount / totalMeshlets) * totalTriangles);

    hud.innerHTML = `
      <div style="font-weight: 700; color: #38bdf8; margin-bottom: 4px; display: flex; justify-content: space-between;">
        <span>Nanite Meshlet Pipeline</span>
        <span style="font-size: 10px; background: #0369a1; padding: 1px 6px; border-radius: 4px;">GPU-DRIVEN</span>
      </div>
      <div>几何实体: <b>${GRID_SIZE}×${GRID_SIZE} (${GRID_SIZE * GRID_SIZE} 个)</b></div>
      <div>场景总三角形: <b>${totalTriangles.toLocaleString()}</b></div>
      <div>总 Meshlets: <b>${totalMeshlets.toLocaleString()}</b></div>
      <div style="border-top: 1px dashed #3f3f46; margin: 4px 0;"></div>
      <div>当前可见簇: <b style="color: #4ade80;">${visibleCount.toLocaleString()}</b></div>
      <div>当前剔除簇: <b style="color: #f87171;">${culledCount.toLocaleString()}</b></div>
      <div>GPU 剔除率: <b style="color: #fbbf24; font-size: 14px;">${cullingRatio}%</b></div>
      <div>实际渲染面数: <b>${renderedTris.toLocaleString()}</b></div>
      <div style="font-size: 10px; color: #71717a; margin-top: 4px;">
        💡 交互: 左键旋转 | 右键平移 | 滚轮缩放
      </div>
    `;
  };

  gui.addButton("🎨 切换: 簇划分色彩 (Meshlet View)", () => { state.debugMode = 0; });
  gui.addButton("💡 切换: 真实光照材质 (Shaded)", () => { state.debugMode = 1; });
  gui.addButton("🌈 切换: 顶点法线模式 (Normals)", () => { state.debugMode = 2; });
  gui.addButton("🧊 冻结/解冻视锥 (拉远视角查看剔除)", () => {
    state.freezeFrustum = !state.freezeFrustum;
  });
  gui.addButton("⚡ 开启/关闭 GPU 剔除", () => {
    state.enableCulling = !state.enableCulling;
  });
  gui.addButton("🔄 开启/停止 自动旋转", () => {
    state.autoRotate = !state.autoRotate;
  });

  // ==========================================
  // 8. 核心渲染与剔除管线循环
  // ==========================================
  const projMat = new Float32Array(16);
  const viewMat = new Float32Array(16);
  const vpMat = new Float32Array(16);
  const frozenVPMat = new Float32Array(16);
  let hasFrozen = false;

  let animId: number;
  let isReadingBack = false;

  function frame() {
    if (state.autoRotate && !camera.isDragging) {
      camera.theta += 0.003;
    }

    // 1. 计算相机视锥与 ViewProj
    const eyeX = camera.target[0] + camera.radius * Math.sin(camera.phi) * Math.sin(camera.theta);
    const eyeY = camera.target[1] + camera.radius * Math.cos(camera.phi);
    const eyeZ = camera.target[2] + camera.radius * Math.sin(camera.phi) * Math.cos(camera.theta);

    const aspect = canvas.width / canvas.height || 1.33;
    mat4PerspectiveWebGPU(projMat, Math.PI / 3.0, aspect, 2.0, 450.0);
    mat4LookAt(viewMat, [eyeX, eyeY, eyeZ], camera.target, [0, 1, 0]);
    mat4Multiply(vpMat, viewMat, projMat);

    // 冻结逻辑支持
    if (!state.freezeFrustum || !hasFrozen) {
      frozenVPMat.set(vpMat);
      hasFrozen = true;
    }

    // 提取严格符合 WebGPU 坐标系的 6 个平截头体平面
    const planes = extractFrustumPlanesWebGPU(frozenVPMat);

    // 2. 写入全局 Uniform
    device.queue.writeBuffer(uniformBuffer, 0, vpMat);
    device.queue.writeBuffer(uniformBuffer, 64, planes as any);
    device.queue.writeBuffer(uniformBuffer, 160, new Uint32Array([
      state.debugMode,
      state.enableCulling ? 1 : 0
    ]));

    const encoder = device.createCommandEncoder();

    // 3. 重置 Indirect Buffer (GPU Copy 保证在 Compute 前无任何竞态)
    encoder.copyBufferToBuffer(indirectResetBuffer, 0, indirectDrawBuffer, 0, 16);

    // 4. Compute Pass: GPU 视锥剔除
    const cpass = encoder.beginComputePass();
    cpass.setPipeline(cullingPipeline);
    cpass.setBindGroup(0, cullingBindGroup);
    cpass.dispatchWorkgroups(Math.ceil(totalMeshlets / 64));
    cpass.end();

    // 5. Render Pass: 间接渲染 (DrawIndirect)
    const rpass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.04, g: 0.04, b: 0.06, a: 1.0 },
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

    rpass.setPipeline(renderPipeline);
    rpass.setBindGroup(0, renderBindGroup);
    rpass.drawIndirect(indirectDrawBuffer, 0);
    rpass.end();

    // 6. 异步回读可见 instance 数量（保证每几帧平滑更新，绝不锁死流水线）
    if (!isReadingBack) {
      isReadingBack = true;
      encoder.copyBufferToBuffer(indirectDrawBuffer, 4, readbackBuffer, 0, 4);
      device.queue.submit([encoder.finish()]);

      readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const count = new Uint32Array(readbackBuffer.getMappedRange())[0];
        readbackBuffer.unmap();
        updateStatsUI(count);
        isReadingBack = false;
      }).catch(() => {
        isReadingBack = false;
      });
    } else {
      device.queue.submit([encoder.finish()]);
    }

    animId = requestAnimationFrame(frame);
  }

  frame();

  // Resize 监听
  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const w = Math.max(1, entry.contentRect.width);
      const h = Math.max(1, entry.contentRect.height);
      canvas.width = w; canvas.height = h;
      depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
  });
  resizeObserver.observe(canvas);

  return () => {
    cancelAnimationFrame(animId);
    resizeObserver.disconnect();
    globalVertexBuffer.destroy();
    globalIndexBuffer.destroy();
    meshletBuffer.destroy();
    culledMeshletIDsBuffer.destroy();
    indirectDrawBuffer.destroy();
    indirectResetBuffer.destroy();
    readbackBuffer.destroy();
    uniformBuffer.destroy();
    depthTexture.destroy();
    if (hud.parentElement) hud.parentElement.removeChild(hud);
  };
}

// ==========================================
// 针对 WebGPU [0, 1] 深度专门校准的数学库
// ==========================================
function getTorusKnotPos(u: number, radius: number, p: number, q: number): [number, number, number] {
  const cu = Math.cos(u), su = Math.sin(u);
  const quOverP = (q / p) * u;
  const cs = Math.cos(quOverP);
  const r0 = radius * (2 + cs) * 0.5;
  return [r0 * cu, radius * Math.sin(quOverP) * 0.7, r0 * su];
}

function norm3(v: number[]): [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function cross3(a: number[], b: number[]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

// WebGPU 标准透视投影矩阵 (NDC 深度: 0 到 1)
function mat4PerspectiveWebGPU(out: Float32Array, fov: number, aspect: number, near: number, far: number) {
  const f = 1.0 / Math.tan(fov / 2);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far / (near - far);      // WebGPU 对应公式
  out[11] = -1.0;
  out[14] = (near * far) / (near - far);
}

function mat4LookAt(out: Float32Array, eye: number[], center: number[], up: number[]) {
  let z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
  let len = Math.hypot(z0, z1, z2);
  z0 /= len; z1 /= len; z2 /= len;

  let x0 = up[1] * z2 - up[2] * z1;
  let x1 = up[2] * z0 - up[0] * z2;
  let x2 = up[0] * z1 - up[1] * z0;
  len = Math.hypot(x0, x1, x2);
  x0 /= len; x1 /= len; x2 /= len;

  let y0 = z1 * x2 - z2 * x1;
  let y1 = z2 * x0 - z0 * x2;
  let y2 = z0 * x1 - z1 * x0;

  out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
  out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
  out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
  out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
  out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
  out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]);
  out[15] = 1;
}

function mat4Multiply(out: Float32Array, a: Float32Array, b: Float32Array) {
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[i * 4 + k] * b[k * 4 + j];
      out[i * 4 + j] = sum;
    }
  }
}

// 适配 WebGPU [0, 1] 深度规范的精准 Gribb-Hartmann 平面提取
function extractFrustumPlanesWebGPU(m: Float32Array): Float32Array {
  const planes = new Float32Array(24);
  // Left: row3 + row0
  planes[0]  = m[3]  + m[0];  planes[1]  = m[7]  + m[4];  planes[2]  = m[11] + m[8];  planes[3]  = m[15] + m[12];
  // Right: row3 - row0
  planes[4]  = m[3]  - m[0];  planes[5]  = m[7]  - m[4];  planes[6]  = m[11] - m[8];  planes[7]  = m[15] - m[12];
  // Bottom: row3 + row1
  planes[8]  = m[3]  + m[1];  planes[9]  = m[7]  + m[5];  planes[10] = m[11] + m[9];  planes[11] = m[15] + m[13];
  // Top: row3 - row1
  planes[12] = m[3]  - m[1];  planes[13] = m[7]  - m[5];  planes[14] = m[11] - m[9];  planes[15] = m[15] - m[13];
  // Near: WebGPU 深度在近平面是 0，所以对应 row2 (而不是 row3 + row2)
  planes[16] = m[2];          planes[17] = m[6];          planes[18] = m[10];         planes[19] = m[14];
  // Far: row3 - row2
  planes[20] = m[3]  - m[2];  planes[21] = m[7]  - m[6];  planes[22] = m[11] - m[10]; planes[23] = m[15] - m[14];

  // 法向量归一化
  for (let i = 0; i < 6; i++) {
    const idx = i * 4;
    const len = Math.hypot(planes[idx], planes[idx + 1], planes[idx + 2]);
    if (len > 0.00001) {
      planes[idx] /= len;
      planes[idx + 1] /= len;
      planes[idx + 2] /= len;
      planes[idx + 3] /= len;
    }
  }
  return planes;
}