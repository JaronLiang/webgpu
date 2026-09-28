// src/examples/gpuCulling.ts
import { Mat4 } from "../utils/math";
import type { SimpleGUI } from "../utils/gui";

function createLookAtMatrix(eye: number[], center: number[], up: number[]): Float32Array {
  const z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
  const lenZ = 1 / (Math.hypot(z0, z1, z2) || 1);
  const zx = z0 * lenZ, zy = z1 * lenZ, zz = z2 * lenZ;

  const x0 = up[1] * zz - up[2] * zy, x1 = up[2] * zx - up[0] * zz, x2 = up[0] * zy - up[1] * zx;
  const lenX = 1 / (Math.hypot(x0, x1, x2) || 1);
  const xx = x0 * lenX, xy = x1 * lenX, xz = x2 * lenX;

  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;

  const out = new Float32Array(16);
  out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
  out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
  out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1;
  return out;
}

// 原地提取视锥体 6 个裁剪平面到 out 数组中
function updateFrustumPlanes(m: Float32Array, out: Float32Array): void {
  const r0 = [m[0], m[4], m[8], m[12]];
  const r1 = [m[1], m[5], m[9], m[13]];
  const r2 = [m[2], m[6], m[10], m[14]];
  const r3 = [m[3], m[7], m[11], m[15]];

  const raw: number[][] = [
    [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]], // Left
    [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]], // Right
    [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]], // Bottom
    [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]], // Top
    [r2[0],         r2[1],         r2[2],         r2[3]        ], // Near (WebGPU 0 <= z <= w)
    [r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]], // Far
  ];

  for (let i = 0; i < 6; i++) {
    const p = raw[i];
    const len = Math.hypot(p[0], p[1], p[2]) || 1;
    out[i * 4 + 0] = p[0] / len;
    out[i * 4 + 1] = p[1] / len;
    out[i * 4 + 2] = p[2] / len;
    out[i * 4 + 3] = p[3] / len;
  }
}

export function runGpuCulling(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  const TOTAL_OBJECTS = 24000;

  // 1. 创建基础立方体网格数据 (带法线，24 顶点，36 索引)
  // prettier-ignore
  const cubeVertices = new Float32Array([
    // 前面 (z = 0.5)
    -0.5, -0.5,  0.5,  0, 0, 1,   0.5, -0.5,  0.5,  0, 0, 1,   0.5,  0.5,  0.5,  0, 0, 1,  -0.5,  0.5,  0.5,  0, 0, 1,
    // 后面 (z = -0.5)
    -0.5, -0.5, -0.5,  0, 0,-1,  -0.5,  0.5, -0.5,  0, 0,-1,   0.5,  0.5, -0.5,  0, 0,-1,   0.5, -0.5, -0.5,  0, 0,-1,
    // 顶面 (y = 0.5)
    -0.5,  0.5, -0.5,  0, 1, 0,  -0.5,  0.5,  0.5,  0, 1, 0,   0.5,  0.5,  0.5,  0, 1, 0,   0.5,  0.5, -0.5,  0, 1, 0,
    // 底面 (y = -0.5)
    -0.5, -0.5, -0.5,  0,-1, 0,   0.5, -0.5, -0.5,  0,-1, 0,   0.5, -0.5,  0.5,  0,-1, 0,  -0.5, -0.5,  0.5,  0,-1, 0,
    // 右面 (x = 0.5)
     0.5, -0.5, -0.5,  1, 0, 0,   0.5,  0.5, -0.5,  1, 0, 0,   0.5,  0.5,  0.5,  1, 0, 0,   0.5, -0.5,  0.5,  1, 0, 0,
    // 左面 (x = -0.5)
    -0.5, -0.5, -0.5, -1, 0, 0,  -0.5, -0.5,  0.5, -1, 0, 0,  -0.5,  0.5,  0.5, -1, 0, 0,  -0.5,  0.5, -0.5, -1, 0, 0,
  ]);
  // prettier-ignore
  const cubeIndices = new Uint16Array([
     0,  1,  2,   0,  2,  3,    4,  5,  6,   4,  6,  7,
     8,  9, 10,   8, 10, 11,   12, 13, 14,  12, 14, 15,
    16, 17, 18,  16, 18, 19,   20, 21, 22,  20, 22, 23
  ]);

  const vBuffer = device.createBuffer({
    size: cubeVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vBuffer, 0, cubeVertices);

  const iBuffer = device.createBuffer({
    size: cubeIndices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(iBuffer, 0, cubeIndices);

  // 2. 生成所有物体的元数据 (pos.xyz, radius, col.rgb, scale)
  const objectData = new Float32Array(TOTAL_OBJECTS * 8);
  for (let i = 0; i < TOTAL_OBJECTS; i++) {
    const idx = i * 8;
    const x = (Math.random() - 0.5) * 160.0;
    const y = (Math.random() - 0.5) * 30.0;
    const z = (Math.random() - 0.5) * 160.0;
    const scale = 0.4 + Math.random() * 0.8;
    const radius = scale * 0.866;

    const r = 0.2 + Math.random() * 0.75;
    const g = 0.2 + Math.random() * 0.75;
    const b = 0.2 + Math.random() * 0.75;

    objectData[idx + 0] = x;
    objectData[idx + 1] = y;
    objectData[idx + 2] = z;
    objectData[idx + 3] = radius;
    objectData[idx + 4] = r;
    objectData[idx + 5] = g;
    objectData[idx + 6] = b;
    objectData[idx + 7] = scale;
  }

  const objectsBuffer = device.createBuffer({
    size: objectData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(objectsBuffer, 0, objectData);

  // 3. 存储可见列表与间接绘制参数
  const visibleIndicesBuffer = device.createBuffer({
    size: TOTAL_OBJECTS * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  // [indexCount (36), instanceCount, firstIndex, baseVertex, firstInstance]
  const indirectDrawBuffer = device.createBuffer({
    size: 20,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const countReadbackBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // 4. Uniform 缓冲区
  const uniformBuffer = device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  let depthTexture = device.createTexture({
    size: [canvas.width || 800, canvas.height || 600],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // 5. Compute Shader: 仅保留剔除管线
  const computeShaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
      planes: array<vec4f, 6>,
      totalCount: u32,
      cullEnable: u32,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct ObjectData {
      pos: vec3f,
      radius: f32,
      col: vec3f,
      scale: f32,
    };
    @group(0) @binding(1) var<storage, read> objects: array<ObjectData>;

    struct DrawIndirectArgs {
      indexCount: u32,
      instanceCount: atomic<u32>,
      firstIndex: u32,
      baseVertex: u32,
      firstInstance: u32,
    };
    @group(0) @binding(2) var<storage, read_write> drawArgs: DrawIndirectArgs;
    @group(0) @binding(3) var<storage, read_write> visibleIndices: array<u32>;

    @compute @workgroup_size(64)
    fn cs_cull(@builtin(global_invocation_id) gid: vec3u) {
      let index = gid.x;
      if (index >= u.totalCount) { return; }

      let obj = objects[index];
      var isVisible = true;

      if (u.cullEnable == 1u) {
        for (var i = 0u; i < 6u; i = i + 1u) {
          let plane = u.planes[i];
          let dist = dot(plane.xyz, obj.pos) + plane.w;
          if (dist < -obj.radius) {
            isVisible = false;
            break;
          }
        }
      }

      if (isVisible) {
        let slot = atomicAdd(&drawArgs.instanceCount, 1u);
        visibleIndices[slot] = index;
      }
    }
  `;

  const cModule = device.createShaderModule({ code: computeShaderCode });

  const cCullPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: cModule, entryPoint: "cs_cull" },
  });

  const computeBindGroup = device.createBindGroup({
    layout: cCullPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: objectsBuffer } },
      { binding: 2, resource: { buffer: indirectDrawBuffer } },
      { binding: 3, resource: { buffer: visibleIndicesBuffer } },
    ],
  });

  // 6. Render Pipeline: 间接多实例立方体渲染
  const renderShaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct ObjectData {
      pos: vec3f,
      radius: f32,
      col: vec3f,
      scale: f32,
    };
    @group(0) @binding(1) var<storage, read> objects: array<ObjectData>;
    @group(0) @binding(2) var<storage, read> visibleIndices: array<u32>;

    struct VertexInput {
      @location(0) pos: vec3f,
      @location(1) norm: vec3f,
    };

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) col: vec3f,
      @location(1) norm: vec3f,
    };

    @vertex
    fn vs_main(in: VertexInput, @builtin(instance_index) instanceIdx: u32) -> VertexOut {
      let objIdx = visibleIndices[instanceIdx];
      let obj = objects[objIdx];

      let worldPos = in.pos * obj.scale + obj.pos;
      var out: VertexOut;
      out.pos = u.viewProj * vec4f(worldPos, 1.0);
      out.col = obj.col;
      out.norm = in.norm;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4f {
      let lightDir = normalize(vec3f(0.5, 0.9, 0.4));
      let diff = max(dot(in.norm, lightDir), 0.15);
      return vec4f(in.col * diff, 1.0);
    }
  `;

  const rModule = device.createShaderModule({ code: renderShaderCode });
  const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: rModule,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
          ],
        },
      ],
    },
    fragment: { module: rModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: objectsBuffer } },
      { binding: 2, resource: { buffer: visibleIndicesBuffer } },
    ],
  });

  // 7. GUI 控制与统计
  const params = {
    cullingEnabled: 1,
    freezeFrustum: 0,
  };

  const statsObj = {
    visibleCount: TOTAL_OBJECTS,
  };

  const defaultCamera = { distance: 55.0, theta: 25, phi: 30, panX: 0.0, panY: 0.0, fov: 60 };
  const camera = { ...defaultCamera };

  gui.add(params, "cullingEnabled", 0, 1, 1).name("GPU 剔除开关 (0:关 1:开)");
  gui.add(params, "freezeFrustum", 0, 1, 1).name("冻结视锥 (0:跟踪 1:冻结)");
  gui.add(statsObj, "visibleCount", 0, TOTAL_OBJECTS, 1).name("通过剔除实例数");

  gui.addButton("一键: 冻结视锥并向左平移观察", () => {
    params.freezeFrustum = 1;
    camera.theta += 45;
    gui.updateDisplay();
  });
  gui.addButton("一键: 恢复正常视角与跟踪", () => {
    params.freezeFrustum = 0;
    Object.assign(camera, defaultCamera);
    gui.updateDisplay();
  });

  gui.add(camera, "distance", 5.0, 180.0, 1.0).name("相机距离 (Dist)");
  gui.add(camera, "phi", -80, 80, 1).name("仰角 (Pitch)");
  gui.add(camera, "theta", -180, 180, 1).name("偏航角 (Yaw)");

  gui.addTextInfo(
    "<b>【GPU Culling 观察技巧】：</b><br>" +
    "1. <b>总物体数：</b> 24,000 个方块存储在 GPU Storage Buffer 中。<br>" +
    "2. <b>冻结视锥实验：</b> 点击『冻结视锥』固定当前裁剪锥体，再拖动旋转相机，可侧向观察到方块被切成视锥金字塔形状！<br>" +
    "3. <b>Indirect Draw：</b> 剔除后仅将可见物块打包并提交绘制，CPU 零开销。"
  );

  // 8. 鼠标交互
  let isDragging = false;
  let dragButton = 0;
  let lastX = 0, lastY = 0;

  const onPointerDown = (e: PointerEvent) => {
    isDragging = true;
    dragButton = e.shiftKey ? 2 : e.button;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    if (dragButton === 0) {
      camera.theta -= dx * 0.4;
      camera.phi = Math.max(-80, Math.min(80, camera.phi + dy * 0.4));
    } else if (dragButton === 2) {
      const f = camera.distance * 0.0015;
      camera.panX -= dx * f;
      camera.panY += dy * f;
    }
    gui.updateDisplay();
  };
  const onPointerUp = (e: PointerEvent) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  };
  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    camera.distance = Math.max(5.0, Math.min(180.0, camera.distance + e.deltaY * 0.05));
    gui.updateDisplay();
  };

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  // 9. 渲染与计算主循环
  let animId: number;
  const uniformData = new Float32Array(64);
  const frozenPlanes = new Float32Array(24);
  let isFrustumInitialized = false;

  // 5 个 uint32 重置模板：indexCount=36, instanceCount=0, firstIndex=0, baseVertex=0, firstInstance=0
  const resetArgs = new Uint32Array([36, 0, 0, 0, 0]);

  // 异步回读 GPU 实际绘制数量更新到 GUI
  let isReading = false;
  async function pollVisibleCount() {
    if (isReading) return;
    isReading = true;
    try {
      await countReadbackBuffer.mapAsync(GPUMapMode.READ);
      const arr = new Uint32Array(countReadbackBuffer.getMappedRange());
      statsObj.visibleCount = arr[0];
      countReadbackBuffer.unmap();
      gui.updateDisplay();
    } catch {}
    isReading = false;
  }

  let frameCount = 0;

  function frame() {
    if (depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eyeX = camera.panX + camera.distance * Math.cos(radPhi) * Math.sin(radTheta);
    const eyeY = camera.panY + camera.distance * Math.sin(radPhi);
    const eyeZ = camera.distance * Math.cos(radPhi) * Math.cos(radTheta);

    const viewMatrix = createLookAtMatrix([eyeX, eyeY, eyeZ], [camera.panX, camera.panY, 0], [0, 1, 0]);
    const aspect = (canvas.width || 800) / (canvas.height || 600);
    const projMatrix = Mat4.perspective((camera.fov * Math.PI) / 180, aspect, 0.1, 300);
    const viewProj = Mat4.multiply(projMatrix, viewMatrix);

    // 原地更新视锥体平面 (若开启冻结则保持不变)
    if (params.freezeFrustum === 0 || !isFrustumInitialized) {
      updateFrustumPlanes(viewProj, frozenPlanes);
      isFrustumInitialized = true;
    }

    // 填充 Uniform
    uniformData.set(viewProj, 0);
    uniformData.set(frozenPlanes, 16);
    const u32View = new Uint32Array(uniformData.buffer);
    u32View[40] = TOTAL_OBJECTS;
    u32View[41] = params.cullingEnabled;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // 关键优化：在队列首部快速重置间接参数，完全消除多管线切换冲突
    device.queue.writeBuffer(indirectDrawBuffer, 0, resetArgs);

    const encoder = device.createCommandEncoder();

    // ========= Pass A: Compute Shader 执行 GPU 剔除 =========
    const cPass = encoder.beginComputePass();
    cPass.setPipeline(cCullPipeline);
    cPass.setBindGroup(0, computeBindGroup);
    cPass.dispatchWorkgroups(Math.ceil(TOTAL_OBJECTS / 64));
    cPass.end();

    // 拷贝 instanceCount (offset 4, size 4) 到回读缓冲区用于统计
    encoder.copyBufferToBuffer(indirectDrawBuffer, 4, countReadbackBuffer, 0, 4);

    // ========= Pass B: Render Pass 进行 Indirect 绘制 =========
    const rPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.08, g: 0.09, b: 0.12, a: 1.0 },
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

    rPass.setPipeline(renderPipeline);
    rPass.setBindGroup(0, renderBindGroup);
    rPass.setVertexBuffer(0, vBuffer);
    rPass.setIndexBuffer(iBuffer, "uint16");

    // 核心调用：间接绘制
    rPass.drawIndexedIndirect(indirectDrawBuffer, 0);
    rPass.end();

    device.queue.submit([encoder.finish()]);

    // 每 15 帧异步更新一次 GUI 统计数值
    frameCount++;
    if (frameCount % 15 === 0) {
      pollVisibleCount();
    }

    animId = requestAnimationFrame(frame);
  }

  frame();

  // 10. 资源清理
  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);
    vBuffer.destroy();
    iBuffer.destroy();
    objectsBuffer.destroy();
    visibleIndicesBuffer.destroy();
    indirectDrawBuffer.destroy();
    countReadbackBuffer.destroy();
    uniformBuffer.destroy();
    depthTexture.destroy();
  };
}