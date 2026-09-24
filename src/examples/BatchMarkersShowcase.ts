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
// 2. Batch 批处理合并标注入口
// ==========================================
export function runBatchMarkersShowcase(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // ----------------------------------------------------
  // 1. 离线生成 2x2 标注纹理图集 (Texture Atlas, 512x512)
  // ----------------------------------------------------
  // [0: 红色定位] [1: 黄色警告]
  // [2: 蓝色闪电] [3: 绿色十字]
  const atlasCanvas = document.createElement("canvas");
  atlasCanvas.width = 512;
  atlasCanvas.height = 512;
  const ctx = atlasCanvas.getContext("2d")!;

  function drawAtlas() {
    ctx.clearRect(0, 0, 512, 512);

    // 图标 0: 红色定位图钉 (左上 [0, 0] ~ [256, 256])
    ctx.save();
    ctx.translate(128, 128);
    ctx.shadowColor = "rgba(239, 68, 68, 0.7)";
    ctx.shadowBlur = 16;
    ctx.fillStyle = "#ef4444";
    ctx.beginPath();
    ctx.arc(0, -30, 55, Math.PI * 0.8, Math.PI * 0.2, true);
    ctx.lineTo(0, 90);
    ctx.closePath();
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(0, -30, 22, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // 图标 1: 黄色警示三角 (右上 [256, 0] ~ [512, 256])
    ctx.save();
    ctx.translate(384, 128);
    ctx.shadowColor = "rgba(245, 158, 11, 0.8)";
    ctx.shadowBlur = 18;
    ctx.fillStyle = "#f59e0b";
    ctx.beginPath();
    ctx.moveTo(0, -85);
    ctx.lineTo(85, 75);
    ctx.lineTo(-85, 75);
    ctx.closePath();
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#111827";
    ctx.fillRect(-7, -25, 14, 52);
    ctx.beginPath();
    ctx.arc(0, 48, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // 图标 2: 科技蓝雷电 (左下 [0, 256] ~ [256, 512])
    ctx.save();
    ctx.translate(128, 384);
    ctx.shadowColor = "rgba(14, 165, 233, 0.9)";
    ctx.shadowBlur = 20;
    ctx.fillStyle = "#0ea5e9";
    ctx.beginPath();
    ctx.arc(0, 0, 75, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(8, -50);
    ctx.lineTo(-30, 5);
    ctx.lineTo(-5, 5);
    ctx.lineTo(-12, 50);
    ctx.lineTo(30, -5);
    ctx.lineTo(5, -5);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // 图标 3: 医疗/设施绿十字 (右下 [256, 256] ~ [512, 512])
    ctx.save();
    ctx.translate(384, 384);
    ctx.shadowColor = "rgba(16, 185, 129, 0.8)";
    ctx.shadowBlur = 18;
    ctx.fillStyle = "#10b981";
    ctx.beginPath();
    ctx.arc(0, 0, 75, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    const cw = 20, cl = 90;
    ctx.rect(-cw / 2, -cl / 2, cw, cl);
    ctx.rect(-cl / 2, -cw / 2, cl, cw);
    ctx.fill();
    ctx.restore();
  }
  drawAtlas();

  const atlasTexture = device.createTexture({
    size: [512, 512, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: atlasCanvas }, { texture: atlasTexture }, [512, 512]);

  // ----------------------------------------------------
  // 2. 场景地面与网格
  // ----------------------------------------------------
  const sceneVerts: number[] = [];
  function pushQuad(p1: number[], p2: number[], p3: number[], p4: number[], n: number[], col: number[]) {
    sceneVerts.push(
      ...p1, ...n, ...col, ...p2, ...n, ...col, ...p3, ...n, ...col,
      ...p1, ...n, ...col, ...p3, ...n, ...col, ...p4, ...n, ...col
    );
  }
  // 广阔深色大地面 (60 x 60)
  const gh = 30;
  pushQuad([-gh, 0,  gh], [ gh, 0,  gh], [ gh, 0, -gh], [-gh, 0, -gh], [0, 1, 0], [0.12, 0.15, 0.20]);

  const sceneVBuffer = device.createBuffer({
    size: sceneVerts.length * 4,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(sceneVBuffer, 0, new Float32Array(sceneVerts));

  // ----------------------------------------------------
  // 3. 构建 Instance 批处理数据 (最高支持 3000 个标注)
  // ----------------------------------------------------
  const MAX_INSTANCES = 3000;
  // 内存布局: pos(3) + iconIdx(1) + size(2) + animPhase(1) + pad(1) = 8 Float (32 字节对齐)
  const instanceData = new Float32Array(MAX_INSTANCES * 8);

  for (let i = 0; i < MAX_INSTANCES; i++) {
    const offset = i * 8;
    // 随机散布在地面周围
    const radius = 2.0 + Math.sqrt(Math.random()) * 26.0;
    const angle = Math.random() * Math.PI * 2;
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    const y = 0.5 + Math.random() * 2.5;

    instanceData[offset + 0] = x;
    instanceData[offset + 1] = y;
    instanceData[offset + 2] = z;
    instanceData[offset + 3] = i % 4; // 均匀分配 4 种图标类型 (0, 1, 2, 3)

    // 随机大小 (0.9 ~ 1.3)
    const s = 0.9 + Math.random() * 0.4;
    instanceData[offset + 4] = s;
    instanceData[offset + 5] = s;

    // 随机相位，产生错落起伏的波动动画
    instanceData[offset + 6] = Math.random() * Math.PI * 2;
    instanceData[offset + 7] = 0.0;
  }

  const instanceBuffer = device.createBuffer({
    size: instanceData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(instanceBuffer, 0, instanceData);

  // 单个 Unit Quad 顶点数据 (Pos:2, UV:2)
  const quadVerts = new Float32Array([
    -0.5, -0.5,  0.0, 1.0,
     0.5, -0.5,  1.0, 1.0,
     0.5,  0.5,  1.0, 0.0,
    -0.5, -0.5,  0.0, 1.0,
     0.5,  0.5,  1.0, 0.0,
    -0.5,  0.5,  0.0, 0.0,
  ]);
  const quadVBuffer = device.createBuffer({
    size: quadVerts.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(quadVBuffer, 0, quadVerts);

  // ----------------------------------------------------
  // 4. 着色器实现 (包含批处理与图集 UV 计算)
  // ----------------------------------------------------
  const shaderWGSL = `
    struct SceneUniforms {
      viewProj: mat4x4f,
      camView: mat4x4f,
      time_scale: vec4f, // x: time, y: globalScale, z: bounceAnim(1/0)
    };

    @group(0) @binding(0) var<uniform> u: SceneUniforms;
    @group(0) @binding(1) var s: sampler;
    @group(0) @binding(2) var atlasTex: texture_2d<f32>;

    // 实体地面
    struct MeshOut {
      @builtin(position) pos: vec4f,
      @location(0) col: vec3f,
    };
    @vertex fn vs_mesh(@location(0) pos: vec3f, @location(1) normal: vec3f, @location(2) col: vec3f) -> MeshOut {
      var o: MeshOut;
      o.pos = u.viewProj * vec4f(pos, 1.0);
      o.col = col;
      return o;
    }
    @fragment fn fs_mesh(in: MeshOut) -> @location(0) vec4f {
      return vec4f(in.col, 1.0);
    }

    // 实例化批处理 Billboard (一次 DrawCall 渲染全场)
    struct MarkerOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_batch(
      // Slot 0: Quad 基础几何 (按顶点步进)
      @location(0) quadPos: vec2f,
      @location(1) quadUV: vec2f,
      // Slot 1: Instance 独立属性 (按实例步进)
      @location(2) instPos: vec3f,
      @location(3) instIconIdx: f32,
      @location(4) instSize: vec2f,
      @location(5) instPhase: f32,
    ) -> MarkerOut {
      var o: MarkerOut;

      // 1. 图集 UV 映射计算 (2x2 纹理图集)
      let idx = u32(round(instIconIdx));
      let col = f32(idx % 2u);
      let row = f32(idx / 2u);
      // 每个小图标占图集的 0.5 宽度和高度
      o.uv = quadUV * 0.5 + vec2f(col * 0.5, row * 0.5);

      // 2. 提取相机局部坐标轴 (Spherical Billboarding)
      let camRight = vec3f(u.camView[0].x, u.camView[1].x, u.camView[2].x);
      let camUp    = vec3f(u.camView[0].y, u.camView[1].y, u.camView[2].y);

      // 3. 悬浮正弦波跳动动画
      var bounce = 0.0;
      if (u.time_scale.z > 0.5) {
        bounce = sin(u.time_scale.x * 2.5 + instPhase) * 0.18;
      }
      let anchor = instPos + vec3f(0.0, bounce, 0.0);

      // 4. 顶点世界坐标展开
      let size = instSize * u.time_scale.y;
      let worldPos = anchor + (quadPos.x * size.x) * camRight + (quadPos.y * size.y) * camUp;

      o.pos = u.viewProj * vec4f(worldPos, 1.0);
      return o;
    }

    @fragment
    fn fs_batch(in: MarkerOut) -> @location(0) vec4f {
      let col = textureSample(atlasTex, s, in.uv);
      if (col.a < 0.05) { discard; }
      return col;
    }
  `;

  const shaderModule = device.createShaderModule({ code: shaderWGSL });

  // 实体渲染管线 (仅需 binding 0)
  const meshPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shaderModule,
      entryPoint: "vs_mesh",
      buffers: [{
        arrayStride: 9 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" },
          { shaderLocation: 2, offset: 24, format: "float32x3" },
        ],
      }],
    },
    fragment: { module: shaderModule, entryPoint: "fs_mesh", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // 批量批处理管线 (合并 quad 顶点缓冲与 instance 实例化缓冲)
  const batchPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shaderModule,
      entryPoint: "vs_batch",
      buffers: [
        // Buffer Slot 0: 单个 Quad 顶点数据 (stepMode: "vertex")
        {
          arrayStride: 4 * 4,
          stepMode: "vertex",
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x2" }, // quadPos
            { shaderLocation: 1, offset: 8, format: "float32x2" }, // quadUV
          ],
        },
        // Buffer Slot 1: 实例独立属性数据 (stepMode: "instance")
        {
          arrayStride: 8 * 4,
          stepMode: "instance",
          attributes: [
            { shaderLocation: 2, offset: 0,  format: "float32x3" }, // instPos
            { shaderLocation: 3, offset: 12, format: "float32"   }, // instIconIdx
            { shaderLocation: 4, offset: 16, format: "float32x2" }, // instSize
            { shaderLocation: 5, offset: 24, format: "float32"   }, // instPhase
          ],
        },
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_batch",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        },
      }],
    },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
  });

  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  const uniformBuffer = device.createBuffer({
    size: 144,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const meshBindGroup = device.createBindGroup({
    layout: meshPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const batchBindGroup = device.createBindGroup({
    layout: batchPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: atlasTexture.createView() },
    ],
  });

  // ----------------------------------------------------
  // 5. 控制状态与 GUI
  // ----------------------------------------------------
  const state = {
    markerCount: 1200,
    markerScale: 1.1,
    bounceAnim: 1,
    autoRotate: 1,
  };

  gui.addTextInfo("🚀 <b>GPU 实例化批处理性能展示:</b><br>• 全场数千个图片标注<b>仅需 1 次 Draw Call</b>。<br>• 采用 2×2 纹理图集，彻底避免状态切换开销。");
  gui.add(state, "markerCount", 100, MAX_INSTANCES, 100).name("标注数量 (实例)");
  gui.add(state, "markerScale", 0.5, 2.0, 0.1).name("全局缩放");
  gui.add(state, "bounceAnim", 0, 1, 1).name("悬浮波浪动画");
  gui.add(state, "autoRotate", 0, 1, 1).name("相机自动环绕");

  const camera = { distance: 28.0, phi: 35, theta: 45 };
  let isDragging = false, lastX = 0, lastY = 0;

  const onPointerDown = (e: PointerEvent) => {
    isDragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.35;
    camera.phi = Math.max(-85, Math.min(85, camera.phi + (e.clientY - lastY) * 0.35));
    lastX = e.clientX; lastY = e.clientY;
  };
  const onPointerUp = (e: PointerEvent) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  };
  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    camera.distance = Math.max(6.0, Math.min(55.0, camera.distance + e.deltaY * 0.02));
  };

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  // ----------------------------------------------------
  // 6. 渲染主循环
  // ----------------------------------------------------
  let animId: number;
  let depthTexture: GPUTexture | null = null;
  const uniformData = new Float32Array(36);

  function frame(timestamp: number) {
    if (!depthTexture || depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    if (state.autoRotate) camera.theta += 0.2;

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.distance * Math.cos(radPhi) * Math.sin(radTheta),
      camera.distance * Math.sin(radPhi),
      camera.distance * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const camView = Math3D.lookAt(eye, [0, 1.0, 0], [0, 1, 0]);
    const camProj = Math3D.perspective((45 * Math.PI) / 180, (canvas.width || 800) / (canvas.height || 600), 0.1, 150.0);
    const camViewProj = Math3D.multiply(camProj, camView);

    // 上传矩阵与动画时间参数
    uniformData.set(camViewProj, 0);
    uniformData.set(camView, 16);
    uniformData[32] = timestamp * 0.001;
    uniformData[33] = state.markerScale;
    uniformData[34] = state.bounceAnim;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.06, g: 0.08, b: 0.12, a: 1.0 },
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

    // 1. 绘制地面 (1 次 Draw Call)
    pass.setPipeline(meshPipeline);
    pass.setBindGroup(0, meshBindGroup);
    pass.setVertexBuffer(0, sceneVBuffer);
    pass.draw(6);

    // 2. 批量渲染数千个标注 (仅需 1 次 Draw Call！)
    pass.setPipeline(batchPipeline);
    pass.setBindGroup(0, batchBindGroup);
    pass.setVertexBuffer(0, quadVBuffer);     // Slot 0: 几何基础网格
    pass.setVertexBuffer(1, instanceBuffer); // Slot 1: 实例化属性流
    // draw(vertexCount, instanceCount, firstVertex, firstInstance)
    pass.draw(6, state.markerCount, 0, 0);

    pass.end();
    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  // ----------------------------------------------------
  // 7. 销毁与资源释放
  // ----------------------------------------------------
  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);

    sceneVBuffer.destroy();
    quadVBuffer.destroy();
    instanceBuffer.destroy();
    uniformBuffer.destroy();
    atlasTexture.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}