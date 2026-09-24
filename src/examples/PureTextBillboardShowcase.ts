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
// 2. 纯文字标签功能示例 (Pure Text Billboard)
// ==========================================
export function runPureTextBillboardShowcase(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // ----------------------------------------------------
  // 1. 生成纯文字透明贴图核心工具 (支持任意中文、加粗、文字外描边)
  // ----------------------------------------------------
  const textCanvas = document.createElement("canvas");
  textCanvas.width = 512;
  textCanvas.height = 128; // 宽高比 4:1
  const ctx = textCanvas.getContext("2d")!;

  function drawPureTextToTexture(
    texture: GPUTexture,
    name: string,
    hp: number,
    color: string = "#ffffff"
  ) {
    ctx.clearRect(0, 0, textCanvas.width, textCanvas.height);

    // 字体配置：兼容 macOS PingFang、Windows 微软雅黑及系统无衬线
    ctx.font = "bold 44px 'PingFang SC', 'Microsoft YaHei', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    const text = `${name}  HP ${Math.round(hp)}%`;

    // 核心技巧：黑色外描边 (Stroke) 确保在任何 3D 亮暗背景下都清晰可见
    ctx.lineJoin = "round";
    ctx.miterLimit = 2;
    ctx.strokeStyle = "rgba(0, 0, 0, 0.95)";
    ctx.lineWidth = 8;
    ctx.strokeText(text, 256, 64);

    // 内部文字填充 (Fill)
    ctx.fillStyle = color;
    ctx.fillText(text, 256, 64);

    // 零拷贝直接上传至 WebGPU 显存
    device.queue.copyExternalImageToTexture(
      { source: textCanvas },
      { texture },
      [textCanvas.width, textCanvas.height]
    );
  }

  function createTextGPUTexture(): GPUTexture {
    return device.createTexture({
      size: [512, 128, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  // ----------------------------------------------------
  // 2. 创建 3D 方块怪物模型 (还原截图中的三台方块)
  // ----------------------------------------------------
  const sceneVerts: number[] = [];
  function pushQuad(p1: number[], p2: number[], p3: number[], p4: number[], n: number[], col: number[]) {
    sceneVerts.push(
      ...p1, ...n, ...col, ...p2, ...n, ...col, ...p3, ...n, ...col,
      ...p1, ...n, ...col, ...p3, ...n, ...col, ...p4, ...n, ...col
    );
  }
  function addBox(cx: number, cy: number, cz: number, s: number, col: number[]) {
    const r = s / 2;
    const x0 = cx - r, x1 = cx + r, y0 = cy - r, y1 = cy + r, z0 = cz - r, z1 = cz + r;
    pushQuad([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], [0, 1, 0], col); // 上
    pushQuad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], [0, -1, 0], col); // 下
    pushQuad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, 1], col);  // 前
    pushQuad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [0, 0, -1], col); // 后
    pushQuad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [-1, 0, 0], col); // 左
    pushQuad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [1, 0, 0], col);  // 右
  }

  // 1. 兽人 (Orc) - 淡紫色方块 (左)
  addBox(-2.6, 0.0, 0.0, 1.2, [0.72, 0.65, 0.82]);
  // 2. 哥布林 (Goblin) - 淡粉色方块 (中)
  addBox( 0.0, 0.0, 0.0, 1.2, [0.85, 0.65, 0.75]);
  // 3. 巨魔 (Troll) - 淡红肉色方块 (右)
  addBox( 2.6, 0.0, 0.0, 1.2, [0.92, 0.68, 0.68]);

  const sceneVBuffer = device.createBuffer({
    size: sceneVerts.length * 4,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(sceneVBuffer, 0, new Float32Array(sceneVerts));

  // 3. 纯文字四边形公告板 Quad (Pos:2, UV:2)
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
  // 4. 着色器实现 (包含实体着色与公告板着色)
  // ----------------------------------------------------
  const shaderWGSL = `
    struct SceneData {
      viewProj: mat4x4f,
      camView: mat4x4f,
      labelAnchor_Scale: vec4f, // xyz: 头顶文字世界坐标, w: 缩放比率
    };

    @group(0) @binding(0) var<uniform> u: SceneData;
    @group(0) @binding(1) var s: sampler;
    @group(0) @binding(2) var t: texture_2d<f32>;

    // --- 方块渲染 ---
    struct MeshOut {
      @builtin(position) pos: vec4f,
      @location(0) normal: vec3f,
      @location(1) color: vec3f,
    };
    @vertex
    fn vs_mesh(@location(0) pos: vec3f, @location(1) normal: vec3f, @location(2) color: vec3f) -> MeshOut {
      var out: MeshOut;
      out.pos = u.viewProj * vec4f(pos, 1.0);
      out.normal = normal;
      out.color = color;
      return out;
    }
    @fragment
    fn fs_mesh(in: MeshOut) -> @location(0) vec4f {
      let L = normalize(vec3f(1.0, 2.0, 1.5));
      let diff = max(dot(normalize(in.normal), L), 0.3);
      return vec4f(in.color * diff, 1.0);
    }

    // --- 纯文字公告板渲染 ---
    struct BBOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_text(@location(0) quadPos: vec2f, @location(1) uv: vec2f) -> BBOut {
      var out: BBOut;
      out.uv = uv;

      // 从视图矩阵中提取相机的 Right 与 Up 矢量，实现始终正对视线
      let camRight = vec3f(u.camView[0].x, u.camView[1].x, u.camView[2].x);
      let camUp    = vec3f(u.camView[0].y, u.camView[1].y, u.camView[2].y);

      let anchor = u.labelAnchor_Scale.xyz;
      let scale  = u.labelAnchor_Scale.w;

      // 贴图尺寸比例 512 : 128 = 4 : 1
      let width  = 2.4 * scale;
      let height = 0.6 * scale;

      // 在相机视平面进行顶点展开
      let worldPos = anchor + (quadPos.x * width) * camRight + (quadPos.y * height) * camUp;
      out.pos = u.viewProj * vec4f(worldPos, 1.0);
      return out;
    }

    @fragment
    fn fs_text(in: BBOut) -> @location(0) vec4f {
      let col = textureSample(t, s, in.uv);
      // 纯文字完全透明区域剔除，提升 GPU 渲染性能
      if (col.a < 0.02) { discard; }
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

  // 纯文字透明公告板渲染管线
  const textPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shaderModule,
      entryPoint: "vs_text",
      buffers: [{
        arrayStride: 4 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x2" },
          { shaderLocation: 1, offset: 8, format: "float32x2" },
        ],
      }],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_text",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        },
      }],
    },
    // 透明物体不写入深度，避免遮挡其它半透明像素
    depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
  });

  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  // ----------------------------------------------------
  // 5. 实体怪物状态与文字贴图初始化
  // ----------------------------------------------------
  const entities = [
    { name: "兽人战士 Orc", hp: 100, pos: [-2.6, 1.25, 0.0], texture: createTextGPUTexture() },
    { name: "哥布林 Goblin", hp: 100, pos: [ 0.0, 1.25, 0.0], texture: createTextGPUTexture() },
    { name: "巨魔萨满 Troll", hp: 100, pos: [ 2.6, 1.25, 0.0], texture: createTextGPUTexture() },
  ];

  // 初始绘制文字
  entities.forEach((e) => drawPureTextToTexture(e.texture, e.name, e.hp));

  // 为每个文字标签创建独立的 Uniform 缓存与 BindGroup
  const textBuffers = entities.map(() =>
    device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  );
  const textBindGroups = textBuffers.map((buf, i) =>
    device.createBindGroup({
      layout: textPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buf } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: entities[i].texture.createView() },
      ],
    })
  );

  const sceneUniformBuffer = device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const sceneBindGroup = device.createBindGroup({
    layout: meshPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sceneUniformBuffer } },
    ],
  });

  // ----------------------------------------------------
  // 6. GUI 控制与动态数值修改
  // ----------------------------------------------------
  const state = {
    orcHP: 100,
    goblinHP: 100,
    trollHP: 100,
    textScale: 1.0,
    autoRotate: 0,
  };

  gui.add(state, "orcHP", 0, 100, 1).name("兽人 HP").onChange((val: number) => {
    drawPureTextToTexture(entities[0].texture, entities[0].name, val);
  });
  gui.add(state, "goblinHP", 0, 100, 1).name("哥布林 HP").onChange((val: number) => {
    drawPureTextToTexture(entities[1].texture, entities[1].name, val);
  });
  gui.add(state, "trollHP", 0, 100, 1).name("巨魔 HP").onChange((val: number) => {
    drawPureTextToTexture(entities[2].texture, entities[2].name, val);
  });

  gui.add(state, "textScale", 0.5, 2.0, 0.1).name("文字大小");
  gui.add(state, "autoRotate", 0, 1, 1).name("相机自动旋转");

  gui.addButton("随机扣血测试", () => {
    state.orcHP = Math.floor(Math.random() * 100);
    state.goblinHP = Math.floor(Math.random() * 100);
    state.trollHP = Math.floor(Math.random() * 100);
    drawPureTextToTexture(entities[0].texture, entities[0].name, state.orcHP);
    drawPureTextToTexture(entities[1].texture, entities[1].name, state.goblinHP);
    drawPureTextToTexture(entities[2].texture, entities[2].name, state.trollHP);
    gui.updateDisplay?.();
  });

  // ----------------------------------------------------
  // 7. 相机控制器
  // ----------------------------------------------------
  const camera = { distance: 7.5, phi: 12, theta: 0 };
  let isDragging = false, lastX = 0, lastY = 0;

  const onPointerDown = (e: PointerEvent) => {
    isDragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.4;
    camera.phi = Math.max(-85, Math.min(85, camera.phi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY;
  };
  const onPointerUp = (e: PointerEvent) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  };
  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    camera.distance = Math.max(3.0, Math.min(20.0, camera.distance + e.deltaY * 0.01));
  };

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  // ----------------------------------------------------
  // 8. 渲染主循环
  // ----------------------------------------------------
  let animId: number;
  let depthTexture: GPUTexture | null = null;
  const uniformData = new Float32Array(36);

  function frame() {
    if (!depthTexture || depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    if (state.autoRotate) camera.theta += 0.3;

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.distance * Math.cos(radPhi) * Math.sin(radTheta),
      camera.distance * Math.sin(radPhi),
      camera.distance * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const camView = Math3D.lookAt(eye, [0, 0.4, 0], [0, 1, 0]);
    const camProj = Math3D.perspective((45 * Math.PI) / 180, (canvas.width || 800) / (canvas.height || 600), 0.1, 50.0);
    const camViewProj = Math3D.multiply(camProj, camView);

    // 更新场景 Uniform
    uniformData.set(camViewProj, 0);
    uniformData.set(camView, 16);
    device.queue.writeBuffer(sceneUniformBuffer, 0, uniformData);

    // 分别更新 3 个文字标签的 Uniform
    for (let i = 0; i < 3; i++) {
      uniformData.set([entities[i].pos[0], entities[i].pos[1], entities[i].pos[2], state.textScale], 32);
      device.queue.writeBuffer(textBuffers[i], 0, uniformData);
    }

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        // 深暗蓝背景色，与游戏/截图保持一致
        clearValue: { r: 0.08, g: 0.13, b: 0.20, a: 1.0 },
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

    // 步骤 1：先绘制不透明方块
    pass.setPipeline(meshPipeline);
    pass.setBindGroup(0, sceneBindGroup);
    pass.setVertexBuffer(0, sceneVBuffer);
    pass.draw(sceneVerts.length / 9);

    // 步骤 2：后绘制纯透明文字标签 (Alpha Blending 必须后画)
    pass.setPipeline(textPipeline);
    pass.setVertexBuffer(0, quadVBuffer);
    for (let i = 0; i < 3; i++) {
      pass.setBindGroup(0, textBindGroups[i]);
      pass.draw(6);
    }

    pass.end();
    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  // ----------------------------------------------------
  // 9. 销毁与资源释放
  // ----------------------------------------------------
  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);

    sceneVBuffer.destroy();
    quadVBuffer.destroy();
    sceneUniformBuffer.destroy();
    textBuffers.forEach((b) => b.destroy());
    entities.forEach((e) => e.texture.destroy());
    if (depthTexture) depthTexture.destroy();
  };
}