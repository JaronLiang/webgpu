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

// 辅助：从离线 HTML5 Canvas 上传纹理至显存
function createTextureFromCanvas(device: GPUDevice, canvas: HTMLCanvasElement): GPUTexture {
  const texture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture(
    { source: canvas },
    { texture },
    [canvas.width, canvas.height]
  );
  return texture;
}

// ============================================================================
// 交互式标注与上方锁定信息框功能入口
// ============================================================================
export function runInteractiveMarkerShowcase(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // ----------------------------------------------------
  // 1. 生成 POI 图标与信息卡片贴图
  // ----------------------------------------------------
  // A. 图标贴图绘制
  function makeMarkerIcon(type: "pin" | "warn" | "server"): GPUTexture {
    const cvs = document.createElement("canvas");
    cvs.width = 128;
    cvs.height = 128;
    const ctx = cvs.getContext("2d")!;
    ctx.clearRect(0, 0, 128, 128);

    if (type === "pin") {
      ctx.shadowColor = "rgba(239, 68, 68, 0.7)";
      ctx.shadowBlur = 12;
      ctx.fillStyle = "#ef4444";
      ctx.beginPath();
      ctx.arc(64, 45, 30, Math.PI * 0.8, Math.PI * 0.2, true);
      ctx.lineTo(64, 115);
      ctx.closePath();
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(64, 45, 12, 0, Math.PI * 2);
      ctx.fill();
    } else if (type === "warn") {
      ctx.shadowColor = "rgba(245, 158, 11, 0.8)";
      ctx.shadowBlur = 14;
      ctx.fillStyle = "#f59e0b";
      ctx.beginPath();
      ctx.moveTo(64, 15);
      ctx.lineTo(115, 110);
      ctx.lineTo(13, 110);
      ctx.closePath();
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.fillStyle = "#111827";
      ctx.fillRect(60, 48, 8, 30);
      ctx.beginPath();
      ctx.arc(64, 93, 5, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.shadowColor = "rgba(14, 165, 233, 0.8)";
      ctx.shadowBlur = 14;
      ctx.fillStyle = "#0ea5e9";
      ctx.beginPath();
      ctx.arc(64, 64, 42, 0, Math.PI * 2);
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 6;
      ctx.beginPath();
      ctx.arc(64, 64, 28, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(64, 64, 10, 0, Math.PI * 2);
      ctx.fill();
    }
    return createTextureFromCanvas(device, cvs);
  }

  // B. 弹出信息卡片绘制 (带指示箭头气泡)
  function makeInfoCard(title: string, line1: string, line2: string, statusColor: string): GPUTexture {
    const cvs = document.createElement("canvas");
    cvs.width = 512;
    cvs.height = 256;
    const ctx = cvs.getContext("2d")!;
    ctx.clearRect(0, 0, 512, 256);

    const pad = 12, w = 512 - pad * 2, h = 200;

    // 绘制卡片背景 + 底部指向箭头
    ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
    ctx.shadowBlur = 16;
    ctx.fillStyle = "rgba(15, 23, 42, 0.95)";
    ctx.strokeStyle = statusColor;
    ctx.lineWidth = 4;

    ctx.beginPath();
    ctx.roundRect(pad, pad, w, h, 16);
    // 底部小三角指示箭头
    ctx.moveTo(256 - 16, pad + h);
    ctx.lineTo(256, pad + h + 24);
    ctx.lineTo(256 + 16, pad + h);
    ctx.fill();
    ctx.stroke();

    ctx.shadowBlur = 0;

    // 状态条装饰
    ctx.fillStyle = statusColor;
    ctx.fillRect(pad + 16, pad + 20, 8, 38);

    // 标题
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 32px 'PingFang SC', 'Microsoft YaHei', sans-serif";
    ctx.fillText(title, pad + 36, pad + 50);

    // 分隔线
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pad + 16, pad + 76);
    ctx.lineTo(pad + w - 16, pad + 76);
    ctx.stroke();

    // 详细信息文本
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "24px sans-serif";
    ctx.fillText(line1, pad + 20, pad + 120);

    ctx.fillStyle = "#94a3b8";
    ctx.font = "22px sans-serif";
    ctx.fillText(line2, pad + 20, pad + 162);

    return createTextureFromCanvas(device, cvs);
  }

  // 标注数据集合
  const markersData = [
    {
      id: 0,
      title: "1号变电主站",
      line1: "电压负荷: 10kV / 运行良好",
      line2: "环境温度: 38.5℃ (正常)",
      color: "#ef4444",
      worldPos: [-4.0, 2.8, -2.0],
      iconTex: makeMarkerIcon("pin"),
      cardTex: makeInfoCard("1号变电主站", "电压负荷: 10kV / 运行良好", "环境温度: 38.5℃ (正常)", "#ef4444"),
    },
    {
      id: 1,
      title: "2号高危储压罐",
      line1: "压力警报: 1.85 MPa (超载)",
      line2: "检修级别: 一级预警处理中",
      color: "#f59e0b",
      worldPos: [3.5, 4.0, 1.5],
      iconTex: makeMarkerIcon("warn"),
      cardTex: makeInfoCard("2号高危储压罐", "压力警报: 1.85 MPa (超载)", "检修级别: 一级预警处理中", "#f59e0b"),
    },
    {
      id: 2,
      title: "边缘计算微机柜",
      line1: "网络状态: 双路由正常连通",
      line2: "GPU负载: 42% | 节点在线",
      color: "#0ea5e9",
      worldPos: [0.0, 1.8, 3.5],
      iconTex: makeMarkerIcon("server"),
      cardTex: makeInfoCard("边缘计算微机柜", "网络状态: 双路由正常连通", "GPU负载: 42% | 节点在线", "#0ea5e9"),
    },
  ];

  // ----------------------------------------------------
  // 2. 场景建筑物与地面网格
  // ----------------------------------------------------
  const sceneVerts: number[] = [];
  function pushQuad(p1: number[], p2: number[], p3: number[], p4: number[], n: number[], col: number[]) {
    sceneVerts.push(
      ...p1, ...n, ...col, ...p2, ...n, ...col, ...p3, ...n, ...col,
      ...p1, ...n, ...col, ...p3, ...n, ...col, ...p4, ...n, ...col
    );
  }
  function addBox(cx: number, cy: number, cz: number, sx: number, sy: number, sz: number, col: number[]) {
    const x0 = cx - sx/2, x1 = cx + sx/2, y0 = cy - sy/2, y1 = cy + sy/2, z0 = cz - sz/2, z1 = cz + sz/2;
    pushQuad([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], [0, 1, 0], col);
    pushQuad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], [0, -1, 0], col);
    pushQuad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, 1], col);
    pushQuad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [0, 0, -1], col);
    pushQuad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [-1, 0, 0], col);
    pushQuad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [1, 0, 0], col);
  }

  // 地面
  addBox(0, -0.2, 0, 24, 0.4, 24, [0.18, 0.22, 0.28]);
  // 三栋对应的设备模型实体
  addBox(-4.0, 1.2, -2.0, 2.2, 2.4, 2.2, [0.35, 0.38, 0.45]);
  addBox( 3.5, 1.8,  1.5, 2.8, 3.6, 2.8, [0.40, 0.44, 0.52]);
  addBox( 0.0, 0.7,  3.5, 2.4, 1.4, 2.4, [0.30, 0.34, 0.42]);

  const sceneVBuffer = device.createBuffer({
    size: sceneVerts.length * 4,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(sceneVBuffer, 0, new Float32Array(sceneVerts));

  // Billboard 平面 Quad 顶点数据 (Pos:2, UV:2)
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
  // 3. 着色器实现
  // ----------------------------------------------------
  const shaderWGSL = `
    struct SceneUniforms {
      viewProj: mat4x4f,
      camView: mat4x4f,
      anchorPos_Offset: vec4f, // xyz: 标注物锚点, w: 沿 camUp 方向的偏移距离
      size: vec4f,             // x: width, y: height
    };

    @group(0) @binding(0) var<uniform> u: SceneUniforms;
    @group(0) @binding(1) var s: sampler;
    @group(0) @binding(2) var t: texture_2d<f32>;

    // 实体管线
    struct MeshOut { @builtin(position) pos: vec4f, @location(0) normal: vec3f, @location(1) col: vec3f };
    @vertex fn vs_mesh(@location(0) pos: vec3f, @location(1) normal: vec3f, @location(2) col: vec3f) -> MeshOut {
      var o: MeshOut;
      o.pos = u.viewProj * vec4f(pos, 1.0);
      o.normal = normal;
      o.col = col;
      return o;
    }
    @fragment fn fs_mesh(in: MeshOut) -> @location(0) vec4f {
      let diff = max(dot(normalize(in.normal), normalize(vec3f(1.0, 2.5, 1.2))), 0.25);
      return vec4f(in.col * diff, 1.0);
    }

    // Billboard 公告板管线 (通用支持图标与上方信息框)
    struct BBOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
    @vertex fn vs_billboard(@location(0) quad: vec2f, @location(1) uv: vec2f) -> BBOut {
      var o: BBOut;
      o.uv = uv;

      // 提取相机局部坐标轴
      let camRight = vec3f(u.camView[0].x, u.camView[1].x, u.camView[2].x);
      let camUp    = vec3f(u.camView[0].y, u.camView[1].y, u.camView[2].y);

      // 关键技巧：让物体沿着 camUp 向上偏移，保证在任何相机俯仰角度下都绝不与下方图标遮挡或偏斜
      let center = u.anchorPos_Offset.xyz + camUp * u.anchorPos_Offset.w;
      let worldPos = center + (quad.x * u.size.x) * camRight + (quad.y * u.size.y) * camUp;

      o.pos = u.viewProj * vec4f(worldPos, 1.0);
      return o;
    }
    @fragment fn fs_billboard(in: BBOut) -> @location(0) vec4f {
      let col = textureSample(t, s, in.uv);
      if (col.a < 0.05) { discard; }
      return col;
    }
  `;

  const shaderModule = device.createShaderModule({ code: shaderWGSL });

  // 1. 实体管线 (只绑定 binding: 0)
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

  // 2. 规范定义 Billboard 共享 BindGroupLayout
  const billboardBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
    ],
  });

  const billboardPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [billboardBindGroupLayout] }),
    vertex: {
      module: shaderModule,
      entryPoint: "vs_billboard",
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
      entryPoint: "fs_billboard",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        },
      }],
    },
    // 信息框与图标均不写入深度，但遵循深度测试
    depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
  });

  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  // 统一资源分配
  const sceneUniformBuffer = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const sceneBindGroup = device.createBindGroup({
    layout: meshPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: sceneUniformBuffer } }],
  });

  // 为每个 Marker 和 InfoCard 构建独立的 Uniform Buffer 与 BindGroup
  const markerBuffers = markersData.map(() => device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  const markerBindGroups = markerBuffers.map((buf, i) =>
    device.createBindGroup({
      layout: billboardBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buf } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: markersData[i].iconTex.createView() },
      ],
    })
  );

  const cardBuffers = markersData.map(() => device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  const cardBindGroups = cardBuffers.map((buf, i) =>
    device.createBindGroup({
      layout: billboardBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buf } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: markersData[i].cardTex.createView() },
      ],
    })
  );

  // ----------------------------------------------------
  // 4. 拾取检测与交互状态
  // ----------------------------------------------------
  let selectedMarkerId: number | null = 0; // 默认选中第 1 个图标展示效果

  const state = {
    selectedName: "1号变电主站",
    autoRotate: 1,
  };

  gui.addTextInfo("💡 <b>操作提示:</b><br>• <b>鼠标单击</b> 场景中的任一 POI 图标即可弹出信息卡片。<br>• 旋转镜头观察：信息卡片始终锁定在图标上方。");
  gui.add(state, "autoRotate", 0, 1, 1).name("相机自动旋转");
  gui.addButton("关闭信息卡片", () => {
    selectedMarkerId = null;
    state.selectedName = "无";
    gui.updateDisplay?.();
  });

  const camera = { distance: 16.0, phi: 30, theta: 45 };
  let isDragging = false, startX = 0, startY = 0, lastX = 0, lastY = 0;

  // 3D 世界坐标投射至 2D 屏幕坐标算法
  function projectWorldToScreen(worldPos: number[], viewProj: Float32Array, w: number, h: number): [number, number, boolean] {
    const x = worldPos[0], y = worldPos[1], z = worldPos[2];
    const clipX = viewProj[0] * x + viewProj[4] * y + viewProj[8] * z + viewProj[12];
    const clipY = viewProj[1] * x + viewProj[5] * y + viewProj[9] * z + viewProj[13];
    const clipW = viewProj[3] * x + viewProj[7] * y + viewProj[11] * z + viewProj[15];

    if (clipW <= 0.0) return [0, 0, false]; // 在相机后方，不可见

    const ndcX = clipX / clipW;
    const ndcY = clipY / clipW;

    const screenX = (ndcX * 0.5 + 0.5) * w;
    const screenY = (1.0 - (ndcY * 0.5 + 0.5)) * h;
    return [screenX, screenY, true];
  }

  let latestViewProj = new Float32Array(16);

  // 拾取与悬停交互事件
  const onPointerDown = (e: PointerEvent) => {
    isDragging = true;
    startX = e.clientX;
    startY = e.clientY;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: PointerEvent) => {
    // 鼠标悬停 (Hover) 检测：检测是否悬停在某个图标上，是则将鼠标指针变成小手
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);

    let isHovering = false;
    for (const m of markersData) {
      const [sx, sy, visible] = projectWorldToScreen(m.worldPos, latestViewProj, canvas.width, canvas.height);
      if (visible && Math.hypot(mouseX - sx, mouseY - sy) < 32) {
        isHovering = true;
        break;
      }
    }
    canvas.style.cursor = isHovering ? "pointer" : "default";

    // 视角拖拽旋转
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.4;
    camera.phi = Math.max(-85, Math.min(85, camera.phi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX;
    lastY = e.clientY;
  };

  const onPointerUp = (e: PointerEvent) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}

    // 如果位移小于 4 像素，判定为“单击点击（Click）”操作
    if (Math.hypot(e.clientX - startX, e.clientY - startY) < 4) {
      const rect = canvas.getBoundingClientRect();
      const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
      const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);

      let hitId: number | null = null;
      let minDis = 36; // 拾取检测范围阈值 (36px 半径)

      for (const m of markersData) {
        const [sx, sy, visible] = projectWorldToScreen(m.worldPos, latestViewProj, canvas.width, canvas.height);
        if (visible) {
          const dis = Math.hypot(mouseX - sx, mouseY - sy);
          if (dis < minDis) {
            minDis = dis;
            hitId = m.id;
          }
        }
      }

      if (hitId !== null) {
        selectedMarkerId = hitId;
        state.selectedName = markersData[hitId].title;
      } else {
        // 点击空白处关闭信息框
        selectedMarkerId = null;
        state.selectedName = "无";
      }
      gui.updateDisplay?.();
    }
  };

  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    camera.distance = Math.max(5.0, Math.min(30.0, camera.distance + e.deltaY * 0.015));
  };

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  // ----------------------------------------------------
  // 5. 渲染循环
  // ----------------------------------------------------
  let animId: number;
  let depthTexture: GPUTexture | null = null;
  const uniformData = new Float32Array(40);

  function frame() {
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
    const camView = Math3D.lookAt(eye, [0, 1.2, 0], [0, 1, 0]);
    const camProj = Math3D.perspective((45 * Math.PI) / 180, (canvas.width || 800) / (canvas.height || 600), 0.1, 80.0);
    const camViewProj = Math3D.multiply(camProj, camView);

    // 保存供拾取计算
    latestViewProj.set(camViewProj);

    // 1. 上传基础场景 Uniform
    uniformData.set(camViewProj, 0);
    uniformData.set(camView, 16);
    device.queue.writeBuffer(sceneUniformBuffer, 0, uniformData);

    // 2. 更新所有标注图标 Uniform (尺寸 1.1 x 1.1)
    for (let i = 0; i < 3; i++) {
      const pos = markersData[i].worldPos;
      uniformData.set([pos[0], pos[1], pos[2], 0.0], 32); // offset = 0
      uniformData.set([1.1, 1.1, 0, 0], 36);              // width, height
      device.queue.writeBuffer(markerBuffers[i], 0, uniformData);
    }

    // 3. 更新选中信息卡片的 Uniform (尺寸 3.2 x 1.6，offset 向上提 1.5 单位)
    if (selectedMarkerId !== null) {
      const pos = markersData[selectedMarkerId].worldPos;
      // 关键：w = 1.5 使得中心沿着 camUp 向上偏移 1.5，刚好让尖角对准图标正中心上方
      uniformData.set([pos[0], pos[1], pos[2], 1.5], 32);
      uniformData.set([3.2, 1.6, 0, 0], 36);
      device.queue.writeBuffer(cardBuffers[selectedMarkerId], 0, uniformData);
    }

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.08, g: 0.10, b: 0.14, a: 1.0 },
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

    // 阶段 1：绘制实体建筑
    pass.setPipeline(meshPipeline);
    pass.setBindGroup(0, sceneBindGroup);
    pass.setVertexBuffer(0, sceneVBuffer);
    pass.draw(sceneVerts.length / 9);

    // 阶段 2：绘制三个 POI 图标
    pass.setPipeline(billboardPipeline);
    pass.setVertexBuffer(0, quadVBuffer);
    for (let i = 0; i < 3; i++) {
      pass.setBindGroup(0, markerBindGroups[i]);
      pass.draw(6);
    }

    // 阶段 3：如果当前选中了标注，绘制浮空信息框
    if (selectedMarkerId !== null) {
      pass.setBindGroup(0, cardBindGroups[selectedMarkerId]);
      pass.draw(6);
    }

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
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);
    canvas.style.cursor = "default";

    sceneVBuffer.destroy();
    quadVBuffer.destroy();
    sceneUniformBuffer.destroy();
    markerBuffers.forEach((b) => b.destroy());
    cardBuffers.forEach((b) => b.destroy());
    markersData.forEach((m) => {
      m.iconTex.destroy();
      m.cardTex.destroy();
    });
    if (depthTexture) depthTexture.destroy();
  };
}