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


// 辅助：从离线 HTML5 Canvas 生成 WebGPU 纹理
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
// 修复示例 2: 3D 图片/POI 标注与锚定针脚 (Image Marker with Anchor Stem)
// ============================================================================
export function runImageMarkerShowcase(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // 1. 矢量离线绘制 3 种高清 POI 图标
  function createIconTexture(type: "pin" | "warning" | "info"): GPUTexture {
    const cvs = document.createElement("canvas");
    cvs.width = 256;
    cvs.height = 256;
    const ctx = cvs.getContext("2d")!;
    ctx.clearRect(0, 0, 256, 256);

    if (type === "pin") {
      ctx.shadowColor = "rgba(239, 68, 68, 0.6)";
      ctx.shadowBlur = 18;
      ctx.fillStyle = "#ef4444";
      ctx.beginPath();
      ctx.arc(128, 90, 60, Math.PI * 0.8, Math.PI * 0.2, true);
      ctx.lineTo(128, 220);
      ctx.closePath();
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(128, 90, 22, 0, Math.PI * 2);
      ctx.fill();
    } else if (type === "warning") {
      ctx.shadowColor = "rgba(245, 158, 11, 0.7)";
      ctx.shadowBlur = 20;
      ctx.fillStyle = "#f59e0b";
      ctx.beginPath();
      ctx.moveTo(128, 30);
      ctx.lineTo(230, 210);
      ctx.lineTo(26, 210);
      ctx.closePath();
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.fillStyle = "#111827";
      ctx.fillRect(122, 90, 12, 55);
      ctx.beginPath();
      ctx.arc(128, 175, 7, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.shadowColor = "rgba(14, 165, 233, 0.8)";
      ctx.shadowBlur = 22;
      ctx.fillStyle = "#0ea5e9";
      ctx.beginPath();
      ctx.arc(128, 128, 80, 0, Math.PI * 2);
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 10;
      ctx.beginPath();
      ctx.arc(128, 128, 55, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(128, 128, 20, 0, Math.PI * 2);
      ctx.fill();
    }
    return createTextureFromCanvas(device, cvs);
  }

  const iconTextures = [
    createIconTexture("pin"),
    createIconTexture("warning"),
    createIconTexture("info"),
  ];

  // 2. 地面与建筑模型
  const sceneVerts: number[] = [];
  function pushQuad(p1: number[], p2: number[], p3: number[], p4: number[], n: number[], col: number[]) {
    sceneVerts.push(...p1, ...n, ...col, ...p2, ...n, ...col, ...p3, ...n, ...col, ...p1, ...n, ...col, ...p3, ...n, ...col, ...p4, ...n, ...col);
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

  addBox(0, -0.2, 0, 30, 0.4, 30, [0.18, 0.20, 0.25]);
  addBox(-5, 1.5, -4, 3, 3, 3, [0.35, 0.40, 0.50]);
  addBox( 4, 2.5,  2, 4, 5, 3, [0.40, 0.45, 0.55]);
  addBox( 2, 1.0, -5, 2, 2, 2, [0.32, 0.36, 0.44]);

  const sceneVBuffer = device.createBuffer({ size: sceneVerts.length * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(sceneVBuffer, 0, new Float32Array(sceneVerts));

  // 3. 标注四边形 (Pos:2, UV:2)
  const quadVerts = new Float32Array([
    -0.5, -0.5,  0.0, 1.0,
     0.5, -0.5,  1.0, 1.0,
     0.5,  0.5,  1.0, 0.0,
    -0.5, -0.5,  0.0, 1.0,
     0.5,  0.5,  1.0, 0.0,
    -0.5,  0.5,  0.0, 0.0,
  ]);
  const quadVBuffer = device.createBuffer({ size: quadVerts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(quadVBuffer, 0, quadVerts);

  // 4. 着色器实现
  const markerShaderWGSL = `
    struct Uniforms {
      viewProj: mat4x4f,
      camView: mat4x4f,
      markerPos_Bounce: vec4f,
      params: vec4f,
    };

    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var s: sampler;
    @group(0) @binding(2) var t: texture_2d<f32>;

    // 实体渲染
    struct MeshOut { @builtin(position) pos: vec4f, @location(0) normal: vec3f, @location(1) col: vec3f };
    @vertex fn vs_mesh(@location(0) pos: vec3f, @location(1) normal: vec3f, @location(2) col: vec3f) -> MeshOut {
      var o: MeshOut;
      o.pos = u.viewProj * vec4f(pos, 1.0);
      o.normal = normal;
      o.col = col;
      return o;
    }
    @fragment fn fs_mesh(in: MeshOut) -> @location(0) vec4f {
      let diff = max(dot(normalize(in.normal), normalize(vec3f(1.0, 2.0, 1.0))), 0.2);
      return vec4f(in.col * diff, 1.0);
    }

    // 锚定针线
    @vertex fn vs_stem(@builtin(vertex_index) vIdx: u32) -> @builtin(position) vec4f {
      let base = u.markerPos_Bounce.xyz;
      let top  = base + vec3f(0.0, 1.2 + u.markerPos_Bounce.w, 0.0);
      var p = base;
      if (vIdx == 1u) { p = top; }
      return u.viewProj * vec4f(p, 1.0);
    }
    @fragment fn fs_stem() -> @location(0) vec4f {
      return vec4f(0.38, 0.85, 0.95, 0.75);
    }

    // 图标 Billboard
    struct MarkerOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
    @vertex fn vs_marker(@location(0) quad: vec2f, @location(1) uv: vec2f) -> MarkerOut {
      var o: MarkerOut;
      o.uv = uv;

      let camRight = vec3f(u.camView[0].x, u.camView[1].x, u.camView[2].x);
      let camUp    = vec3f(u.camView[0].y, u.camView[1].y, u.camView[2].y);
      let anchor   = u.markerPos_Bounce.xyz + vec3f(0.0, 1.6 + u.markerPos_Bounce.w, 0.0);
      let size     = 1.4 * u.params.x;

      let worldPos = anchor + (quad.x * size) * camRight + (quad.y * size) * camUp;
      o.pos = u.viewProj * vec4f(worldPos, 1.0);
      return o;
    }
    @fragment fn fs_marker(in: MarkerOut) -> @location(0) vec4f {
      let col = textureSample(t, s, in.uv);
      if (col.a < 0.05) { discard; }
      return col;
    }
  `;

  const module = device.createShaderModule({ code: markerShaderWGSL });

  // 实体管线 (auto 推导出仅含 binding:0)
  const meshPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module, entryPoint: "vs_mesh",
      buffers: [{ arrayStride: 9 * 4, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x3" },
      ]}],
    },
    fragment: { module, entryPoint: "fs_mesh", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // 【核心修复】：为 stem 和 marker 创建显式的共享布局，避免 layout: "auto" 不兼容问题
  const markerBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
    ],
  });

  const markerPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [markerBindGroupLayout],
  });

  const stemPipeline = device.createRenderPipeline({
    layout: markerPipelineLayout,
    vertex: { module, entryPoint: "vs_stem" },
    fragment: {
      module, entryPoint: "fs_stem",
      targets: [{
        format,
        blend: { color: { srcFactor: "src-alpha", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } }
      }],
    },
    primitive: { topology: "line-list" },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less", format: "depth24plus" },
  });

  const markerPipeline = device.createRenderPipeline({
    layout: markerPipelineLayout,
    vertex: {
      module, entryPoint: "vs_marker",
      buffers: [{ arrayStride: 4 * 4, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x2" },
        { shaderLocation: 1, offset: 8, format: "float32x2" },
      ]}],
    },
    fragment: {
      module, entryPoint: "fs_marker",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" }
        },
      }],
    },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
  });

  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  const markerAnchors = [
    [-5.0, 3.0, -4.0],
    [ 4.0, 5.0,  2.0],
    [ 2.0, 2.0, -5.0],
  ];

  const markerBuffers = markerAnchors.map(() => device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  const markerBindGroups = markerBuffers.map((buf, i) =>
    device.createBindGroup({
      layout: markerBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buf } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: iconTextures[i].createView() },
      ],
    })
  );

  const sceneUniformBuffer = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // 【核心修复】：meshPipeline 只绑定包含 binding 0 的 Uniform
  const sceneBindGroup = device.createBindGroup({
    layout: meshPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sceneUniformBuffer } },
    ],
  });

  const state = { markerScale: 1.0, bounceAnim: 1, autoRotate: 1 };
  gui.add(state, "markerScale", 0.5, 2.5, 0.1).name("图标大小");
  gui.add(state, "bounceAnim", 0, 1, 1).name("上下悬浮动画");
  gui.add(state, "autoRotate", 0, 1, 1).name("相机自动环绕");

  const camera = { distance: 18.0, phi: 35, theta: 30 };
  let isDragging = false, lastX = 0, lastY = 0;
  const onPointerDown = (e: PointerEvent) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.4;
    camera.phi = Math.max(-85, Math.min(85, camera.phi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY;
  };
  const onPointerUp = (e: PointerEvent) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} };
  const onWheel = (e: WheelEvent) => { e.preventDefault(); camera.distance = Math.max(6.0, Math.min(40.0, camera.distance + e.deltaY * 0.015)); };

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  let animId: number;
  let depthTexture: GPUTexture | null = null;
  const uniformData = new Float32Array(40);

  function frame(timestamp: number) {
    if (!depthTexture || depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({ size: [canvas.width || 800, canvas.height || 600], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }

    if (state.autoRotate) camera.theta += 0.25;

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [ camera.distance * Math.cos(radPhi) * Math.sin(radTheta), camera.distance * Math.sin(radPhi), camera.distance * Math.cos(radPhi) * Math.cos(radTheta) ];
    const camView = Math3D.lookAt(eye, [0, 1.5, 0], [0, 1, 0]);
    const camProj = Math3D.perspective((45 * Math.PI) / 180, (canvas.width || 800) / (canvas.height || 600), 0.1, 100.0);
    const camViewProj = Math3D.multiply(camProj, camView);

    uniformData.set(camViewProj, 0);
    uniformData.set(camView, 16);
    device.queue.writeBuffer(sceneUniformBuffer, 0, uniformData);

    const bounce = state.bounceAnim ? Math.sin(timestamp * 0.003) * 0.25 : 0.0;

    for (let i = 0; i < 3; i++) {
      uniformData.set([markerAnchors[i][0], markerAnchors[i][1], markerAnchors[i][2], bounce], 32);
      uniformData.set([state.markerScale, 0, 0, 0], 36);
      device.queue.writeBuffer(markerBuffers[i], 0, uniformData);
    }

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.08, g: 0.10, b: 0.14, a: 1.0 },
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });

    // 1. 实体建筑
    pass.setPipeline(meshPipeline);
    pass.setBindGroup(0, sceneBindGroup);
    pass.setVertexBuffer(0, sceneVBuffer);
    pass.draw(sceneVerts.length / 9);

    // 2. 锚定连接线 (复用 markerBindGroups[i]，因 shared layout 保证安全)
    pass.setPipeline(stemPipeline);
    for (let i = 0; i < 3; i++) {
      pass.setBindGroup(0, markerBindGroups[i]);
      pass.draw(2);
    }

    // 3. 悬浮 POI 图标
    pass.setPipeline(markerPipeline);
    pass.setVertexBuffer(0, quadVBuffer);
    for (let i = 0; i < 3; i++) {
      pass.setBindGroup(0, markerBindGroups[i]);
      pass.draw(6);
    }

    pass.end();
    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }
  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);
    sceneVBuffer.destroy();
    quadVBuffer.destroy();
    sceneUniformBuffer.destroy();
    markerBuffers.forEach(b => b.destroy());
    iconTextures.forEach(t => t.destroy());
    if (depthTexture) depthTexture.destroy();
  };
}