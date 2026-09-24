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
// 修复示例 1: 3D 文字标签公告板 (Text Billboard)
// ============================================================================
export function runTextBillboardShowcase(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // 1. 离线 Canvas 绘制 3 种高分辨率文字标签 (带半透明背景与圆角设计)
  function makeTextLabelCanvas(title: string, sub: string, tagColor: string): HTMLCanvasElement {
    const cvs = document.createElement("canvas");
    cvs.width = 512;
    cvs.height = 160;
    const ctx = cvs.getContext("2d")!;

    ctx.fillStyle = "rgba(18, 24, 38, 0.88)";
    ctx.strokeStyle = tagColor;
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.roundRect(10, 10, 492, 140, 24);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = tagColor;
    ctx.beginPath();
    ctx.arc(45, 60, 14, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 34px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillText(title, 75, 72);

    ctx.fillStyle = "#94a3b8";
    ctx.font = "24px sans-serif";
    ctx.fillText(sub, 75, 118);

    return cvs;
  }

  const labelTextures = [
    createTextureFromCanvas(device, makeTextLabelCanvas("核心机组 A-01", "运行中 • 42.5 ℃", "#10b981")),
    createTextureFromCanvas(device, makeTextLabelCanvas("主变电整流站", "负荷 92% • 高压警告", "#ef4444")),
    createTextureFromCanvas(device, makeTextLabelCanvas("备用储能电堆", "状态良好 • 蓄电 98%", "#3b82f6")),
  ];

  // 2. 场景底座几何体 (Pos:3, Normal:3, Color:3)
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

  addBox(0, -0.3, 0, 24, 0.6, 24, [0.15, 0.17, 0.22]);
  addBox(-4.5, 1.0, -1.0, 2.4, 2.0, 2.4, [0.3, 0.35, 0.45]);
  addBox( 4.5, 1.5,  0.5, 2.6, 3.0, 2.6, [0.3, 0.35, 0.45]);
  addBox( 0.0, 0.8,  4.0, 3.0, 1.6, 3.0, [0.3, 0.35, 0.45]);

  const sceneVBuffer = device.createBuffer({ size: sceneVerts.length * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(sceneVBuffer, 0, new Float32Array(sceneVerts));

  // 3. 公告板四边形平面 (Pos:2, UV:2)
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
  const shaderWGSL = `
    struct SceneUniforms {
      viewProj: mat4x4f,
      camView: mat4x4f,
      labelPos_Scale: vec4f,
    };

    @group(0) @binding(0) var<uniform> u: SceneUniforms;
    @group(0) @binding(1) var s: sampler;
    @group(0) @binding(2) var t: texture_2d<f32>;

    // 实体着色 (只用到 binding 0)
    struct MeshVertexOut {
      @builtin(position) pos: vec4f,
      @location(0) normal: vec3f,
      @location(1) color: vec3f,
    };
    @vertex
    fn vs_mesh(@location(0) pos: vec3f, @location(1) normal: vec3f, @location(2) color: vec3f) -> MeshVertexOut {
      var out: MeshVertexOut;
      out.pos = u.viewProj * vec4f(pos, 1.0);
      out.normal = normal;
      out.color = color;
      return out;
    }
    @fragment
    fn fs_mesh(in: MeshVertexOut) -> @location(0) vec4f {
      let L = normalize(vec3f(1.0, 2.0, 1.5));
      let diff = max(dot(normalize(in.normal), L), 0.25);
      return vec4f(in.color * diff, 1.0);
    }

    // Billboard 公告板着色
    struct BBVertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_billboard(@location(0) quadPos: vec2f, @location(1) uv: vec2f) -> BBVertexOut {
      var out: BBVertexOut;
      out.uv = uv;

      let camRight = vec3f(u.camView[0].x, u.camView[1].x, u.camView[2].x);
      let camUp    = vec3f(u.camView[0].y, u.camView[1].y, u.camView[2].y);

      let anchor = u.labelPos_Scale.xyz;
      let scale  = u.labelPos_Scale.w;
      
      let width  = 3.2 * scale;
      let height = 1.0 * scale;

      let worldOffset = (quadPos.x * width) * camRight + ((quadPos.y + 0.5) * height) * camUp;
      let worldPos = anchor + worldOffset;

      out.pos = u.viewProj * vec4f(worldPos, 1.0);
      return out;
    }

    @fragment
    fn fs_billboard(in: BBVertexOut) -> @location(0) vec4f {
      let col = textureSample(t, s, in.uv);
      if (col.a < 0.05) { discard; }
      return col;
    }
  `;

  const module = device.createShaderModule({ code: shaderWGSL });

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

  const billboardPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module, entryPoint: "vs_billboard",
      buffers: [{ arrayStride: 4 * 4, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x2" },
        { shaderLocation: 1, offset: 8, format: "float32x2" },
      ]}],
    },
    fragment: {
      module, entryPoint: "fs_billboard",
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

  const labelPositions = [
    [-4.5, 2.1, -1.0],
    [ 4.5, 3.1,  0.5],
    [ 0.0, 1.7,  4.0],
  ];

  const labelBuffers = labelPositions.map(() => device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  const labelBindGroups = labelBuffers.map((buf, i) =>
    device.createBindGroup({
      layout: billboardPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buf } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: labelTextures[i].createView() },
      ],
    })
  );

  const sceneUniformBuffer = device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // 【核心修复】：meshPipeline 只用到了 binding: 0，不再传入多余的 binding 1 和 2
  const sceneBindGroup = device.createBindGroup({
    layout: meshPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sceneUniformBuffer } },
    ],
  });

  const state = { labelScale: 0.9, autoRotate: 1 };
  gui.add(state, "labelScale", 0.4, 2.0, 0.1).name("标签尺寸");
  gui.add(state, "autoRotate", 0, 1, 1).name("相机自动旋转");

  const camera = { distance: 16.0, phi: 30, theta: 45 };
  let isDragging = false, lastX = 0, lastY = 0;
  const onPointerDown = (e: PointerEvent) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.4;
    camera.phi = Math.max(-85, Math.min(85, camera.phi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY;
  };
  const onPointerUp = (e: PointerEvent) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} };
  const onWheel = (e: WheelEvent) => { e.preventDefault(); camera.distance = Math.max(5.0, Math.min(30.0, camera.distance + e.deltaY * 0.015)); };

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  let animId: number;
  let depthTexture: GPUTexture | null = null;
  const uniformData = new Float32Array(36);

  function frame() {
    if (!depthTexture || depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({ size: [canvas.width || 800, canvas.height || 600], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }

    if (state.autoRotate) camera.theta += 0.3;

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [ camera.distance * Math.cos(radPhi) * Math.sin(radTheta), camera.distance * Math.sin(radPhi), camera.distance * Math.cos(radPhi) * Math.cos(radTheta) ];
    const camView = Math3D.lookAt(eye, [0, 0.5, 0], [0, 1, 0]);
    const camProj = Math3D.perspective((45 * Math.PI) / 180, (canvas.width || 800) / (canvas.height || 600), 0.1, 80.0);
    const camViewProj = Math3D.multiply(camProj, camView);

    uniformData.set(camViewProj, 0);
    uniformData.set(camView, 16);
    device.queue.writeBuffer(sceneUniformBuffer, 0, uniformData);

    for (let i = 0; i < 3; i++) {
      uniformData.set([labelPositions[i][0], labelPositions[i][1], labelPositions[i][2], state.labelScale], 32);
      device.queue.writeBuffer(labelBuffers[i], 0, uniformData);
    }

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.07, g: 0.08, b: 0.11, a: 1.0 },
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });

    // 渲染实体背景
    pass.setPipeline(meshPipeline);
    pass.setBindGroup(0, sceneBindGroup);
    pass.setVertexBuffer(0, sceneVBuffer);
    pass.draw(sceneVerts.length / 9);

    // 渲染透明标签
    pass.setPipeline(billboardPipeline);
    pass.setVertexBuffer(0, quadVBuffer);
    for (let i = 0; i < 3; i++) {
      pass.setBindGroup(0, labelBindGroups[i]);
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
    labelBuffers.forEach(b => b.destroy());
    labelTextures.forEach(t => t.destroy());
    if (depthTexture) depthTexture.destroy();
  };
}
