// src/examples/render_to_texture.ts
import { Mat4 } from "../utils/math";
import type { SimpleGUI } from "../utils/gui";

// 视图矩阵辅助函数
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

// 动态生成一张带标记的高清 PNG 贴图 (也可换成外部图片 URL)
async function createSourceImageBitmap(): Promise<ImageBitmap> {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 512;
  const ctx = c.getContext("2d")!;

  // 绘制炫彩格子背景
  const block = 64;
  for (let y = 0; y < 512; y += block) {
    for (let x = 0; x < 512; x += block) {
      ctx.fillStyle = (x / block + y / block) % 2 === 0 ? "#ff5e7e" : "#00d2fc";
      ctx.fillRect(x, y, block, block);
    }
  }

  // 叠加图形与明确的标识文字，证明这是贴图源
  ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
  ctx.fillRect(30, 180, 452, 150);

  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 8;
  ctx.strokeRect(20, 20, 472, 472);

  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 52px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("INPUT PNG", 256, 230);
  ctx.font = "bold 34px sans-serif";
  ctx.fillText("TEXTURE SOURCE", 256, 285);

  return await createImageBitmap(c);
}

export function runRenderToTexture(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // =========================================================================
  // 1. 基础几何体：带 UV 纹理坐标的立方体 (Pos: 3, UV: 2)
  // =========================================================================
  // prettier-ignore
  const cubeVertexData = new Float32Array([
    // 前面 (Z+)
    -1,-1, 1, 0,1,   1,-1, 1, 1,1,   1, 1, 1, 1,0,  -1, 1, 1, 0,0,
    // 后面 (Z-)
     1,-1,-1, 0,1,  -1,-1,-1, 1,1,  -1, 1,-1, 1,0,   1, 1,-1, 0,0,
    // 顶面 (Y+)
    -1, 1, 1, 0,1,   1, 1, 1, 1,1,   1, 1,-1, 1,0,  -1, 1,-1, 0,0,
    // 底面 (Y-)
    -1,-1,-1, 0,1,   1,-1,-1, 1,1,   1,-1, 1, 1,0,  -1,-1, 1, 0,0,
    // 右面 (X+)
     1,-1, 1, 0,1,   1,-1,-1, 1,1,   1, 1,-1, 1,0,   1, 1, 1, 0,0,
    // 左面 (X-)
    -1,-1,-1, 0,1,  -1,-1, 1, 1,1,  -1, 1, 1, 1,0,  -1, 1,-1, 0,0,
  ]);

  // prettier-ignore
  const cubeIndexData = new Uint16Array([
    0,1,2, 0,2,3,       4,5,6, 4,6,7,       8,9,10, 8,10,11,
    12,13,14, 12,14,15, 16,17,18, 16,18,19, 20,21,22, 20,22,23
  ]);

  const cubeVBuffer = device.createBuffer({
    size: cubeVertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(cubeVBuffer, 0, cubeVertexData);

  const cubeIBuffer = device.createBuffer({
    size: cubeIndexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(cubeIBuffer, 0, cubeIndexData);

  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  // =========================================================================
  // 2. 输入源纹理加载 (使用 copyExternalImageToTexture 将 PNG/Bitmap 写入 GPU)
  // =========================================================================
  const sourceTexture = device.createTexture({
    size: [512, 512],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  createSourceImageBitmap().then((bitmap) => {
    device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture: sourceTexture },
      [bitmap.width, bitmap.height]
    );
  });

  // =========================================================================
  // 3. 核心中间件：离屏 RTT 纹理 (Pass A 绘制目标，Pass B 采样源)
  // =========================================================================
  const RTT_SIZE = 512;
  const rttFormat: GPUTextureFormat = "rgba8unorm";

  const rttColorTexture = device.createTexture({
    size: [RTT_SIZE, RTT_SIZE],
    format: rttFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  const rttDepthTexture = device.createTexture({
    size: [RTT_SIZE, RTT_SIZE],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // 主画布上屏专用的深度缓冲
  let mainDepthTexture = device.createTexture({
    size: [canvas.width || 800, canvas.height || 600],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // =========================================================================
  // 4. 通用着色器与管线布局 (Pass A 和 Pass B 共享相同的贴图着色器)
  // =========================================================================
  const shaderCode = `
    struct Uniforms { mvp: mat4x4f };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var s: sampler;
    @group(0) @binding(2) var t: texture_2d<f32>;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f
    };

    @vertex fn vs_main(@location(0) pos: vec3f, @location(1) uv: vec2f) -> VertexOut {
      var out: VertexOut;
      out.pos = u.mvp * vec4f(pos, 1.0);
      out.uv = uv;
      return out;
    }

    @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
      return textureSample(t, s, uv);
    }
  `;
  const shaderModule = device.createShaderModule({ code: shaderCode });

  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

  // Pass A 管线 (输出目标: rttFormat)
  const pipelineA = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [{ arrayStride: 20, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x2" }] }],
    },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format: rttFormat }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // Pass B 管线 (输出目标: canvas format)
  const pipelineB = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [{ arrayStride: 20, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x2" }] }],
    },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // Pass A: 采样原始 PNG 贴图 (sourceTexture)
  const uniformBufferA = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bindGroupA = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: uniformBufferA } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: sourceTexture.createView() },
    ],
  });

  // Pass B: 采样 Pass A 动态生成的离屏纹理 (rttColorTexture)
  const uniformBufferB = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bindGroupB = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: uniformBufferB } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: rttColorTexture.createView() },
    ],
  });

  // =========================================================================
  // 5. 相机与 GUI 控制
  // =========================================================================
  const config = {
    speedA: 1.5,
    speedB: 0.5,
    cameraDist: 4.5,
  };

  gui.add(config, "speedA", 0, 4, 0.1).name("内层(PassA)转速");
  gui.add(config, "speedB", 0, 2, 0.1).name("外层(PassB)自转");
  gui.add(config, "cameraDist", 2.0, 10.0, 0.1).name("相机距离");

  gui.addTextInfo(
    "<b>流转流程验证：</b><br>" +
    "1. <b>源图片 PNG</b>: 加载带有 'INPUT PNG' 标签的贴图。<br>" +
    "2. <b>Pass A (RTT)</b>: 把源图片贴在一个小立方体上，离屏渲染到 512×512 纹理。<br>" +
    "3. <b>Pass B (上屏)</b>: 主视口大立方体采用 Pass A 的动态纹理做材质贴图，形成画中画！"
  );

  // 鼠标交互控制主相机
  let theta = 30, phi = 20;
  let isDragging = false, lastX = 0, lastY = 0;
  const onPointerDown = (e: PointerEvent) => {
    isDragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    theta -= (e.clientX - lastX) * 0.5;
    phi = Math.max(-85, Math.min(85, phi + (e.clientY - lastY) * 0.5));
    lastX = e.clientX;
    lastY = e.clientY;
  };
  const onPointerUp = (e: PointerEvent) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  };
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);

  // =========================================================================
  // 6. 渲染循环
  // =========================================================================
  let animId: number;
  let timeA = 0;
  let timeB = 0;

  function frame() {
    timeA += 0.02 * config.speedA;
    timeB += 0.01 * config.speedB;

    // 检查画布调整
    if (mainDepthTexture.width !== canvas.width || mainDepthTexture.height !== canvas.height) {
      mainDepthTexture.destroy();
      mainDepthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    // --- Pass A 变换矩阵 (小立方体高速旋转) ---
    const projA = Mat4.perspective((60 * Math.PI) / 180, 1.0, 0.1, 100);
    const eyeA = [Math.sin(timeA) * 3.2, 1.8, Math.cos(timeA) * 3.2];
    const viewA = createLookAtMatrix(eyeA, [0, 0, 0], [0, 1, 0]);
    const mvpA = Mat4.multiply(projA, viewA);
    device.queue.writeBuffer(uniformBufferA, 0, mvpA.buffer as ArrayBuffer);

    // --- Pass B 变换矩阵 (主立方体) ---
    const aspectB = (canvas.width || 800) / (canvas.height || 600);
    const projB = Mat4.perspective((50 * Math.PI) / 180, aspectB, 0.1, 100);
    const radTheta = ((theta + timeB * 20) * Math.PI) / 180;
    const radPhi = (phi * Math.PI) / 180;
    const eyeB = [
      config.cameraDist * Math.cos(radPhi) * Math.sin(radTheta),
      config.cameraDist * Math.sin(radPhi),
      config.cameraDist * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const viewB = createLookAtMatrix(eyeB, [0, 0, 0], [0, 1, 0]);
    const mvpB = Mat4.multiply(projB, viewB);
    device.queue.writeBuffer(uniformBufferB, 0, mvpB.buffer as ArrayBuffer);

    const encoder = device.createCommandEncoder();

    // -----------------------------------------------------------------------
    // 【Pass A】：离屏渲染到 rttColorTexture (使用原始 PNG 纹理)
    // -----------------------------------------------------------------------
    const passA = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: rttColorTexture.createView(),
          clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
          loadOp: "clear",
          storeOp: "store", // 必须保留结果供下一 Pass 采样
        },
      ],
      depthStencilAttachment: {
        view: rttDepthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "discard",
      },
    });
    passA.setPipeline(pipelineA);
    passA.setBindGroup(0, bindGroupA);
    passA.setVertexBuffer(0, cubeVBuffer);
    passA.setIndexBuffer(cubeIBuffer, "uint16");
    passA.drawIndexed(36);
    passA.end();

    // -----------------------------------------------------------------------
    // 【Pass B】：上屏渲染到 Canvas (采样 Pass A 刚刚绘制的动态纹理)
    // -----------------------------------------------------------------------
    const passB = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.06, g: 0.06, b: 0.09, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: mainDepthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "discard",
      },
    });
    passB.setPipeline(pipelineB);
    passB.setBindGroup(0, bindGroupB); // 使用 rttColorTexture 的 BindGroup
    passB.setVertexBuffer(0, cubeVBuffer);
    passB.setIndexBuffer(cubeIBuffer, "uint16");
    passB.drawIndexed(36);
    passB.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  frame();

  // =========================================================================
  // 7. 销毁与解绑
  // =========================================================================
  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    cubeVBuffer.destroy();
    cubeIBuffer.destroy();
    uniformBufferA.destroy();
    uniformBufferB.destroy();
    sourceTexture.destroy();
    rttColorTexture.destroy();
    rttDepthTexture.destroy();
    mainDepthTexture.destroy();
  };
}