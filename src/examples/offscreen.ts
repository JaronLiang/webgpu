// src/examples/offscreen.ts
import { Mat4 } from "../utils/math";

export function runOffscreen(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement
) {
  // 1. 立方体网格
  // prettier-ignore
  const cubeVerts = new Float32Array([
    // 前面 (UV: 0,0 到 1,1)
    -1,-1, 1, 0,1,   1,-1, 1, 1,1,   1, 1, 1, 1,0,  -1, 1, 1, 0,0,
    -1,-1,-1, 1,1,  -1, 1,-1, 1,0,   1, 1,-1, 0,0,   1,-1,-1, 0,1,
    -1, 1,-1, 0,1,  -1, 1, 1, 0,0,   1, 1, 1, 1,0,   1, 1,-1, 1,1,
    -1,-1,-1, 0,0,   1,-1,-1, 1,0,   1,-1, 1, 1,1,  -1,-1, 1, 0,1,
     1,-1,-1, 1,1,   1, 1,-1, 1,0,   1, 1, 1, 0,0,   1,-1, 1, 0,1,
    -1,-1,-1, 0,1,  -1,-1, 1, 0,0,  -1, 1, 1, 1,0,  -1, 1,-1, 1,1,
  ]);
  // prettier-ignore
  const cubeIndices = new Uint16Array([
    0,1,2, 0,2,3,       4,5,6, 4,6,7,       8,9,10, 8,10,11,
    12,13,14, 12,14,15, 16,17,18, 16,18,19, 20,21,22, 20,22,23
  ]);

  const vBuffer = device.createBuffer({ size: cubeVerts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vBuffer, 0, cubeVerts);
  const iBuffer = device.createBuffer({ size: cubeIndices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(iBuffer, 0, cubeIndices);

  // 2. 【核心】创建离屏纹理 (Offscreen Render Target)
  const offscreenSize = 512;
  const offscreenTexture = device.createTexture({
    size: [offscreenSize, offscreenSize],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  const offscreenDepth = device.createTexture({
    size: [offscreenSize, offscreenSize], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const mainDepth = device.createTexture({
    size: [canvas.width, canvas.height], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  // 3. 着色器: 离屏画彩虹色立方体，主屏贴上离屏纹理
  const shaderCode = `
    struct Uniforms { mvp: mat4x4f };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
      @location(1) modelPos: vec3f,
    };

    @vertex
    fn vs_main(@location(0) pos: vec3f, @location(1) uv: vec2f) -> VertexOut {
      var out: VertexOut;
      out.pos = u.mvp * vec4f(pos, 1.0);
      out.uv = uv;
      out.modelPos = pos;
      return out;
    }

    // 离屏渲染着色：根据坐标生成程序化渐变彩虹色
    @fragment
    fn fs_offscreen(in: VertexOut) -> @location(0) vec4f {
      return vec4f(in.modelPos * 0.5 + 0.5, 1.0);
    }

    // 主屏幕着色：采样离屏纹理贴图
    @group(0) @binding(1) var s: sampler;
    @group(0) @binding(2) var t: texture_2d<f32>;
    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4f {
      return textureSample(t, s, in.uv);
    }
  `;
  const module = device.createShaderModule({ code: shaderCode });

  const uBufferOffscreen = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const uBufferMain = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const pipelineOffscreen = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module, entryPoint: "vs_main",
      buffers: [{ arrayStride: 20, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x2" }] }],
    },
    fragment: { module, entryPoint: "fs_offscreen", targets: [{ format: "rgba8unorm" }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const pipelineMain = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module, entryPoint: "vs_main",
      buffers: [{ arrayStride: 20, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x2" }] }],
    },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const bgOffscreen = device.createBindGroup({
    layout: pipelineOffscreen.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uBufferOffscreen } }],
  });

  const bgMain = device.createBindGroup({
    layout: pipelineMain.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uBufferMain } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: offscreenTexture.createView() },
    ],
  });

  let animId: number;
  let rotOff = 0;
  let rotMain = 0;

  const proj = Mat4.perspective((60 * Math.PI) / 180, canvas.width / canvas.height, 0.1, 100);
  const offscreenProj = Mat4.perspective((60 * Math.PI) / 180, 1.0, 0.1, 100);

  function frame() {
    rotOff += 1.5;
    rotMain += 0.6;

    // 更新离屏立方体矩阵
    const mvpOff = Mat4.multiply(offscreenProj, Mat4.createModelView(rotOff, rotOff * 0.7, -3.2));
    device.queue.writeBuffer(uBufferOffscreen, 0, mvpOff.buffer as ArrayBuffer);

    // 更新主屏立方体矩阵
    const mvpMain = Mat4.multiply(proj, Mat4.createModelView(rotMain * 0.4, rotMain, -4.0));
    device.queue.writeBuffer(uBufferMain, 0, mvpMain.buffer as ArrayBuffer);

    const encoder = device.createCommandEncoder();

    // --- Pass 1: 离屏通道 (渲染到 offscreenTexture) ---
    const pass1 = encoder.beginRenderPass({
      colorAttachments: [{
        view: offscreenTexture.createView(),
        clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1.0 },
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: offscreenDepth.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });
    pass1.setPipeline(pipelineOffscreen);
    pass1.setBindGroup(0, bgOffscreen);
    pass1.setVertexBuffer(0, vBuffer);
    pass1.setIndexBuffer(iBuffer, "uint16");
    pass1.drawIndexed(36);
    pass1.end();

    // --- Pass 2: 主通道 (把离屏纹理贴在画布的主立方体上) ---
    const pass2 = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.12, g: 0.12, b: 0.18, a: 1.0 },
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: mainDepth.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });
    pass2.setPipeline(pipelineMain);
    pass2.setBindGroup(0, bgMain);
    pass2.setVertexBuffer(0, vBuffer);
    pass2.setIndexBuffer(iBuffer, "uint16");
    pass2.drawIndexed(36);
    pass2.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  frame();

  return () => {
    cancelAnimationFrame(animId);
    vBuffer.destroy();
    iBuffer.destroy();
    uBufferOffscreen.destroy();
    uBufferMain.destroy();
    offscreenTexture.destroy();
    offscreenDepth.destroy();
    mainDepth.destroy();
  };
}