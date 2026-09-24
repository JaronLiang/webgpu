// src/examples/reversedZ.ts
export function runReversedZ(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement
) {
  // 1. 生成极远处相距仅 0.002 的两层重叠薄片 (专门测试远端 Z-Fighting)
  // prettier-ignore
  const vertexData = new Float32Array([
    // 底层薄片 (青色, Z = -200.0)
    -20, -20, -200.000,  0.0, 0.8, 0.9,
     20, -20, -200.000,  0.0, 0.8, 0.9,
     20,  20, -200.000,  0.0, 0.8, 0.9,
    -20, -20, -200.000,  0.0, 0.8, 0.9,
     20,  20, -200.000,  0.0, 0.8, 0.9,
    -20,  20, -200.000,  0.0, 0.8, 0.9,

    // 顶层薄片 (洋红, Z = -199.998，距离底层只有 0.002)
    -15, -15, -199.998,  0.9, 0.1, 0.5,
     15, -15, -199.998,  0.9, 0.1, 0.5,
     15,  15, -199.998,  0.9, 0.1, 0.5,
    -15, -15, -199.998,  0.9, 0.1, 0.5,
     15,  15, -199.998,  0.9, 0.1, 0.5,
    -15,  15, -199.998,  0.9, 0.1, 0.5,
  ]);

  const vBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vBuffer, 0, vertexData);

  // 2. 构造 Standard 与 Reversed-Z 投影矩阵
  const fovy = (60 * Math.PI) / 180;
  const halfW = Math.floor(canvas.width / 2);
  const aspect = halfW / canvas.height;
  const near = 0.1;
  const far = 300.0;
  const f = 1.0 / Math.tan(fovy / 2);

  // Standard-Z: Near 映射到 0，Far 映射到 1
  const standardProj = new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far / (near - far), -1,
    0, 0, (near * far) / (near - far), 0,
  ]);

  // Reversed-Z: Near 映射到 1，Far 映射到 0
  const reversedProj = new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, near / (far - near), -1,
    0, 0, (near * far) / (far - near), 0,
  ]);

  const uBufferStandard = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const uBufferReversed = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uBufferStandard, 0, standardProj);
  device.queue.writeBuffer(uBufferReversed, 0, reversedProj);

  // 3. 【核心修复】：深度纹理物理大小必须与画布全尺寸完全相同，不可除以2！
  const depthTexStandard = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const depthTexReversed = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const shaderCode = `
    @group(0) @binding(0) var<uniform> proj: mat4x4f;
    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec3f,
    };
    @vertex
    fn vs_main(@location(0) pos: vec3f, @location(1) col: vec3f) -> VertexOut {
      var out: VertexOut;
      out.pos = proj * vec4f(pos, 1.0);
      out.color = col;
      return out;
    }
    @fragment
    fn fs_main(@location(0) col: vec3f) -> @location(0) vec4f {
      return vec4f(col, 1.0);
    }
  `;
  const module = device.createShaderModule({ code: shaderCode });

  // Standard-Z 管线: depthCompare 为 less
  const pipelineStandard = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module, entryPoint: "vs_main",
      buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x3" }] }],
    },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // Reversed-Z 管线: depthCompare 为 greater
  const pipelineReversed = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module, entryPoint: "vs_main",
      buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x3" }] }],
    },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "greater", format: "depth24plus" },
  });

  const bgStandard = device.createBindGroup({
    layout: pipelineStandard.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uBufferStandard } }],
  });
  const bgReversed = device.createBindGroup({
    layout: pipelineReversed.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uBufferReversed } }],
  });

  const encoder = device.createCommandEncoder();
  const currentView = context.getCurrentTexture().createView();

  // --- 左半屏: Standard-Z ---
  const passStandard = encoder.beginRenderPass({
    colorAttachments: [{
      view: currentView,
      clearValue: { r: 0.1, g: 0.1, b: 0.12, a: 1.0 },
      loadOp: "clear", storeOp: "store",
    }],
    depthStencilAttachment: {
      view: depthTexStandard.createView(),
      depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
    },
  });
  // 通过 setViewport 控制只在左半边绘制
  passStandard.setViewport(0, 0, halfW, canvas.height, 0, 1);
  passStandard.setPipeline(pipelineStandard);
  passStandard.setBindGroup(0, bgStandard);
  passStandard.setVertexBuffer(0, vBuffer);
  passStandard.draw(12);
  passStandard.end();

  // --- 右半屏: Reversed-Z ---
  const passReversed = encoder.beginRenderPass({
    colorAttachments: [{
      view: currentView,
      loadOp: "load", storeOp: "store", // 保留左半屏内容
    }],
    depthStencilAttachment: {
      view: depthTexReversed.createView(),
      depthClearValue: 0.0, depthLoadOp: "clear", depthStoreOp: "store", // 初始清空为 0.0
    },
  });
  // 通过 setViewport 控制只在右半边绘制
  passReversed.setViewport(halfW, 0, canvas.width - halfW, canvas.height, 0, 1);
  passReversed.setPipeline(pipelineReversed);
  passReversed.setBindGroup(0, bgReversed);
  passReversed.setVertexBuffer(0, vBuffer);
  passReversed.draw(12);
  passReversed.end();

  device.queue.submit([encoder.finish()]);

  return () => {
    vBuffer.destroy();
    uBufferStandard.destroy();
    uBufferReversed.destroy();
    depthTexStandard.destroy();
    depthTexReversed.destroy();
  };
}