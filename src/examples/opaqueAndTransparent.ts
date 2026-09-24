// src/examples/opaqueAndTransparent.ts
export function runOpaqueAndTransparent(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 步骤 1: 编写 WGSL 着色器
  // =========================================================================
  const shaderCode = `
    struct VertexInput {
      @location(0) position: vec3f,
      @location(1) color: vec4f, // 包含 Alpha 通道
    };

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) color: vec4f,
    };

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
      var out: VertexOutput;
      // 简单透视效果映射
      out.position = vec4f(in.position.xy, in.position.z * 0.5 + 0.5, 1.0);
      out.color = in.color;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      return in.color;
    }
  `;

  // =========================================================================
  // 步骤 2: 顶点数据布局 (Pos: vec3f, Color: vec4f)
  // =========================================================================
  // 1. 不透明物体：实心红色矩形 (位于 Z = 0.6 深处)
  const opaqueVertices = new Float32Array([
    //   X,     Y,    Z,     R,   G,   B,   A
    -0.6,  0.6, 0.6,   1.0, 0.2, 0.2, 1.0,
    -0.6, -0.6, 0.6,   1.0, 0.2, 0.2, 1.0,
     0.6, -0.6, 0.6,   1.0, 0.2, 0.2, 1.0,

    -0.6,  0.6, 0.6,   1.0, 0.2, 0.2, 1.0,
     0.6, -0.6, 0.6,   1.0, 0.2, 0.2, 1.0,
     0.6,  0.6, 0.6,   1.0, 0.2, 0.2, 1.0,
  ]);

  // 2. 半透明物体：从远到近排序 (Z = 0.2 蓝色玻璃，Z = -0.2 黄色玻璃)
  const transparentVertices = new Float32Array([
    // [中景] 偏左侧半透明海蓝色四边形 (Z = 0.2, Alpha = 0.55)
    -0.7,  0.4, 0.2,   0.1, 0.6, 1.0, 0.55,
    -0.7, -0.7, 0.2,   0.1, 0.6, 1.0, 0.55,
     0.2, -0.7, 0.2,   0.1, 0.6, 1.0, 0.55,
    -0.7,  0.4, 0.2,   0.1, 0.6, 1.0, 0.55,
     0.2, -0.7, 0.2,   0.1, 0.6, 1.0, 0.55,
     0.2,  0.4, 0.2,   0.1, 0.6, 1.0, 0.55,

    // [前景] 偏右侧半透明金黄色四边形 (Z = -0.2, Alpha = 0.6)
    -0.2,  0.7, -0.2,  1.0, 0.8, 0.1, 0.6,
    -0.2, -0.4, -0.2,  1.0, 0.8, 0.1, 0.6,
     0.7, -0.4, -0.2,  1.0, 0.8, 0.1, 0.6,
    -0.2,  0.7, -0.2,  1.0, 0.8, 0.1, 0.6,
     0.7, -0.4, -0.2,  1.0, 0.8, 0.1, 0.6,
     0.7,  0.7, -0.2,  1.0, 0.8, 0.1, 0.6,
  ]);

  const opaqueVertexBuffer = device.createBuffer({
    size: opaqueVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(opaqueVertexBuffer, 0, opaqueVertices);

  const transparentVertexBuffer = device.createBuffer({
    size: transparentVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(transparentVertexBuffer, 0, transparentVertices);

  // =========================================================================
  // 步骤 3: 创建深度纹理 (用于多通道深度遮挡检测)
  // =========================================================================
  const canvas = context.canvas as HTMLCanvasElement;
  const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // =========================================================================
  // 步骤 4 & 5: 显式创建 PipelineLayout
  // =========================================================================
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [] });
  const shaderModule = device.createShaderModule({ code: shaderCode });

  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 7 * Float32Array.BYTES_PER_ELEMENT, // 3个Pos + 4个RGBA
    stepMode: "vertex",
    attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" },
      { shaderLocation: 1, offset: 3 * Float32Array.BYTES_PER_ELEMENT, format: "float32x4" },
    ],
  };

  // =========================================================================
  // 步骤 6A: 创建 不透明管线 (写深度，无混合)
  // =========================================================================
  const opaquePipeline = device.createRenderPipeline({
    label: "Opaque-Pipeline",
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [vertexBufferLayout],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }], // 默认不开启混合
    },
    depthStencil: {
      depthWriteEnabled: true, // 👈 不透明物体开启深度写入
      depthCompare: "less",
      format: "depth24plus",
    },
    primitive: { topology: "triangle-list" },
  });

  // =========================================================================
  // 步骤 6B: 创建 半透明管线 (关闭深度写入，开启 Alpha 混合)
  // =========================================================================
  const transparentPipeline = device.createRenderPipeline({
    label: "Transparent-Pipeline",
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [vertexBufferLayout],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [
        {
          format,
          // 👈 核心：标准 SrcAlpha / OneMinusSrcAlpha 混合方程
          blend: {
            color: {
              srcFactor: "src-alpha",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
          },
        },
      ],
    },
    depthStencil: {
      depthWriteEnabled: false, // 👈 核心：半透明物体必须关闭深度写入！
      depthCompare: "less",     // 👈 但保留深度测试，使其被不透明红色方块挡住
      format: "depth24plus",
    },
    primitive: { topology: "triangle-list" },
  });

  // =========================================================================
  // 步骤 7: 录制与多阶段绘制指令
  // =========================================================================
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
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

  // 通道 1: 优先绘制不透明物体
  pass.setPipeline(opaquePipeline);
  pass.setVertexBuffer(0, opaqueVertexBuffer);
  pass.draw(6, 1, 0, 0);

  // 通道 2: 切换到半透明管线，从远到近绘制透明物体
  pass.setPipeline(transparentPipeline);
  pass.setVertexBuffer(0, transparentVertexBuffer);
  pass.draw(12, 1, 0, 0); // 绘制两块交叉叠放的半透明玻璃片

  pass.end();
  device.queue.submit([encoder.finish()]);

  // =========================================================================
  // 步骤 8: 显存资源彻底释放
  // =========================================================================
  return () => {
    opaqueVertexBuffer.destroy();
    transparentVertexBuffer.destroy();
    depthTexture.destroy();
  };
}