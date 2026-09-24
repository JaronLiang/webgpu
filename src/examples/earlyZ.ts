// src/examples/earlyZ.ts
export function runEarlyZ(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // 1. 创建顶点数据 (包含 X, Y, Z，模拟 3D 深度)
  // 近处的红色小三角形 (Z = 0.2) 和 远处的蓝色大三角形 (Z = 0.8)
  const vertices = new Float32Array([
    // --- 远处的蓝色大三角形 (Z = 0.8) ---
    -0.8,  0.8, 0.8,
    -0.8, -0.8, 0.8,
     0.8, -0.8, 0.8,
    // --- 近处的红色小三角形 (Z = 0.2) ---
    -0.4,  0.4, 0.2,
    -0.4, -0.4, 0.2,
     0.4, -0.4, 0.2,
  ]);

  const vertexBuffer = device.createBuffer({
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertices);

  // 2. 创建深度纹理 (Depth Texture) 用于深度测试
  const canvasTexture = context.getCurrentTexture();
  const depthTexture = device.createTexture({
    size: [canvasTexture.width, canvasTexture.height],
    format: "depth24plus", // 标准深度格式
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // 3. 着色器代码 (引入顶点颜色标识)
  const shaderCode = `
    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec3f,
    };

    @vertex
    fn vs_main(@location(0) pos: vec3f, @builtin(vertex_index) vIdx: u32) -> VertexOut {
      var out: VertexOut;
      out.pos = vec4f(pos, 1.0);
      
      // 根据顶点索引判断颜色：前3个点(蓝色)，后3个点(红色)
      if (vIdx < 3u) {
        out.color = vec3f(0.2, 0.4, 1.0); // 远处的蓝色
      } else {
        out.color = vec3f(1.0, 0.2, 0.2); // 近处的红色
      }
      return out;
    }

    @fragment
    fn fs_main(@location(0) color: vec3f) -> @location(0) vec4f {
      // 提示：由于开启了深度测试并且近处的红色三角形先渲染 (或者深度更小)，
      // 蓝色三角形被遮挡的部分，GPU硬件会自动触发 Early-Z 丢弃，根本不会执行到这里。
      return vec4f(color, 1.0);
    }
  `;
  const module = device.createShaderModule({ code: shaderCode });

  // 4. 创建渲染管线 (启用 depthStencil)
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [] });
  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module,
      entryPoint: "vs_main",
      buffers: [{
        arrayStride: 3 * 4,
        attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
      }],
    },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
    // 【关键】配置深度测试
    depthStencil: {
      depthWriteEnabled: true, // 允许写入深度
      depthCompare: "less",    // 深度值较小的 (更近的) 像素通过测试
      format: "depth24plus",
    },
  });

  // 5. 渲染流程
  const encoder = device.createCommandEncoder();
  
  // 最佳实践：为了最大化 Early-Z 收益，在实际绘制时，通常将物体按距离从近到远排序。
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
      loadOp: "clear",
      storeOp: "store",
    }],
    // 【关键】挂载深度缓冲区
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1.0, // 初始化深度为最远 (1.0)
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });
  
  pass.setPipeline(pipeline);
  pass.setVertexBuffer(0, vertexBuffer);
  
  // 先画近处红色三角形 (3个顶点，从索引 3 开始)
  pass.draw(3, 1, 3, 0); 
  // 再画远处蓝色三角形 (3个顶点，从索引 0 开始) -> 被遮挡处触发 Early-Z
  pass.draw(3, 1, 0, 0); 

  pass.end();
  device.queue.submit([encoder.finish()]);

  // 6. 释放资源
  return () => {
    vertexBuffer.destroy();
    depthTexture.destroy();
  };
}