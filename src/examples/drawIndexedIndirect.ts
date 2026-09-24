// src/examples/drawIndexedIndirect.ts
export function runDrawIndexedIndirect(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 步骤 1: 编写 WGSL 着色器
  // =========================================================================
  const shaderCode = `
    struct VertexInput {
      @location(0) position: vec2f,
    };

    @vertex
    fn vs_main(in: VertexInput) -> @builtin(position) vec4f {
      return vec4f(in.position, 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
      return vec4f(0.9, 0.3, 0.6, 1.0); // 粉色矩形
    }
  `;

  // =========================================================================
  // 步骤 2: 创建 顶点 和 索引 Buffer (画一个正方形需要 4 个点, 6 个索引)
  // =========================================================================
  const vertexData = new Float32Array([
    -0.5,  0.5, // 0: 左上
    -0.5, -0.5, // 1: 左下
     0.5, -0.5, // 2: 右下
     0.5,  0.5, // 3: 右上
  ]);
  const vertexBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  const indexData = new Uint16Array([0, 1, 2, 0, 2, 3]);
  const indexBuffer = device.createBuffer({
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indexBuffer, 0, indexData);

  // =========================================================================
  // 步骤 3: 创建 Indexed Indirect Buffer
  // 格式: 5 个 u32 [indexCount, instanceCount, firstIndex, baseVertex, firstInstance]
  // =========================================================================
  const indirectData = new Uint32Array([6, 1, 0, 0, 0]); // 6个索引点，1个实例
  const indirectBuffer = device.createBuffer({
    label: "IndexedIndirectBuffer",
    size: indirectData.byteLength,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST, // 👈 间接标识
  });
  device.queue.writeBuffer(indirectBuffer, 0, indirectData);

  // =========================================================================
  // 步骤 4: 显式创建布局与管线
  // =========================================================================
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [] });
  const shaderModule = device.createShaderModule({ code: shaderCode });
  
  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 2 * Float32Array.BYTES_PER_ELEMENT,
    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
  };

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: shaderModule, entryPoint: "vs_main", buffers: [vertexBufferLayout] },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // =========================================================================
  // 步骤 5: 录制 Indexed Indirect 渲染指令
  // =========================================================================
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
      loadOp: "clear", storeOp: "store",
    }],
  });
  
  pass.setPipeline(pipeline);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.setIndexBuffer(indexBuffer, "uint16"); // 👈 绑定索引缓冲
  // 👈 使用 drawIndexedIndirect 从 GPU 显存直接读取绘制参数
  pass.drawIndexedIndirect(indirectBuffer, 0); 
  
  pass.end();
  device.queue.submit([encoder.finish()]);

  return () => {
    vertexBuffer.destroy();
    indexBuffer.destroy();
    indirectBuffer.destroy();
  };
}