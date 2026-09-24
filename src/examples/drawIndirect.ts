// src/examples/drawIndirect.ts
export function runDrawIndirect(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 步骤 1: 编写 WGSL 着色器
  // =========================================================================
  const shaderCode = `
    @vertex
    fn vs_main(@builtin(vertex_index) vIdx: u32) -> @builtin(position) vec4f {
      var pos = array<vec2f, 3>(
        vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5)
      );
      return vec4f(pos[vIdx], 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
      return vec4f(0.2, 0.8, 0.4, 1.0); // 绿色
    }
  `;

  // =========================================================================
  // 步骤 2: 创建 Indirect Buffer
  // 格式: 4 个 u32 [vertexCount, instanceCount, firstVertex, firstInstance]
  // =========================================================================
  const indirectData = new Uint32Array([3, 1, 0, 0]); // 画 3 个点，1 个实例
  const indirectBuffer = device.createBuffer({
    label: "IndirectBuffer",
    size: indirectData.byteLength,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST, // 👈 必须包含 INDIRECT 标签
  });
  device.queue.writeBuffer(indirectBuffer, 0, indirectData);

  // =========================================================================
  // 步骤 3: 显式创建空布局与管线
  // =========================================================================
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [] });
  const shaderModule = device.createShaderModule({ code: shaderCode });

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: shaderModule, entryPoint: "vs_main" },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // =========================================================================
  // 步骤 4: 录制 Indirect 渲染指令
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
  // 👈 使用 drawIndirect 读取 GPU buffer 中的指令，不再由 CPU 传参
  pass.drawIndirect(indirectBuffer, 0); 
  pass.end();
  
  device.queue.submit([encoder.finish()]);

  return () => { indirectBuffer.destroy(); };
}