// src/examples/alignment.ts
export function runMemoryAlignment(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 步骤 1: 编写 WGSL 着色器 (展示真实的内存 Padding 规则)
  // =========================================================================
  const shaderCode = `
    struct AlignedUniform {
      // vec3f 对齐要求 16 字节，占用 12 字节（字节 0~11）
      color: vec3f, 
      
      // 下一个是 vec2f，对齐要求是 8 字节。
      // 当前在第 12 字节，不是 8 的倍数，因此 WGSL 会自动填充 4 字节的 Padding (字节 12~15)。
      // 所以 offset 实际从第 16 字节开始（字节 16~23）
      offset: vec2f,

      // f32 对齐要求 4 字节。当前在第 24 字节，刚好是 4 的倍数，无需 Padding。
      // scale 占用字节 24~27
      scale: f32,
    }; // 结构体总对齐取最大值(16)，所以总大小会被补齐到 32 字节

    @group(0) @binding(0) var<uniform> uData: AlignedUniform;

    @vertex
    fn vs_main(@builtin(vertex_index) vIdx: u32) -> @builtin(position) vec4f {
      var pos = array<vec2f, 3>(
        vec2f(0.0, 0.5),
        vec2f(-0.5, -0.5),
        vec2f(0.5, -0.5)
      );
      // 应用 Uniform 中的偏移和缩放
      let p = pos[vIdx] * uData.scale + uData.offset;
      return vec4f(p, 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
      return vec4f(uData.color, 1.0);
    }
  `;

  // =========================================================================
  // 步骤 2: 在 JS 中严格匹配 WGSL 的内存排布来构造 Uniform Buffer
  // =========================================================================
  // 结构体总大小 32 字节，对应 8 个 Float32
  const uniformData = new Float32Array(8);
  
  // 字节 0~11 (Index 0, 1, 2): color (红色)
  uniformData[0] = 1.0; 
  uniformData[1] = 0.2; 
  uniformData[2] = 0.3; 

  // 字节 12~15 (Index 3): 👈 这里就是 WGSL 自动产生的 PADDING 区域！必须跳过！
  // uniformData[3] 保持默认的 0 即可

  // 字节 16~23 (Index 4, 5): offset (向右上角偏移)
  uniformData[4] = 0.2; 
  uniformData[5] = 0.2; 

  // 字节 24~27 (Index 6): scale (缩放 50%)
  uniformData[6] = 0.5; 

  // 字节 28~31 (Index 7): 结尾的 PADDING

  const uniformBuffer = device.createBuffer({
    label: "Align-UniformBuffer",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  // =========================================================================
  // 步骤 3: 显式创建布局与管线
  // =========================================================================
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const shaderModule = device.createShaderModule({ code: shaderCode });

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: shaderModule, entryPoint: "vs_main" },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // =========================================================================
  // 步骤 4: 渲染与清理
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
  pass.setBindGroup(0, bindGroup);
  pass.draw(3, 1, 0, 0); // 应该能看到一个偏向右上的粉红色缩放三角形
  pass.end();
  device.queue.submit([encoder.finish()]);

  return () => { uniformBuffer.destroy(); };
}