// src/examples/triangle.ts
export function runTriangle(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 步骤 1: 编写 WGSL 着色器
  // 说明：不再使用内置索引硬编码，而是通过 @location 接收顶点输入，通过 @group/@binding 接收 Uniform
  // =========================================================================
  const shaderCode = `
    // 外部传入的 Uniform 配置（例如：颜色增强/乘法因子）
    struct UniformConfig {
      colorMultiplier: vec4f,
    };

    @group(0) @binding(0) var<uniform> uConfig: UniformConfig;

    // 顶点输入结构体（对应 VertexBuffer 的布局）
    struct VertexInput {
      @location(0) position: vec2f,
      @location(1) color: vec3f,
    };

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) color: vec4f,
    };

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
      var out: VertexOutput;
      out.position = vec4f(in.position, 0.0, 1.0);
      // 将顶点自身的颜色与外部 Uniform 的调节因子相乘
      out.color = vec4f(in.color, 1.0) * uConfig.colorMultiplier;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      return in.color;
    }
  `;

  // =========================================================================
  // 步骤 2: 创建并填充 Vertex Buffer (交错存储：[X, Y, R, G, B])
  // =========================================================================
  // 每个顶点占 5 个 float (2个位置 + 3个RGB)
  // Float32Array: 3 个顶点共 15 个浮点数
  const vertexData = new Float32Array([
    //   X,     Y,      R,   G,   B
     0.0,   0.5,    1.0, 0.0, 0.0,  // 顶部顶点 (红色)
    -0.5,  -0.5,    0.0, 1.0, 0.0,  // 左下顶点 (绿色)
     0.5,  -0.5,    0.0, 0.0, 1.0,  // 右下顶点 (蓝色)
  ]);

  const vertexBuffer = device.createBuffer({
    label: "Triangle-VertexBuffer",
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // =========================================================================
  // 步骤 3: 创建并填充 Uniform Buffer (用于向 BindGroup 传递外部配置参数)
  // =========================================================================
  // 传递 vec4f(1.0, 1.0, 1.0, 1.0)，占 16 字节
  const uniformData = new Float32Array([1.0, 1.0, 1.0, 1.0]);
  const uniformBuffer = device.createBuffer({
    label: "Triangle-UniformBuffer",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  // =========================================================================
  // 步骤 4: 显式创建 BindGroupLayout (定义资源契约：组 0 槽位 0 是一个只读 Uniform Buffer)
  // =========================================================================
  const bindGroupLayout = device.createBindGroupLayout({
    label: "Triangle-BindGroupLayout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: {
          type: "uniform",
        },
      },
    ],
  });

  // =========================================================================
  // 步骤 5: 显式创建 PipelineLayout (代替 layout: "auto")
  // =========================================================================
  const pipelineLayout = device.createPipelineLayout({
    label: "Triangle-PipelineLayout",
    bindGroupLayouts: [bindGroupLayout],
  });

  // =========================================================================
  // 步骤 6: 创建实际的 BindGroup (将 UniformBuffer 绑定至对应的 BindGroupLayout)
  // =========================================================================
  const bindGroup = device.createBindGroup({
    label: "Triangle-BindGroup",
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: uniformBuffer,
        },
      },
    ],
  });

  // =========================================================================
  // 步骤 7: 创建渲染管线 (RenderPipeline)
  // 显式指定：pipelineLayout、Shader Module、顶点缓冲区布局(VertexBufferLayout)
  // =========================================================================
  const shaderModule = device.createShaderModule({
    label: "Triangle-ShaderModule",
    code: shaderCode,
  });

  // 定义单个顶点缓冲区的内存对齐与属性偏移
  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 5 * Float32Array.BYTES_PER_ELEMENT, // 步长：5 * 4 = 20 字节
    stepMode: "vertex",
    attributes: [
      {
        shaderLocation: 0, // 对应 shader 中的 @location(0) position: vec2f
        offset: 0,         // 从第 0 字节开始读
        format: "float32x2",
      },
      {
        shaderLocation: 1, // 对应 shader 中的 @location(1) color: vec3f
        offset: 2 * Float32Array.BYTES_PER_ELEMENT, // 偏移 8 字节开始读
        format: "float32x3",
      },
    ],
  };

  const pipeline = device.createRenderPipeline({
    label: "Triangle-RenderPipeline",
    layout: pipelineLayout, // 👈 显式传入 PipelineLayout，不再是 "auto"
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [vertexBufferLayout], // 👈 显式配置顶点输入格式
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "none",
    },
  });

  // =========================================================================
  // 步骤 8: 录制指令并提交渲染 (CommandEncoder & RenderPass)
  // =========================================================================
  const encoder = device.createCommandEncoder({ label: "Triangle-Encoder" });
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });

  // 绑定管线
  pass.setPipeline(pipeline);
  // 绑定资源组（Group 0）
  pass.setBindGroup(0, bindGroup);
  // 绑定顶点缓冲（Slot 0）
  pass.setVertexBuffer(0, vertexBuffer);
  // 执行绘制 (3个顶点, 1个实例)
  pass.draw(3, 1, 0, 0);

  pass.end();

  // 提交 GPU 指令队列
  device.queue.submit([encoder.finish()]);

  // =========================================================================
  // 步骤 9: 清理与显存释放回调
  // 当用户切换到其他用例时执行，彻底释放 GPUBuffer 显存
  // =========================================================================
  return () => {
    vertexBuffer.destroy();
    uniformBuffer.destroy();
  };
}