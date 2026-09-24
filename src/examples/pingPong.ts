
// src/examples/pingPong.ts

export function runPingPong(
  device: GPUDevice,
  context: GPUCanvasContext,
  presentationFormat: GPUTextureFormat
) {
  const width = 256;
  const height = 256;
  const simFormat: GPUTextureFormat = "rgba8unorm";

  // -------------------------------------------------------------
  // 1. 基础顶点缓冲 (全屏四边形: 两个三角形)
  // -------------------------------------------------------------
  const quadVertices = new Float32Array([
    // x,    y,    u,   v
    -1.0,  1.0,  0.0, 0.0,
    -1.0, -1.0,  0.0, 1.0,
     1.0, -1.0,  1.0, 1.0,
    -1.0,  1.0,  0.0, 0.0,
     1.0, -1.0,  1.0, 1.0,
     1.0,  1.0,  1.0, 0.0,
  ]);
  const vertexBuffer = device.createBuffer({
    size: quadVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, quadVertices);

  // -------------------------------------------------------------
  // 2. 创建乒乓双缓冲纹理 (Ping 和 Pong)
  // -------------------------------------------------------------
  const createSimTexture = () => device.createTexture({
    size: [width, height],
    format: simFormat,
    // 既要在片元着色器中采样(TEXTURE_BINDING)，又要作为渲染目标输出(RENDER_ATTACHMENT)
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
  });

  const textureA = createSimTexture();
  const textureB = createSimTexture();
  const viewA = textureA.createView();
  const viewB = textureB.createView();

  // 给 Texture A 初始化一个中央亮点
  const initialData = new Uint8Array(width * height * 4);
  const cx = width / 2, cy = height / 2;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const dist = Math.hypot(x - cx, y - cy);
      if (dist < 15) {
        initialData[idx + 0] = 255; // R
        initialData[idx + 1] = 120; // G
        initialData[idx + 2] = 50;  // B
        initialData[idx + 3] = 255; // A
      }
    }
  }
  device.queue.writeTexture(
    { texture: textureA },
    initialData,
    { bytesPerRow: width * 4 },
    [width, height]
  );

  // 采样器 (双线性插值实现平滑扩散)
  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  // 控制衰减与时间步长的 Uniform Buffer
  const simParams = new Float32Array([0.985, 0.0]); // [衰减系数, padding]
  const simUniformBuffer = device.createBuffer({
    size: 16, // WebGPU uniform 至少 16 字节对齐
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(simUniformBuffer, 0, simParams);

  // -------------------------------------------------------------
  // 3. 手动定义 BindGroupLayout (不用 auto)
  // -------------------------------------------------------------
  
  // (1) 模拟管线的布局: 需要 采样器 + 上一帧纹理 + 参数Buffer
  const simBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ],
  });

  // (2) 显示管线的布局: 只需要 采样器 + 最终纹理
  const displayBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
  });

  // -------------------------------------------------------------
  // 4. 着色器代码与多 Pipeline 构建
  // -------------------------------------------------------------

  // 通用顶点着色器代码
  const vertexShaderCommon = `
    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };
    @vertex
    fn vs_main(@location(0) pos: vec2f, @location(1) uv: vec2f) -> VertexOut {
      var out: VertexOut;
      out.pos = vec4f(pos, 0.0, 1.0);
      out.uv = uv;
      return out;
    }
  `;

  // Pipeline 1: 模拟更新着色器 (邻域扩散 + 衰减反馈)
  const simShaderModule = device.createShaderModule({
    code: `
      ${vertexShaderCommon}
      @group(0) @binding(0) var uSampler: sampler;
      @group(0) @binding(1) var srcTexture: texture_2d<f32>;
      @group(0) @binding(2) var<uniform> decay: vec2f;

      @fragment
      fn fs_sim(@location(0) uv: vec2f) -> @location(0) vec4f {
        let texel = 1.0 / 256.0;
        // 采样自身及四周微小偏移，形成热扩散/模糊反馈
        var col = textureSample(srcTexture, uSampler, uv) * 0.4;
        col += textureSample(srcTexture, uSampler, uv + vec2f(texel, 0.0)) * 0.15;
        col += textureSample(srcTexture, uSampler, uv - vec2f(texel, 0.0)) * 0.15;
        col += textureSample(srcTexture, uSampler, uv + vec2f(0.0, texel)) * 0.15;
        col += textureSample(srcTexture, uSampler, uv - vec2f(0.0, texel)) * 0.15;

        // 乘以衰减系数，随着时间向四周扩散并缓慢消逝
        return col * decay.x;
      }
    `,
  });

  // Pipeline 2: 屏幕展示着色器 (颜色映射/增强后输出到画布)
  const displayShaderModule = device.createShaderModule({
    code: `
      ${vertexShaderCommon}
      @group(0) @binding(0) var uSampler: sampler;
      @group(0) @binding(1) var finalTexture: texture_2d<f32>;

      @fragment
      fn fs_display(@location(0) uv: vec2f) -> @location(0) vec4f {
        let color = textureSample(finalTexture, uSampler, uv);
        // 色彩增强后输出到屏幕
        return vec4f(color.rgb * 1.5, 1.0);
      }
    `,
  });

  // 创建 PipelineLayout
  const simPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [simBindGroupLayout],
  });
  const displayPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [displayBindGroupLayout],
  });

  // 顶点属性布局
  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 16,
    attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x2" },
      { shaderLocation: 1, offset: 8, format: "float32x2" },
    ],
  };

  // 创建两个不同的 Pipeline
  const simPipeline = device.createRenderPipeline({
    layout: simPipelineLayout,
    vertex: { module: simShaderModule, entryPoint: "vs_main", buffers: [vertexBufferLayout] },
    fragment: { module: simShaderModule, entryPoint: "fs_sim", targets: [{ format: simFormat }] },
    primitive: { topology: "triangle-list" },
  });

  const displayPipeline = device.createRenderPipeline({
    layout: displayPipelineLayout,
    vertex: { module: displayShaderModule, entryPoint: "vs_main", buffers: [vertexBufferLayout] },
    fragment: { module: displayShaderModule, entryPoint: "fs_display", targets: [{ format: presentationFormat }] },
    primitive: { topology: "triangle-list" },
  });

  // -------------------------------------------------------------
  // 5. 准备乒乓阶段的各个 BindGroup
  // -------------------------------------------------------------
  
  // 模拟阶段: A -> B (读 A，写 B)
  const simBindGroupA = device.createBindGroup({
    layout: simBindGroupLayout,
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: viewA },
      { binding: 2, resource: { buffer: simUniformBuffer } },
    ],
  });

  // 模拟阶段: B -> A (读 B，写 A)
  const simBindGroupB = device.createBindGroup({
    layout: simBindGroupLayout,
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: viewB },
      { binding: 2, resource: { buffer: simUniformBuffer } },
    ],
  });

  // 显示阶段: 显示 A
  const displayBindGroupA = device.createBindGroup({
    layout: displayBindGroupLayout,
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: viewA },
    ],
  });

  // 显示阶段: 显示 B
  const displayBindGroupB = device.createBindGroup({
    layout: displayBindGroupLayout,
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: viewB },
    ],
  });

  // -------------------------------------------------------------
  // 6. 帧渲染循环 (乒乓交替 + 动态切换 Pipeline)
  // -------------------------------------------------------------
  let step = 0;
  let animationFrameId: number;

  function frame() {
    // 判定当前轮次的源和目标：
    // step 为偶数时：从 A 计算写入 B；最终屏幕显示 B
    // step 为奇数时：从 B 计算写入 A；最终屏幕显示 A
    const isEven = step % 2 === 0;
    const simBindGroup = isEven ? simBindGroupA : simBindGroupB;
    const renderTargetView = isEven ? viewB : viewA;
    const displayBindGroup = isEven ? displayBindGroupB : displayBindGroupA;

    const encoder = device.createCommandEncoder();

    // =============== PASS 1: 使用 simPipeline 进行模拟 ===============
    const simPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: renderTargetView, // 写入目标纹理
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    // 切换到模拟管线
    simPass.setPipeline(simPipeline);
    simPass.setVertexBuffer(0, vertexBuffer);
    simPass.setBindGroup(0, simBindGroup);
    simPass.draw(6);
    simPass.end();

    // =============== PASS 2: 切换到 displayPipeline 渲染至屏幕 ===============
    const displayPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(), // 写入 Canvas
        clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    // 切换到显示管线
    displayPass.setPipeline(displayPipeline);
    displayPass.setVertexBuffer(0, vertexBuffer);
    displayPass.setBindGroup(0, displayBindGroup);
    displayPass.draw(6);
    displayPass.end();

    device.queue.submit([encoder.finish()]);

    step++;
    animationFrameId = requestAnimationFrame(frame);
  }

  animationFrameId = requestAnimationFrame(frame);

  // 7. 销毁与清理资源
  return () => {
    cancelAnimationFrame(animationFrameId);
    textureA.destroy();
    textureB.destroy();
    vertexBuffer.destroy();
    simUniformBuffer.destroy();
  };
}