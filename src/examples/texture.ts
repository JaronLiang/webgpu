// src/examples/texture.ts
export function runTexture(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // 1. 创建顶点与纹理坐标数据 (带 UV)
  const vertices = new Float32Array([
    -0.6,  0.6, 0.0, 0.0,
    -0.6, -0.6, 0.0, 1.0,
     0.6, -0.6, 1.0, 1.0,
    -0.6,  0.6, 0.0, 0.0,
     0.6, -0.6, 1.0, 1.0,
     0.6,  0.6, 1.0, 0.0,
  ]);

  const vertexBuffer = device.createBuffer({
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertices);

  // 2. 生成 8x8 棋盘格像素数据
  const texSize = 8;
  const texData = new Uint8Array(texSize * texSize * 4);
  for (let y = 0; y < texSize; y++) {
    for (let x = 0; x < texSize; x++) {
      const isBlack = (x + y) % 2 === 0;
      const offset = (y * texSize + x) * 4;
      const c = isBlack ? 255 : 40;
      texData[offset + 0] = c;       // R
      texData[offset + 1] = c;       // G
      texData[offset + 2] = isBlack ? 255 : 200; // B (微蓝)
      texData[offset + 3] = 255;     // A
    }
  }

  // 3. 创建 GPU 纹理并写入数据
  const texture = device.createTexture({
    size: [texSize, texSize],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture },
    texData,
    { bytesPerRow: texSize * 4 },
    [texSize, texSize]
  );

  // 4. 创建采样器
  const sampler = device.createSampler({
    magFilter: "nearest",
    minFilter: "nearest",
  });

  // 5. 【新增】创建 Uniform Buffer (传入一个缩放比或平移等参数)
  // 这里传入一个 4维向量 (scaleX, scaleY, offsetX, offsetY)
  const uniformData = new Float32Array([
    1.0, 1.0,   // scale: 不缩放
    0.1, 0.1    // offset: 顶点向右上微移
  ]);
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  // 6. 着色器代码 (引入 Uniform Buffer 声明)
  const shaderCode = `
    struct Uniforms {
      scale: vec2f,
      offset: vec2f,
    };

    @group(0) @binding(0) var mySampler: sampler;
    @group(0) @binding(1) var myTexture: texture_2d<f32>;
    @group(0) @binding(2) var<uniform> myUniforms: Uniforms; // 通过 buffer 绑定的数据

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@location(0) pos: vec2f, @location(1) uv: vec2f) -> VertexOut {
      var out: VertexOut;
      // 利用 Uniform Buffer 传入的参数做顶点平移与缩放
      let transformedPos = pos * myUniforms.scale + myUniforms.offset;
      out.pos = vec4f(transformedPos, 0.0, 1.0);
      out.uv = uv;
      return out;
    }

    @fragment
    fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
      return textureSample(myTexture, mySampler, uv);
    }
  `;
  const module = device.createShaderModule({ code: shaderCode });

  // 7. 【关键】手动定义 BindGroupLayout (不用 layout: "auto")
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: "filtering" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: "float", viewDimension: "2d" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.VERTEX, // uniform 在顶点着色器中使用
        buffer: {
          type: "uniform",
          hasDynamicOffset: false,
          minBindingSize: uniformData.byteLength,
        },
      },
    ],
  });

  // 8. 手动创建 PipelineLayout
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  // 9. 创建渲染管线 (注入手动定义的 pipelineLayout)
  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout, // 替换原先的 layout: "auto"
    vertex: {
      module,
      entryPoint: "vs_main",
      buffers: [{
        arrayStride: 4 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x2" },
          { shaderLocation: 1, offset: 8, format: "float32x2" },
        ],
      }],
    },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // 10. 创建 BindGroup (严格对应手动定义的 bindGroupLayout)
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout, // 使用手动创建的 bindGroupLayout
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: texture.createView() },
      {
        binding: 2,
        resource: {
          buffer: uniformBuffer,
          offset: 0,
          size: uniformData.byteLength,
        },
      },
    ],
  });

  // 11. 渲染流程
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.15, g: 0.15, b: 0.18, a: 1.0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass.setPipeline(pipeline);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.setBindGroup(0, bindGroup);
  pass.draw(6);
  pass.end();
  device.queue.submit([encoder.finish()]);

  // 12. 释放资源
  return () => {
    texture.destroy();
    vertexBuffer.destroy();
    uniformBuffer.destroy();
  };
}