// src/examples/multiplePipelines.ts
export function runMultiplePipelines(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement
) {
  // 1. 初始化 20,000 个粒子数据: pos(vec2f), vel(vec2f) = 16 字节每个粒子
  const particleCount = 20000;
  const initialData = new Float32Array(particleCount * 4);
  for (let i = 0; i < particleCount; i++) {
    const angle = Math.random() * Math.PI * 2;
    const r = Math.sqrt(Math.random()) * 0.7;
    initialData[i * 4 + 0] = Math.cos(angle) * r; // Pos X
    initialData[i * 4 + 1] = Math.sin(angle) * r; // Pos Y
    initialData[i * 4 + 2] = -Math.sin(angle) * 0.2; // Vel X (旋转初速度)
    initialData[i * 4 + 3] = Math.cos(angle) * 0.2;  // Vel Y
  }

  // 【核心交互点】同一块 Buffer，既是计算管线的 STORAGE，又是渲染管线的 VERTEX！
  const particleBuffer = device.createBuffer({
    size: initialData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(particleBuffer, 0, initialData);

  // 2. 管线 1: Compute Pipeline (计算引力与速度物理更新)
  const computeShader = `
    struct Particle {
      pos: vec2f,
      vel: vec2f,
    };
    @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

    @compute @workgroup_size(64)
    fn cs_main(@builtin(global_invocation_id) id: vec3u) {
      let idx = id.x;
      if (idx >= ${particleCount}u) { return; }

      var p = particles[idx];
      // 向中心点施加重力
      let dist = max(length(p.pos), 0.1);
      let force = -normalize(p.pos) * (0.15 / (dist * dist));
      p.vel += force * 0.0005;
      p.pos += p.vel * 0.016;

      particles[idx] = p;
    }
  `;
  const computePipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: computeShader }), entryPoint: "cs_main" },
  });
  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: particleBuffer } }],
  });

  // 3. 管线 2: Render Pipeline (发光粒子加法混合渲染)
  const renderShader = `
    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec4f,
    };

    @vertex
    fn vs_main(@location(0) pos: vec2f, @location(1) vel: vec2f) -> VertexOut {
      var out: VertexOut;
      out.pos = vec4f(pos, 0.0, 1.0);
      let speed = length(vel) * 2.5;
      out.color = vec4f(mix(vec3f(0.1, 0.5, 1.0), vec3f(1.0, 0.3, 0.1), speed), 0.7);
      return out;
    }

    @fragment
    fn fs_main(@location(0) col: vec4f) -> @location(0) vec4f {
      return col;
    }
  `;
  const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: renderShader }), entryPoint: "vs_main",
      buffers: [{
        arrayStride: 16,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x2" }, // pos
          { shaderLocation: 1, offset: 8, format: "float32x2" }, // vel
        ],
      }],
    },
    fragment: {
      module: device.createShaderModule({ code: renderShader }), entryPoint: "fs_main",
      targets: [{
        format,
        // 加法混合 (Additive Blending) 展现发光星系效果
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
          alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
        },
      }],
    },
    primitive: { topology: "point-list" },
  });

  let animId: number;

  function frame() {
    const encoder = device.createCommandEncoder();

    // 【步骤 1】调度计算管线：并发更新 20000 颗粒子物理状态
    const cPass = encoder.beginComputePass();
    cPass.setPipeline(computePipeline);
    cPass.setBindGroup(0, computeBindGroup);
    cPass.dispatchWorkgroups(Math.ceil(particleCount / 64));
    cPass.end();

    // 【步骤 2】调度渲染管线：直接从刚算完的 particleBuffer 绘制！
    const rPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.02, g: 0.02, b: 0.05, a: 1.0 },
        loadOp: "clear", storeOp: "store",
      }],
    });
    rPass.setPipeline(renderPipeline);
    rPass.setVertexBuffer(0, particleBuffer); // 无任何数据拷贝传输
    rPass.draw(particleCount);
    rPass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  frame();

  return () => {
    cancelAnimationFrame(animId);
    particleBuffer.destroy();
  };
}