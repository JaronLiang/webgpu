// src/examples/computeParticles.ts

export function runComputeParticles(
  device: GPUDevice,
  context: GPUCanvasContext,
  presentationFormat: GPUTextureFormat
) {
  // 1. 粒子系统基础参数
  const NUM_PARTICLES = 65536; // 64K 粒子
  const WORKGROUP_SIZE = 64;   // 计算工作组大小

  // 每个粒子包含: pos(vec2f), vel(vec2f) = 4 个 float (16 字节)
  const particleData = new Float32Array(NUM_PARTICLES * 4);
  for (let i = 0; i < NUM_PARTICLES; i++) {
    const idx = i * 4;
    // 随机极坐标生成圆形星系形态
    const angle = Math.random() * Math.PI * 2;
    const r = Math.sqrt(Math.random()) * 0.7; // 半径
    
    particleData[idx + 0] = Math.cos(angle) * r; // pos.x
    particleData[idx + 1] = Math.sin(angle) * r; // pos.y
    // 初速度（垂直于径向形成旋转力）
    particleData[idx + 2] = -Math.sin(angle) * r * 0.5; // vel.x
    particleData[idx + 3] =  Math.cos(angle) * r * 0.5; // vel.y
  }

  // 2. 创建粒子 Buffer (关键：既作为 Compute 的 Storage，又作为 Render 的 Vertex)
  const particleBuffer = device.createBuffer({
    size: particleData.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.VERTEX |
      GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(particleBuffer, 0, particleData);

  // 3. 创建时间与物理控制 Uniform Buffer
  // 结构: deltaTime(f32), centerAttract(f32), padding(vec2f) -> 16 字节对齐
  const simParams = new Float32Array([0.016, 0.5, 0.0, 0.0]);
  const uniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, simParams);

  // -------------------------------------------------------------
  // 4. 手动定义 Compute 阶段的 BindGroupLayout 与 Pipeline
  // -------------------------------------------------------------
  const computeBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0, // 粒子存储缓冲 (读写)
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 1, // 控制参数 Uniform
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  });

  const computePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [computeBindGroupLayout],
  });

  const computeShader = `
    struct Particle {
      pos: vec2f,
      vel: vec2f,
    };

    struct Params {
      dt: f32,
      attraction: f32,
    };

    @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
    @group(0) @binding(1) var<uniform> params: Params;

    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn cs_main(@builtin(global_invocation_id) id: vec3u) {
      let index = id.x;
      if (index >= ${NUM_PARTICLES}u) {
        return;
      }

      var p = particles[index];

      // 向中心产生引力加速度
      let dist = length(p.pos) + 0.05; // 避免除以 0
      let force = -normalize(p.pos) * (params.attraction / (dist * dist));

      // 欧拉积分更新速度与位置
      p.vel += force * params.dt;
      // 微小阻尼防止粒子发散过快
      p.vel *= 0.998;
      p.pos += p.vel * params.dt;

      // 如果粒子飞出边界，让其重置回中心附近
      if (abs(p.pos.x) > 1.2 || abs(p.pos.y) > 1.2) {
        p.pos = -p.pos * 0.2;
        p.vel = vec2f(-p.vel.y, p.vel.x) * 0.5;
      }

      particles[index] = p;
    }
  `;

  const computePipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: {
      module: device.createShaderModule({ code: computeShader }),
      entryPoint: "cs_main",
    },
  });

  const computeBindGroup = device.createBindGroup({
    layout: computeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: particleBuffer } },
      { binding: 1, resource: { buffer: uniformBuffer } },
    ],
  });

  // -------------------------------------------------------------
  // 5. 手动定义 Render 阶段的 Layout 与 Pipeline
  // -------------------------------------------------------------
  
  // 渲染阶段只需要从 VertexBuffer 中取数据，不需要 BindGroup，PipelineLayout 置空
  const renderPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [],
  });

  const renderShader = `
    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) speed: f32,
    };

    @vertex
    fn vs_main(@location(0) pos: vec2f, @location(1) vel: vec2f) -> VertexOut {
      var out: VertexOut;
      out.pos = vec4f(pos, 0.0, 1.0);
      out.speed = length(vel); // 用速度来计算颜色
      return out;
    }

    @fragment
    fn fs_main(@location(0) speed: f32) -> @location(0) vec4f {
      // 速度慢时呈青蓝色，速度快时呈亮橙黄色
      let slowColor = vec3f(0.1, 0.6, 1.0);
      let fastColor = vec3f(1.0, 0.8, 0.2);
      let col = mix(slowColor, fastColor, clamp(speed * 2.0, 0.0, 1.0));
      return vec4f(col, 0.85);
    }
  `;

  const renderPipeline = device.createRenderPipeline({
    layout: renderPipelineLayout,
    vertex: {
      module: device.createShaderModule({ code: renderShader }),
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 16, // 每个粒子 4 个 float
          stepMode: "vertex",
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x2" }, // @location(0) pos
            { shaderLocation: 1, offset: 8, format: "float32x2" }, // @location(1) vel
          ],
        },
      ],
    },
    fragment: {
      module: device.createShaderModule({ code: renderShader }),
      entryPoint: "fs_main",
      targets: [
        {
          format: presentationFormat,
          // 开启轻度 Alpha 混合，让密集粒子产生发光感
          blend: {
            color: {
              srcFactor: "src-alpha",
              dstFactor: "one", // 加法混合（发光）
              operation: "add",
            },
            alpha: {
              srcFactor: "zero",
              dstFactor: "one",
              operation: "add",
            },
          },
        },
      ],
    },
    primitive: {
      topology: "point-list", // 点图元
    },
  });

  // -------------------------------------------------------------
  // 6. 帧循环 (Compute -> Render 流水线切换)
  // -------------------------------------------------------------
  let animId: number;

  function frame() {
    const encoder = device.createCommandEncoder();

    // ================= PASS 1: 计算调度 (Compute Pass) =================
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    // 派发工作组 (Dispatch)
    computePass.dispatchWorkgroups(Math.ceil(NUM_PARTICLES / WORKGROUP_SIZE));
    computePass.end();

    // ================= PASS 2: 粒子绘制 (Render Pass) =================
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.02, g: 0.02, b: 0.05, a: 1.0 }, // 深空黑背景
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(renderPipeline);
    // 把刚刚被 Compute Shader 更新过的 Buffer 直接挂载为 VertexBuffer
    renderPass.setVertexBuffer(0, particleBuffer);
    renderPass.draw(NUM_PARTICLES);
    renderPass.end();

    // 提交命令至 GPU 队列执行
    device.queue.submit([encoder.finish()]);

    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  // 7. 销毁资源
  return () => {
    cancelAnimationFrame(animId);
    particleBuffer.destroy();
    uniformBuffer.destroy();
  };
}