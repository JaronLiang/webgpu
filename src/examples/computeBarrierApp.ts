/// <reference types="@webgpu/types" />

export async function runComputeBarrierApp(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  context.configure({
    device,
    format,
    alphaMode: "premultiplied",
  });

  // 1. 修复后的 WGSL 计算着色器 (符合 Uniform Control Flow 规则)
  const computeShaderSource = `
    struct Uniforms {
      resolution: vec2f,
      time: f32,
      pad: f32,
    };

    @group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
    @group(0) @binding(1) var rwTexture: texture_storage_2d<r32float, read_write>;
    @group(0) @binding(2) var<storage, read_write> globalStorageBuf: array<f32>;
    @group(0) @binding(3) var<uniform> u: Uniforms;

    fn hash(p: vec2f) -> f32 {
      return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
    }

    @compute @workgroup_size(16, 16, 1)
    fn main(
      @builtin(global_invocation_id) global_id: vec3u,
      @builtin(local_invocation_id) local_id: vec3u,
      @builtin(workgroup_id) group_id: vec3u
    ) {
      let dims = textureDimensions(outputTex);
      let coord = global_id.xy;
      
      // ✅ 消除 early return：使用安全坐标与布尔标志保证一致性控制流 (Uniform Control Flow)
      let in_bounds = coord.x < dims.x && coord.y < dims.y;
      let safe_coord = min(coord, dims - vec2u(1u, 1u));
      let linearIdx = safe_coord.y * dims.x + safe_coord.x;
      let uv = vec2f(safe_coord) / u.resolution;

      // =========================================================================
      // 【阶段 1】：写入 Storage Buffer
      // =========================================================================
      if (in_bounds) {
        let wave1 = sin(uv.x * 14.0 + u.time * 2.5);
        let wave2 = cos(uv.y * 14.0 - u.time * 2.0);
        let rawVal = (wave1 + wave2) * 0.5 + hash(uv + fract(u.time * 0.05)) * 0.15;
        globalStorageBuf[linearIdx] = rawVal;
      }

      // 🌟【核心屏障 1：storageBarrier() 必须位于顶层】
      storageBarrier();
      workgroupBarrier();

      // =========================================================================
      // 【阶段 2】：从 Storage Buffer 跨线程读取，写入 read_write Storage Texture
      // =========================================================================
      if (in_bounds) {
        var blurVal = 0.0;
        let kernel = array<f32, 3>(0.25, 0.5, 0.25);
        for (var dx = -1; dx <= 1; dx++) {
          let sampleX = clamp(i32(safe_coord.x) + dx, 0, i32(dims.x - 1u));
          let readIdx = safe_coord.y * dims.x + u32(sampleX);
          blurVal += globalStorageBuf[readIdx] * kernel[dx + 1];
        }
        textureStore(rwTexture, safe_coord, vec4f(blurVal, 0.0, 0.0, 0.0));
      }

      // 🌟【核心屏障 2：textureBarrier() 必须位于顶层】
      textureBarrier();
      workgroupBarrier();

      // =========================================================================
      // 【阶段 3】：从 read_write Storage Texture 原位回读并输出到呈现纹理
      // =========================================================================
      if (in_bounds) {
        let c_up    = textureLoad(rwTexture, vec2u(safe_coord.x, max(safe_coord.y, 1u) - 1u)).r;
        let c_down  = textureLoad(rwTexture, vec2u(safe_coord.x, min(safe_coord.y + 1u, dims.y - 1u))).r;
        let c_left  = textureLoad(rwTexture, vec2u(max(safe_coord.x, 1u) - 1u, safe_coord.y)).r;
        let c_right = textureLoad(rwTexture, vec2u(min(safe_coord.x + 1u, dims.x - 1u), safe_coord.y)).r;
        let current = textureLoad(rwTexture, safe_coord).r;

        let edge = sqrt(pow(c_right - c_left, 2.0) + pow(c_down - c_up, 2.0)) * 6.0;

        let baseColor = vec3f(0.08, 0.15, 0.28) * (current * 0.5 + 0.5);
        let neonEdge  = vec3f(0.1, 0.9, 0.8) * edge;
        let coreColor = vec3f(0.9, 0.2, 0.6) * pow(edge, 2.0);

        let finalRgb = baseColor + neonEdge + coreColor;
        textureStore(outputTex, safe_coord, vec4f(finalRgb, 1.0));
      }
    }
  `;

  // 2. 呈现着色器 (全屏 Blit 到画布)
  const blitShaderSource = `
    @vertex
    fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
      var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0)
      );
      return vec4f(pos[vid], 0.0, 1.0);
    }

    @group(0) @binding(0) var tex: texture_2d<f32>;
    @group(0) @binding(1) var smp: sampler;

    @fragment
    fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
      let dims = vec2f(textureDimensions(tex));
      let uv = p.xy / dims;
      return textureSample(tex, smp, uv);
    }
  `;

  // 3. 构建管线
  const computeModule = device.createShaderModule({ code: computeShaderSource });
  const blitModule = device.createShaderModule({ code: blitShaderSource });

  const computePipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: computeModule, entryPoint: "main" },
  });

  const blitPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: blitModule, entryPoint: "vs" },
    fragment: {
      module: blitModule,
      entryPoint: "fs",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  const uniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  let storageBuffer: GPUBuffer | null = null;
  let intermediateTexture: GPUTexture | null = null;
  let outputTexture: GPUTexture | null = null;
  let computeBindGroup: GPUBindGroup | null = null;
  let blitBindGroup: GPUBindGroup | null = null;

  let currentW = 0;
  let currentH = 0;

  function ensureSize(width: number, height: number) {
    if (currentW === width && currentH === height && outputTexture) return;
    currentW = width;
    currentH = height;

    if (storageBuffer) storageBuffer.destroy();
    if (intermediateTexture) intermediateTexture.destroy();
    if (outputTexture) outputTexture.destroy();

    storageBuffer = device.createBuffer({
      size: width * height * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    intermediateTexture = device.createTexture({
      size: [width, height, 1],
      format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING,
    });

    outputTexture = device.createTexture({
      size: [width, height, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    computeBindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: outputTexture.createView() },
        { binding: 1, resource: intermediateTexture.createView() },
        { binding: 2, resource: { buffer: storageBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });

    blitBindGroup = device.createBindGroup({
      layout: blitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: outputTexture.createView() },
        { binding: 1, resource: sampler },
      ],
    });
  }

  let isRunning = true;
  const startTime = performance.now();

  function render() {
    if (!isRunning) return;

    const canvas = context.canvas as HTMLCanvasElement;
    const width = Math.max(1, canvas.width);
    const height = Math.max(1, canvas.height);
    ensureSize(width, height);

    const time = (performance.now() - startTime) * 0.001;
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Float32Array([width, height, time, 0.0])
    );

    const encoder = device.createCommandEncoder();

    // 1) Compute Pass
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup!);
    computePass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
    computePass.end();

    // 2) Render Pass
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(blitPipeline);
    renderPass.setBindGroup(0, blitBindGroup!);
    renderPass.draw(3, 1, 0, 0);
    renderPass.end();

    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);

  return () => {
    isRunning = false;
    if (storageBuffer) storageBuffer.destroy();
    if (intermediateTexture) intermediateTexture.destroy();
    if (outputTexture) outputTexture.destroy();
    uniformBuffer.destroy();
  };
}