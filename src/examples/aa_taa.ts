// src/examples/aa_taa.ts
export function runTAA(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // 1. Pass 1 着色器 (带 Jitter 亚像素抖动光线追踪)
  const sceneShader = `
    struct Uniforms {
      resolution: vec2f,
      jitter: vec2f,
      time: f32,
      pad0: f32, pad1: f32, pad2: f32,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
      var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
      var out: VertexOutput;
      out.position = vec4f(pos[id], 0.0, 1.0);
      out.uv = pos[id];
      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      var uv = in.uv;
      // 引入亚像素 Jitter 抖动偏移
      uv += (u.jitter * 2.0) / u.resolution;
      uv.x *= u.resolution.x / u.resolution.y;

      let ro = vec3f(sin(u.time * 0.3) * 2.8, 0.45, cos(u.time * 0.3) * 2.8);
      let forward = normalize(-ro);
      let right = normalize(cross(vec3f(0.0, 1.0, 0.0), forward));
      let up = cross(forward, right);
      let rd = normalize(uv.x * right + uv.y * up + 1.8 * forward);

      let b = dot(ro, rd);
      let c = dot(ro, ro) - 0.95 * 0.95;
      let h = b * b - c;

      if (h < 0.0) {
        return vec4f(0.03, 0.04, 0.05, 1.0);
      }

      let d = -b - sqrt(h);
      let P = ro + rd * d;
      let N = normalize(P);
      let R = reflect(rd, N);

      // 高亮反射
      let lDir1 = normalize(vec3f(0.7, 0.7, 0.4));
      let s1 = pow(max(dot(R, lDir1), 0.0), 48.0) * 12.0;
      let lDir2 = normalize(vec3f(-0.8, 0.2, -0.5));
      let s2 = pow(max(dot(R, lDir2), 0.0), 24.0) * 4.0;

      let gold = vec3f(1.0, 0.76, 0.28);
      let col = gold * 0.15 + (vec3f(s1) * vec3f(1.0, 0.9, 0.7) + vec3f(s2) * vec3f(0.3, 0.6, 1.0)) * gold;
      return vec4f(col, 1.0);
    }
  `;

  // 2. Pass 2 着色器 (TAA 历史帧融合 + 3x3 邻域色彩截断 Clamping)
  const taaShader = `
    @group(0) @binding(0) var uSampler: sampler;
    @group(0) @binding(1) var uCurrentTex: texture_2d<f32>;
    @group(0) @binding(2) var uHistoryTex: texture_2d<f32>;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
      var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
      var out: VertexOutput;
      out.position = vec4f(pos[id], 0.0, 1.0);
      out.uv = pos[id] * 0.5 + 0.5;
      out.uv.y = 1.0 - out.uv.y;
      return out;
    }

    fn aces(color: vec3f) -> vec3f {
      let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
      return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3f(0.0), vec3f(1.0));
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let dims = vec2f(textureDimensions(uCurrentTex));
      let invSize = 1.0 / dims;

      let curColor = textureSample(uCurrentTex, uSampler, in.uv).rgb;

      // 提取 3x3 邻域计算 AABB 色彩包围盒
      var boxMin = curColor;
      var boxMax = curColor;

      for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
          let s = textureSample(uCurrentTex, uSampler, in.uv + vec2f(f32(x), f32(y)) * invSize).rgb;
          boxMin = min(boxMin, s);
          boxMax = max(boxMax, s);
        }
      }

      // 采样历史帧并将颜色 Clamp 在当前 3x3 区域内 (消除残影核心)
      var histColor = textureSample(uHistoryTex, uSampler, in.uv).rgb;
      histColor = clamp(histColor, boxMin, boxMax);

      // 指数移动平均融合 (90% 历史 + 10% 当前)
      let resolved = mix(histColor, curColor, 0.10);

      // 输出色调映射
      let mapped = pow(aces(resolved), vec3f(1.0 / 2.2));
      return vec4f(mapped, 1.0);
    }
  `;

  // 3. 构建 Halton 序列 (2, 3) 抖动发生器
  function halton(index: number, base: number): number {
    let result = 0;
    let f = 1 / base;
    let i = index;
    while (i > 0) {
      result += f * (i % base);
      i = Math.floor(i / base);
      f /= base;
    }
    return result;
  }

  const jitterSamples = 16;
  const jitters: [number, number][] = [];
  for (let i = 1; i <= jitterSamples; i++) {
    jitters.push([halton(i, 2) - 0.5, halton(i, 3) - 0.5]);
  }

  // 4. 创建资源和管线
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  const uniformBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sceneBGL = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });
  const scenePipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [sceneBGL] }),
    vertex: { module: device.createShaderModule({ code: sceneShader }), entryPoint: "vs_main" },
    fragment: { module: device.createShaderModule({ code: sceneShader }), entryPoint: "fs_main", targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list" },
  });
  const sceneBG = device.createBindGroup({
    layout: sceneBGL,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const taaBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
  });
  const taaPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [taaBGL] }),
    vertex: { module: device.createShaderModule({ code: taaShader }), entryPoint: "vs_main" },
    fragment: { module: device.createShaderModule({ code: taaShader }), entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // Ping-Pong 历史纹理
  let currentTex: GPUTexture | null = null;
  let historyTex: [GPUTexture, GPUTexture] | null = null;
  let taaBindGroups: [GPUBindGroup, GPUBindGroup] | null = null;
  let frameIndex = 0;

  function resize(w: number, h: number) {
    if (currentTex) currentTex.destroy();
    if (historyTex) { historyTex[0].destroy(); historyTex[1].destroy(); }

    currentTex = device.createTexture({
      size: [w, h],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    historyTex = [
      device.createTexture({
        size: [w, h],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      }),
      device.createTexture({
        size: [w, h],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      }),
    ];

    taaBindGroups = [
      device.createBindGroup({
        layout: taaBGL,
        entries: [
          { binding: 0, resource: sampler },
          { binding: 1, resource: currentTex.createView() },
          { binding: 2, resource: historyTex[0].createView() },
        ],
      }),
      device.createBindGroup({
        layout: taaBGL,
        entries: [
          { binding: 0, resource: sampler },
          { binding: 1, resource: currentTex.createView() },
          { binding: 2, resource: historyTex[1].createView() },
        ],
      }),
    ];
  }

  const canvas = context.canvas as HTMLCanvasElement;
  resize(canvas.width, canvas.height);

  let animId: number;
  const uniformData = new Float32Array(8);

  function render() {
    if (canvas.width !== currentTex?.width || canvas.height !== currentTex?.height) {
      resize(canvas.width, canvas.height);
    }

    const jitter = jitters[frameIndex % jitterSamples];
    uniformData[0] = canvas.width;
    uniformData[1] = canvas.height;
    uniformData[2] = jitter[0];
    uniformData[3] = jitter[1];
    uniformData[4] = performance.now() * 0.001;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const readIdx = frameIndex % 2;
    const writeIdx = 1 - readIdx;

    const encoder = device.createCommandEncoder();

    // 1. Pass 1: 渲染带亚像素抖动的单帧
    const pass1 = encoder.beginRenderPass({
      colorAttachments: [{
        view: currentTex!.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass1.setPipeline(scenePipeline);
    pass1.setBindGroup(0, sceneBG);
    pass1.draw(3);
    pass1.end();

    // 2. Pass 2: TAA 历史混合并输出到屏幕与下一个历史帧
    const pass2 = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }
      ],
    });
    pass2.setPipeline(taaPipeline);
    pass2.setBindGroup(0, taaBindGroups![readIdx]);
    pass2.draw(3);
    pass2.end();

    // 将本帧渲染记录同步保存到下一个历史纹理中备用
    encoder.copyTextureToTexture(
      { texture: currentTex! },
      { texture: historyTex![writeIdx] },
      [canvas.width, canvas.height, 1]
    );

    device.queue.submit([encoder.finish()]);
    frameIndex++;
    animId = requestAnimationFrame(render);
  }
  render();

  return () => {
    cancelAnimationFrame(animId);
    currentTex?.destroy();
    historyTex?.[0].destroy();
    historyTex?.[1].destroy();
    uniformBuffer.destroy();
  };
}