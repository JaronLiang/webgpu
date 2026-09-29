// src/examples/aa_fxaa.ts
export function runFXAA(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // 1. Pass 1 着色器 (光追球体场景)
  const sceneShader = `
    struct Config { resolution: vec2f, time: f32, pad: f32 };
    @group(0) @binding(0) var<uniform> u: Config;

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
      uv.x *= u.resolution.x / u.resolution.y;

      let ro = vec3f(sin(u.time * 0.4) * 2.8, 0.4, cos(u.time * 0.4) * 2.8);
      let forward = normalize(-ro);
      let right = normalize(cross(vec3f(0.0, 1.0, 0.0), forward));
      let up = cross(forward, right);
      let rd = normalize(uv.x * right + uv.y * up + 1.8 * forward);

      let b = dot(ro, rd);
      let c = dot(ro, ro) - 0.95 * 0.95;
      let h = b * b - c;

      if (h < 0.0) {
        return vec4f(0.04, 0.05, 0.07, 1.0);
      }

      let d = -b - sqrt(h);
      let P = ro + rd * d;
      let N = normalize(P);
      let R = reflect(rd, N);

      // 强高对比度光源高光 (容易产生严重锯齿)
      let sun1 = pow(max(dot(R, normalize(vec3f(1.0, 1.0, 0.5))), 0.0), 64.0) * 10.0;
      let sun2 = pow(max(dot(R, normalize(vec3f(-1.0, 0.2, -0.5))), 0.0), 32.0) * 4.0;
      let gold = vec3f(1.0, 0.76, 0.28);
      let col = gold * 0.2 + (vec3f(sun1) + vec3f(sun2) * vec3f(0.3, 0.7, 1.0)) * gold;

      // 输出 RGB 以及 Alpha 通道存放的感知灰度 Luma (FXAA 必需)
      let luma = dot(col, vec3f(0.299, 0.587, 0.114));
      return vec4f(col, luma);
    }
  `;

  // 2. Pass 2 着色器 (标准高效 FXAA 核心算法)
  const fxaaShader = `
    @group(0) @binding(0) var uSampler: sampler;
    @group(0) @binding(1) var uTexture: texture_2d<f32>;

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

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let dims = vec2f(textureDimensions(uTexture));
      let invSize = 1.0 / dims;

      // 采样当前像素及 4 邻域 Luma
      let m = textureSample(uTexture, uSampler, in.uv);
      let lumaM = m.a;
      let lumaNW = textureSample(uTexture, uSampler, in.uv + vec2f(-1.0, -1.0) * invSize).a;
      let lumaNE = textureSample(uTexture, uSampler, in.uv + vec2f( 1.0, -1.0) * invSize).a;
      let lumaSW = textureSample(uTexture, uSampler, in.uv + vec2f(-1.0,  1.0) * invSize).a;
      let lumaSE = textureSample(uTexture, uSampler, in.uv + vec2f( 1.0,  1.0) * invSize).a;

      let lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
      let lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
      let lumaRange = lumaMax - lumaMin;

      // 若未达到边缘对比度阈值，直接返回原色
      if (lumaRange < max(0.05, lumaMax * 0.125)) {
        return vec4f(m.rgb, 1.0);
      }

      // 计算梯度主方向
      var dir = vec2f(
        -((lumaNW + lumaNE) - (lumaSW + lumaSE)),
        ((lumaNW + lumaSW) - (lumaNE + lumaSE))
      );
      let dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.03125, 0.0078125);
      let rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
      dir = min(vec2f(8.0), max(vec2f(-8.0), dir * rcpDirMin)) * invSize;

      // 沿边缘混色采样
      let rgbA = 0.5 * (
        textureSample(uTexture, uSampler, in.uv + dir * (1.0 / 3.0 - 0.5)).rgb +
        textureSample(uTexture, uSampler, in.uv + dir * (2.0 / 3.0 - 0.5)).rgb
      );
      let rgbB = rgbA * 0.5 + 0.25 * (
        textureSample(uTexture, uSampler, in.uv + dir * -0.5).rgb +
        textureSample(uTexture, uSampler, in.uv + dir * 0.5).rgb
      );

      let lumaB = dot(rgbB, vec3f(0.299, 0.587, 0.114));
      if (lumaB < lumaMin || lumaB > lumaMax) {
        return vec4f(rgbA, 1.0);
      }
      return vec4f(rgbB, 1.0);
    }
  `;

  // 3. 构建中间贴图与管线
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  let sceneTexture: GPUTexture | null = null;
  let fxaaBindGroup: GPUBindGroup | null = null;

  const configBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sceneBGL = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });
  const scenePipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [sceneBGL] }),
    vertex: { module: device.createShaderModule({ code: sceneShader }), entryPoint: "vs_main" },
    fragment: { module: device.createShaderModule({ code: sceneShader }), entryPoint: "fs_main", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" },
  });
  const sceneBG = device.createBindGroup({
    layout: sceneBGL,
    entries: [{ binding: 0, resource: { buffer: configBuffer } }],
  });

  const fxaaBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
  });
  const fxaaPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [fxaaBGL] }),
    vertex: { module: device.createShaderModule({ code: fxaaShader }), entryPoint: "vs_main" },
    fragment: { module: device.createShaderModule({ code: fxaaShader }), entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  function resize(w: number, h: number) {
    if (sceneTexture) sceneTexture.destroy();
    sceneTexture = device.createTexture({
      size: [w, h],
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    fxaaBindGroup = device.createBindGroup({
      layout: fxaaBGL,
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: sceneTexture.createView() },
      ],
    });
  }

  const canvas = context.canvas as HTMLCanvasElement;
  resize(canvas.width, canvas.height);

  let animId: number;
  const configData = new Float32Array(4);

  function render() {
    if (canvas.width !== sceneTexture?.width || canvas.height !== sceneTexture?.height) {
      resize(canvas.width, canvas.height);
    }

    configData[0] = canvas.width;
    configData[1] = canvas.height;
    configData[2] = performance.now() * 0.001;
    device.queue.writeBuffer(configBuffer, 0, configData);

    const encoder = device.createCommandEncoder();

    // Pass 1: 渲染原始场景至 sceneTexture
    const pass1 = encoder.beginRenderPass({
      colorAttachments: [{
        view: sceneTexture!.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass1.setPipeline(scenePipeline);
    pass1.setBindGroup(0, sceneBG);
    pass1.draw(3);
    pass1.end();

    // Pass 2: 应用 FXAA 抗锯齿后处理输出到屏幕
    const pass2 = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass2.setPipeline(fxaaPipeline);
    pass2.setBindGroup(0, fxaaBindGroup!);
    pass2.draw(3);
    pass2.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(render);
  }
  render();

  return () => {
    cancelAnimationFrame(animId);
    sceneTexture?.destroy();
    configBuffer.destroy();
  };
}