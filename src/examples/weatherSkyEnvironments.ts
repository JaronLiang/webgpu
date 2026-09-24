// src/examples/weatherSkyEnvironments.ts
export function runWeatherSkyEnvironments(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 步骤 1: 编写 WGSL 着色器
  // =========================================================================
  const shaderCode = `
    struct UniformConfig {
      resolution: vec2f,
      time: f32,
      // 0: 晴天(Clear/Sunny), 1: 黄昏(Sunset), 2: 黑夜(Night), 3: 阴天多云(Overcast)
      weatherMode: f32, 
    };

    @group(0) @binding(0) var<uniform> uConfig: UniformConfig;

    struct VertexInput {
      @location(0) position: vec2f,
    };

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
      var out: VertexOutput;
      out.position = vec4f(in.position, 0.0, 1.0);
      out.uv = in.position;
      return out;
    }

    fn hash(p: vec2f) -> f32 {
      return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
    }

    fn noise(p: vec2f) -> f32 {
      let i = floor(p);
      let f = fract(p);
      let u = f * f * (3.0 - 2.0 * f);
      return mix(mix(hash(i + vec2f(0.0, 0.0)), hash(i + vec2f(1.0, 0.0)), u.x),
                 mix(hash(i + vec2f(0.0, 1.0)), hash(i + vec2f(1.0, 1.0)), u.x), u.y);
    }

    // 分形布朗运动 (fBm)，用于生成云朵
    fn fbm(p: vec2f) -> f32 {
      var v = 0.0;
      var a = 0.5;
      var shift = vec2f(100.0);
      var pMut = p;
      for (var i = 0; i < 4; i++) {
        v += a * noise(pMut);
        pMut = pMut * 2.0 + shift;
        a *= 0.5;
      }
      return v;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      var uv = in.uv;
      uv.x *= uConfig.resolution.x / uConfig.resolution.y;

      let mode = i32(floor(uConfig.weatherMode + 0.5)) % 4;
      var sky = vec3f(0.0);

      // ==========================================
      // 模式 0: 晴天 (Sunny / Clear Day)
      // ==========================================
      if (mode == 0) {
        let sunPos = vec2f(0.4, 0.4);
        let d = length(uv - sunPos);
        let sunDisc = smoothstep(0.08, 0.07, d);
        let sunGlow = 0.15 / (d + 0.1);
        let skyGradient = mix(vec3f(0.5, 0.7, 0.95), vec3f(0.15, 0.4, 0.85), uv.y * 0.5 + 0.5);
        sky = skyGradient + (vec3f(1.0, 0.9, 0.6) * sunDisc) + (vec3f(0.8, 0.6, 0.3) * sunGlow * 0.5);
      }

      // ==========================================
      // 模式 1: 黄昏 / 日落 (Sunset)
      // ==========================================
      else if (mode == 1) {
        let sunPos = vec2f(0.0, -0.2); // 太阳接近地平线
        let d = length(uv - sunPos);
        let sunDisc = smoothstep(0.1, 0.09, d);
        let sunsetGradient = mix(vec3f(0.9, 0.35, 0.1), vec3f(0.15, 0.1, 0.3), uv.y * 0.5 + 0.5);
        sky = sunsetGradient + (vec3f(1.0, 0.8, 0.3) * sunDisc) + vec3f(0.8, 0.2, 0.05) * (0.2 / (d + 0.1));
      }

      // ==========================================
      // 模式 2: 黑夜 / 繁星与月牙 (Night & Moon)
      // ==========================================
      else if (mode == 2) {
        // 夜空基础渐变
        sky = mix(vec3f(0.02, 0.03, 0.06), vec3f(0.0, 0.0, 0.02), uv.y * 0.5 + 0.5);
        
        // 闪烁繁星
        let starGrid = uv * 35.0;
        let starId = floor(starGrid);
        let starHash = hash(starId);
        if (starHash > 0.96) {
          let twinkle = sin(uConfig.time * 3.0 + starHash * 10.0) * 0.5 + 0.5;
          let starDist = length(fract(starGrid) - 0.5);
          sky += vec3f(smoothstep(0.15, 0.0, starDist) * twinkle * 0.8);
        }

        // 月牙渲染 (用两个相交圆相减)
        let moonPos = vec2f(-0.5, 0.4);
        let d1 = length(uv - moonPos);
        let d2 = length(uv - (moonPos + vec2f(0.05, 0.03)));
        let moonBody = smoothstep(0.12, 0.11, d1);
        let moonCut = smoothstep(0.11, 0.12, d2);
        let crescent = max(0.0, moonBody - moonCut);
        sky += vec3f(0.9, 0.95, 1.0) * crescent + vec3f(0.1, 0.2, 0.3) * (0.05 / (d1 + 0.1));
      }

      // ==========================================
      // 模式 3: 阴天多云 (Overcast & Clouds)
      // ==========================================
      else if (mode == 3) {
        // 阴天低沉灰白色调
        let baseSky = mix(vec3f(0.55, 0.58, 0.6), vec3f(0.3, 0.33, 0.38), uv.y * 0.5 + 0.5);
        // 缓慢流动的云层
        let cloudUV = uv * 2.0 + vec2f(uConfig.time * 0.05, 0.0);
        let cloudCover = fbm(cloudUV);
        let clouds = smoothstep(0.3, 0.8, cloudCover);
        sky = mix(baseSky, vec3f(0.2, 0.22, 0.25), clouds * 0.75);
      }

      return vec4f(sky, 1.0);
    }
  `;

  // =========================================================================
  // 步骤 2: 创建全屏顶点缓冲
  // =========================================================================
  const vertexData = new Float32Array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
  const vertexBuffer = device.createBuffer({
    label: "Sky-VertexBuffer",
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // =========================================================================
  // 步骤 3: 创建并对齐 Uniform Buffer
  // =========================================================================
  const uniformData = new Float32Array(4); // resolution(2), time(1), weatherMode(1)
  const uniformBuffer = device.createBuffer({
    label: "Sky-UniformBuffer",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 步骤 4 & 5: 显式创建 BindGroupLayout & PipelineLayout
  // =========================================================================
  const bindGroupLayout = device.createBindGroupLayout({
    label: "Sky-BindGroupLayout",
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "Sky-PipelineLayout",
    bindGroupLayouts: [bindGroupLayout],
  });

  const bindGroup = device.createBindGroup({
    label: "Sky-BindGroup",
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // =========================================================================
  // 步骤 7: 创建渲染管线
  // =========================================================================
  const shaderModule = device.createShaderModule({ code: shaderCode });
  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 2 * Float32Array.BYTES_PER_ELEMENT,
    stepMode: "vertex",
    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
  };

  const pipeline = device.createRenderPipeline({
    label: "Sky-Pipeline",
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [vertexBufferLayout],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });

  // =========================================================================
  // 步骤 8: 渲染循环 (每 4 秒轮换一次天气状态)
  // =========================================================================
  let animId: number;
  const startTime = performance.now();

  function render() {
    const canvas = context.canvas as HTMLCanvasElement;
    const elapsed = (performance.now() - startTime) / 1000;
    
    // 每 4 秒切换一个天气模式 (0: 晴天 -> 1: 黄昏 -> 2: 夜晚 -> 3: 阴天多云)
    const currentMode = Math.floor(elapsed / 4.0) % 4;

    uniformData[0] = canvas.width;
    uniformData[1] = canvas.height;
    uniformData[2] = elapsed;
    uniformData[3] = currentMode;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(3, 1, 0, 0);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(render);
  }
  render();

  return () => {
    cancelAnimationFrame(animId);
    vertexBuffer.destroy();
    uniformBuffer.destroy();
  };
}