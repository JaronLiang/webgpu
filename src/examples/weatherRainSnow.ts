// src/examples/weatherRainSnow.ts
export function runWeatherRainSnow(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 步骤 1: 编写 WGSL 着色器 (修复 let 不可变变量赋值错误)
  // =========================================================================
  const shaderCode = `
    struct UniformConfig {
      resolution: vec2f,
      time: f32,
      weatherType: f32, // 0.0 为雨，1.0 为雪
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

    fn hash21(p: vec2f) -> f32 {
      var p3 = fract(vec3f(p.xyx) * 0.1031);
      p3 += dot(p3, p3.yzx + 33.33);
      return fract((p3.x + p3.y) * p3.z);
    }

    // 雨丝模拟
    fn renderRain(uv: vec2f, t: f32) -> f32 {
      var rainCol = 0.0;
      for (var i = 0; i < 3; i++) {
        let scale = 25.0 + f32(i) * 15.0;
        let speed = 14.0 + f32(i) * 5.0;
        let slantedUV = vec2f(uv.x + uv.y * 0.18, uv.y);
        
        // 👈 核心修复：必须使用 var 声明，否则 st.y += 会抛出 immutable 错误
        var st = slantedUV * vec2f(scale, scale * 0.1);
        st.y += t * speed;

        let id = floor(st);
        let gv = fract(st) - 0.5;
        let n = hash21(id);

        if (n > 0.65) {
          let drop = smoothstep(0.06, 0.0, abs(gv.x)) * smoothstep(0.5, -0.5, gv.y);
          rainCol += drop * (0.4 + f32(i) * 0.2);
        }
      }
      return rainCol;
    }

    // 雪花模拟
    fn renderSnow(uv: vec2f, t: f32) -> f32 {
      var snowCol = 0.0;
      for (var i = 0; i < 3; i++) {
        let scale = 9.0 + f32(i) * 6.0;
        let speed = 1.2 + f32(i) * 0.7;
        
        var st = uv * scale;
        st.x += sin(t * 1.5 + f32(i) * 1.8) * 0.35;
        st.y += t * speed;

        let id = floor(st);
        let gv = fract(st) - 0.5;
        let n = hash21(id);

        if (n > 0.7) {
          let dist = length(gv);
          let flake = smoothstep(0.18, 0.0, dist);
          snowCol += flake * (0.5 + f32(i) * 0.2);
        }
      }
      return snowCol;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      var uv = in.uv;
      uv.x *= uConfig.resolution.x / uConfig.resolution.y;

      let t = uConfig.time;
      let bgGradient = mix(vec3f(0.06, 0.08, 0.12), vec3f(0.01, 0.02, 0.03), uv.y * 0.5 + 0.5);

      let rain = renderRain(uv, t);
      let snow = renderSnow(uv, t);

      let weatherAlpha = mix(rain, snow, uConfig.weatherType);
      let particleColor = mix(vec3f(0.7, 0.8, 1.0), vec3f(0.95, 0.98, 1.0), uConfig.weatherType);

      let col = bgGradient + particleColor * weatherAlpha;
      return vec4f(col, 1.0);
    }
  `;

  // =========================================================================
  // 步骤 2: 全屏顶点缓冲
  // =========================================================================
  const vertexData = new Float32Array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
  const vertexBuffer = device.createBuffer({
    label: "Weather-VertexBuffer",
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // =========================================================================
  // 步骤 3: 显式 Uniform 对齐
  // =========================================================================
  const uniformData = new Float32Array(4); // resolution(2), time(1), weatherType(1)
  const uniformBuffer = device.createBuffer({
    label: "Weather-UniformBuffer",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 步骤 4 & 5: 显式 BindGroupLayout 与 PipelineLayout
  // =========================================================================
  const bindGroupLayout = device.createBindGroupLayout({
    label: "Weather-BindGroupLayout",
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "Weather-PipelineLayout",
    bindGroupLayouts: [bindGroupLayout],
  });

  const bindGroup = device.createBindGroup({
    label: "Weather-BindGroup",
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
    label: "Weather-Pipeline",
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
  // 步骤 8: 渲染循环
  // =========================================================================
  let animId: number;
  const startTime = performance.now();

  function render() {
    const canvas = context.canvas as HTMLCanvasElement;
    const elapsed = (performance.now() - startTime) / 1000;
    
    // 平滑在雨天与雪天之间切换 (周期为 8 秒)
    const cycle = (Math.sin(elapsed * 0.4) + 1.0) * 0.5;

    uniformData[0] = canvas.width;
    uniformData[1] = canvas.height;
    uniformData[2] = elapsed;
    uniformData[3] = cycle;

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