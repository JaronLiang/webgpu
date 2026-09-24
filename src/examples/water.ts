// src/examples/water.ts
export function runWater(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 步骤 1: 编写 WGSL 水面着色器
  // =========================================================================
  const shaderCode = `
    struct UniformConfig {
      resolution: vec2f,
      time: f32,
      _pad: f32,
    };

    @group(0) @binding(0) var<uniform> uConfig: UniformConfig;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) vIdx: u32) -> VertexOutput {
      var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
      var out: VertexOutput;
      out.position = vec4f(pos[vIdx], 0.0, 1.0);
      out.uv = pos[vIdx];
      return out;
    }

    // 叠加不同频率和方向的连续水波函数
    fn getWaveHeight(p: vec2f, t: f32) -> f32 {
      var h = 0.0;
      var pos = p * 1.5;
      
      // 主波
      h += sin(pos.x * 0.8 + t * 2.0 + pos.y * 0.5) * 0.12;
      // 次级斜交波
      h += sin(pos.x * 1.5 - t * 2.5 - pos.y * 1.2) * 0.06;
      // 微风波纹
      h += cos(pos.x * 3.2 + t * 4.0 + pos.y * 2.8) * 0.025;
      h += sin(pos.x * 6.5 - t * 5.5 + pos.y * 5.0) * 0.012;
      return h;
    }

    // 利用微小偏移差分推导波浪表面法线
    fn getWaveNormal(p: vec2f, t: f32) -> vec3f {
      let eps = 0.01;
      let hCenter = getWaveHeight(p, t);
      let hRight  = getWaveHeight(p + vec2f(eps, 0.0), t);
      let hUp     = getWaveHeight(p + vec2f(0.0, eps), t);

      let dx = (hRight - hCenter) / eps;
      let dz = (hUp - hCenter) / eps;

      return normalize(vec3f(-dx, 1.0, -dz));
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      var uv = in.uv;
      uv.x *= uConfig.resolution.x / uConfig.resolution.y;

      let ro = vec3f(0.0, 1.2, -2.5); // 摄像机位于海平面上方 1.2 米处
      let rd = normalize(vec3f(uv.x, uv.y - 0.2, 1.5)); // 略微俯视水面

      // 射线与基准水平面 y = 0 求交
      if (rd.y >= -0.01) {
        // 水天交界线上方的蔚蓝天空渐变
        let skyCol = mix(vec3f(0.55, 0.75, 0.95), vec3f(0.15, 0.4, 0.8), rd.y * 2.0);
        return vec4f(skyCol, 1.0);
      }

      // 计算视线击中水面的距离
      let tWater = -ro.y / rd.y;
      let hitPos = ro + rd * tWater;

      // 采样当前水面法线和高度
      let N = getWaveNormal(hitPos.xz, uConfig.time);
      let waveH = getWaveHeight(hitPos.xz, uConfig.time);
      let V = -rd;

      // 太阳方位与光照
      let sunDir = normalize(vec3f(0.6, 0.45, 0.8));
      let sunColor = vec3f(1.0, 0.95, 0.8);

      // 1. 菲涅尔效应 (视线越平，天空反射越强；垂直看越清澈)
      let NdotV = max(dot(N, V), 0.0);
      let fresnel = 0.04 + 0.96 * pow(1.0 - NdotV, 5.0);

      // 2. 太阳高光镜面反射 (波光粼粼)
      let R = reflect(-sunDir, N);
      let spec = pow(max(dot(R, V), 0.0), 128.0) * 2.5;

      // 3. 水体颜色渐变 (深海蓝与浅碧绿)
      let deepWater = vec3f(0.02, 0.08, 0.22);
      let shallowWater = vec3f(0.05, 0.35, 0.45);
      let waterAlbedo = mix(deepWater, shallowWater, clamp(waveH * 2.0 + 0.5, 0.0, 1.0));

      // 4. 水面反射的天空颜色
      let skyReflection = mix(vec3f(0.6, 0.75, 0.9), vec3f(0.2, 0.5, 0.85), clamp(R.y, 0.0, 1.0));

      // 5. 浪尖白沫 (Wave Foam)
      let foam = smoothstep(0.1, 0.15, waveH);
      let foamColor = vec3f(0.9, 0.95, 1.0);

      // 最终合成：水色 + 菲涅尔天空反射 + 耀眼阳光 + 浪尖泡沫
      var finalColor = mix(waterAlbedo, skyReflection, fresnel);
      finalColor += spec * sunColor;
      finalColor = mix(finalColor, foamColor, foam * 0.75);

      // 远景雾气淡化
      let fogFactor = clamp(tWater * 0.03, 0.0, 1.0);
      let horizonSky = vec3f(0.6, 0.75, 0.9);
      finalColor = mix(finalColor, horizonSky, fogFactor);

      return vec4f(finalColor, 1.0);
    }
  `;

  // =========================================================================
  // 步骤 2: 全屏三角形
  // =========================================================================
  const vertexData = new Float32Array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
  const vertexBuffer = device.createBuffer({
    label: "Water-VertexBuffer",
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // =========================================================================
  // 步骤 3: 显式 Uniform 对齐 (16 字节)
  // =========================================================================
  const uniformData = new Float32Array(4);
  const uniformBuffer = device.createBuffer({
    label: "Water-UniformBuffer",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 步骤 4 & 5: 显式创建 BindGroupLayout 与 PipelineLayout
  // =========================================================================
  const bindGroupLayout = device.createBindGroupLayout({
    label: "Water-BindGroupLayout",
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "Water-PipelineLayout",
    bindGroupLayouts: [bindGroupLayout],
  });

  const bindGroup = device.createBindGroup({
    label: "Water-BindGroup",
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // =========================================================================
  // 步骤 6: 创建渲染管线
  // =========================================================================
  const shaderModule = device.createShaderModule({ code: shaderCode });
  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 2 * Float32Array.BYTES_PER_ELEMENT,
    stepMode: "vertex",
    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
  };

  const pipeline = device.createRenderPipeline({
    label: "Water-Pipeline",
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
  // 步骤 7: 动态水波渲染循环
  // =========================================================================
  let animId: number;
  const startTime = performance.now();

  function render() {
    const canvas = context.canvas as HTMLCanvasElement;
    uniformData[0] = canvas.width;
    uniformData[1] = canvas.height;
    uniformData[2] = (performance.now() - startTime) / 1000;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.15, b: 0.25, a: 1 },
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