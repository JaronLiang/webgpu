// src/examples/raymarching.ts
export function runRaymarching(
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
      _pad: f32,
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

    // 球体 SDF
    fn sdSphere(p: vec3f, r: f32) -> f32 {
      return length(p) - r;
    }

    // 平滑并集（Smooth Min），使两个形状融合为液体质感
    fn smin(a: f32, b: f32, k: f32) -> f32 {
      let h = max(k - abs(a - b), 0.0) / k;
      return min(a, b) - h * h * k * 0.25;
    }

    // 场景的全局距离场
    fn map(p: vec3f) -> f32 {
      // 地面
      let floorDist = p.y + 0.8;
      
      // 动态移动的小球
      let sphere1 = sdSphere(p - vec3f(0.0, 0.0, 0.0), 0.8);
      let sphere2 = sdSphere(p - vec3f(sin(uConfig.time * 2.0) * 1.2, cos(uConfig.time * 2.0) * 0.4, 0.0), 0.45);
      let sphere3 = sdSphere(p - vec3f(0.0, sin(uConfig.time * 1.5) * 0.8, cos(uConfig.time * 1.5) * 0.6), 0.35);

      let objects = smin(sphere1, smin(sphere2, sphere3, 0.3), 0.3);
      return min(floorDist, objects);
    }

    // 数值差分计算法线
    fn calcNormal(p: vec3f) -> vec3f {
      let e = vec2f(0.001, 0.0);
      return normalize(vec3f(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
      ));
    }

    // 软阴影算法
    fn softShadow(ro: vec3f, rd: vec3f, mint: f32, maxt: f32, k: f32) -> f32 {
      var res = 1.0;
      var t = mint;
      for (var i = 0; i < 24; i++) {
        let h = map(ro + rd * t);
        if (h < 0.001) { return 0.0; }
        res = min(res, k * h / t);
        t += clamp(h, 0.02, 0.2);
        if (t > maxt) { break; }
      }
      return clamp(res, 0.0, 1.0);
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      var uv = in.uv;
      uv.x *= uConfig.resolution.x / uConfig.resolution.y;

      let ro = vec3f(0.0, 1.2, -3.2); // 相机位置 (微俯视)
      let ta = vec3f(0.0, 0.0, 0.0);  // 观察目标
      let ww = normalize(ta - ro);
      let uu = normalize(cross(ww, vec3f(0.0, 1.0, 0.0)));
      let vv = normalize(cross(uu, ww));
      let rd = normalize(uv.x * uu + uv.y * vv + 1.8 * ww);

      var t = 0.0;
      var hit = false;
      for (var i = 0; i < 80; i++) {
        let d = map(ro + rd * t);
        if (d < 0.001) { hit = true; break; }
        if (t > 15.0) { break; }
        t += d;
      }

      if (!hit) {
        // 背景渐变
        return vec4f(mix(vec3f(0.2, 0.3, 0.5), vec3f(0.05, 0.05, 0.1), in.uv.y * 0.5 + 0.5), 1.0);
      }

      let p = ro + rd * t;
      let n = calcNormal(p);
      let lightDir = normalize(vec3f(1.5, 2.5, -1.5));
      
      // 漫反射与阴影
      let diff = max(dot(n, lightDir), 0.0);
      let shadow = softShadow(p + n * 0.002, lightDir, 0.02, 5.0, 16.0);
      let amb = 0.5 + 0.5 * n.y; // 简单的半球环境光

      // 区分地面棋盘格和主体颜色
      var baseColor = vec3f(0.8, 0.3, 0.2);
      if (p.y < -0.79) {
        let f = fract(p.x * 2.0) > 0.5;
        let g = fract(p.z * 2.0) > 0.5;
        let c = select(0.2, 0.6, (f != g));
        baseColor = vec3f(c);
      }

      let col = baseColor * (diff * shadow + amb * 0.2);
      return vec4f(col, 1.0);
    }
  `;

  // =========================================================================
  // 步骤 2: 填充全屏顶点缓冲
  // =========================================================================
  const vertexData = new Float32Array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
  const vertexBuffer = device.createBuffer({
    label: "Raymarch-VertexBuffer",
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // =========================================================================
  // 步骤 3: 显式创建 Uniform Buffer
  // =========================================================================
  const uniformData = new Float32Array(4);
  const uniformBuffer = device.createBuffer({
    label: "Raymarch-UniformBuffer",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 步骤 4 & 5: 显式创建 BindGroupLayout & PipelineLayout
  // =========================================================================
  const bindGroupLayout = device.createBindGroupLayout({
    label: "Raymarch-BindGroupLayout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "Raymarch-PipelineLayout",
    bindGroupLayouts: [bindGroupLayout],
  });

  // =========================================================================
  // 步骤 6: 创建 BindGroup
  // =========================================================================
  const bindGroup = device.createBindGroup({
    label: "Raymarch-BindGroup",
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // =========================================================================
  // 步骤 7: 创建渲染管线
  // =========================================================================
  const shaderModule = device.createShaderModule({
    label: "Raymarch-ShaderModule",
    code: shaderCode,
  });

  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 2 * Float32Array.BYTES_PER_ELEMENT,
    stepMode: "vertex",
    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
  };

  const pipeline = device.createRenderPipeline({
    label: "Raymarch-Pipeline",
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
  // 步骤 8: 循环绘制
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

  // =========================================================================
  // 步骤 9: 清理与显存释放回调
  // =========================================================================
  return () => {
    cancelAnimationFrame(animId);
    vertexBuffer.destroy();
    uniformBuffer.destroy();
  };
}