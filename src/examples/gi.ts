// src/examples/giOptimized.ts
export function runGIOptimized(
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

    // 场景 SDF 几何计算
    fn map(p: vec3f) -> vec2f { // 返回: vec2f(最小距离, 材质ID: 1=红墙, 2=绿墙, 3=白墙, 4=白球)
      let leftWall  = p.x + 2.0;    // x = -2.0
      let rightWall = 2.0 - p.x;    // x =  2.0
      let floorDist = p.y + 1.2;    // y = -1.2
      let ceilDist  = 1.8 - p.y;    // y =  1.8
      let backWall  = 3.2 - p.z;    // z =  3.2

      let sphereCenter = vec3f(0.0, -0.45 + sin(uConfig.time * 1.5) * 0.12, 1.2);
      let sphereDist = length(p - sphereCenter) - 0.75;

      var d = sphereDist;
      var matId = 4.0; // 悬浮白球

      if (leftWall < d)  { d = leftWall;  matId = 1.0; }
      if (rightWall < d) { d = rightWall; matId = 2.0; }
      if (floorDist < d) { d = floorDist; matId = 3.0; }
      if (ceilDist < d)  { d = ceilDist;  matId = 3.0; }
      if (backWall < d)  { d = backWall;  matId = 3.0; }

      return vec2f(d, matId);
    }

    fn calcNormal(p: vec3f) -> vec3f {
      let e = vec2f(0.0008, 0.0);
      return normalize(vec3f(
        map(p + e.xyy).x - map(p - e.xyy).x,
        map(p + e.yxy).x - map(p - e.yxy).x,
        map(p + e.yyx).x - map(p - e.yyx).x
      ));
    }

    // 5-Tap 接触环境光遮蔽 (Ambient Occlusion)
    fn calcAO(p: vec3f, n: vec3f) -> f32 {
      var occ = 0.0;
      var sca = 1.0;
      for (var i = 1; i <= 5; i++) {
        let hr = 0.01 + 0.12 * f32(i) / 5.0;
        let d = map(p + n * hr).x;
        occ += (hr - d) * sca;
        sca *= 0.75;
      }
      return clamp(1.0 - 2.0 * occ, 0.0, 1.0);
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      var uv = in.uv;
      uv.x *= uConfig.resolution.x / uConfig.resolution.y;

      let ro = vec3f(0.0, 0.0, -1.8);
      let rd = normalize(vec3f(uv, 1.55));

      // 1. 强化版光线求交 (96步 + 0.9x 防掠射过度步进，消除球体边缘黑边虚线)
      var t = 0.0;
      var hitMatId = 0.0;
      var hit = false;

      for (var i = 0; i < 96; i++) {
        let res = map(ro + rd * t);
        if (res.x < 0.0008) {
          hit = true;
          hitMatId = res.y;
          break;
        }
        t += res.x * 0.92;
        if (t > 9.0) { break; }
      }

      if (!hit) {
        return vec4f(0.05, 0.05, 0.06, 1.0);
      }

      let p = ro + rd * t;
      let n = calcNormal(p);

      // 基础固有材质色
      var albedo = vec3f(0.85);
      if (hitMatId == 1.0) { albedo = vec3f(0.95, 0.08, 0.08); } // 纯红墙
      if (hitMatId == 2.0) { albedo = vec3f(0.08, 0.95, 0.08); } // 纯绿墙

      // 2. 直接光照计算 (顶部柔光面板)
      let lightPos = vec3f(0.0, 1.6, 1.2);
      let toLight = lightPos - p;
      let lightDist = length(toLight);
      let lightDir = normalize(toLight);
      let diff = max(dot(n, lightDir), 0.0) / (1.0 + lightDist * 0.25);

      // 3. 核心改进：连续物理辐射度 (Continuous Radiosity GI)
      // 计算到左侧红墙与右侧绿墙的连续平滑投影权重，根除离散采样造成的“阶梯同心条纹”
      let distToLeftWall = max(p.x + 2.0, 0.05);
      let distToRightWall = max(2.0 - p.x, 0.05);

      // 朝向左侧的法线会接收左墙反弹的漫反射红光
      let leftWallBleed = vec3f(1.0, 0.02, 0.02) * max(-n.x * 0.7 + 0.3, 0.0) / (distToLeftWall * distToLeftWall + 0.8);
      // 朝向右侧的法线会接收右墙反弹的漫反射绿光
      let rightWallBleed = vec3f(0.02, 1.0, 0.02) * max( n.x * 0.7 + 0.3, 0.0) / (distToRightWall * distToRightWall + 0.8);

      let indirectGI = (leftWallBleed + rightWallBleed) * 0.85;

      // 4. 环境遮蔽
      let ao = calcAO(p, n);

      // 5. 最终着色合成 + Gamma 增强
      let ambient = vec3f(0.05);
      var finalColor = albedo * (diff * vec3f(1.1) + indirectGI + ambient) * ao;

      // ACES 色调映射
      finalColor = finalColor / (finalColor + vec3f(0.18)) * 1.15;
      return vec4f(finalColor, 1.0);
    }
  `;

  // =========================================================================
  // 步骤 2: 全屏顶点缓冲
  // =========================================================================
  const vertexData = new Float32Array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
  const vertexBuffer = device.createBuffer({
    label: "GIOpt-VertexBuffer",
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // =========================================================================
  // 步骤 3: Uniform Buffer (16 字节对齐)
  // =========================================================================
  const uniformData = new Float32Array(4);
  const uniformBuffer = device.createBuffer({
    label: "GIOpt-UniformBuffer",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 步骤 4 & 5: 显式创建 BindGroupLayout & PipelineLayout
  // =========================================================================
  const bindGroupLayout = device.createBindGroupLayout({
    label: "GIOpt-BindGroupLayout",
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "GIOpt-PipelineLayout",
    bindGroupLayouts: [bindGroupLayout],
  });

  const bindGroup = device.createBindGroup({
    label: "GIOpt-BindGroup",
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
    label: "GIOpt-Pipeline",
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

  return () => {
    cancelAnimationFrame(animId);
    vertexBuffer.destroy();
    uniformBuffer.destroy();
  };
}