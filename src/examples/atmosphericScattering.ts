// src/examples/atmosphericScattering.ts
export function runAtmosphericScattering(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 步骤 1: WGSL 着色器 - 物理大气层光线步进算法
  // =========================================================================
  const shaderCode = `
    struct Uniforms {
      resolution: vec2f,
      padding: vec2f, // 保证 sunDir 以 16 字节对齐
      sunDir: vec4f,
    };

    @group(0) @binding(0) var<uniform> uData: Uniforms;

    // 常量：模拟地球和大气层的物理半径 (缩放比例)
    const PLANET_RADIUS: f32 = 6371e3;
    const ATMOS_RADIUS: f32 = 6471e3;
    const RAYLEIGH_HEIGHT: f32 = 8e3;
    const MIE_HEIGHT: f32 = 1.2e3;
    
    // 散射系数
    const betaR = vec3f(5.5e-6, 13.0e-6, 22.4e-6); // 瑞利散射系数 (偏向蓝色)
    const betaM = vec3f(21e-6);                   // 米氏散射系数

    // 求射线与球体的交点距离
    fn sphereIntersect(ro: vec3f, rd: vec3f, radius: f32) -> vec2f {
      let b = dot(ro, rd);
      let c = dot(ro, ro) - radius * radius;
      var h = b * b - c;
      if (h < 0.0) { return vec2f(-1.0); }
      h = sqrt(h);
      return vec2f(-b - h, -b + h);
    }

    @vertex
    fn vs_main(@builtin(vertex_index) vIdx: u32) -> @builtin(position) vec4f {
      var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
      return vec4f(pos[vIdx], 0.0, 1.0);
    }

    @fragment
    fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      // NDC 坐标
      var uv = (fragCoord.xy / uData.resolution.xy) * 2.0 - 1.0;
      uv.y = -uv.y;

      // 模拟相机位于地表上方 1 米处
      let ro = vec3f(0.0, PLANET_RADIUS + 1.0, 0.0);
      let rd = normalize(vec3f(uv.x, uv.y + 0.2, 1.0)); // 略微仰视
      let sunDir = normalize(uData.sunDir.xyz);

      // 与大气层的交点
      let atmosIntersection = sphereIntersect(ro, rd, ATMOS_RADIUS);
      if (atmosIntersection.y < 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }

      let tMin = max(0.0, atmosIntersection.x);
      let tMax = atmosIntersection.y;
      
      // 与地球的交点，如果穿过地球，则提前终止光线
      let planetIntersection = sphereIntersect(ro, rd, PLANET_RADIUS);
      var endT = tMax;
      if (planetIntersection.x > 0.0) { endT = min(endT, planetIntersection.x); }

      // 积分步数配置 (为主射线步进16次)
      let steps = 16;
      let stepSize = (endT - tMin) / f32(steps);
      var currentT = tMin;

      // 累加的光学厚度和颜色
      var depthR = 0.0;
      var depthM = 0.0;
      var color = vec3f(0.0);

      // 角度相关相位计算
      let mu = dot(rd, sunDir);
      let phaseR = 3.0 / (16.0 * 3.14159) * (1.0 + mu * mu);
      let g = 0.76;
      let phaseM = 3.0 / (8.0 * 3.14159) * ((1.0 - g * g) * (1.0 + mu * mu)) / 
                   ((2.0 + g * g) * pow(1.0 + g * g - 2.0 * g * mu, 1.5));

      // 嵌套的主循环
      for (var i = 0; i < steps; i++) {
        let p = ro + rd * (currentT + stepSize * 0.5); // 步进中点
        let height = length(p) - PLANET_RADIUS; // 当前高度
        
        let hr = exp(-height / RAYLEIGH_HEIGHT) * stepSize;
        let hm = exp(-height / MIE_HEIGHT) * stepSize;
        depthR += hr;
        depthM += hm;

        // 次级光线: 计算当前点到太阳的光学厚度 (为了性能，我们简化只算 4 步)
        let sunIntersect = sphereIntersect(p, sunDir, ATMOS_RADIUS);
        let lightStepSize = sunIntersect.y / 4.0;
        var lightT = 0.0;
        var lightDepthR = 0.0;
        var lightDepthM = 0.0;
        
        for (var j = 0; j < 4; j++) {
          let pl = p + sunDir * (lightT + lightStepSize * 0.5);
          let lHeight = length(pl) - PLANET_RADIUS;
          lightDepthR += exp(-lHeight / RAYLEIGH_HEIGHT) * lightStepSize;
          lightDepthM += exp(-lHeight / MIE_HEIGHT) * lightStepSize;
          lightT += lightStepSize;
        }

        // 光学总厚度及吸收
        let tau = betaR * (depthR + lightDepthR) + betaM * 1.1 * (depthM + lightDepthM);
        let attenuation = exp(-tau);
        
        // 累加单次散射
        color += attenuation * (hr * betaR * phaseR + hm * betaM * phaseM);
        currentT += stepSize;
      }
      
      // 色调映射曝光控制 (Exposure tone mapping)
      color = vec3f(1.0) - exp(-color * 20.0); 
      // Gamma 校正
      color = pow(color, vec3f(1.0 / 2.2)); 

      return vec4f(color, 1.0);
    }
  `;

  // =========================================================================
  // 步骤 2: 构造 Uniform Buffer (显式对齐 vec4f)
  // =========================================================================
  // Uniforms = vec2f(8 bytes) + vec2f(padding 8 bytes) + vec4f(16 bytes) = 32 bytes
  const uniformData = new Float32Array(8);
  const uniformBuffer = device.createBuffer({
    label: "Atmosphere-Uniform",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 步骤 3: 显式创建 BindGroupLayout 与 PipelineLayout
  // =========================================================================
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const shaderModule = device.createShaderModule({ code: shaderCode });

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: shaderModule, entryPoint: "vs_main" },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // =========================================================================
  // 步骤 4: 渲染循环 (动态太阳模拟日出日落)
  // =========================================================================
  let animationFrameId: number;
  let startTime = performance.now();

  function frame() {
    const canvas = context.canvas as HTMLCanvasElement;
    const now = performance.now();
    const time = (now - startTime) / 1000.0;

    uniformData[0] = canvas.width;
    uniformData[1] = canvas.height;
    // 2, 3 为 padding 保留 0
    
    // 让太阳沿 Y-Z 平面缓慢升降，实现昼夜交替的视觉效果
    // y > 0 是白天，y < 0 降至地平线以下是黑夜
    const sunAngle = time * 0.3; 
    uniformData[4] = 0.0;                       // X 
    uniformData[5] = Math.sin(sunAngle);        // Y (高度)
    uniformData[6] = -Math.cos(sunAngle);       // Z 
    uniformData[7] = 0.0;
    
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear", storeOp: "store",
      }],
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3, 1, 0, 0); // 渲染全屏
    pass.end();
    
    device.queue.submit([encoder.finish()]);
    animationFrameId = requestAnimationFrame(frame);
  }
  
  frame();

  // 清理资源，停止动画
  return () => {
    cancelAnimationFrame(animationFrameId);
    uniformBuffer.destroy();
  };
}