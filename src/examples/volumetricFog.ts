// src/examples/volumetricFog.ts
export function runVolumetricFog(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 步骤 1: WGSL 着色器 - 包含 3D 光线步进和体积累加算法
  // =========================================================================
  const shaderCode = `
    struct Uniforms {
      resolution: vec2f,
      time: f32,
      // 填充对齐
      _pad: f32, 
      lightPos: vec4f,
    };

    @group(0) @binding(0) var<uniform> uData: Uniforms;

    // 一个简单的 3D 伪随机噪声函数
    fn hash(p: vec3f) -> f32 {
      var p3 = fract(p * 0.1031);
      p3 += dot(p3, p3.yzx + 33.33);
      return fract((p3.x + p3.y) * p3.z);
    }
    
    // 获取指定位置的雾密度 (结合高度指数衰减与噪声)
    fn get_density(p: vec3f) -> f32 {
      // 基础高度雾：越低越浓
      let heightFog = exp(-p.y * 1.5);
      // 增加时间动态噪声扰动
      let noise = sin(p.x * 2.0 + uData.time) * cos(p.z * 2.0 + uData.time) * 0.5 + 0.5;
      return clamp(heightFog * noise, 0.0, 1.0);
    }

    @vertex
    fn vs_main(@builtin(vertex_index) vIdx: u32) -> @builtin(position) vec4f {
      // 绘制全屏三角形
      var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
      return vec4f(pos[vIdx], 0.0, 1.0);
    }

    @fragment
    fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      // 将屏幕坐标映射到 [-1, 1] 的 NDC 坐标
      var uv = (fragCoord.xy / uData.resolution.xy) * 2.0 - 1.0;
      uv.y = -uv.y;
      
      // 简单的透视相机射线生成
      let ro = vec3f(0.0, 1.0, uData.time * 0.5); // 相机位置随时间向前移动
      let rd = normalize(vec3f(uv, 1.5)); // 射线方向
      
      // 光线步进参数
      var t = 0.0;
      let max_t = 10.0;
      let steps = 40;
      let step_size = max_t / f32(steps);
      
      var transmittance = 1.0;  // 光的透射率 (1.0表示完全透明)
      var scatteredLight = vec3f(0.0); // 累加的颜色
      let lightColor = vec3f(1.0, 0.8, 0.6); // 暖色光
      
      for(var i = 0; i < steps; i++) {
        if (transmittance < 0.01) { break; } // 优化：光线已经穿透不了了
        
        let p = ro + rd * t;
        let density = get_density(p);
        
        if (density > 0.01) {
          // 步进的衰减量
          let extinction = density * 1.5;
          let stepTransmittance = exp(-extinction * step_size);
          transmittance *= stepTransmittance;
          
          // 计算当前点到光源的方向，用于产生体积光的各向异性（相位函数简写）
          let dirToLight = normalize(uData.lightPos.xyz - p);
          let sunScattering = max(dot(rd, dirToLight), 0.0);
          // 模拟 Henyey-Greenstein 相位函数高光
          let phase = pow(sunScattering, 4.0) * 0.8 + 0.2; 
          
          // 累加内散射光
          scatteredLight += density * step_size * transmittance * phase * lightColor;
        }
        t += step_size;
      }
      
      // 最终颜色 = 雾的颜色 + 背景颜色(天空深蓝色) * 最终透射率
      let bgColor = vec3f(0.05, 0.1, 0.2);
      let finalColor = scatteredLight + bgColor * transmittance;
      
      // 简单的色调映射
      let mapped = finalColor / (finalColor + vec3f(1.0));
      return vec4f(mapped, 1.0);
    }
  `;

  // =========================================================================
  // 步骤 2: 构造 Uniform Buffer
  // =========================================================================
  // uniform 占用 8 个 f32 (32字节)
  const uniformData = new Float32Array(8);
  const uniformBuffer = device.createBuffer({
    label: "VolumetricFog-Uniform",
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
  // 步骤 4: 渲染循环
  // =========================================================================
  let animationFrameId: number;
  let startTime = performance.now();

  function frame() {
    const canvas = context.canvas as HTMLCanvasElement;
    const now = performance.now();
    const time = (now - startTime) / 1000.0;

    // 填充 Uniform 数据
    uniformData[0] = canvas.width;
    uniformData[1] = canvas.height;
    uniformData[2] = time;
    // uniformData[3] 是 padding
    
    // 让光源位置随时间圆周运动
    uniformData[4] = Math.sin(time * 0.5) * 5.0; // Light X
    uniformData[5] = 2.0;                        // Light Y
    uniformData[6] = Math.cos(time * 0.5) * 5.0; // Light Z
    // uniformData[7] 是 padding
    
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
    pass.draw(3, 1, 0, 0); // 渲染全屏三角形
    pass.end();
    
    device.queue.submit([encoder.finish()]);
    animationFrameId = requestAnimationFrame(frame);
  }
  
  frame();

  // 清理回调
  return () => {
    cancelAnimationFrame(animationFrameId);
    uniformBuffer.destroy();
  };
}