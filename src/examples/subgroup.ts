// src/examples/subgroup.ts
export function runSubgroup(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // 检查设备是否支持 subgroups
  const hasSubgroups = device.features.has("subgroups");
  
  if (!hasSubgroups) {
    console.warn(
      "当前设备或浏览器尚未开启 subgroups 特性！\n" +
      "提示: 请尝试在 Chrome 浏览器中访问 chrome://flags，\n" +
      "搜索并开启 'Unsafe WebGPU' 后重启浏览器再试。\n" +
      "⚠️ 现已为您降级到普通渐变渲染模式，以保证画面正常输出。"
    );
  }

  // =========================================================================
  // 步骤 1: 根据是否支持特性，选择不同的着色器代码 (优雅降级)
  // =========================================================================
  
  // 包含 enable subgroups; 的真实代码
  const subgroupShader = `
    enable subgroups; 

    @vertex
    fn vs_main(@builtin(vertex_index) vIdx: u32) -> @builtin(position) vec4f {
      var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0)
      );
      return vec4f(pos[vIdx], 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
      // 获取当前线程在 SIMD subgroup 中的 ID
      let id = f32(subgroupInvocationID);
      let size = f32(subgroupSize);
      
      // 可视化: 每个硬件执行束(Wavefront/Warp) 都会显示不同的渐变带
      let normalized = id / size;
      return vec4f(normalized, 1.0 - normalized, 0.5, 1.0);
    }
  `;

  // 降级代码：模拟相似的渐变视觉效果，但不依赖 subgroup API
  const fallbackShader = `
    @vertex
    fn vs_main(@builtin(vertex_index) vIdx: u32) -> @builtin(position) vec4f {
      var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0)
      );
      return vec4f(pos[vIdx], 0.0, 1.0);
    }

    @fragment
    fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      // 通过屏幕坐标模拟一个渐变色条纹，表示 Fallback 状态
      let normalized = (fragCoord.x % 64.0) / 64.0; 
      return vec4f(normalized, 0.2, 1.0 - normalized, 1.0); // 蓝紫色调表示降级
    }
  `;

  const shaderCode = hasSubgroups ? subgroupShader : fallbackShader;

  // =========================================================================
  // 步骤 2: 显式创建布局与管线
  // =========================================================================
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [] });
  const shaderModule = device.createShaderModule({ code: shaderCode });

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: shaderModule, entryPoint: "vs_main" },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // =========================================================================
  // 步骤 3: 渲染
  // =========================================================================
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: "clear", storeOp: "store",
    }],
  });
  
  pass.setPipeline(pipeline);
  pass.draw(3, 1, 0, 0); // 全屏三角形覆盖画布
  pass.end();
  
  device.queue.submit([encoder.finish()]);

  return () => {}; // 无需清理外部 Buffer
}