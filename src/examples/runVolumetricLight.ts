// src/examples/volumetricLight.ts
export function runVolumetricLight(
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
      _pad: f32, // 16 字节对齐
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
      out.uv = in.position; // 范围 [-1, 1]
      return out;
    }

    // 球体几何相交测试，用于充当遮挡物 (Occluder)
    fn hitSphere(ro: vec3f, rd: vec3f, center: vec3f, radius: f32) -> bool {
      let oc = ro - center;
      let b = dot(oc, rd);
      let c = dot(oc, oc) - radius * radius;
      return (b * b - c) > 0.0 && b < 0.0;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      var uv = in.uv;
      uv.x *= uConfig.resolution.x / uConfig.resolution.y;

      let ro = vec3f(0.0, 0.0, -2.5); // 相机位置
      let rd = normalize(vec3f(uv, 1.5)); // 视线方向

      // 动态旋转的光源位置
      let lightPos = vec3f(sin(uConfig.time * 0.8) * 1.5, 0.5, 2.5);
      // 位于原点的黑色遮挡球体
      let sphereCenter = vec3f(0.0, 0.0, 0.8);
      let sphereRadius = 0.7;

      let steps = 48;
      let maxDist = 6.0;
      let stepSize = maxDist / f32(steps);
      
      var accumulatedFog = vec3f(0.0);
      var t = 0.0;

      // 光线步进：采样光柱
      for (var i = 0; i < steps; i++) {
        let p = ro + rd * t;
        
        // 沿当前采样点向光源发射阴影射线
        let toLight = lightPos - p;
        let lightDist = length(toLight);
        let lightDir = normalize(toLight);

        // 如果被中间的球体遮挡，则当前采样点处于体积阴影内
        let inShadow = hitSphere(p, lightDir, sphereCenter, sphereRadius);
        
        if (!inShadow) {
          // 距离平方反比衰减
          let attenuation = 1.0 / (lightDist * lightDist + 0.1);
          // 简化的前向散射相位函数 (Henyey-Greenstein)
          let cosTheta = dot(rd, lightDir);
          let phase = (1.0 - 0.25) / pow(1.25 - cosTheta, 1.5);
          
          accumulatedFog += vec3f(1.0, 0.7, 0.3) * attenuation * phase * stepSize * 0.4;
        }
        t += stepSize;
      }

      // 如果视线击中实体球体本身，渲染黑色剪影
      if (hitSphere(ro, rd, sphereCenter, sphereRadius)) {
        return vec4f(accumulatedFog * 0.2, 1.0);
      }

      // 背景暗部叠加体积光
      let finalColor = accumulatedFog + vec3f(0.02, 0.02, 0.04);
      // Tone mapping
      return vec4f(finalColor / (finalColor + vec3f(1.0)), 1.0);
    }
  `;

  // =========================================================================
  // 步骤 2: 填充覆盖屏幕的大三角形 (覆盖 NDC: [-1,-1] 到 [3,-1], [-1,3])
  // =========================================================================
  const vertexData = new Float32Array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
  const vertexBuffer = device.createBuffer({
    label: "VolumetricLight-VertexBuffer",
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // =========================================================================
  // 步骤 3: 创建 Uniform Buffer (显式 16 字节对齐)
  // =========================================================================
  const uniformData = new Float32Array(4); // resolution(2) + time(1) + pad(1)
  const uniformBuffer = device.createBuffer({
    label: "VolumetricLight-UniformBuffer",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 步骤 4: 显式创建 BindGroupLayout
  // =========================================================================
  const bindGroupLayout = device.createBindGroupLayout({
    label: "VolumetricLight-BindGroupLayout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
    ],
  });

  // =========================================================================
  // 步骤 5: 显式创建 PipelineLayout
  // =========================================================================
  const pipelineLayout = device.createPipelineLayout({
    label: "VolumetricLight-PipelineLayout",
    bindGroupLayouts: [bindGroupLayout],
  });

  // =========================================================================
  // 步骤 6: 创建 BindGroup
  // =========================================================================
  const bindGroup = device.createBindGroup({
    label: "VolumetricLight-BindGroup",
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // =========================================================================
  // 步骤 7: 创建渲染管线
  // =========================================================================
  const shaderModule = device.createShaderModule({
    label: "VolumetricLight-ShaderModule",
    code: shaderCode,
  });

  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 2 * Float32Array.BYTES_PER_ELEMENT,
    stepMode: "vertex",
    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
  };

  const pipeline = device.createRenderPipeline({
    label: "VolumetricLight-Pipeline",
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
  // 步骤 8: 动画驱动渲染指令
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