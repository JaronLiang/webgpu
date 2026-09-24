// src/examples/iblFixed.ts
export function runIBL(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 步骤 1: 编写 WGSL 着色器 (严格 32 字节对齐)
  // =========================================================================
  const shaderCode = `
    struct UniformConfig {
      resolution: vec2f, // 0~7
      time: f32,         // 8~11
      metallic: f32,     // 12~15
      roughness: f32,    // 16~19
      pad0: f32,         // 20~23 (标量单精度，对齐为 4 字节)
      pad1: f32,         // 24~27
      pad2: f32,         // 28~31 -> 结构体总计精确为 32 字节！
    };

    @group(0) @binding(0) var<uniform> uConfig: UniformConfig;
    @group(0) @binding(1) var uSampler: sampler;
    @group(0) @binding(2) var uEnvMap: texture_cube<f32>;

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

    fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
      return F0 + (vec3f(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      var uv = in.uv;
      uv.x *= uConfig.resolution.x / uConfig.resolution.y;

      let ro = vec3f(0.0, 0.0, -2.5);
      let rd = normalize(vec3f(uv, 1.8));

      let sphereRadius = 0.85;
      let b = dot(ro, rd);
      let c = dot(ro, ro) - sphereRadius * sphereRadius;
      let h = b * b - c;

      // 分支内使用 textureSampleLevel，安全且无偏导数警告
      if (h < 0.0) {
        let envColor = textureSampleLevel(uEnvMap, uSampler, rd, 0.0).rgb;
        return vec4f(envColor, 1.0);
      }

      let t = -b - sqrt(h);
      let p = ro + rd * t;
      let N = normalize(p);
      let V = -rd;
      let R = reflect(-V, N);

      let albedo = vec3f(0.95, 0.75, 0.3); // 金色 PBR
      let F0 = mix(vec3f(0.04), albedo, uConfig.metallic);
      let NdotV = max(dot(N, V), 0.0);
      let F = fresnelSchlick(NdotV, F0);

      // 漫反射 IBL
      let irradiance = textureSampleLevel(uEnvMap, uSampler, N, 0.0).rgb;
      let diffuse = irradiance * albedo;

      // 镜面高光 IBL
      let specular = textureSampleLevel(uEnvMap, uSampler, R, 0.0).rgb;

      let kS = F;
      let kD = (vec3f(1.0) - kS) * (1.0 - uConfig.metallic);
      let color = kD * diffuse + specular * (F * 0.85 + 0.15);

      let finalColor = color / (color + vec3f(1.0));
      return vec4f(finalColor, 1.0);
    }
  `;

  // =========================================================================
  // 步骤 2: 构建 6 面 Cubemap
  // =========================================================================
  const faceSize = 16;
  const cubemapTexture = device.createTexture({
    label: "IBL-CubemapTexture",
    size: [faceSize, faceSize, 6],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  const faceColors = [
    [255, 140, 50, 255],  // +X 暖橙
    [180, 60, 220, 255],  // -X 紫霞
    [80, 160, 255, 255],  // +Y 湛蓝天顶
    [30, 25, 20, 255],    // -Y 棕黑大地
    [40, 90, 200, 255],   // +Z 深蓝
    [120, 210, 240, 255], // -Z 青空
  ];

  for (let face = 0; face < 6; face++) {
    const data = new Uint8Array(faceSize * faceSize * 4);
    const color = faceColors[face];
    for (let i = 0; i < faceSize * faceSize; i++) {
      data[i * 4 + 0] = color[0];
      data[i * 4 + 1] = color[1];
      data[i * 4 + 2] = color[2];
      data[i * 4 + 3] = color[3];
    }
    device.queue.writeTexture(
      { texture: cubemapTexture, origin: [0, 0, face] },
      data,
      { bytesPerRow: faceSize * 4, rowsPerImage: faceSize },
      [faceSize, faceSize, 1]
    );
  }

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  // =========================================================================
  // 步骤 3: 严格 32 字节显存配置
  // =========================================================================
  const uniformData = new Float32Array(8); // 8 * 4 = 32 字节
  uniformData[3] = 0.95; // metallic
  uniformData[4] = 0.05; // roughness
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength, // 精确 32 字节，满足 Pipeline 需求
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 步骤 4 & 5: 显式创建 BindGroupLayout 与 PipelineLayout
  // =========================================================================
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "cube" } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: cubemapTexture.createView({ dimension: "cube" }) },
    ],
  });

  // =========================================================================
  // 步骤 6: 创建管线
  // =========================================================================
  const shaderModule = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: shaderModule, entryPoint: "vs_main" },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // =========================================================================
  // 步骤 7: 渲染循环
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
        clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3, 1, 0, 0);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(render);
  }
  render();

  return () => {
    cancelAnimationFrame(animId);
    cubemapTexture.destroy();
    uniformBuffer.destroy();
  };
}