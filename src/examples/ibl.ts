// src/examples/iblFixed.ts
export function runIBL(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 1. WGSL 着色器：物理级金属 IBL + 影棚级色调映射
  // =========================================================================
  const shaderCode = `
    struct UniformConfig {
      resolution: vec2f,
      time: f32,
      metallic: f32,
      roughness: f32,
      pad0: f32,
      pad1: f32,
      pad2: f32,
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
      var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0)
      );
      var out: VertexOutput;
      out.position = vec4f(pos[vIdx], 0.0, 1.0);
      out.uv = pos[vIdx];
      return out;
    }

    // 粗糙度抑制的 Fresnel-Schlick 方程
    fn fresnelSchlickRoughness(cosTheta: f32, F0: vec3f, roughness: f32) -> vec3f {
      let maxF = max(vec3f(1.0 - roughness), F0);
      return F0 + (maxF - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    }

    // 电影级 ACES Tone Mapping
    fn acesToneMapping(color: vec3f) -> vec3f {
      let a = 2.51;
      let b = 0.03;
      let c = 2.43;
      let d = 0.59;
      let e = 0.14;
      return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3f(0.0), vec3f(1.0));
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      var uv = in.uv;
      let aspect = uConfig.resolution.x / uConfig.resolution.y;
      uv.x *= aspect;

      // 摄像机缓缓环绕
      let t_cam = uConfig.time * 0.25;
      let camDist = 2.8;
      let ro = vec3f(sin(t_cam) * camDist, 0.45, cos(t_cam) * camDist);
      let sphereCenter = vec3f(0.0, 0.0, 0.0);

      let forward = normalize(sphereCenter - ro);
      let right = normalize(cross(vec3f(0.0, 1.0, 0.0), forward));
      let up = cross(forward, right);
      let rd = normalize(uv.x * right + uv.y * up + 1.8 * forward);

      // 球体相交测试
      let sphereRadius = 0.95;
      let oc = ro - sphereCenter;
      let b = dot(oc, rd);
      let c = dot(oc, oc) - sphereRadius * sphereRadius;
      let h = b * b - c;

      // 背景：直接显示影棚环境
      if (h < 0.0) {
        let envColor = textureSampleLevel(uEnvMap, uSampler, rd, 0.0).rgb;
        let sky = pow(acesToneMapping(envColor * 0.8), vec3f(1.0 / 2.2));
        return vec4f(sky, 1.0);
      }

      // 计算着色点
      let dist = -b - sqrt(h);
      let P = ro + rd * dist;
      let N = normalize(P - sphereCenter);
      let V = -rd;
      let R = reflect(-V, N);

      let NdotV = max(dot(N, V), 0.001);

      // 质感极佳的皇家黄金材质参数 (可替换为铬合金: vec3f(0.96, 0.96, 0.97))
      let albedo = vec3f(1.00, 0.76, 0.28);
      let metallic = clamp(uConfig.metallic, 0.0, 1.0);
      let roughness = clamp(uConfig.roughness, 0.02, 0.95);

      let F0 = mix(vec3f(0.04), albedo, metallic);
      let F = fresnelSchlickRoughness(NdotV, F0, roughness);

      // 漫反射辐照度 (Irradiance) -> 采样最模糊的一级 Mip
      let irradiance = textureSampleLevel(uEnvMap, uSampler, N, 6.0).rgb;
      let diffuse = irradiance * albedo;

      // 镜面反射 (Specular) -> 根据粗糙度连续映射 Mip 级别
      let specMip = roughness * 6.0;
      let prefilteredColor = textureSampleLevel(uEnvMap, uSampler, R, specMip).rgb;

      // Karis 镜面积分环境拟合 (Split-Sum Environment BRDF)
      let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
      let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
      let r = roughness * c0 + c1;
      let a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
      let envBRDF = vec2f(-1.04, 1.04) * a004 + r.zw;
      let specular = prefilteredColor * (F * envBRDF.x + envBRDF.y);

      // 能量守恒组合
      let kD = (vec3f(1.0) - F) * (1.0 - metallic);
      let litColor = kD * diffuse + specular;

      // 底部环境光遮蔽微调 (Fake AO: 让球体底部更有分量感)
      let ao = clamp(0.35 + 0.65 * N.y, 0.2, 1.0);
      let finalLit = litColor * ao;

      // 色调映射与 sRGB 校正
      let mapped = acesToneMapping(finalLit);
      let finalColor = pow(mapped, vec3f(1.0 / 2.2));

      return vec4f(finalColor, 1.0);
    }
  `;

  // =========================================================================
  // 2. 生成无锯齿的高级专业摄影棚 (Studio Environment) HDR Cubemap
  // =========================================================================
  const faceSize = 256; // 提升分辨率至 256，画面极其细腻
  const mipLevelCount = 7;

  const cubemapTexture = device.createTexture({
    label: "Studio-HDR-Cubemap",
    size: [faceSize, faceSize, 6],
    mipLevelCount,
    format: "rgba16float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  // 解析式摄影棚光照（纯平滑数学函数，无任何随机采样，杜绝锯齿马赛克）
  function evaluateStudioHDR(dir: [number, number, number], roughnessFilter: number): [number, number, number] {
    const len = Math.hypot(dir[0], dir[1], dir[2]);
    const x = dir[0] / len;
    const y = dir[1] / len;
    const z = dir[2] / len;

    // 1. 摄影棚深空无缝背景环幕（柔和上下过渡，无生硬切线）
    const t = 0.5 * (y + 1.0);
    let r = 0.03 + 0.08 * t;
    let g = 0.04 + 0.10 * t;
    let b = 0.06 + 0.14 * t;

    // 地面微妙微光
    if (y < 0.0) {
      const groundGlow = Math.exp(y * 4.0) * 0.08;
      r += groundGlow * 0.5;
      g += groundGlow * 0.6;
      b += groundGlow * 0.8;
    }

    // 2. 主柔光箱 (Key Softbox Light): 位于右上前方，暖白色矩形大光源
    {
      const boxDir = [0.65, 0.65, 0.4];
      const dotVal = x * boxDir[0] + y * boxDir[1] + z * boxDir[2];
      if (dotVal > 0.0) {
        // 高斯平滑扩散衰减
        const dist = Math.acos(Math.min(1.0, dotVal));
        const blur = 0.15 + roughnessFilter * 0.45;
        const spot = Math.exp(-(dist * dist) / (2.0 * blur * blur));
        r += spot * 7.5;
        g += spot * 6.5;
        b += spot * 5.0;
      }
    }

    // 3. 辅补光条 (Fill Light): 位于左侧偏后，冷青色柔光板，增强立体轮廓对比
    {
      const fillDir = [-0.85, 0.2, -0.48];
      const dotVal = x * fillDir[0] + y * fillDir[1] + z * fillDir[2];
      if (dotVal > 0.0) {
        const dist = Math.acos(Math.min(1.0, dotVal));
        const blur = 0.22 + roughnessFilter * 0.5;
        const spot = Math.exp(-(dist * dist) / (2.0 * blur * blur));
        r += spot * 1.5;
        g += spot * 3.0;
        b += spot * 5.5;
      }
    }

    // 4. 摄影棚顶灯光环 (Ceiling Light Ring): 给金属球顶部提供精致弧形高光
    {
      const ringY = Math.abs(y - 0.85);
      const blur = 0.08 + roughnessFilter * 0.35;
      const ring = Math.exp(-(ringY * ringY) / (2.0 * blur * blur));
      r += ring * 1.8;
      g += ring * 2.0;
      b += ring * 2.4;
    }

    return [r, g, b];
  }

  // Float32 -> Float16 转换
  function toHalf(val: number): number {
    const f32 = new Float32Array([val]);
    const u32 = new Uint32Array(f32.buffer)[0];
    const sign = (u32 >> 16) & 0x8000;
    let exp = ((u32 >> 23) & 0xff) - (127 - 15);
    let mant = u32 & 0x007fffff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | (mant >> 13);
  }

  function getCubeDir(face: number, u: number, v: number): [number, number, number] {
    const uc = 2.0 * u - 1.0;
    const vc = -(2.0 * v - 1.0);
    switch (face) {
      case 0: return [1.0, vc, -uc];
      case 1: return [-1.0, vc, uc];
      case 2: return [uc, 1.0, -vc];
      case 3: return [uc, -1.0, vc];
      case 4: return [uc, vc, 1.0];
      default: return [-uc, vc, -1.0];
    }
  }

  // 逐级生成平滑无噪点的 Prefiltered Mipmaps
  for (let level = 0; level < mipLevelCount; level++) {
    const size = Math.max(1, faceSize >> level);
    const roughnessFilter = level / (mipLevelCount - 1);

    for (let face = 0; face < 6; face++) {
      const data = new Uint16Array(size * size * 4);

      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const u = (x + 0.5) / size;
          const v = (y + 0.5) / size;

          const dir = getCubeDir(face, u, v);
          const rgb = evaluateStudioHDR(dir, roughnessFilter);

          const idx = (y * size + x) * 4;
          data[idx + 0] = toHalf(rgb[0]);
          data[idx + 1] = toHalf(rgb[1]);
          data[idx + 2] = toHalf(rgb[2]);
          data[idx + 3] = toHalf(1.0);
        }
      }

      device.queue.writeTexture(
        { texture: cubemapTexture, mipLevel: level, origin: [0, 0, face] },
        data,
        { bytesPerRow: size * 4 * 2, rowsPerImage: size },
        [size, size, 1]
      );
    }
  }

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    mipmapFilter: "linear",
  });

  // =========================================================================
  // 3. 配置 Uniform 缓冲区 (32 字节对齐)
  // =========================================================================
  const uniformData = new Float32Array(8);
  uniformData[3] = 1.0;   // 100% 纯金属
  uniformData[4] = 0.08;  // 超高精细度镜面 (低粗糙度展示高级倒影)
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 4. 管线与绑定组
  // =========================================================================
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "cube" } },
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module: device.createShaderModule({ code: shaderCode }),
      entryPoint: "vs_main",
    },
    fragment: {
      module: device.createShaderModule({ code: shaderCode }),
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
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
  // 5. 渲染循环
  // =========================================================================
  let animId: number;
  const startTime = performance.now();

  function render() {
    const canvas = context.canvas as HTMLCanvasElement;
    uniformData[0] = canvas.width;
    uniformData[1] = canvas.height;
    uniformData[2] = (performance.now() - startTime) / 1000;

    // 让粗糙度在 0.05 ~ 0.22 之间轻柔呼吸变化，直观感受金属光泽与柔光倒影
    uniformData[4] = 0.12 + Math.sin(uniformData[2] * 0.6) * 0.07;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.02, g: 0.02, b: 0.03, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
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