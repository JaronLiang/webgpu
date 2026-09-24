import type { SimpleGUI } from "../utils/gui";

// ==========================================
// 1. 基础 3D 矩阵数学 (免第三方依赖)
// ==========================================
namespace Math3D {
  export function lookAt(eye: number[], center: number[], up: number[]): Float32Array {
    const z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
    const lenZ = 1 / (Math.hypot(z0, z1, z2) || 1);
    const zx = z0 * lenZ, zy = z1 * lenZ, zz = z2 * lenZ;

    const x0 = up[1] * zz - up[2] * zy, x1 = up[2] * zx - up[0] * zz, x2 = up[0] * zy - up[1] * zx;
    const lenX = 1 / (Math.hypot(x0, x1, x2) || 1);
    const xx = x0 * lenX, xy = x1 * lenX, xz = x2 * lenX;

    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;

    const out = new Float32Array(16);
    out[0] = xx;  out[1] = yx;  out[2] = zx;  out[3] = 0;
    out[4] = xy;  out[5] = yy;  out[6] = zy;  out[7] = 0;
    out[8] = xz;  out[9] = yz;  out[10] = zz; out[11] = 0;
    out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    out[15] = 1;
    return out;
  }

  export function perspective(fovRad: number, aspect: number, near: number, far: number): Float32Array {
    const f = 1.0 / Math.tan(fovRad / 2.0);
    const nf = 1 / (near - far);
    const out = new Float32Array(16);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = far * nf;
    out[11] = -1.0;
    out[14] = near * far * nf;
    return out;
  }

  export function multiply(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      const ai0 = a[i], ai1 = a[i + 4], ai2 = a[i + 8], ai3 = a[i + 12];
      out[i]      = ai0 * b[0]  + ai1 * b[1]  + ai2 * b[2]  + ai3 * b[3];
      out[i + 4]  = ai0 * b[4]  + ai1 * b[5]  + ai2 * b[6]  + ai3 * b[7];
      out[i + 8]  = ai0 * b[8]  + ai1 * b[9]  + ai2 * b[10] + ai3 * b[11];
      out[i + 12] = ai0 * b[12] + ai1 * b[13] + ai2 * b[14] + ai3 * b[15];
    }
    return out;
  }
}

// ==========================================
// 2. Compute Shader 火焰粒子系统主入口
// ==========================================
export function runFireTextureAnimationShowcase(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  const PARTICLE_COUNT = 8192; // 8192 颗物理粒子

  // ----------------------------------------------------
  // 1. WGSL: 计算着色器 (物理更新) + 渲染着色器 (预乘混合)
  // ----------------------------------------------------
  const shaderWGSL = `
    struct Uniforms {
      viewProj: mat4x4f,
      camPos: vec4f,
      params: vec4f,   // x: dt, y: time, z: flameHeight, w: smokeDensity
      dynamics: vec4f, // x: windX, y: windZ, z: turbulence, w: particleCount
    };

    struct Particle {
      pos: vec3f,
      life: f32,       // 剩余寿命
      vel: vec3f,
      maxLife: f32,    // 总寿命
      size: f32,
      pType: f32,      // 0.0: 火焰, 1.0: 浓烟, 2.0: 火星
      rot: f32,
      pad: f32,
    };

    // --- Compute 阶段绑定 ---
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var<storage, read_write> simParticles: array<Particle>;

    // 伪随机生成器
    fn hash31(p: f32) -> vec3f {
      var p3 = fract(vec3f(p * 0.1031, p * 0.1030, p * 0.0973));
      p3 += dot(p3, p3.yzx + 33.33);
      return fract((p3.xxy + p3.yzz) * p3.zyx);
    }

    @compute @workgroup_size(64)
    fn cs_main(@builtin(global_invocation_id) id: vec3u) {
      let idx = id.x;
      if (idx >= u32(u.dynamics.w)) { return; }

      var p = simParticles[idx];
      let dt = clamp(u.params.x, 0.001, 0.05);
      p.life -= dt;

      // 粒子重生逻辑
      if (p.life <= 0.0) {
        let seed = hash31(f32(idx) * 1.618 + u.params.y * 3.14);
        let discAngle = seed.x * 6.28318;
        let discRadius = sqrt(seed.y) * 0.9;

        p.pos = vec3f(cos(discAngle) * discRadius, 0.0, sin(discAngle) * discRadius);
        p.rot = seed.z * 6.28318;

        // 粒子种类分布: 60% 烈焰, 28% 浓烟, 12% 火星
        if (seed.z < 0.60) {
          // A. 烈焰粒子
          p.pType = 0.0;
          p.maxLife = 0.5 + seed.y * 0.6;
          p.life = p.maxLife;
          p.size = 0.45 + seed.x * 0.35;
          p.vel = vec3f((seed.x - 0.5) * 0.6, 2.8 + seed.y * 2.2, (seed.y - 0.5) * 0.6);
        } else if (seed.z < 0.88) {
          // B. 浓烟粒子 (寿命更长，自底向上升腾)
          p.pType = 1.0;
          p.maxLife = 1.6 + seed.y * 1.2;
          p.life = p.maxLife;
          p.size = 0.5 + seed.x * 0.4;
          p.pos.y += 0.4; // 烟雾略微偏上生成
          p.vel = vec3f((seed.x - 0.5) * 0.8, 1.8 + seed.y * 1.5, (seed.y - 0.5) * 0.8);
        } else {
          // C. 飞溅火星 (高初速细小粒子)
          p.pType = 2.0;
          p.maxLife = 1.2 + seed.y * 1.0;
          p.life = p.maxLife;
          p.size = 0.06 + seed.x * 0.05;
          p.vel = vec3f((seed.x - 0.5) * 2.2, 4.2 + seed.y * 3.0, (seed.y - 0.5) * 2.2);
        }
      } else {
        // 物理动力学模拟
        let progress = 1.0 - (p.life / p.maxLife);

        // 浮力与重力对流
        if (p.pType == 0.0) {
          // 火焰：向上加速后逐渐衰减
          p.vel.y += (4.0 * u.params.z - p.vel.y * 1.2) * dt;
          p.size *= (1.0 - dt * 0.4);
        } else if (p.pType == 1.0) {
          // 烟雾：阻力减速并向外膨胀
          p.vel.y += (1.5 - p.vel.y * 0.8) * dt;
          p.size += dt * 0.65; // 浓烟剧烈扩散体积
        } else {
          // 火星：受空气阻力并自由扰动
          p.vel.y += (1.0 - p.vel.y * 0.5) * dt;
        }

        // 风力与三维湍流影响
        let turb = vec3f(
          sin(p.pos.y * 3.0 + u.params.y * 4.0 + f32(idx) * 0.1),
          0.0,
          cos(p.pos.y * 3.0 + u.params.y * 3.5 + f32(idx) * 0.1)
        ) * u.dynamics.z;

        let wind = vec3f(u.dynamics.x, 0.0, u.dynamics.y);
        p.pos += (p.vel + wind + turb) * dt;
        p.rot += 1.2 * dt;
      }

      simParticles[idx] = p;
    }

    // --- Render 阶段绑定 ---
    @group(0) @binding(1) var<storage, read> rendParticles: array<Particle>;

    struct VertexOutput {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
      @location(1) progress: f32,
      @location(2) pType: f32,
    };

    // 无顶点缓冲区：着色器内置四边形拓扑
    @vertex
    fn vs_render(
      @builtin(vertex_index) vIdx: u32,
      @builtin(instance_index) iIdx: u32
    ) -> VertexOutput {
      var o: VertexOutput;
      let p = rendParticles[iIdx];

      // 粒子已死亡或尺寸为0时抛出不可见四边形
      if (p.life <= 0.0) {
        o.pos = vec4f(0.0);
        return o;
      }

      var quadOffsets = array<vec2f, 6>(
        vec2f(-0.5, -0.5), vec2f( 0.5, -0.5), vec2f( 0.5,  0.5),
        vec2f(-0.5, -0.5), vec2f( 0.5,  0.5), vec2f(-0.5,  0.5)
      );
      var quadUVs = array<vec2f, 6>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
        vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0)
      );

      let off = quadOffsets[vIdx];
      o.uv = quadUVs[vIdx];
      o.progress = 1.0 - (p.life / p.maxLife);
      o.pType = p.pType;

      // 自旋转矩阵计算
      let cosR = cos(p.rot);
      let sinR = sin(p.rot);
      let rotatedOff = vec2f(off.x * cosR - off.y * sinR, off.x * sinR + off.y * cosR) * p.size;

      // 视线正交 Billboard (面向摄像机)
      let look = normalize(u.camPos.xyz - p.pos);
      let right = normalize(cross(vec3f(0.0, 1.0, 0.0), look));
      let up = cross(look, right);

      let worldPos = p.pos + (right * rotatedOff.x + up * rotatedOff.y);
      o.pos = u.viewProj * vec4f(worldPos, 1.0);
      return o;
    }

    @fragment
    fn fs_render(in: VertexOutput) -> @location(0) vec4f {
      let d = length(in.uv - vec2f(0.5));
      if (d > 0.5) { discard; }

      // 软粒子径向衰减
      let softEdge = smoothstep(0.5, 0.0, d);

      // --- 1. 火焰渲染 (加色发光) ---
      if (in.pType < 0.5) {
        let coreCol = vec3f(1.0, 0.95, 0.8);
        let midCol  = vec3f(1.0, 0.42, 0.03);
        let endCol  = vec3f(0.85, 0.08, 0.01);

        var col = vec3f(0.0);
        if (in.progress < 0.3) {
          col = mix(coreCol, midCol, in.progress / 0.3);
        } else {
          col = mix(midCol, endCol, (in.progress - 0.3) / 0.7);
        }

        // 焰心热辐射增益
        let intense = (exp(-d * 4.5) * 1.5 + softEdge * 0.8) * (1.0 - in.progress * 0.85);
        let fireRGB = col * intense * 2.2;
        
        // 预乘格式：Alpha 为 0.0 实现纯加色混合
        return vec4f(fireRGB, 0.0);
      }

      // --- 2. 浓烟渲染 (遮挡吸收) ---
      if (in.pType < 1.5) {
        // 烟雾色彩：暗褐灰向炭黑色渐变
        let smokeBase = vec3f(0.08, 0.075, 0.07);
        
        // 烟雾生成时淡入，消散时淡出
        let fade = smoothstep(0.0, 0.25, in.progress) * smoothstep(1.0, 0.4, in.progress);
        let alpha = softEdge * fade * 0.32 * u.params.w;
        if (alpha < 0.005) { discard; }

        // 预乘 Alpha 输出
        return vec4f(smokeBase * alpha, alpha);
      }

      // --- 3. 火星渲染 (高亮微粒) ---
      let sparkCol = mix(vec3f(1.0, 0.9, 0.4), vec3f(1.0, 0.25, 0.02), in.progress);
      let sparkGlow = exp(-d * 8.0) * 3.5;
      return vec4f(sparkCol * sparkGlow, 0.0);
    }
  `;

  const shaderModule = device.createShaderModule({ code: shaderWGSL });

  // ----------------------------------------------------
  // 2. 缓冲分配与初始数据装载
  // ----------------------------------------------------
  // Uniform 缓冲 (128 bytes)
  const uniformBuffer = device.createBuffer({
    size: 128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Particle 结构体尺寸: 48 bytes
  const particleBufferSize = PARTICLE_COUNT * 48;
  const initialParticleData = new Float32Array(PARTICLE_COUNT * 12);

  // 预热粒子初始生命周期 (错开出生时间，避免第0帧爆发)
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const offset = i * 12;
    initialParticleData[offset + 3] = Math.random() * 1.5; // life
    initialParticleData[offset + 7] = 2.0;                  // maxLife
  }

  const particleBuffer = device.createBuffer({
    size: particleBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(particleBuffer, 0, initialParticleData as unknown as BufferSource);

  // ----------------------------------------------------
  // 3. Compute 与 Render 独立管线绑定配置
  // ----------------------------------------------------
  // Compute 阶段: storage (read_write)
  const computeBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });

  const computeBindGroup = device.createBindGroup({
    layout: computeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
    ],
  });

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
    compute: { module: shaderModule, entryPoint: "cs_main" },
  });

  // Render 阶段: read-only-storage (安全合规跨阶段只读访问)
  const renderBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });

  const renderBindGroup = device.createBindGroup({
    layout: renderBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
    ],
  });

  // 预乘 Alpha 混合模式：实现火焰加色发光与浓烟半透明遮挡共存
  const premultipliedBlend: GPUBlendState = {
    color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  };

  const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
    vertex: { module: shaderModule, entryPoint: "vs_render" },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_render",
      targets: [{ format, blend: premultipliedBlend }],
    },
    primitive: { topology: "triangle-list" },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
  });

  // ----------------------------------------------------
  // 4. GUI 控制与交互
  // ----------------------------------------------------
  const state = {
    flameHeight: 1.2,
    smokeDensity: 1.0,
    windX: 0.35,
    windZ: 0.0,
    turbulence: 0.45,
    autoRotate: 1,
  };

  gui.addTextInfo("🔥 <b>Compute Shader 烈焰与浓烟系统:</b><br>• <b>GPU 并行演算</b>: 8192 颗粒子全程在 GPU 端解算。<br>• <b>预乘混合特效</b>: 单次 Pass 融合自发光焰火与吸光浓烟。");
  gui.add(state, "flameHeight", 0.5, 3.0, 0.1).name("火焰蹿升高度");
  gui.add(state, "smokeDensity", 0.0, 2.0, 0.1).name("燃烧浓烟浓度");
  gui.add(state, "windX", -2.0, 2.0, 0.05).name("横向风速 (X)");
  gui.add(state, "windZ", -2.0, 2.0, 0.05).name("纵向风速 (Z)");
  gui.add(state, "turbulence", 0.0, 1.2, 0.05).name("热空气湍流");
  gui.add(state, "autoRotate", 0, 1, 1).name("相机自动旋转");

  const camera = { distance: 9.0, phi: 18, theta: 30 };
  let isDragging = false, lastX = 0, lastY = 0;

  canvas.addEventListener("pointerdown", (e) => {
    isDragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.4;
    camera.phi = Math.max(-85, Math.min(85, camera.phi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY;
  });
  canvas.addEventListener("pointerup", (e) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.distance = Math.max(3.0, Math.min(30.0, camera.distance + e.deltaY * 0.015));
  }, { passive: false });

  // ----------------------------------------------------
  // 5. 渲染循环 (Compute -> Render)
  // ----------------------------------------------------
  let animId: number;
  let depthTexture: GPUTexture | null = null;
  const uniformData = new Float32Array(32);
  let lastTimestamp = performance.now();

  function frame(timestamp: number) {
    const dt = Math.min((timestamp - lastTimestamp) * 0.001, 0.033);
    lastTimestamp = timestamp;

    if (!depthTexture || depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    if (state.autoRotate) camera.theta += 0.25;

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.distance * Math.cos(radPhi) * Math.sin(radTheta),
      Math.max(0.5, camera.distance * Math.sin(radPhi)),
      camera.distance * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    // 聚焦于火焰中心偏上位置
    const camView = Math3D.lookAt(eye, [0, 1.8, 0], [0, 1, 0]);
    const camProj = Math3D.perspective((45 * Math.PI) / 180, (canvas.width || 800) / (canvas.height || 600), 0.1, 100.0);
    const camViewProj = Math3D.multiply(camProj, camView);

    // 更新 Uniform
    uniformData.set(camViewProj, 0);                 // 0~15: viewProj
    uniformData.set([eye[0], eye[1], eye[2], 1.0], 16); // 16~19: camPos
    uniformData[20] = dt;                            // params.x: deltaTime
    uniformData[21] = timestamp * 0.001;             // params.y: totalTime
    uniformData[22] = state.flameHeight;            // params.z: flameHeight
    uniformData[23] = state.smokeDensity;           // params.w: smokeDensity
    uniformData[24] = state.windX;                   // dynamics.x: windX
    uniformData[25] = state.windZ;                   // dynamics.y: windZ
    uniformData[26] = state.turbulence;              // dynamics.z: turbulence
    uniformData[27] = PARTICLE_COUNT;                // dynamics.w: count

    device.queue.writeBuffer(uniformBuffer, 0, uniformData as unknown as BufferSource);

    const encoder = device.createCommandEncoder();

    // 步骤 1: Compute Pass - 并行推进粒子物理演化
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 64));
    computePass.end();

    // 步骤 2: Render Pass - 渲染混合后的火焰与浓烟
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        // 极暗环境衬托高对比烈焰与微透浓烟
        clearValue: { r: 0.015, g: 0.015, b: 0.02, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    // 绘制 6 个顶点 (无顶点缓冲四边形) x PARTICLE_COUNT 实例
    renderPass.draw(6, PARTICLE_COUNT, 0, 0);

    renderPass.end();
    device.queue.submit([encoder.finish()]);

    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  // ----------------------------------------------------
  // 6. 销毁与资源清理
  // ----------------------------------------------------
  return () => {
    cancelAnimationFrame(animId);
    uniformBuffer.destroy();
    particleBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}