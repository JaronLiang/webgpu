// src/examples/smokeFireExplosionDemo.ts
import type { SimpleGUI } from "../utils/gui";

export async function runSmokeFireExplosionDemo(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  const GRID_W = 256;
  const GRID_H = 256;
  const COMPUTE_WG_SIZE = 16;
  const PARTICLE_COUNT = 16384;
  const PARTICLE_WG_SIZE = 64;

  // =====================================================================
  // 1. WGSL Compute: 烟雾流体网格计算
  // =====================================================================
  const smokeComputeShader = `
    struct SimParams {
      gridSize: vec2f,
      dt: f32,
      diffusion: f32,
      viscosity: f32,
      dissipation: f32,
      buoyancy: f32,
      time: f32,
      mousePos: vec2f,
      mouseVel: vec2f,
      triggerExplosion: f32,
      centerBurnIntensity: f32,
      pad0: f32,
      pad1: f32,
    };

    @group(0) @binding(0) var<uniform> sim: SimParams;
    @group(0) @binding(1) var sSampler: sampler;
    @group(0) @binding(2) var tSrcVelocity: texture_2d<f32>;
    @group(0) @binding(3) var tDstVelocity: texture_storage_2d<rgba16float, write>;
    @group(0) @binding(4) var tSrcDensity: texture_2d<f32>;
    @group(0) @binding(5) var tDstDensity: texture_storage_2d<rgba16float, write>;

    fn hash2(p: vec2f) -> f32 {
      return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453123);
    }

    @compute @workgroup_size(${COMPUTE_WG_SIZE}, ${COMPUTE_WG_SIZE})
    fn cs_inject_forces(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= u32(sim.gridSize.x) || id.y >= u32(sim.gridSize.y)) { return; }
      let coord = vec2i(id.xy);
      let uv = (vec2f(id.xy) + 0.5) / sim.gridSize;

      var vel = textureLoad(tSrcVelocity, coord, 0).xy;
      var dens = textureLoad(tSrcDensity, coord, 0).r;

      // 1. 屏幕正中心 (0.5, 0.55) 烟雾与热源
      let center = vec2f(0.5, 0.55);
      let distToCenter = length((uv - center) * vec2f(1.0, 1.4));
      if (distToCenter < 0.08) {
        let factor = (1.0 - distToCenter / 0.08);
        dens = min(dens + factor * sim.centerBurnIntensity * sim.dt * 4.0, 2.5);
        let noiseOffset = hash2(uv * 10.0 + sim.time) * 0.4 - 0.2;
        vel += vec2f(noiseOffset, -0.85) * factor * sim.dt * 20.0;
      }

      // 2. 向上热浮力
      vel.y -= dens * sim.buoyancy * sim.dt;

      // 3. 爆炸冲击推力
      if (sim.triggerExplosion > 0.5) {
        let diff = uv - sim.mousePos;
        let d = length(diff);
        if (d < 0.25 && d > 1e-4) {
          let push = normalize(diff) * (1.0 - d / 0.25) * 8.0;
          vel += push;
          dens = min(dens + (1.0 - d / 0.25) * 1.5, 3.0);
        }
      }

      textureStore(tDstVelocity, coord, vec4f(vel, 0.0, 1.0));
      textureStore(tDstDensity, coord, vec4f(dens, 0.0, 0.0, 1.0));
    }

    @compute @workgroup_size(${COMPUTE_WG_SIZE}, ${COMPUTE_WG_SIZE})
    fn cs_diffuse_density(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= u32(sim.gridSize.x) || id.y >= u32(sim.gridSize.y)) { return; }
      let coord = vec2i(id.xy);

      let alpha = 1.0 / (sim.diffusion * sim.dt + 1e-5);
      let beta = 4.0 + alpha;

      let l = textureLoad(tSrcDensity, max(coord - vec2i(1, 0), vec2i(0)), 0).r;
      let r = textureLoad(tSrcDensity, min(coord + vec2i(1, 0), vec2i(sim.gridSize) - 1), 0).r;
      let t = textureLoad(tSrcDensity, max(coord - vec2i(0, 1), vec2i(0)), 0).r;
      let b = textureLoad(tSrcDensity, min(coord + vec2i(0, 1), vec2i(sim.gridSize) - 1), 0).r;
      let c = textureLoad(tSrcDensity, coord, 0).r;

      let newDens = (l + r + t + b + alpha * c) / beta;
      textureStore(tDstDensity, coord, vec4f(newDens, 0.0, 0.0, 1.0));
    }

    @compute @workgroup_size(${COMPUTE_WG_SIZE}, ${COMPUTE_WG_SIZE})
    fn cs_advect(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= u32(sim.gridSize.x) || id.y >= u32(sim.gridSize.y)) { return; }
      let coord = vec2i(id.xy);
      let uv = (vec2f(id.xy) + 0.5) / sim.gridSize;

      let vel = textureLoad(tSrcVelocity, coord, 0).xy;
      let backtracedUV = uv - vel * (sim.dt / sim.gridSize);

      let advectedVel = textureSampleLevel(tSrcVelocity, sSampler, backtracedUV, 0.0).xy;
      var advectedDens = textureSampleLevel(tSrcDensity, sSampler, backtracedUV, 0.0).r;

      advectedDens *= sim.dissipation;

      textureStore(tDstVelocity, coord, vec4f(advectedVel, 0.0, 1.0));
      textureStore(tDstDensity, coord, vec4f(advectedDens, 0.0, 0.0, 1.0));
    }
  `;

  // =====================================================================
  // 2. WGSL Compute: 火焰与爆炸粒子 (48 字节严格对齐)
  // =====================================================================
  const particleComputeShader = `
    struct Particle {
      pos: vec2f,
      vel: vec2f,
      life: f32,
      maxLife: f32,
      size: f32,
      pType: f32,
    };

    // 严格 48 字节对齐 (3 个 vec4 = 12 个 32位 float)
    struct ParticleParams {
      // vec4 1 (offset 0..16)
      dt: f32,
      time: f32,
      particleCount: u32,
      triggerExplosion: f32,
      // vec4 2 (offset 16..32)
      explodePos: vec2f,
      pad0: f32,
      pad1: f32,
      // vec4 3 (offset 32..48)
      pad2: f32,
      pad3: f32,
      pad4: f32,
      pad5: f32,
    };

    @group(0) @binding(0) var<uniform> pp: ParticleParams;
    @group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

    fn pcg_hash(v: u32) -> u32 {
      var state = v * 747796405u + 2891336453u;
      var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
      return (word >> 22u) ^ word;
    }

    fn randFloat(seed: u32) -> f32 {
      return f32(pcg_hash(seed) & 0x00FFFFFFu) / f32(0x01000000);
    }

    @compute @workgroup_size(${PARTICLE_WG_SIZE})
    fn cs_update_particles(@builtin(global_invocation_id) id: vec3u) {
      let idx = id.x;
      if (idx >= pp.particleCount) { return; }

      var p = particles[idx];
      let seed = idx + u32(pp.time * 1000.0);

      // 引爆逻辑
      if (pp.triggerExplosion > 0.5 && idx < (pp.particleCount / 2u)) {
        let angle = randFloat(seed) * 6.283185;
        let speed = 0.5 + randFloat(seed + 1u) * 1.8;
        p.pos = pp.explodePos;
        p.vel = vec2f(cos(angle), sin(angle)) * speed;
        p.life = 1.0;
        p.maxLife = 0.4 + randFloat(seed + 2u) * 0.8;
        p.size = 0.02 + randFloat(seed + 3u) * 0.025;
        p.pType = 1.0;
      } else {
        p.life -= pp.dt / max(p.maxLife, 0.01);

        if (p.life <= 0.0) {
          // 在中心持续循环重生为火焰粒子
          let rAngle = randFloat(seed) * 6.283185;
          let rDist = randFloat(seed + 1u) * 0.04;
          p.pos = vec2f(0.5, 0.56) + vec2f(cos(rAngle), sin(rAngle)) * rDist;
          
          let vx = (randFloat(seed + 2u) - 0.5) * 0.15;
          let vy = -0.35 - randFloat(seed + 3u) * 0.45;
          p.vel = vec2f(vx, vy);

          p.maxLife = 0.5 + randFloat(seed + 4u) * 0.7;
          p.life = 1.0;
          p.size = 0.025 + randFloat(seed + 5u) * 0.035;
          p.pType = 0.0;
        } else {
          if (p.pType > 0.5) {
            p.vel.y += 0.9 * pp.dt;
            p.vel *= 0.96;
          } else {
            p.vel.y -= 0.45 * pp.dt;
            p.vel.x += (randFloat(seed) - 0.5) * 0.25 * pp.dt;
          }
          p.pos += p.vel * pp.dt;
        }
      }

      particles[idx] = p;
    }
  `;

  // =====================================================================
  // 3. WGSL Render: 渲染着色器
  // =====================================================================
  const renderShaderCode = `
    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_smoke(@builtin(vertex_index) id: u32) -> VertexOut {
      var out: VertexOut;
      let x = f32((id << 1u) & 2u);
      let y = f32(id & 2u);
      out.uv = vec2f(x * 0.5, 1.0 - y * 0.5);
      out.pos = vec4f(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
      return out;
    }

    @group(0) @binding(0) var sRenderSampler: sampler;
    @group(0) @binding(1) var tDensityTexture: texture_2d<f32>;

    @fragment
    fn fs_smoke(in: VertexOut) -> @location(0) vec4f {
      let d = textureSample(tDensityTexture, sRenderSampler, in.uv).r;
      let bg = vec3f(0.04, 0.04, 0.06);
      let smokeColor = vec3f(0.18, 0.17, 0.22);
      let innerGlow = vec3f(0.95, 0.4, 0.08);

      var col = mix(bg, smokeColor, clamp(d * 1.5, 0.0, 0.85));
      col += innerGlow * pow(clamp(d - 0.25, 0.0, 2.0), 2.2) * 0.7;
      return vec4f(col, 1.0);
    }

    struct Particle {
      pos: vec2f,
      vel: vec2f,
      life: f32,
      maxLife: f32,
      size: f32,
      pType: f32,
    };

    @group(0) @binding(0) var<storage, read> particles: array<Particle>;

    struct ParticleVOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
      @location(1) life: f32,
      @location(2) pType: f32,
    };

    @vertex
    fn vs_particle(
      @builtin(vertex_index) vIdx: u32,
      @builtin(instance_index) iIdx: u32
    ) -> ParticleVOut {
      var quadOffset = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
      );

      let p = particles[iIdx];
      let offset = quadOffset[vIdx] * p.size * (0.35 + 0.65 * p.life);
      
      let clipCenter = p.pos * 2.0 - 1.0;
      let finalPos = vec2f(clipCenter.x, -clipCenter.y) + offset;

      var out: ParticleVOut;
      out.pos = vec4f(finalPos, 0.0, 1.0);
      out.uv = quadOffset[vIdx];
      out.life = p.life;
      out.pType = p.pType;
      return out;
    }

    @fragment
    fn fs_particle(in: ParticleVOut) -> @location(0) vec4f {
      let r2 = dot(in.uv, in.uv);
      if (r2 > 1.0) { discard; }
      
      let intensity = exp(-r2 * 3.5) * in.life;
      var heatColor: vec3f;

      if (in.pType > 0.5) {
        heatColor = mix(vec3f(1.0, 0.4, 0.05), vec3f(2.0, 1.6, 1.0), in.life);
      } else {
        let t = in.life;
        let c0 = vec3f(0.85, 0.05, 0.0);
        let c1 = vec3f(1.0, 0.55, 0.02);
        let c2 = vec3f(1.6, 1.4, 0.9);
        heatColor = mix(c0, c1, clamp(t * 1.8, 0.0, 1.0));
        heatColor = mix(heatColor, c2, clamp((t - 0.55) * 2.5, 0.0, 1.0));
      }

      return vec4f(heatColor * intensity * 1.8, 1.0);
    }
  `;

  // =====================================================================
  // 4. 显式创建 Buffers、BindGroupLayout 与 PipelineLayout
  // =====================================================================
  const sLinear = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  function createFieldTexture(): GPUTexture {
    return device.createTexture({
      size: [GRID_W, GRID_H],
      format: "rgba16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    });
  }

  const velTex = [createFieldTexture(), createFieldTexture()];
  const densTex = [createFieldTexture(), createFieldTexture()];

  // Smoke UBO: 16 个 float = 64 字节
  const smokeUBO = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // 【核心修复】：Particle UBO 明确分配 48 字节，满足对齐要求
  const particleUBO = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // 初始化粒子数据，使首帧就拥有分布在中心的火苗
  const initialParticleData = new Float32Array(PARTICLE_COUNT * 8);
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const o = i * 8;
    initialParticleData[o + 0] = 0.5 + (Math.random() - 0.5) * 0.05; // pos.x
    initialParticleData[o + 1] = 0.56 + (Math.random() - 0.5) * 0.05; // pos.y
    initialParticleData[o + 2] = (Math.random() - 0.5) * 0.15;        // vel.x
    initialParticleData[o + 3] = -0.3 - Math.random() * 0.4;          // vel.y
    initialParticleData[o + 4] = Math.random();                       // life
    initialParticleData[o + 5] = 0.6 + Math.random() * 0.8;           // maxLife
    initialParticleData[o + 6] = 0.02 + Math.random() * 0.03;         // size
    initialParticleData[o + 7] = 0.0;                                 // pType
  }

  const particleBuffer = device.createBuffer({
    size: initialParticleData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(particleBuffer, 0, initialParticleData);

  // Layouts
  const smokeComputeBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", minBindingSize: 64 } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float" } },
    ],
  });
  const smokeComputeLayout = device.createPipelineLayout({ bindGroupLayouts: [smokeComputeBGL] });

  const particleComputeBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", minBindingSize: 48 } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const particleComputeLayout = device.createPipelineLayout({ bindGroupLayouts: [particleComputeBGL] });

  const smokeRenderBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
  });
  const smokeRenderLayout = device.createPipelineLayout({ bindGroupLayouts: [smokeRenderBGL] });

  const particleRenderBGL = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }],
  });
  const particleRenderLayout = device.createPipelineLayout({ bindGroupLayouts: [particleRenderBGL] });

  // =====================================================================
  // 5. 创建 Pipelines
  // =====================================================================
  const smokeModule = device.createShaderModule({ code: smokeComputeShader });
  const particleModule = device.createShaderModule({ code: particleComputeShader });
  const renderModule = device.createShaderModule({ code: renderShaderCode });

  const pipeInject = device.createComputePipeline({ layout: smokeComputeLayout, compute: { module: smokeModule, entryPoint: "cs_inject_forces" } });
  const pipeDiffuse = device.createComputePipeline({ layout: smokeComputeLayout, compute: { module: smokeModule, entryPoint: "cs_diffuse_density" } });
  const pipeAdvect = device.createComputePipeline({ layout: smokeComputeLayout, compute: { module: smokeModule, entryPoint: "cs_advect" } });

  const pipeUpdateParticles = device.createComputePipeline({
    layout: particleComputeLayout,
    compute: { module: particleModule, entryPoint: "cs_update_particles" },
  });

  const pipeRenderSmoke = device.createRenderPipeline({
    layout: smokeRenderLayout,
    vertex: { module: renderModule, entryPoint: "vs_smoke" },
    fragment: { module: renderModule, entryPoint: "fs_smoke", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const pipeRenderParticles = device.createRenderPipeline({
    layout: particleRenderLayout,
    vertex: { module: renderModule, entryPoint: "vs_particle" },
    fragment: {
      module: renderModule,
      entryPoint: "fs_particle",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
          alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
        },
      }],
    },
    primitive: { topology: "triangle-list" },
  });

  // BindGroups
  const bgAtoB = device.createBindGroup({
    layout: smokeComputeBGL,
    entries: [
      { binding: 0, resource: { buffer: smokeUBO } },
      { binding: 1, resource: sLinear },
      { binding: 2, resource: velTex[0].createView() },
      { binding: 3, resource: velTex[1].createView() },
      { binding: 4, resource: densTex[0].createView() },
      { binding: 5, resource: densTex[1].createView() },
    ],
  });

  const bgBtoA = device.createBindGroup({
    layout: smokeComputeBGL,
    entries: [
      { binding: 0, resource: { buffer: smokeUBO } },
      { binding: 1, resource: sLinear },
      { binding: 2, resource: velTex[1].createView() },
      { binding: 3, resource: velTex[0].createView() },
      { binding: 4, resource: densTex[1].createView() },
      { binding: 5, resource: densTex[0].createView() },
    ],
  });

  const particleComputeBG = device.createBindGroup({
    layout: particleComputeBGL,
    entries: [
      { binding: 0, resource: { buffer: particleUBO } },
      { binding: 1, resource: { buffer: particleBuffer } },
    ],
  });

  const smokeRenderBGA = device.createBindGroup({
    layout: smokeRenderBGL,
    entries: [{ binding: 0, resource: sLinear }, { binding: 1, resource: densTex[0].createView() }],
  });
  const smokeRenderBGB = device.createBindGroup({
    layout: smokeRenderBGL,
    entries: [{ binding: 0, resource: sLinear }, { binding: 1, resource: densTex[1].createView() }],
  });

  const particleRenderBG = device.createBindGroup({
    layout: particleRenderBGL,
    entries: [{ binding: 0, resource: { buffer: particleBuffer } }],
  });

  // =====================================================================
  // 6. GUI 与鼠标点击交互
  // =====================================================================
  const params = {
    diffusion: 0.12,
    buoyancy: 0.38,
    dissipation: 0.993,
    centerBurnIntensity: 1.0,
    explode: () => triggerExplosionAt(0.5, 0.55),
  };

  gui.add(params, "diffusion", 0.0, 0.5, 0.01).name("烟雾扩散速率");
  gui.add(params, "buoyancy", 0.0, 1.0, 0.01).name("热浮力上升");
  gui.add(params, "dissipation", 0.98, 0.999, 0.001).name("消散延缓");
  gui.add(params, "centerBurnIntensity", 0.0, 3.0, 0.1).name("中心火焰功率");
  gui.addButton("💥 引爆中心炸弹", params.explode);
  gui.addTextInfo("🔥 <b>中心升腾烟雾 + 16384 发光粒子</b><br>点击屏幕任意位置可实时在该处引爆火花与冲击波。");

  let explosionRequested = false;
  let explodeCoord = [0.5, 0.55];

  function triggerExplosionAt(x: number, y: number) {
    explosionRequested = true;
    explodeCoord = [x, y];
  }

  const onPointerDown = (e: PointerEvent) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    triggerExplosionAt(x, y);
  };
  canvas.addEventListener("pointerdown", onPointerDown);

  // =====================================================================
  // 7. 主循环调度
  // =====================================================================
  let animId: number;
  let currentPingPong = 0;
  let clockTime = 0;

  function frame() {
    clockTime += 0.016;

    // 1. 上传 Smoke Uniforms (16 个 float = 64 字节)
    const sData = new Float32Array(16);
    sData[0] = GRID_W;
    sData[1] = GRID_H;
    sData[2] = 0.016;
    sData[3] = params.diffusion;
    sData[4] = 0.0;
    sData[5] = params.dissipation;
    sData[6] = params.buoyancy;
    sData[7] = clockTime;
    sData[8] = explodeCoord[0];
    sData[9] = explodeCoord[1];
    sData[10] = 0.0;
    sData[11] = 0.0;
    sData[12] = explosionRequested ? 1.0 : 0.0;
    sData[13] = params.centerBurnIntensity;
    device.queue.writeBuffer(smokeUBO, 0, sData);

    // 2. 上传 Particle Uniforms (12 个 float = 48 字节)
    const pData = new Float32Array(12);
    pData[0] = 0.016;
    pData[1] = clockTime;
    new Uint32Array(pData.buffer)[2] = PARTICLE_COUNT;
    pData[3] = explosionRequested ? 1.0 : 0.0;
    pData[4] = explodeCoord[0];
    pData[5] = explodeCoord[1];
    device.queue.writeBuffer(particleUBO, 0, pData);

    const encoder = device.createCommandEncoder();
    const workX = Math.ceil(GRID_W / COMPUTE_WG_SIZE);
    const workY = Math.ceil(GRID_H / COMPUTE_WG_SIZE);

    // Compute Pass: 烟雾与粒子推进
    {
      const cpass = encoder.beginComputePass();
      cpass.setPipeline(pipeInject);
      cpass.setBindGroup(0, currentPingPong === 0 ? bgAtoB : bgBtoA);
      cpass.dispatchWorkgroups(workX, workY);
      currentPingPong = 1 - currentPingPong;

      for (let i = 0; i < 3; i++) {
        cpass.setPipeline(pipeDiffuse);
        cpass.setBindGroup(0, currentPingPong === 0 ? bgAtoB : bgBtoA);
        cpass.dispatchWorkgroups(workX, workY);
        currentPingPong = 1 - currentPingPong;
      }

      cpass.setPipeline(pipeAdvect);
      cpass.setBindGroup(0, currentPingPong === 0 ? bgAtoB : bgBtoA);
      cpass.dispatchWorkgroups(workX, workY);
      currentPingPong = 1 - currentPingPong;

      // 更新粒子物理
      cpass.setPipeline(pipeUpdateParticles);
      cpass.setBindGroup(0, particleComputeBG);
      cpass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / PARTICLE_WG_SIZE));

      cpass.end();
    }

    // Render Pass: 烟雾底层 + 粒子加法混合顶层
    {
      const rpass = encoder.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0.03, g: 0.03, b: 0.04, a: 1.0 },
          storeOp: "store",
        }],
      });

      rpass.setPipeline(pipeRenderSmoke);
      rpass.setBindGroup(0, currentPingPong === 0 ? smokeRenderBGA : smokeRenderBGB);
      rpass.draw(3);

      rpass.setPipeline(pipeRenderParticles);
      rpass.setBindGroup(0, particleRenderBG);
      rpass.draw(6, PARTICLE_COUNT, 0, 0);

      rpass.end();
    }

    device.queue.submit([encoder.finish()]);

    explosionRequested = false;
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);

    velTex[0].destroy();
    velTex[1].destroy();
    densTex[0].destroy();
    densTex[1].destroy();
    smokeUBO.destroy();
    particleUBO.destroy();
    particleBuffer.destroy();
  };
}