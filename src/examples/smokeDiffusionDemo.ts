// src/examples/smokeDiffusionDemo.ts
import type { SimpleGUI } from "../utils/gui";

export async function runSmokeDiffusionDemo(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  const GRID_W = 256;
  const GRID_H = 256;
  const WORKGROUP_SIZE = 16;

  // =====================================================================
  // 1. WGSL Compute 着色器 (流体网格计算)
  // =====================================================================
  const computeShaderCode = `
    struct SimParams {
      gridSize: vec2f,
      dt: f32,
      diffusion: f32,
      viscosity: f32,
      dissipation: f32,
      buoyancy: f32,
      pad0: f32,
      mousePos: vec2f,
      mouseVel: vec2f,
      mouseDown: f32,
      pad1: f32,
      pad2: f32,
      pad3: f32,
    };

    @group(0) @binding(0) var<uniform> sim: SimParams;
    @group(0) @binding(1) var sSampler: sampler;
    @group(0) @binding(2) var tSrcVelocity: texture_2d<f32>;
    @group(0) @binding(3) var tDstVelocity: texture_storage_2d<rgba16float, write>;
    @group(0) @binding(4) var tSrcDensity: texture_2d<f32>;
    @group(0) @binding(5) var tDstDensity: texture_storage_2d<rgba16float, write>;

    // Pass 1: 外力注入 (底部连续烟源 + 热浮力 + 鼠标交互)
    @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
    fn cs_inject_forces(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= u32(sim.gridSize.x) || id.y >= u32(sim.gridSize.y)) { return; }
      let coord = vec2i(id.xy);
      let uv = (vec2f(id.xy) + 0.5) / sim.gridSize;

      var vel = textureLoad(tSrcVelocity, coord, 0).xy;
      var dens = textureLoad(tSrcDensity, coord, 0).r;

      // 1. 底部连续产生烟雾源
      let sourcePos = vec2f(0.5, 0.9);
      let distToSource = length((uv - sourcePos) * vec2f(1.0, 2.0));
      if (distToSource < 0.04) {
        dens = min(dens + 2.0 * sim.dt, 2.0);
        vel += vec2f(sin(sim.dt * 10.0 + uv.x * 20.0) * 0.1, -0.6) * sim.dt * 20.0;
      }

      // 2. 烟雾浮力 (向上)
      vel.y -= dens * sim.buoyancy * sim.dt;

      // 3. 鼠标交互
      if (sim.mouseDown > 0.5) {
        let mDist = length((uv - sim.mousePos) * vec2f(sim.gridSize.x / sim.gridSize.y, 1.0));
        if (mDist < 0.05) {
          let forceFactor = (1.0 - mDist / 0.05);
          vel += sim.mouseVel * forceFactor * 5.0;
          dens = min(dens + forceFactor * 1.5, 2.0);
        }
      }

      textureStore(tDstVelocity, coord, vec4f(vel, 0.0, 1.0));
      textureStore(tDstDensity, coord, vec4f(dens, 0.0, 0.0, 1.0));
    }

    // Pass 2: 雅可比迭代扩散
    @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
    fn cs_diffuse_density(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= u32(sim.gridSize.x) || id.y >= u32(sim.gridSize.y)) { return; }
      let coord = vec2i(id.xy);

      let alpha = (1.0 / (sim.diffusion * sim.dt + 1e-6));
      let beta = 4.0 + alpha;

      let l = textureLoad(tSrcDensity, max(coord - vec2i(1, 0), vec2i(0)), 0).r;
      let r = textureLoad(tSrcDensity, min(coord + vec2i(1, 0), vec2i(sim.gridSize) - 1), 0).r;
      let t = textureLoad(tSrcDensity, max(coord - vec2i(0, 1), vec2i(0)), 0).r;
      let b = textureLoad(tSrcDensity, min(coord + vec2i(0, 1), vec2i(sim.gridSize) - 1), 0).r;
      let c = textureLoad(tSrcDensity, coord, 0).r;

      let newDens = (l + r + t + b + alpha * c) / beta;
      textureStore(tDstDensity, coord, vec4f(newDens, 0.0, 0.0, 1.0));
    }

    // Pass 3: 半拉格朗日对流 (Advection)
    @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
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
  // 2. WGSL Render 着色器 (独立模块，隔离 bindings)
  // =====================================================================
  const renderShaderCode = `
    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) id: u32) -> VertexOut {
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
    fn fs_main(in: VertexOut) -> @location(0) vec4f {
      let d = textureSample(tDensityTexture, sRenderSampler, in.uv).r;

      // 墨水/火焰热度色彩映射阶梯
      let c0 = vec3f(0.02, 0.02, 0.04);
      let c1 = vec3f(0.1, 0.35, 0.85);
      let c2 = vec3f(1.0, 0.45, 0.1);
      let c3 = vec3f(1.0, 0.98, 0.9);

      var color = mix(c0, c1, clamp(d * 2.5, 0.0, 1.0));
      color = mix(color, c2, clamp((d - 0.4) * 2.5, 0.0, 1.0));
      color = mix(color, c3, clamp((d - 0.8) * 3.0, 0.0, 1.0));

      return vec4f(color, 1.0);
    }
  `;

  // =====================================================================
  // 3. 纹理、采样器与双缓冲配置
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
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_DST,
    });
  }

  const velTex = [createFieldTexture(), createFieldTexture()];
  const densTex = [createFieldTexture(), createFieldTexture()];

  // Uniform Buffer (16 个 float = 64 字节)
  const simUBO = device.createBuffer({
    size: 16 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =====================================================================
  // 4. 显式创建 BindGroupLayout 与 PipelineLayout (无 layout: "auto")
  // =====================================================================
  const computeModule = device.createShaderModule({ code: computeShaderCode });
  const renderModule = device.createShaderModule({ code: renderShaderCode });

  // 1. Compute BindGroupLayout
  const computeBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float" } },
    ],
  });

  const computePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [computeBindGroupLayout],
  });

  // 2. Render BindGroupLayout (严格匹配 renderShaderCode)
  // binding 0 -> sampler
  // binding 1 -> texture
  const renderBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
  });

  const renderPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [renderBindGroupLayout],
  });

  // 构建 Compute Pipelines
  const pipelineInject = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: { module: computeModule, entryPoint: "cs_inject_forces" },
  });

  const pipelineDiffuse = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: { module: computeModule, entryPoint: "cs_diffuse_density" },
  });

  const pipelineAdvect = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: { module: computeModule, entryPoint: "cs_advect" },
  });

  // 构建 Render Pipeline
  const renderPipeline = device.createRenderPipeline({
    layout: renderPipelineLayout,
    vertex: { module: renderModule, entryPoint: "vs_main" },
    fragment: { module: renderModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // =====================================================================
  // 5. 显式创建交替 Ping-Pong BindGroups
  // =====================================================================
  const bgAtoB = device.createBindGroup({
    layout: computeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: simUBO } },
      { binding: 1, resource: sLinear },
      { binding: 2, resource: velTex[0].createView() },
      { binding: 3, resource: velTex[1].createView() },
      { binding: 4, resource: densTex[0].createView() },
      { binding: 5, resource: densTex[1].createView() },
    ],
  });

  const bgBtoA = device.createBindGroup({
    layout: computeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: simUBO } },
      { binding: 1, resource: sLinear },
      { binding: 2, resource: velTex[1].createView() },
      { binding: 3, resource: velTex[0].createView() },
      { binding: 4, resource: densTex[1].createView() },
      { binding: 5, resource: densTex[0].createView() },
    ],
  });

  const renderBGA = device.createBindGroup({
    layout: renderBindGroupLayout,
    entries: [
      { binding: 0, resource: sLinear },
      { binding: 1, resource: densTex[0].createView() },
    ],
  });

  const renderBGB = device.createBindGroup({
    layout: renderBindGroupLayout,
    entries: [
      { binding: 0, resource: sLinear },
      { binding: 1, resource: densTex[1].createView() },
    ],
  });

  // =====================================================================
  // 6. 交互控制与 GUI
  // =====================================================================
  const params = {
    diffusion: 0.15,
    buoyancy: 0.35,
    dissipation: 0.992,
    iterations: 4,
  };

  gui.add(params, "diffusion", 0.0, 1.0, 0.01).name("烟雾扩散系数");
  gui.add(params, "buoyancy", 0.0, 1.0, 0.01).name("热空气浮力");
  gui.add(params, "dissipation", 0.95, 0.999, 0.001).name("留存/消散率");
  gui.add(params, "iterations", 1, 10, 1).name("扩散迭代轮数");
  gui.addTextInfo("💨 <b>GPU 烟雾扩散/流体动力学</b><br>在画布上按住鼠标拖动可施加气流与烟雾。");

  let isMouseDown = false;
  let mouseX = 0.5, mouseY = 0.5;
  let mouseVelX = 0, mouseVelY = 0;

  const onPointerMove = (e: PointerEvent) => {
    const rect = canvas.getBoundingClientRect();
    const currX = (e.clientX - rect.left) / rect.width;
    const currY = (e.clientY - rect.top) / rect.height;
    mouseVelX = currX - mouseX;
    mouseVelY = currY - mouseY;
    mouseX = currX;
    mouseY = currY;
  };
  const onPointerDown = (e: PointerEvent) => {
    isMouseDown = true;
    onPointerMove(e);
  };
  const onPointerUp = () => {
    isMouseDown = false;
    mouseVelX = 0;
    mouseVelY = 0;
  };

  canvas.addEventListener("pointerdown", onPointerDown);
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);

  // =====================================================================
  // 7. 主循环调度
  // =====================================================================
  let animId: number;
  let currentPingPong = 0;

  function frame() {
    const simData = new Float32Array(16);
    simData[0] = GRID_W;
    simData[1] = GRID_H;
    simData[2] = 0.016;
    simData[3] = params.diffusion;
    simData[4] = 0.0;
    simData[5] = params.dissipation;
    simData[6] = params.buoyancy;
    simData[7] = 0.0;
    simData[8] = mouseX;
    simData[9] = mouseY;
    simData[10] = mouseVelX;
    simData[11] = mouseVelY;
    simData[12] = isMouseDown ? 1.0 : 0.0;
    device.queue.writeBuffer(simUBO, 0, simData);

    mouseVelX *= 0.85;
    mouseVelY *= 0.85;

    const encoder = device.createCommandEncoder();
    const workX = Math.ceil(GRID_W / WORKGROUP_SIZE);
    const workY = Math.ceil(GRID_H / WORKGROUP_SIZE);

    // Pass 1: 外力/烟源注入
    {
      const cpass = encoder.beginComputePass();
      cpass.setPipeline(pipelineInject);
      cpass.setBindGroup(0, currentPingPong === 0 ? bgAtoB : bgBtoA);
      cpass.dispatchWorkgroups(workX, workY);
      cpass.end();
      currentPingPong = 1 - currentPingPong;
    }

    // Pass 2: 雅可比迭代扩散
    for (let i = 0; i < params.iterations; i++) {
      const cpass = encoder.beginComputePass();
      cpass.setPipeline(pipelineDiffuse);
      cpass.setBindGroup(0, currentPingPong === 0 ? bgAtoB : bgBtoA);
      cpass.dispatchWorkgroups(workX, workY);
      cpass.end();
      currentPingPong = 1 - currentPingPong;
    }

    // Pass 3: 半拉格朗日对流平流
    {
      const cpass = encoder.beginComputePass();
      cpass.setPipeline(pipelineAdvect);
      cpass.setBindGroup(0, currentPingPong === 0 ? bgAtoB : bgBtoA);
      cpass.dispatchWorkgroups(workX, workY);
      cpass.end();
      currentPingPong = 1 - currentPingPong;
    }

    // Pass 4: 呈现渲染输出
    {
      const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          storeOp: "store",
        }],
      });
      renderPass.setPipeline(renderPipeline);
      // 传递当前最新数据的纹理 BindGroup
      renderPass.setBindGroup(0, currentPingPong === 0 ? renderBGA : renderBGB);
      renderPass.draw(3);
      renderPass.end();
    }

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);

    velTex[0].destroy();
    velTex[1].destroy();
    densTex[0].destroy();
    densTex[1].destroy();
    simUBO.destroy();
  };
}