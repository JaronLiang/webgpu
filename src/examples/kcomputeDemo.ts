import type { SimpleGUI } from "../utils/gui";

// ======================== 1. KCompute 核心架构 ========================
export class Tensor {
  public buffer: GPUBuffer;
  public size: number;
  public byteLength: number;
  private device: GPUDevice;

  constructor(device: GPUDevice, dataOrSize: Float32Array | Uint32Array | number) {
    this.device = device;
    const isArray = typeof dataOrSize !== "number";
    this.size = isArray ? dataOrSize.length : dataOrSize;
    this.byteLength = Math.max(16, this.size * 4);

    this.buffer = device.createBuffer({
      size: this.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: isArray,
    });

    if (isArray) {
      if (dataOrSize instanceof Uint32Array) {
        new Uint32Array(this.buffer.getMappedRange()).set(dataOrSize);
      } else {
        new Float32Array(this.buffer.getMappedRange()).set(dataOrSize);
      }
      this.buffer.unmap();
    }
  }

  // 异步回读 GPU 内存至 CPU
  async toCPU(): Promise<Float32Array> {
    const readBuffer = this.device.createBuffer({
      size: this.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this.buffer, 0, readBuffer, 0, this.byteLength);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();
    readBuffer.destroy();
    return result;
  }

  destroy() {
    this.buffer.destroy();
  }
}

export class KCompute {
  public device: GPUDevice;
  private pipelineCache = new Map<string, GPUComputePipeline>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  tensor(dataOrSize: Float32Array | Uint32Array | number): Tensor {
    return new Tensor(this.device, dataOrSize);
  }

  run(config: {
    shader: string;
    inputs: (Tensor | GPUBuffer)[];
    uniformBuffer?: GPUBuffer;
    workgroups: [number, number?, number?];
  }) {
    let pipeline = this.pipelineCache.get(config.shader);
    if (!pipeline) {
      pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: {
          module: this.device.createShaderModule({ code: config.shader }),
          entryPoint: "main",
        },
      });
      this.pipelineCache.set(config.shader, pipeline);
    }

    const bindGroupEntries: GPUBindGroupEntry[] = [];
    let bindingIdx = 0;

    if (config.uniformBuffer) {
      bindGroupEntries.push({
        binding: bindingIdx++,
        resource: { buffer: config.uniformBuffer },
      });
    }

    for (const item of config.inputs) {
      const buffer = item instanceof Tensor ? item.buffer : item;
      bindGroupEntries.push({
        binding: bindingIdx++,
        resource: { buffer },
      });
    }

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindGroupEntries,
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      config.workgroups[0],
      config.workgroups[1] || 1,
      config.workgroups[2] || 1
    );
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

// ======================== 2. 算子与可视化展示 ========================
export function runComputePro(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  const kc = new KCompute(device);

  // 1. 创建高性能科技感悬浮终端
  const term = document.createElement("div");
  term.style.cssText = `
    position: absolute; left: 20px; bottom: 20px; width: 620px; max-height: 320px;
    background: rgba(11, 15, 25, 0.92); color: #38bdf8; font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
    font-size: 12px; line-height: 1.6; padding: 16px; border-radius: 12px;
    border: 1px solid rgba(56, 189, 248, 0.25); overflow-y: auto; pointer-events: auto; z-index: 100;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6); backdrop-filter: blur(8px);
  `;
  term.innerHTML = `
    <div style="color:#f59e0b;font-weight:bold;margin-bottom:6px;border-bottom:1px solid #1e293b;padding-bottom:4px;">
      ⚡ WebGPU KCompute 并行计算引擎就绪 | GPU: ${device.label || "Hardware Accelerated"}
    </div>
    <div style="color:#94a3b8;">• 点击右上角算子按钮触发并行计算与硬件基准性能测评</div>
  `;
  canvas.parentElement?.appendChild(term);

  function log(msg: string) {
    const item = document.createElement("div");
    item.style.marginTop = "8px";
    item.innerHTML = msg;
    term.appendChild(item);
    term.scrollTop = term.scrollHeight;
  }

  // 2. 高精度 Turbo 科学热力图着色器
  const renderShader = `
    struct VOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };
    @vertex fn vs(@builtin(vertex_index) vid: u32) -> VOut {
      var p = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
      );
      var o: VOut;
      o.pos = vec4f(p[vid], 0.0, 1.0);
      o.uv = (p[vid] + 1.0) * 0.5;
      return o;
    }

    @group(0) @binding(0) var<storage, read> data: array<f32>;
    @group(0) @binding(1) var<uniform> dim: vec2f;

    // 经典 Turbo 热力色谱算法 (避免单调蓝/白过曝)
    fn turboColormap(x: f32) -> vec3f {
      let t = clamp(x, 0.0, 1.0);
      let r = 0.1357 + t * ( 4.6154 + t * ( -42.66 + t * ( 132.13 + t * ( -152.94 + t * 59.28 ))));
      let g = 0.0914 + t * ( 2.1942 + t * (   4.84 + t * ( -14.18 + t * (    4.27 + t *  2.83 ))));
      let b = 0.1067 + t * (12.6419 + t * ( -60.58 + t * ( 110.36 + t * (  -89.90 + t * 27.35 ))));
      return clamp(vec3f(r, g, b), vec3f(0.0), vec3f(1.0));
    }

    @fragment fn fs(in: VOut) -> @location(0) vec4f {
      let x = u32(clamp(in.uv.x * dim.x, 0.0, dim.x - 1.0));
      let y = u32(clamp(in.uv.y * dim.y, 0.0, dim.y - 1.0));
      let val = data[y * u32(dim.x) + x];

      let color = turboColormap(val);
      return vec4f(color, 1.0);
    }
  `;
  const renderModule = device.createShaderModule({ code: renderShader });
  const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: renderModule, entryPoint: "vs" },
    fragment: { module: renderModule, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const renderDimUBO = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const VIS_SIZE = 512;
  const visTensor = kc.tensor(VIS_SIZE * VIS_SIZE);
  device.queue.writeBuffer(renderDimUBO, 0, new Float32Array([VIS_SIZE, VIS_SIZE, 0, 0]));

  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: visTensor.buffer } },
      { binding: 1, resource: { buffer: renderDimUBO } },
    ],
  });

  function drawCanvas() {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.01, g: 0.02, b: 0.04, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(renderPipeline);
    pass.setBindGroup(0, renderBindGroup);
    pass.draw(6);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  // ======================== 3. WGSL 算子实现 ========================
  // 算子 A: 2D 复杂波动与干涉场热力计算 (修正类型对齐)
  const waveShader = `
    struct Params {
      dim: f32,
      time: f32,
      _p1: f32,
      _p2: f32,
    };
    @group(0) @binding(0) var<uniform> p: Params;
    @group(0) @binding(1) var<storage, read_write> outMap: array<f32>;

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
      let uDim = u32(p.dim);
      if (gid.x >= uDim || gid.y >= uDim) { return; }

      let uv = (vec2f(gid.xy) / p.dim - 0.5) * 12.0;

      // 三源动态相干波干涉场模拟
      let src1 = vec2f(sin(p.time * 1.2) * 3.0, cos(p.time * 0.9) * 3.0);
      let src2 = vec2f(cos(p.time * 0.8) * 4.0, sin(p.time * 1.1) * 2.5);
      let src3 = vec2f(0.0, 0.0);

      let d1 = length(uv - src1);
      let d2 = length(uv - src2);
      let d3 = length(uv - src3);

      let w1 = sin(d1 * 3.5 - p.time * 5.0) / (d1 * 0.35 + 1.0);
      let w2 = sin(d2 * 4.0 - p.time * 6.5) / (d2 * 0.35 + 1.0);
      let w3 = cos(d3 * 2.5 - p.time * 4.0) / (d3 * 0.25 + 1.0);

      let field = (w1 + w2 + w3) * 0.5 + 0.5;
      outMap[gid.y * uDim + gid.x] = clamp(field, 0.0, 1.0);
    }
  `;

  // 复用固定 Uniform 缓冲，杜绝每帧动态分配造成的开销
  const waveUniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // 算子 B: 一维百万级向量并行 FMA 融合乘加 (y = scale * x^2 + bias)
  const vectorShader = `
    struct Params { count: u32, scale: f32, bias: f32, _pad: f32 };
    @group(0) @binding(0) var<uniform> p: Params;
    @group(0) @binding(1) var<storage, read> inputX: array<f32>;
    @group(0) @binding(2) var<storage, read_write> outputY: array<f32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
      let idx = gid.x;
      if (idx >= p.count) { return; }
      let x = inputX[idx];
      outputY[idx] = fma(x * p.scale, x, p.bias);
    }
  `;

  // 算子 C: 密集矩阵乘法 GEMM (C = A x B)
  const gemmShader = `
    struct Params { N: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
    @group(0) @binding(0) var<uniform> p: Params;
    @group(0) @binding(1) var<storage, read> A: array<f32>;
    @group(0) @binding(2) var<storage, read> B: array<f32>;
    @group(0) @binding(3) var<storage, read_write> C: array<f32>;

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
      let r = gid.y;
      let c = gid.x;
      let n = p.N;
      if (r >= n || c >= n) { return; }

      var acc: f32 = 0.0;
      let rOffset = r * n;
      for (var k = 0u; k < n; k = k + 1u) {
        acc += A[rOffset + k] * B[k * n + c];
      }
      C[rOffset + c] = acc;
    }
  `;

  // ======================== 4. 控制逻辑与高精度测评 ========================
  let isWaveSimActive = true;
  let waveTime = 0.0;

  async function testVectorActivation() {
    const N = 2_000_000;
    log(`⏳ <b>[1D 向量算子]</b> 正在生成 ${N.toLocaleString()} 个单精度浮点数...`);

    const cpuIn = new Float32Array(N);
    for (let i = 0; i < N; i++) cpuIn[i] = (i % 1000) * 0.001;

    // CPU 基准
    const c0 = performance.now();
    const cpuOut = new Float32Array(N);
    for (let i = 0; i < N; i++) cpuOut[i] = (cpuIn[i] * 2.5) * cpuIn[i] + 10.0;
    const cpuMs = performance.now() - c0;

    // GPU 执行
    const g0 = performance.now();
    const tensorX = kc.tensor(cpuIn);
    const tensorY = kc.tensor(N);

    const uniformBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const uniformData = new ArrayBuffer(16);
    new Uint32Array(uniformData, 0, 1)[0] = N;
    new Float32Array(uniformData, 4, 1)[0] = 2.5;
    new Float32Array(uniformData, 8, 1)[0] = 10.0;
    device.queue.writeBuffer(uniformBuf, 0, uniformData);

    const uploadMs = performance.now() - g0;

    const tKernel = performance.now();
    kc.run({
      shader: vectorShader,
      uniformBuffer: uniformBuf,
      inputs: [tensorX, tensorY],
      workgroups: [Math.ceil(N / 256)],
    });
    const kernelMs = performance.now() - tKernel;

    const tRead = performance.now();
    const gpuOut = await tensorY.toCPU();
    const readMs = performance.now() - tRead;
    const totalGpuMs = performance.now() - g0;

    tensorX.destroy();
    tensorY.destroy();
    uniformBuf.destroy();

    // 精度验证
    let maxDiff = 0;
    for (let i = 0; i < 2000; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(gpuOut[i] - cpuOut[i]));
    }

    const dataMB = (N * 4 * 2) / (1024 * 1024); // 读+写吞吐
    const bandwidthGBs = (dataMB / 1024) / (totalGpuMs / 1000);

    log(
      `🚀 <b style="color:#22c55e;">[向量完成] 规模: ${(N / 1e6).toFixed(1)}M Float32</b>\n` +
      `  • <b>JS CPU:</b> ${cpuMs.toFixed(2)} ms\n` +
      `  • <b>WebGPU 总用时:</b> <span style="color:#f59e0b;">${totalGpuMs.toFixed(2)} ms</span> ` +
      `[上传: ${uploadMs.toFixed(1)}ms | 计算: ${kernelMs.toFixed(1)}ms | 回读: ${readMs.toFixed(1)}ms]\n` +
      `  • <b>内存等效吞吐:</b> ${bandwidthGBs.toFixed(2)} GB/s | <b>精度绝对误差:</b> ${maxDiff.toExponential(3)} (精准)`
    );
  }

  async function testGEMM() {
    const N = 512;
    log(`⏳ <b>[GEMM 密集矩阵乘]</b> 分配并计算 ${N}x${N} (${(N*N).toLocaleString()} 元素, 复杂度 O(N³))...`);

    const total = N * N;
    const matA = new Float32Array(total);
    const matB = new Float32Array(total);
    for (let i = 0; i < total; i++) {
      matA[i] = (i % 17) * 0.05;
      matB[i] = (i % 13) * 0.04;
    }

    const g0 = performance.now();
    const tA = kc.tensor(matA);
    const tB = kc.tensor(matB);
    const tC = kc.tensor(total);

    const uniformBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([N, 0, 0, 0]));

    kc.run({
      shader: gemmShader,
      uniformBuffer: uniformBuf,
      inputs: [tA, tB, tC],
      workgroups: [Math.ceil(N / 16), Math.ceil(N / 16)],
    });

    const gpuResult = await tC.toCPU();
    const totalGpuMs = performance.now() - g0;

    // CPU 抽样校验中间行列
    const testR = 128, testC = 256;
    let cpuRef = 0;
    for (let k = 0; k < N; k++) {
      cpuRef += matA[testR * N + k] * matB[k * N + testC];
    }
    const gpuVal = gpuResult[testR * N + testC];

    tA.destroy();
    tB.destroy();
    tC.destroy();
    uniformBuf.destroy();

    // GFLOPs 计算: 2 * N^3 次浮点操作
    const gflops = (2 * Math.pow(N, 3) / (totalGpuMs / 1000)) / 1e9;

    log(
      `🔥 <b style="color:#ec4899;">[GEMM 完成] 规模: ${N}x${N}</b>\n` +
      `  • <b>GPU 耗时 (含回读):</b> <span style="color:#f59e0b;">${totalGpuMs.toFixed(2)} ms</span>\n` +
      `  • <b>有效浮点算力:</b> <span style="color:#10b981;font-weight:bold;">${gflops.toFixed(2)} GFLOPS</span>\n` +
      `  • <b>抽样校验 [${testR}, ${testC}]:</b> GPU=${gpuVal.toFixed(4)} | CPU=${cpuRef.toFixed(4)} (误差: ${Math.abs(gpuVal - cpuRef).toExponential(3)})`
    );
  }

  // ======================== 5. 解决右上角按钮 UI 错位 ========================
  // 在右上方挂载专用的 Glassmorphism 算子控制栏（彻底杜绝 SimpleGUI 识别为 Slider）
  const controlPanel = document.createElement("div");
  controlPanel.style.cssText = `
    position: absolute; right: 20px; top: 20px; z-index: 1000;
    display: flex; flex-direction: column; gap: 8px; width: 260px;
    background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.12); padding: 14px; border-radius: 12px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
  `;
  controlPanel.innerHTML = `
    <div style="font-size: 13px; font-weight: bold; color: #f8fafc; margin-bottom: 4px; display: flex; align-items: center; justify-content: space-between;">
      <span>⚡ KCompute 控制台</span>
      <span id="sim-status-badge" style="font-size: 10px; background: #059669; color: white; padding: 2px 6px; border-radius: 4px;">模拟中</span>
    </div>
  `;

  function createBtn(label: string, color: string, onClick: () => void) {
    const btn = document.createElement("button");
    btn.innerHTML = label;
    btn.style.cssText = `
      background: ${color}; color: #ffffff; border: none; outline: none;
      padding: 8px 12px; border-radius: 6px; font-size: 12px; font-weight: 600;
      cursor: pointer; text-align: left; transition: all 0.2s ease;
      box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    `;
    btn.onmouseover = () => (btn.style.filter = "brightness(1.15)");
    btn.onmouseout = () => (btn.style.filter = "brightness(1.0)");
    btn.onclick = onClick;
    return btn;
  }

  const btnVector = createBtn("▶ 运行 200万级向量算子 (FMA)", "linear-gradient(135deg, #0284c7, #2563eb)", testVectorActivation);
  const btnGEMM = createBtn("▶ 运行 512x512 GEMM 矩阵乘法", "linear-gradient(135deg, #7c3aed, #9333ea)", testGEMM);
  const btnToggle = createBtn("⏯ 切换热力图模拟", "linear-gradient(135deg, #475569, #334155)", () => {
    isWaveSimActive = !isWaveSimActive;
    const badge = document.getElementById("sim-status-badge");
    if (badge) {
      badge.innerText = isWaveSimActive ? "模拟中" : "已暂停";
      badge.style.background = isWaveSimActive ? "#059669" : "#dc2626";
    }
  });
  const btnClear = createBtn("🗑 清空调试终端", "linear-gradient(135deg, #1e293b, #0f172a)", () => {
    term.innerHTML = `<div style="color:#64748b;">⚡ 控制台输出已清空</div>`;
  });

  controlPanel.appendChild(btnVector);
  controlPanel.appendChild(btnGEMM);
  controlPanel.appendChild(btnToggle);
  controlPanel.appendChild(btnClear);
  canvas.parentElement?.appendChild(controlPanel);

  // ======================== 6. 实时热力图渲染帧循环 ========================
  let animId: number;

  function frame() {
    if (isWaveSimActive) {
      waveTime += 0.016;

      // 更新统一 Uniform 缓冲区 (dim, time)
      device.queue.writeBuffer(
        waveUniformBuffer,
        0,
        new Float32Array([VIS_SIZE, waveTime, 0, 0])
      );

      // 计算着色器更新干涉场 Tensor
      kc.run({
        shader: waveShader,
        uniformBuffer: waveUniformBuffer,
        inputs: [visTensor],
        workgroups: [Math.ceil(VIS_SIZE / 16), Math.ceil(VIS_SIZE / 16)],
      });
    }

    // 渲染着色器将 Tensor 映射为高动态科学热力图
    drawCanvas();
    animId = requestAnimationFrame(frame);
  }
  frame();

  // 初始启动执行一次基准验证
  setTimeout(testVectorActivation, 400);

  return () => {
    cancelAnimationFrame(animId);
    term.remove();
    controlPanel.remove();
    visTensor.destroy();
    renderDimUBO.destroy();
    waveUniformBuffer.destroy();
  };
}