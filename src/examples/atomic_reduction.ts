// src/examples/atomic_reduction.ts
import type { SimpleGUI } from "../utils/gui";

export function runAtomicReduction(
  device: GPUDevice, 
  context: GPUCanvasContext, 
  format: GPUTextureFormat, 
  canvas: HTMLCanvasElement, 
  gui: SimpleGUI
) {
  const log = (msg: string) => console.log(`[AtomicReduction] ${msg}`);
  log("正在初始化 WebGPU 并行原子归约示例...");

  // 1. 数据准备：1677 万个 32 位无符号整数 (约 67.1 MB)
  const elementCount = 16 * 1024 * 1024;
  
  // 2. 创建所需 GPU Buffers
  const inputBuffer = device.createBuffer({
    label: "Input Data Buffer",
    size: elementCount * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const outputBuffer = device.createBuffer({
    label: "Atomic Sum Output Buffer",
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const readBuffer = device.createBuffer({
    label: "Readback Staging Buffer",
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const uniformBuffer = device.createBuffer({
    label: "Uniform Params Buffer",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([elementCount, 0, 0, 0]));

  // 3. 计算着色器 (WGSL)：网格跨步循环 + 两级原子规约
  const computeShader = `
    struct Uniforms {
      count: u32,
      _pad1: u32, _pad2: u32, _pad3: u32,
    };

    @group(0) @binding(0) var<storage, read> data: array<u32>;
    @group(0) @binding(1) var<storage, read_write> globalSum: atomic<u32>;
    @group(0) @binding(2) var<uniform> uParams: Uniforms;

    var<workgroup> localSum: atomic<u32>;

    @compute @workgroup_size(256)
    fn main(
      @builtin(global_invocation_id) global_id: vec3u,
      @builtin(local_invocation_id) local_id: vec3u,
      @builtin(num_workgroups) num_workgroups: vec3u
    ) {
      // 1. 网格跨步循环：充分摊销线程派发开销，消除数据规模与硬件派发上限的耦合
      let total_threads = num_workgroups.x * 256u;
      var myVal = 0u;

      for (var i = global_id.x; i < uParams.count; i += total_threads) {
        myVal += data[i];
      }

      // 2. 第一级归约：工作组内累加
      atomicAdd(&localSum, myVal);

      // 3. 屏障等待组内全部线程统计完毕
      workgroupBarrier();

      // 4. 第二级归约：仅由 0 号线程将局部总和汇总至全局
      if (local_id.x == 0u) {
        let groupTotal = atomicLoad(&localSum);
        atomicAdd(&globalSum, groupTotal);
      }
    }
  `;

  // 创建 Pipeline 并排查 Shader 语法
  const shaderModule = device.createShaderModule({ 
    label: "Reduction Shader", 
    code: computeShader 
  });
  
  shaderModule.getCompilationInfo().then((info) => {
    for (const msg of info.messages) {
      console.warn(`[WGSL ${msg.type}] 行 ${msg.lineNum}:${msg.linePos} - ${msg.message}`);
    }
  });

  const computePipeline = device.createComputePipeline({
    label: "Reduction Pipeline",
    layout: "auto",
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });

  const bindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
      { binding: 2, resource: { buffer: uniformBuffer } },
    ],
  });

  // 4. 调试与展示面板
  let isRunning = false;
  let lastStatus = "就绪";

  // 在界面上创建一个可动态更新的状态容器
  const statsContainer = document.createElement("div");
  statsContainer.style.cssText = `
    font-family: monospace; font-size: 12px; line-height: 1.6;
    background: #18181b; padding: 10px; border-radius: 6px;
    margin-top: 8px; border: 1px solid #27272a; color: #e4e4e7;
  `;

  const updateStatsUI = (stats: {
    status: string;
    elements: number;
    expected: number;
    actual: number;
    gpuTime?: string;
    readTime?: string;
    totalTime?: string;
  }) => {
    lastStatus = stats.status;
    statsContainer.innerHTML = `
      <div style="font-weight: bold; margin-bottom: 4px; color: ${stats.status.includes('成功') ? '#4ade80' : stats.status.includes('中') ? '#38bdf8' : '#f87171'};">
        ● 状态: ${stats.status}
      </div>
      <div>数据总量: <b>${(stats.elements / 1024 / 1024).toFixed(1)} M</b> (${stats.elements.toLocaleString()} 元素)</div>
      <div>理论预期: <b>${stats.expected.toLocaleString()}</b></div>
      <div>实际还原: <b>${stats.actual ? stats.actual.toLocaleString() : '-'}</b></div>
      <div style="margin-top: 4px; color: #a1a1aa; border-top: 1px dashed #3f3f46; padding-top: 4px;">
        GPU计算耗时: ${stats.gpuTime ?? '-'} ms<br>
        内存回读耗时: ${stats.readTime ?? '-'} ms<br>
        总体耗时: ${stats.totalTime ?? '-'} ms
      </div>
    `;
  };

  // 核心执行函数
  async function executeCompute() {
    if (isRunning) return;
    isRunning = true;
    updateStatsUI({
      status: "⏳ 正在初始化数据并计算...",
      elements: elementCount,
      expected: elementCount,
      actual: 0
    });

    try {
      const overallStart = performance.now();

      // a. 初始化输入数据 (全部填充为 1，总和即 elementCount)
      log(`正在向显存注入 ${(elementCount * 4 / 1024 / 1024).toFixed(1)}MB 数据...`);
      const inputData = new Uint32Array(elementCount);
      inputData.fill(1);
      device.queue.writeBuffer(inputBuffer, 0, inputData);

      // 清零全局计数器
      device.queue.writeBuffer(outputBuffer, 0, new Uint32Array([0]));

      // b. 录制并派发 Compute Pass
      const gpuStart = performance.now();
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(computePipeline);
      pass.setBindGroup(0, bindGroup);
      
      // 派发固定 4096 个工作组 (4096 * 256 = 1,048,576 线程并发)
      pass.dispatchWorkgroups(4096);
      pass.end();

      // 将结果从 Storage Buffer 拷贝到 MapRead Buffer
      encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, 4);
      device.queue.submit([encoder.finish()]);
      
      // 等待 GPU 队列执行完毕
      await device.queue.onSubmittedWorkDone();
      const gpuEnd = performance.now();

      // c. 回读结果到 CPU
      const readStart = performance.now();
      await readBuffer.mapAsync(GPUMapMode.READ);
      const res = new Uint32Array(readBuffer.getMappedRange());
      const finalSum = res[0];
      readBuffer.unmap();
      const readEnd = performance.now();

      const overallEnd = performance.now();

      const success = finalSum === elementCount;
      const statusText = success ? "✅ 计算成功 (结果完全一致)" : "❌ 计算结果不匹配";

      log(`[完成] 预期: ${elementCount}, 实际: ${finalSum}, GPU耗时: ${(gpuEnd - gpuStart).toFixed(2)}ms`);

      updateStatsUI({
        status: statusText,
        elements: elementCount,
        expected: elementCount,
        actual: finalSum,
        gpuTime: (gpuEnd - gpuStart).toFixed(2),
        readTime: (readEnd - readStart).toFixed(2),
        totalTime: (overallEnd - overallStart).toFixed(2)
      });
    } catch (err: any) {
      console.error("[Compute Error]", err);
      updateStatsUI({
        status: "💥 异常: " + (err?.message || "执行失败"),
        elements: elementCount,
        expected: elementCount,
        actual: 0
      });
    } finally {
      isRunning = false;
    }
  }

  // 绑定 GUI 控件（严格适配现有 SimpleGUI 方法，避免调用未实现的 listen）
  gui.addButton("🚀 发起 GPU 并行原子求和", executeCompute);
  gui.addTextInfo("<b>GPU 网格跨步两级还原</b><br>基于工作组共享内存 `localSum` 与全局原子操作。");
  
  // 挂载动态更新的调试面板
  (gui as any).container.appendChild(statsContainer);
  updateStatsUI({
    status: "就绪",
    elements: elementCount,
    expected: elementCount,
    actual: 0
  });

  // 初始自动运行一次
  setTimeout(executeCompute, 100);

  // 5. 渲染循环：根据状态反馈动态背景，解决“纯黑/无感知”问题
  let animId: number;
  let t = 0;
  function render() {
    t += 0.04;
    const pulse = (Math.sin(t) + 1) * 0.5;

    let r = 0.08, g = 0.09, b = 0.12;
    if (lastStatus.includes("成功")) {
      // 成功呈现深绿微光脉冲
      r = 0.04 + pulse * 0.02;
      g = 0.12 + pulse * 0.04;
      b = 0.08 + pulse * 0.02;
    } else if (lastStatus.includes("中")) {
      // 计算中呈现蓝色脉冲
      r = 0.05;
      g = 0.10 + pulse * 0.06;
      b = 0.18 + pulse * 0.08;
    }

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r, g, b, a: 1.0 },
        loadOp: "clear",
        storeOp: "store"
      }]
    });
    pass.end();
    device.queue.submit([encoder.finish()]);

    animId = requestAnimationFrame(render);
  }
  render();

  // 清理函数
  return () => {
    cancelAnimationFrame(animId);
    inputBuffer.destroy();
    outputBuffer.destroy();
    readBuffer.destroy();
    uniformBuffer.destroy();
    if (statsContainer.parentElement) {
      statsContainer.parentElement.removeChild(statsContainer);
    }
  };
}