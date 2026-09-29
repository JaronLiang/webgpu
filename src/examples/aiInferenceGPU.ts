// src/examples/aiInferenceGPU.ts

export async function initWebGPUDevice(): Promise<{ device: GPUDevice; adapter: GPUAdapter }> {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU not supported on this browser.");

  // 请求最大显存限制以支持大型 AI 模型权重
  const requiredLimits: Record<string, number> = {};
  if (adapter.limits.maxStorageBufferBindingSize) {
    requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
  }
  if (adapter.limits.maxBufferSize) {
    requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
  }

  const device = await adapter.requestDevice({ requiredLimits });
  return { device, adapter };
}

export function runAIInference(device: GPUDevice) {
  // =========================================================================
  // 1. WGSL: 核心 AI 算子 (Linear + Bias + ReLU)
  // 这是现代神经网络 (如 Transformer 的 FFN 层) 最核心的计算单元
  // =========================================================================
  const aiComputeShaderCode = /* wgsl */ `
    struct ShapeUniform {
      M: u32, // Batch Size (输入 token 数量)
      N: u32, // Out Features (输出维度)
      K: u32, // In Features (输入维度)
      _pad: u32,
    };

    @group(0) @binding(0) var<uniform> uShape: ShapeUniform;
    @group(0) @binding(1) var<storage, read> inputX: array<f32>;    // Shape: [M, K]
    @group(0) @binding(2) var<storage, read> weightsW: array<f32>;  // Shape: [K, N]
    @group(0) @binding(3) var<storage, read> biasB: array<f32>;     // Shape: [N]
    @group(0) @binding(4) var<storage, read_write> outputY: array<f32>; // Shape: [M, N]

    // 使用 16x16 的 2D Workgroup，适合矩阵乘法分块
    @compute @workgroup_size(16, 16)
    fn cs_linear_relu(
      @builtin(global_invocation_id) global_id: vec3u
    ) {
      let row = global_id.y; // 对应 M (Batch)
      let col = global_id.x; // 对应 N (Out Features)

      if (row >= uShape.M || col >= uShape.N) {
        return;
      }

      // 矩阵乘法点积计算: dot(X[row, :], W[:, col])
      var sum = 0.0;
      for (var i = 0u; i < uShape.K; i = i + 1u) {
        let x_val = inputX[row * uShape.K + i];
        let w_val = weightsW[i * uShape.N + col];
        sum = sum + (x_val * w_val);
      }

      // 加上偏置 (Bias)
      sum = sum + biasB[col];

      // 激活函数 (ReLU)
      let activated = max(0.0, sum);

      // 写入输出 Tensor
      outputY[row * uShape.N + col] = activated;
    }
  `;

  // =========================================================================
  // 2. 管线配置
  // =========================================================================
  const moduleAI = device.createShaderModule({ code: aiComputeShaderCode });

  const bglCompute = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });

  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bglCompute] }),
    compute: { module: moduleAI, entryPoint: "cs_linear_relu" },
  });

  // =========================================================================
  // 3. 模拟大模型参数与 Buffer 分配
  // =========================================================================
  // 设定一个典型的 LLM 单层维度：输入 1 个 Token，4096 维，输出 4096 维
  const M = 1;      // Batch Size
  const K = 4096;   // In Features
  const N = 4096;   // Out Features

  // 创建 Uniform 缓冲区
  const shapeBuffer = device.createBuffer({
    size: 16, // 4 x u32
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(shapeBuffer, 0, new Uint32Array([M, N, K, 0]));

  // 分配 Tensor 显存
  const inputBuffer = device.createBuffer({ size: M * K * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const weightBuffer = device.createBuffer({ size: K * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const biasBuffer = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outputBuffer = device.createBuffer({ size: M * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  
  // Staging Buffer: 用于将 GPU 的 output 读回到 CPU (JS 端)
  const readbackBuffer = device.createBuffer({
    size: M * N * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    layout: bglCompute,
    entries: [
      { binding: 0, resource: { buffer: shapeBuffer } },
      { binding: 1, resource: { buffer: inputBuffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
      { binding: 3, resource: { buffer: biasBuffer } },
      { binding: 4, resource: { buffer: outputBuffer } },
    ],
  });

  // 模拟加载 AI 权重 (用随机数填充)
  function initMockWeights() {
    logUI("Allocating and initializing weights (64MB) on GPU...");
    const weights = new Float32Array(K * N);
    const bias = new Float32Array(N);
    for (let i = 0; i < weights.length; i++) weights[i] = (Math.random() - 0.5) * 0.1;
    for (let i = 0; i < bias.length; i++) bias[i] = (Math.random() - 0.5) * 0.1;

    device.queue.writeBuffer(weightBuffer, 0, weights);
    device.queue.writeBuffer(biasBuffer, 0, bias);
    logUI(`✅ Weights loaded: Shape [${K}, ${N}]`);
  }

  // =========================================================================
  // 4. 推理执行核心方法
  // =========================================================================
  async function forwardPass(inputArray: Float32Array): Promise<Float32Array> {
    // 1. 将输入向量 (Prompt) 写入 GPU
    device.queue.writeBuffer(inputBuffer, 0, inputArray as any) ;

    const encoder = device.createCommandEncoder();
    
    // 2. 开启 Compute Pass 派发 AI 算子
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    
    // 按照 2D Workgroup 派发：(列数/16, 行数/16)
    const wgX = Math.ceil(N / 16);
    const wgY = Math.ceil(M / 16);
    pass.dispatchWorkgroups(wgX, wgY, 1);
    pass.end();

    // 3. 将推理结果拷贝到 Staging Buffer 准备回读
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, M * N * 4);
    
    // 提交指令
    device.queue.submit([encoder.finish()]);

    // 4. 从显存读取结果回 CPU 内存 (在真实 AI 中，拿回来之后进行 ArgMax 取概率最大的字)
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readbackBuffer.getMappedRange());
    
    // WebGPU 必须复制一份，因为 unmap 后会导致原始 view 失效
    const finalResult = new Float32Array(result); 
    readbackBuffer.unmap();

    return finalResult;
  }

  // =========================================================================
  // 5. UI 与控制器
  // =========================================================================
  const ui = document.createElement("div");
  ui.style.cssText = "position:absolute;top:12px;left:12px;background:rgba(20,20,20,0.9);color:#fff;padding:16px;border-radius:8px;font-family:monospace;width:350px;z-index:99;box-shadow: 0 4px 12px rgba(0,0,0,0.5);";
  ui.innerHTML = `
    <div style="font-weight:bold;margin-bottom:12px;color:#00e5ff;font-size:14px;">🧠 WebGPU AI Inference</div>
    <div style="font-size:12px;color:#aaa;margin-bottom:12px;">Model: Linear(in=4096, out=4096) + ReLU<br/>Params: ~16.7 Million (64MB)</div>
    
    <button id="btnInit" style="width:100%;padding:8px;background:#4CAF50;color:white;border:none;border-radius:4px;cursor:pointer;margin-bottom:8px;">1. Load Mock Weights</button>
    <button id="btnInfer" style="width:100%;padding:8px;background:#2196F3;color:white;border:none;border-radius:4px;cursor:pointer;margin-bottom:8px;" disabled>2. Run Single Inference</button>
    
    <div id="ai-logs" style="margin-top:12px;height:200px;overflow-y:auto;background:#000;padding:8px;border-radius:4px;font-size:11px;color:#0f0;line-height:1.4;"></div>
  `;
  document.body.appendChild(ui);

  const logDiv = document.getElementById("ai-logs")!;
  const btnInit = document.getElementById("btnInit") as HTMLButtonElement;
  const btnInfer = document.getElementById("btnInfer") as HTMLButtonElement;

  function logUI(msg: string) {
    logDiv.innerHTML += `> ${msg}<br/>`;
    logDiv.scrollTop = logDiv.scrollHeight;
  }

  btnInit.onclick = () => {
    btnInit.disabled = true;
    setTimeout(() => { // 让 UI 渲染一下
      initMockWeights();
      btnInfer.disabled = false;
    }, 50);
  };

  btnInfer.onclick = async () => {
    btnInfer.disabled = true;
    
    // 伪造一组输入特征 (比如 LLM 的 Token Embedding)
    const mockInput = new Float32Array(M * K);
    for (let i = 0; i < mockInput.length; i++) mockInput[i] = Math.random();

    logUI("Running Inference Pass...");
    const start = performance.now();
    
    // 执行前向传播并读取结果
    const output = await forwardPass(mockInput);
    
    const end = performance.now();
    logUI(`⚡ Inference time: ${(end - start).toFixed(2)} ms`);
    
    // 打印输出的前 5 个神经元激活值作为演示
    const sample = Array.from(output.slice(0, 5)).map(v => v.toFixed(4));
    logUI(`Output Tensor (first 5): [${sample.join(", ")} ...]`);
    logUI("-------------------------");
    
    btnInfer.disabled = false;
  };

  logUI("WebGPU AI Module Ready.");

  return () => {
    shapeBuffer.destroy();
    inputBuffer.destroy();
    weightBuffer.destroy();
    biasBuffer.destroy();
    outputBuffer.destroy();
    readbackBuffer.destroy();
    document.body.removeChild(ui);
  };
}