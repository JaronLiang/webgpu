// src/examples/aiInferenceGPU.ts

export async function initWebGPUDevice(): Promise<{ device: GPUDevice; adapter: GPUAdapter }> {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU not supported on this browser.");

  const requiredLimits: Record<string, number> = {};
  if (adapter.limits.maxStorageBufferBindingSize) {
    requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
  }
  if (adapter.limits.maxComputeWorkgroupStorageSize) {
    requiredLimits.maxComputeWorkgroupStorageSize = adapter.limits.maxComputeWorkgroupStorageSize; // 确保 Shared Memory 足够
  }
  const device = await adapter.requestDevice({ requiredLimits });
  return { device, adapter };
}

export function runAIInferenceExt(device: GPUDevice) {
  // =========================================================================
  // 1. WGSL: 高性能 Tiled GEMM 融合算子 (Linear + Bias + GELU)
  // 完美解决内存墙问题：16x16 线程组共享缓存，将全局显存读取降低了 16 倍！
  // =========================================================================
  const tiledGemmShaderCode = /* wgsl */ `
    struct ShapeUniform {
      M: u32, // Batch Size / Sequence Length
      N: u32, // Out Features
      K: u32, // In Features
      activation: u32, // 0: None, 1: GELU
    };

    @group(0) @binding(0) var<uniform> uShape: ShapeUniform;
    @group(0) @binding(1) var<storage, read> inputX: array<f32>;    
    @group(0) @binding(2) var<storage, read> weightsW: array<f32>;  
    @group(0) @binding(3) var<storage, read> biasB: array<f32>;     
    @group(0) @binding(4) var<storage, read_write> outputY: array<f32>; 

    // TILE 尺寸 (16x16)
    const TILE_SIZE = 16u;
    
    // 【核心跨越】：声明 Workgroup 级别的共享内存 (Shared Memory)
    var<workgroup> tileX: array<f32, 256>; // 16 * 16
    var<workgroup> tileW: array<f32, 256>; 

    // 高精度近似 GELU 激活函数 (现代大模型首选)
    fn gelu(x: f32) -> f32 {
      let sqrt_2_over_pi = 0.7978845608;
      return 0.5 * x * (1.0 + tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)));
    }

    @compute @workgroup_size(16, 16)
    fn cs_tiled_fused_gemm(
      @builtin(global_invocation_id) global_id: vec3u,
      @builtin(local_invocation_id) local_id: vec3u
    ) {
      let row = global_id.y;
      let col = global_id.x;
      let local_row = local_id.y;
      let local_col = local_id.x;

      var sum = 0.0;
      let numTiles = (uShape.K + TILE_SIZE - 1u) / TILE_SIZE;

      for (var t = 0u; t < numTiles; t = t + 1u) {
        // 1. 协同加载 Input 到 Shared Memory (避免重复读 VRAM)
        let tile_k_x = t * TILE_SIZE + local_col;
        if (row < uShape.M && tile_k_x < uShape.K) {
          tileX[local_row * TILE_SIZE + local_col] = inputX[row * uShape.K + tile_k_x];
        } else {
          tileX[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // 2. 协同加载 Weight 到 Shared Memory
        let tile_k_w = t * TILE_SIZE + local_row;
        if (tile_k_w < uShape.K && col < uShape.N) {
          tileW[local_row * TILE_SIZE + local_col] = weightsW[tile_k_w * uShape.N + col];
        } else {
          tileW[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // 3. 等待组内所有 256 个线程把数据搬运完毕
        workgroupBarrier();

        // 4. 从高速缓存进行局部矩阵乘法累加！(性能起飞)
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
          sum = sum + tileX[local_row * TILE_SIZE + k] * tileW[k * TILE_SIZE + local_col];
        }

        // 5. 等待运算结束，准备加载下一个 Tile
        workgroupBarrier();
      }

      // 【核心跨越】：算子融合 (Kernel Fusion)
      // 计算完 GEMM 后，不写回 VRAM，而是直接在寄存器里 + Bias，做 GELU，最后再写回。
      if (row < uShape.M && col < uShape.N) {
        sum = sum + biasB[col];
        
        if (uShape.activation == 1u) {
           sum = gelu(sum);
        }
        outputY[row * uShape.N + col] = sum;
      }
    }
  `;

  // =========================================================================
  // 2. 管线与绑定组布局
  // =========================================================================
  const moduleAI = device.createShaderModule({ code: tiledGemmShaderCode });

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
    compute: { module: moduleAI, entryPoint: "cs_tiled_fused_gemm" },
  });

  // =========================================================================
  // 3. 模型参数分配 (模拟 LLaMA 的一个 FFN Block)
  // =========================================================================
  // 为了体现 Tiled 的优势，我们将 Batch Size 提大，模拟 64 个 Token 同时推理
  const M = 64;       // Sequence Length (Tokens)
  const D_MODEL = 4096; // 模型维度
  const D_FFN = 4096;   // 隐藏层维度 (由于显存限制演示，设为一样，实际多为 11008)

  // -- Layer 1 Uniform: Linear(4096 -> 4096) + GELU
  const shapeBuffer1 = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(shapeBuffer1, 0, new Uint32Array([M, D_FFN, D_MODEL, 1])); // 1 = GELU

  // -- Layer 2 Uniform: Linear(4096 -> 4096) + No Activation
  const shapeBuffer2 = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(shapeBuffer2, 0, new Uint32Array([M, D_MODEL, D_FFN, 0])); // 0 = None

  // 1. 输入数据 Buffer
  const inputBuffer = device.createBuffer({ size: M * D_MODEL * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  
  // 2. Layer 1 权重和偏置
  const weight1Buffer = device.createBuffer({ size: D_MODEL * D_FFN * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bias1Buffer = device.createBuffer({ size: D_FFN * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  
  // 3. 隐藏层 Buffer (长驻 GPU，不与 CPU 交互！)
  const hiddenBuffer = device.createBuffer({ size: M * D_FFN * 4, usage: GPUBufferUsage.STORAGE });

  // 4. Layer 2 权重和偏置
  const weight2Buffer = device.createBuffer({ size: D_FFN * D_MODEL * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bias2Buffer = device.createBuffer({ size: D_MODEL * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  
  // 5. 最终输出 Buffer & 回读区
  const outputBuffer = device.createBuffer({ size: M * D_MODEL * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readbackBuffer = device.createBuffer({ size: M * D_MODEL * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  // 组装绑定组
  const bindGroupLayer1 = device.createBindGroup({
    layout: bglCompute,
    entries: [
      { binding: 0, resource: { buffer: shapeBuffer1 } },
      { binding: 1, resource: { buffer: inputBuffer } },    // From CPU Input
      { binding: 2, resource: { buffer: weight1Buffer } },
      { binding: 3, resource: { buffer: bias1Buffer } },
      { binding: 4, resource: { buffer: hiddenBuffer } },   // Output to Hidden VRAM
    ],
  });

  const bindGroupLayer2 = device.createBindGroup({
    layout: bglCompute,
    entries: [
      { binding: 0, resource: { buffer: shapeBuffer2 } },
      { binding: 1, resource: { buffer: hiddenBuffer } },   // Read from Hidden VRAM (Operator Graph GraphLink)
      { binding: 2, resource: { buffer: weight2Buffer } },
      { binding: 3, resource: { buffer: bias2Buffer } },
      { binding: 4, resource: { buffer: outputBuffer } },   // Final Output
    ],
  });

  function initMockWeights() {
    logUI("Allocating and initializing weights (128MB) on GPU...");
    // Mock Random Weights
    const w1 = new Float32Array(D_MODEL * D_FFN);
    const b1 = new Float32Array(D_FFN);
    const w2 = new Float32Array(D_FFN * D_MODEL);
    const b2 = new Float32Array(D_MODEL);
    for (let i = 0; i < w1.length; i++) { w1[i] = (Math.random() - 0.5) * 0.05; w2[i] = (Math.random() - 0.5) * 0.05; }
    for (let i = 0; i < b1.length; i++) { b1[i] = (Math.random() - 0.5) * 0.05; b2[i] = (Math.random() - 0.5) * 0.05; }

    device.queue.writeBuffer(weight1Buffer, 0, w1);
    device.queue.writeBuffer(bias1Buffer, 0, b1);
    device.queue.writeBuffer(weight2Buffer, 0, w2);
    device.queue.writeBuffer(bias2Buffer, 0, b2);
    logUI(`✅ Weights loaded: 2 Layers, Total Params ~33.5M`);
  }

  // =========================================================================
  // 4. 完整的 Operator Graph (计算图) 执行
  // =========================================================================
  async function forwardPassGraph(inputArray: Float32Array): Promise<Float32Array> {
    // 1. 将 Token Embedding 发送给 GPU
    device.queue.writeBuffer(inputBuffer, 0, inputArray as any);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);

    // 【核心跨越】：不中断 GPU 管线，连续派发多个融合算子！
    // --- 执行 Layer 1 (Input -> Hidden) ---
    pass.setBindGroup(0, bindGroupLayer1);
    pass.dispatchWorkgroups(Math.ceil(D_FFN / 16), Math.ceil(M / 16), 1);

    // --- 执行 Layer 2 (Hidden -> Output) ---
    pass.setBindGroup(0, bindGroupLayer2);
    pass.dispatchWorkgroups(Math.ceil(D_MODEL / 16), Math.ceil(M / 16), 1);

    pass.end();

    // 只在图的最后一步将 Logits 拷贝出来
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, M * D_MODEL * 4);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readbackBuffer.getMappedRange());
    const finalResult = new Float32Array(result); 
    readbackBuffer.unmap();

    return finalResult;
  }

  // =========================================================================
  // 5. UI 与控制器
  // =========================================================================
  const ui = document.createElement("div");
  ui.style.cssText = "position:absolute;top:12px;left:12px;background:rgba(10,15,20,0.95);color:#fff;padding:16px;border-radius:8px;font-family:monospace;width:400px;z-index:99;box-shadow: 0 4px 12px rgba(0,0,0,0.8);border: 1px solid #333;";
  ui.innerHTML = `
    <div style="font-weight:bold;margin-bottom:12px;color:#00ffcc;font-size:15px;">🧠 WebGPU AI: Tiled Fused Graph</div>
    <div style="font-size:11px;color:#aaa;margin-bottom:12px;line-height:1.5;">
      <b>Architecture:</b> 2-Layer FFN Sub-Graph<br/>
      <b>Graph Flow:</b> Input(64x4K) &rarr; <span style="color:#f39c12">L1(W,B,GELU)</span> &rarr; Hidden(GPU) &rarr; <span style="color:#e74c3c">L2(W,B)</span> &rarr; Logits<br/>
      <b>Optims:</b> 16x16 Shared Memory Tile, Kernel Fusion
    </div>
    
    <button id="btnInit" style="width:100%;padding:10px;background:#27ae60;color:white;border:none;border-radius:4px;cursor:pointer;margin-bottom:8px;font-weight:bold;">1. Load LLaMA FFN Weights (128MB)</button>
    <button id="btnInfer" style="width:100%;padding:10px;background:#2980b9;color:white;border:none;border-radius:4px;cursor:pointer;margin-bottom:8px;font-weight:bold;" disabled>2. Run End-to-End Inference Graph</button>
    
    <div id="ai-logs" style="margin-top:12px;height:250px;overflow-y:auto;background:#000;padding:10px;border-radius:4px;font-size:11px;color:#2ecc71;line-height:1.6;"></div>
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
    setTimeout(() => { 
      initMockWeights();
      btnInfer.disabled = false;
    }, 50);
  };

  btnInfer.onclick = async () => {
    btnInfer.disabled = true;
    
    // 模拟 64 个 Token 的批量输入
    const mockInput = new Float32Array(M * D_MODEL);
    for (let i = 0; i < mockInput.length; i++) mockInput[i] = Math.random();

    logUI("🚀 Dispatching fused compute graph to GPU...");
    const start = performance.now();
    
    const output = await forwardPassGraph(mockInput);
    
    const end = performance.now();
    // Tiled GEMM 在处理大矩阵乘法时，性能会比 Naive 提升几倍甚至十几倍！
    logUI(`⚡ Graph Exec Time: <b style="color:#ffeb3b">${(end - start).toFixed(2)} ms</b>`);
    logUI(`📊 Total MACs: ~2.14 Billion operations!`);
    
    const sample = Array.from(output.slice(0, 8)).map(v => v.toFixed(4));
    logUI(`Output Logits (Token 0, first 8):<br/> [${sample.join(", ")} ...]`);
    logUI("-------------------------");
    
    btnInfer.disabled = false;
  };

  logUI("Advanced WebGPU AI runtime initialized.");

  return () => {
    shapeBuffer1.destroy(); shapeBuffer2.destroy();
    inputBuffer.destroy();
    weight1Buffer.destroy(); bias1Buffer.destroy();
    hiddenBuffer.destroy();
    weight2Buffer.destroy(); bias2Buffer.destroy();
    outputBuffer.destroy(); readbackBuffer.destroy();
    document.body.removeChild(ui);
  };
}