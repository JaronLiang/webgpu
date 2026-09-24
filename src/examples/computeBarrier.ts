// src/examples/computeBarrier.ts

export async function runComputeBarrier(device: GPUDevice, context: GPUCanvasContext, format: string) {
  // 定义规模：4 个工作组，每组 64 个线程，共 256 个元素
  const WORKGROUP_SIZE = 64;
  const NUM_GROUPS = 4;
  const TOTAL_ELEMENTS = WORKGROUP_SIZE * NUM_GROUPS; // 256

  // 1. 初始化原始数据：[0, 1, 2, ..., 255]
  const inputData = new Float32Array(TOTAL_ELEMENTS);
  for (let i = 0; i < TOTAL_ELEMENTS; i++) {
    inputData[i] = i;
  }

  // 2. 创建 Storage Buffer (输入缓冲与输出缓冲)
  const inputBuffer = device.createBuffer({
    size: inputData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuffer, 0, inputData);

  const outputBuffer = device.createBuffer({
    size: inputData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // 用于将结果读回 CPU 的暂存缓冲 (Staging Buffer)
  const stagingBuffer = device.createBuffer({
    size: inputData.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // -------------------------------------------------------------
  // 3. WGSL 计算着色器 (包含 Barrier 与全部 4 个 Built-in 变量)
  // -------------------------------------------------------------
  const shaderCode = `
    // 声明工作组共享内存 (Workgroup Shared Memory)
    // 这块内存在片上高速缓存(SRAM)中，同组的 64 个线程均可访问
    var<workgroup> sharedData: array<f32, ${WORKGROUP_SIZE}>;

    @group(0) @binding(0) var<storage, read>  inputBuf: array<f32>;
    @group(0) @binding(1) var<storage, read_write> outputBuf: array<f32>;

    @compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
    fn cs_main(
      @builtin(global_invocation_id)   global_id   : vec3u, // 全局 3D 坐标
      @builtin(workgroup_id)           group_id    : vec3u, // 当前工作组 3D 坐标
      @builtin(local_invocation_id)    local_id    : vec3u, // 组内 3D 相对坐标
      @builtin(local_invocation_index) local_index : u32    // 组内 1D 线性索引
    ) {
      let g_idx = global_id.x;

      // -----------------------------------------------------------
      // 阶段 1：协同加载数据到共享内存
      // -----------------------------------------------------------
      // 使用 local_index 索引共享内存，将全局数据拉取到片上
      sharedData[local_index] = inputBuf[g_idx];

      // ===========================================================
      // 核心：执行与内存屏障 (Barrier)
      // 保证该工作组内的所有 64 个线程全部完成上述写入后，才允许向下执行！
      // 任何线程越过屏障时，都能安全读取其他线程刚刚写入 sharedData 的数据。
      // ===========================================================
      workgroupBarrier();

      // -----------------------------------------------------------
      // 阶段 2：基于共享内存进行协同计算与顺序重排
      // -----------------------------------------------------------
      // 利用 local_id 和 local_index 进行组内反转：
      // 每个线程读取对称位置邻居的数据 (例如 index 0 读 index 63 的数据)
      let reversedLocalIndex = (${WORKGROUP_SIZE}u - 1u) - local_index;
      let neighborValue = sharedData[reversedLocalIndex];

      // 结合 4 个内置变量构建特征结果写入输出：
      // 这里将：反转后的数值 + group_id 偏移 + local_id 校验
      // 证明全部 4 个内置属性均正确生效
      let groupOffset = f32(group_id.x) * 1000.0;
      let isIdConsistent = f32(local_id.x == local_index); // local_id.x 与 local_index 应完全一致

      outputBuf[g_idx] = neighborValue + groupOffset + isIdConsistent;
    }
  `;

  // -------------------------------------------------------------
  // 4. 手动定义 BindGroupLayout 与 PipelineLayout (无 auto)
  // -------------------------------------------------------------
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }, // 只读 storage
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },           // 读写 storage
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const computePipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: device.createShaderModule({ code: shaderCode }),
      entryPoint: "cs_main",
    },
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
    ],
  });

  // -------------------------------------------------------------
  // 5. 命令录制与调度
  // -------------------------------------------------------------
  const encoder = device.createCommandEncoder();

  // (1) Compute Pass
  const pass = encoder.beginComputePass();
  pass.setPipeline(computePipeline);
  pass.setBindGroup(0, bindGroup);
  // 调度 4 个工作组，每个工作组有 64 个线程
  pass.dispatchWorkgroups(NUM_GROUPS);
  pass.end();

  // (2) 将结果从 GPU 输出缓冲拷贝到可供 CPU 读取的暂存缓冲
  encoder.copyBufferToBuffer(
    outputBuffer, 0,
    stagingBuffer, 0,
    inputData.byteLength
  );

  device.queue.submit([encoder.finish()]);

  // -------------------------------------------------------------
  // 6. 回读 GPU 结果并在 Console 打印验证
  // -------------------------------------------------------------
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(stagingBuffer.getMappedRange());

  console.log("=== WebGPU Compute Barrier 顺序执行结果验证 ===");
  console.log("原始输入数据 [0~7]:", inputData.slice(0, 8));
  // 工作组 0 内原本是 0~63，被倒序后且加了 1.0 (isIdConsistent):
  // 理论上 result[0] 应该等于 inputData[63] + 0(group 0) + 1 = 64
  console.log("处理后输出数据 [0~7] (组内反转 + 同步生效):", result.slice(0, 8));
  console.log("处理后输出数据 [64~71] (第1工作组，基数+1000):", result.slice(64, 72));

  stagingBuffer.unmap();

  // 7. 清理显存资源
  return () => {
    inputBuffer.destroy();
    outputBuffer.destroy();
    stagingBuffer.destroy();
  };
}