/// <reference types="@webgpu/types" />

export async function runSubgroup(
  defaultDevice: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  let device = defaultDevice;

  // 1. 检查当前 device 是否启用了 "subgroups" 特性
  // 如果外部传入的 device 未启用，我们尝试向 adapter 单独请求一个启用了 subgroups 的独立 device
  if (!device.features.has("subgroups")) {
    console.warn("⚠️ 当前 GPUDevice 未启用 'subgroups' 特性，尝试重新协商请求...");
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter || !adapter.features.has("subgroups")) {
      const errMsg = "❌ 你的 GPU / 浏览器环境暂不支持 WebGPU 'subgroups' 硬件特性扩展。\n请在 Chrome 126+ 中确认支持，或在 chrome://flags 中检查 WebGPU 试验性支持。";
      console.error(errMsg);
      alert(errMsg);
      return;
    }
    // 启用 subgroups 特性
    device = await adapter.requestDevice({
      requiredFeatures: ["subgroups"],
    });
  }

  // 重新绑定 context 到有效 device
  context.configure({
    device: device,
    format: format,
    alphaMode: "premultiplied",
  });

  // 2. 准备用于回读真实硬件 Subgroup 参数的 Buffer
  // 结构：[subgroup_size, total_subgroups_executed]
  const infoBufferSize = 2 * 4;
  const gpuInfoBuffer = device.createBuffer({
    size: infoBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const readbackBuffer = device.createBuffer({
    size: infoBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // 统一 Uniform Buffer (时间与分辨率)
  const uniformBuffer = device.createBuffer({
    size: 16, // vec2f resolution, f32 time, f32 pad
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // 3. WGSL 计算着色器：利用 subgroup 内置变量进行并行生成
  const computeShaderCode = `
    // 【核心声明】：启用 subgroups 特性扩展
    enable subgroups;

    struct Uniforms {
      resolution: vec2f,
      time: f32,
      pad: f32,
    };

    struct HardwareInfo {
      actualSubgroupSize: u32,
      activeSubgroupCount: atomic<u32>,
    };

    @group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
    @group(0) @binding(1) var<uniform> u: Uniforms;
    @group(0) @binding(2) var<storage, read_write> info: HardwareInfo;

    // 工作组定义为 16x16 = 256 个线程
    @compute @workgroup_size(16, 16, 1)
    fn main(
      @builtin(global_invocation_id) global_id: vec3u,
      @builtin(local_invocation_id) local_id: vec3u,
      // 🌟【核心变量 1】：当前线程在所属 Subgroup (Warp/Wave) 中的物理车道号 (Lane ID: 0 ~ size-1)
      @builtin(subgroup_invocation_id) lane_id: u32,
      // 🌟【核心变量 2】：当前硬件的 Subgroup 大小 (通常 NVIDIA 为 32，AMD/Apple/Intel 为 32 或 64)
      @builtin(subgroup_size) sg_size: u32
    ) {
      let dims = textureDimensions(outputTex);
      if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
      }

      // 1. 回读硬件信息：仅由每个 Subgroup 内的首个线程(Leader: lane_id == 0) 执行
      if (lane_id == 0u) {
        info.actualSubgroupSize = sg_size;
        atomicAdd(&info.activeSubgroupCount, 1u);
      }

      // 2. 利用 Subgroup 拓扑结构计算可视化图案：
      // lane_ratio (0.0 ~ 1.0): 线程在其所属 SIMD 硬件束内的归一化位置
      let lane_ratio = f32(lane_id) / f32(sg_size - 1u);

      // 计算所属的全局 Subgroup 序号 (用于给不同 Subgroup 赋予不同底色)
      let linear_workgroup_idx = local_id.y * 16u + local_id.x;
      let subgroup_idx_in_workgroup = linear_workgroup_idx / sg_size;

      // 动态波形相位
      let uv = vec2f(global_id.xy) / u.resolution;
      let wave = sin(uv.x * 20.0 + u.time * 3.0 + f32(subgroup_idx_in_workgroup)) * 0.5 + 0.5;

      // 3. 着色逻辑：
      // R 通道：车道在 Subgroup 内部的索引渐变 (高亮显示 Lane 边界)
      // G 通道：Subgroup 的动态波动
      // B 通道：Subgroup 大小特征指示 (区分 32 还是 64 大小)
      var color: vec3f;
      if (lane_id == 0u) {
        // 每个 Subgroup 的 Leader 线程用纯白点标识出来
        color = vec3f(1.0, 1.0, 1.0);
      } else {
        color = vec3f(
          lane_ratio, 
          wave * 0.8 + 0.2, 
          f32(sg_size) / 64.0
        );
      }

      // 写入存储纹理
      textureStore(outputTex, global_id.xy, vec4f(color, 1.0));
    }
  `;

  // 4. WGSL 全屏呈现着色器：把计算纹理 Blit 到画布
  const blitShaderCode = `
    @vertex
    fn vsMain(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
      var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0)
      );
      return vec4f(pos[vid], 0.0, 1.0);
    }

    @group(0) @binding(0) var tex: texture_2d<f32>;
    @group(0) @binding(1) var smp: sampler;

    @fragment
    fn fsMain(@builtin(position) coord: vec4f) -> @location(0) vec4f {
      let dims = vec2f(textureDimensions(tex));
      let uv = coord.xy / dims;
      return textureSample(tex, smp, uv);
    }
  `;

  // 5. 编译着色器模块
  const computeModule = device.createShaderModule({ code: computeShaderCode });
  const blitModule = device.createShaderModule({ code: blitShaderCode });

  // 6. 创建计算管线
  const computePipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: computeModule, entryPoint: "main" },
  });

  // 7. 创建渲染管线 (用于将生成的 Texture 呈现到屏幕)
  const blitPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: blitModule, entryPoint: "vsMain" },
    fragment: {
      module: blitModule,
      entryPoint: "fsMain",
      targets: [{ format: format }],
    },
    primitive: { topology: "triangle-list" },
  });

  const sampler = device.createSampler({
    magFilter: "nearest",
    minFilter: "nearest",
  });

  // 创建动态离屏 Storage Texture
  let storageTexture: GPUTexture | null = null;
  let computeBindGroup: GPUBindGroup | null = null;
  let blitBindGroup: GPUBindGroup | null = null;
  let currentWidth = 0;
  let currentHeight = 0;

  function updateTextures(width: number, height: number) {
    if (currentWidth === width && currentHeight === height && storageTexture) return;
    currentWidth = width;
    currentHeight = height;

    if (storageTexture) storageTexture.destroy();

    storageTexture = device.createTexture({
      size: [width, height, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    computeBindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: storageTexture.createView() },
        { binding: 1, resource: { buffer: uniformBuffer } },
        { binding: 2, resource: { buffer: gpuInfoBuffer } },
      ],
    });

    blitBindGroup = device.createBindGroup({
      layout: blitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: storageTexture.createView() },
        { binding: 1, resource: sampler },
      ],
    });
  }

  // 8. 执行一次回读，读取真实的硬件 Subgroup Size 打印到控制台
  let hasReadHardwareInfo = false;

  let isRunning = true;
  let startTime = performance.now();

  function frame() {
    if (!isRunning) return;

    const canvas = context.canvas as HTMLCanvasElement;
    const width = Math.max(1, canvas.width);
    const height = Math.max(1, canvas.height);
    updateTextures(width, height);

    // 更新 Uniform
    const time = (performance.now() - startTime) * 0.001;
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Float32Array([width, height, time, 0.0])
    );

    const encoder = device.createCommandEncoder();

    // 1) 执行 Compute Pass
    const cPass = encoder.beginComputePass();
    cPass.setPipeline(computePipeline);
    cPass.setBindGroup(0, computeBindGroup!);
    // 按照 16x16 分派工作组
    const workgroupsX = Math.ceil(width / 16);
    const workgroupsY = Math.ceil(height / 16);
    cPass.dispatchWorkgroups(workgroupsX, workgroupsY);
    cPass.end();

    // 2) 如果首次执行，将硬件信息拷贝出来回读
    if (!hasReadHardwareInfo) {
      encoder.copyBufferToBuffer(gpuInfoBuffer, 0, readbackBuffer, 0, infoBufferSize);
    }

    // 3) 执行 Render Pass 绘制到画布
    const rPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          storeOp: "store",
        },
      ],
    });
    rPass.setPipeline(blitPipeline);
    rPass.setBindGroup(0, blitBindGroup!);
    rPass.draw(3, 1, 0, 0);
    rPass.end();

    device.queue.submit([encoder.finish()]);

    // 4) 异步回读硬件 Subgroup 数据
    if (!hasReadHardwareInfo) {
      hasReadHardwareInfo = true;
      readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const array = new Uint32Array(readbackBuffer.getMappedRange());
        const detectedSubgroupSize = array[0];
        const totalSubgroups = array[1];
        readbackBuffer.unmap();

        console.log(
          `%c[WebGPU Subgroup Info]%c\n` +
          `• 硬件真实 Subgroup Size: ${detectedSubgroupSize}\n` +
          `• 当前一帧触发的 Subgroups 数量: ${totalSubgroups}\n` +
          `• 硬件架构类型推断: ${detectedSubgroupSize === 32 ? "NVIDIA Warp (32) 或 AMD/Intel/Apple 模式 (32)" : "AMD Wave64 或 Intel/Apple (64)"}`,
          "color: #38bdf8; font-weight: bold; font-size: 13px;",
          "color: #a1a1aa; font-size: 12px;"
        );
      }).catch(err => {
        console.warn("回读 Subgroup 硬件信息失败:", err);
      });
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);

  // 9. 返回清理函数
  return () => {
    isRunning = false;
    if (storageTexture) storageTexture.destroy();
    uniformBuffer.destroy();
    gpuInfoBuffer.destroy();
    readbackBuffer.destroy();
  };
}