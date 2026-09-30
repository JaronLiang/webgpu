// ==========================================
// 1. KCompute 核心：轻量化 WebGPU GPGPU 引擎
// ==========================================

export interface TensorDescriptor {
  data?: Float32Array | Uint32Array | Int32Array;
  size?: number; // 元素个数
  usage?: GPUBufferUsageFlags;
}

export class Tensor {
  public buffer: GPUBuffer;
  public size: number; // 元素总个数
  public byteLength: number;
  private device: GPUDevice;

  constructor(device: GPUDevice, desc: TensorDescriptor) {
    this.device = device;
    this.size = desc.data ? desc.data.length : (desc.size || 0);
    this.byteLength = this.size * 4;

    this.buffer = device.createBuffer({
      size: Math.max(16, this.byteLength), // WebGPU 至少需要 16 字节
      usage: (desc.usage ?? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST)),
      mappedAtCreation: !!desc.data,
    });

    if (desc.data) {
      const ArrayType = desc.data.constructor as new (buf: ArrayBuffer) => any;
      new ArrayType(this.buffer.getMappedRange()).set(desc.data);
      this.buffer.unmap();
    }
  }

  // 异步拉取回 CPU 内存 (读回结果)
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

  /**
   * 创建一个 Tensor (GPU 显存数组)
   */
  tensor(dataOrSize: Float32Array | number): Tensor {
    if (typeof dataOrSize === "number") {
      return new Tensor(this.device, { size: dataOrSize });
    }
    return new Tensor(this.device, { data: dataOrSize });
  }

  /**
   * 核心算子执行器：传入 WGSL 代码、输入输出 Buffer 数组和工作组维度
   */
  run(config: {
    shader: string;
    inputs: (Tensor | GPUBuffer)[];
    uniforms?: Float32Array | Uint32Array;
    workgroups: [number, number?, number?];
  }) {
    let pipeline = this.pipelineCache.get(config.shader);
    if (!pipeline) {
      const module = this.device.createShaderModule({ code: config.shader });
      pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      this.pipelineCache.set(config.shader, pipeline);
    }

    // 动态装配 BindGroup
    const bindGroupEntries: GPUBindGroupEntry[] = [];
    let bindingIdx = 0;

    // 1. 如果有 Uniform
    let uniformBuffer: GPUBuffer | null = null;
    if (config.uniforms) {
      uniformBuffer = this.device.createBuffer({
        size: Math.max(16, config.uniforms.byteLength),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(uniformBuffer, 0, config.uniforms as any);
      bindGroupEntries.push({
        binding: bindingIdx++,
        resource: { buffer: uniformBuffer },
      });
    }

    // 2. 挂载所有的 Tensor / Storage Buffer
    for (const item of config.inputs) {
      const buf = item instanceof Tensor ? item.buffer : item;
      bindGroupEntries.push({
        binding: bindingIdx++,
        resource: { buffer: buf },
      });
    }

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindGroupEntries,
    });

    const commandEncoder = this.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      config.workgroups[0],
      config.workgroups[1] || 1,
      config.workgroups[2] || 1
    );
    pass.end();

    this.device.queue.submit([commandEncoder.finish()]);

    // 临时 Uniform 缓冲区及时销毁
    if (uniformBuffer) {
      // 保证提交完成后销毁（或者放入池中管理）
      setTimeout(() => uniformBuffer?.destroy(), 200);
    }
  }
}