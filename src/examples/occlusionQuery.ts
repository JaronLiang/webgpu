// src/examples/occlusionQuery.ts
import { Mat4 } from "../utils/math";

export function runOcclusionQuery(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement
) {
  // 1. 创建通用立方体几何数据
  // prettier-ignore
  const vertexData = new Float32Array([
    -1,-1, 1,  1,-1, 1,  1, 1, 1, -1, 1, 1, // 前
    -1,-1,-1, -1, 1,-1,  1, 1,-1,  1,-1,-1, // 后
    -1, 1,-1, -1, 1, 1,  1, 1, 1,  1, 1,-1, // 顶
    -1,-1,-1,  1,-1,-1,  1,-1, 1, -1,-1, 1, // 底
     1,-1,-1,  1, 1,-1,  1, 1, 1,  1,-1, 1, // 右
    -1,-1,-1, -1,-1, 1, -1, 1, 1, -1, 1,-1, // 左
  ]);
  // prettier-ignore
  const indexData = new Uint16Array([
    0,1,2, 0,2,3,       4,5,6, 4,6,7,       8,9,10, 8,10,11,
    12,13,14, 12,14,15, 16,17,18, 16,18,19, 20,21,22, 20,22,23
  ]);

  const vBuffer = device.createBuffer({ size: vertexData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vBuffer, 0, vertexData);
  const iBuffer = device.createBuffer({ size: indexData.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(iBuffer, 0, indexData);

  // 2. Uniform 缓冲 (支持渲染多个物体，每个物体有单独的 MVP 矩阵和颜色)
  // [0..15]: MVP 矩阵, [16..19]: 颜色 vec4
  const boxCount = 3; // 1个大挡板 + 2个后方动态盒子
  const uniformBuffers = Array.from({ length: boxCount }, () =>
    device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  );

  // 3. 【核心】创建遮挡查询集 (QuerySet) 与数据读取缓冲
  const queryCount = 2; // 只对后方2个盒子进行查询
  const querySet = device.createQuerySet({
    type: "occlusion",
    count: queryCount,
  });

  // 用于接收 GPU 写入查询结果的缓冲区 (每个查询结果为 64位无符号整型 u64 = 8 字节)
  const resolveBuffer = device.createBuffer({
    size: queryCount * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });

  // 用于 CPU 异步回读结果的缓冲区
  const readbackBuffer = device.createBuffer({
    size: queryCount * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // 4. 深度纹理
  let depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // 5. 着色器与管线
  const shaderCode = `
    struct Uniforms {
      mvp: mat4x4f,
      color: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec4f,
    };

    @vertex
    fn vs_main(@location(0) pos: vec3f) -> VertexOut {
      var out: VertexOut;
      out.pos = u.mvp * vec4f(pos, 1.0);
      out.color = u.color;
      return out;
    }

    @fragment
    fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
      return color;
    }
  `;
  const module = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vs_main",
      buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }],
    },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const bindGroups = uniformBuffers.map((buf) =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: buf } }],
    })
  );

  // 6. 渲染与动态查询逻辑
  let animId: number;
  let time = 0;
  let isReading = false;
  // 记录后方两个物体的可见状态
  const boxVisibility = [true, true];

  const proj = Mat4.perspective((60 * Math.PI) / 180, canvas.width / canvas.height, 0.1, 100);

  function frame() {
    time += 0.02;

    // --- 1. 更新遮挡物与后方盒子矩阵 ---
    // 物体 0: 前方大遮挡板 (灰色, Z = -3.5, 尺寸缩放加大)
    const occluderModel = Mat4.multiply(Mat4.translation(0, 0, -3.5), Mat4.scaling(1.4, 1.8, 0.2));
    const occluderMVP = Mat4.multiply(proj, occluderModel);
    device.queue.writeBuffer(uniformBuffers[0], 0, occluderMVP.buffer as ArrayBuffer);
    device.queue.writeBuffer(uniformBuffers[0], 64, new Float32Array([0.4, 0.4, 0.4, 1.0]));

    // 物体 1 & 2: 后方运动的盒子 (Z = -6.0)
    for (let i = 0; i < 2; i++) {
      const offsetX = Math.sin(time + i * Math.PI) * 2.8;
      const model = Mat4.multiply(Mat4.translation(offsetX, (i - 0.5) * 1.2, -6.0), Mat4.scaling(0.5, 0.5, 0.5));
      const mvp = Mat4.multiply(proj, model);
      device.queue.writeBuffer(uniformBuffers[i + 1], 0, mvp.buffer as ArrayBuffer);

      // 根据上一帧 Occlusion Query 的结果上色：可见=绿色，完全被挡住=红色 (便于直观辨认)
      const color = boxVisibility[i] ? [0.2, 0.9, 0.2, 1.0] : [0.9, 0.1, 0.1, 1.0];
      device.queue.writeBuffer(uniformBuffers[i + 1], 64, new Float32Array(color));
    }

    const encoder = device.createCommandEncoder();
    // 【关键】声明 renderPass 使用该 querySet
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.1, b: 0.14, a: 1.0 },
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
      },
      occlusionQuerySet: querySet, // 绑定查询集
    });

    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, vBuffer);
    pass.setIndexBuffer(iBuffer, "uint16");

    // 首先绘制前方的大挡板 (写入深度缓冲)
    pass.setBindGroup(0, bindGroups[0]);
    pass.drawIndexed(36);

    // 绘制后方的两个动态盒子，并用 occlusionQuery 包裹！
    for (let i = 0; i < 2; i++) {
      pass.beginOcclusionQuery(i); // 开始第 i 个查询
      pass.setBindGroup(0, bindGroups[i + 1]);
      pass.drawIndexed(36);
      pass.endOcclusionQuery();   // 结束第 i 个查询
    }
    pass.end();

    // 【关键】将查询结果从硬件 QuerySet 解算到 GPU 缓冲中
    encoder.resolveQuerySet(querySet, 0, queryCount, resolveBuffer, 0);

    // 将结果从 resolveBuffer 拷贝到 readbackBuffer 以便 CPU 读取
    if (!isReading) {
      encoder.copyBufferToBuffer(resolveBuffer, 0, readbackBuffer, 0, queryCount * 8);
    }

    device.queue.submit([encoder.finish()]);

    // CPU 异步回读结果 (非阻塞)
    if (!isReading) {
      isReading = true;
      readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const results = new BigUint64Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();
        // 如果像素通过数 > 0 说明通过了深度测试，即物体可见
        boxVisibility[0] = results[0] > 0n;
        boxVisibility[1] = results[1] > 0n;
        isReading = false;
      });
    }

    animId = requestAnimationFrame(frame);
  }

  frame();

  return () => {
    cancelAnimationFrame(animId);
    vBuffer.destroy();
    iBuffer.destroy();
    uniformBuffers.forEach((b) => b.destroy());
    resolveBuffer.destroy();
    readbackBuffer.destroy();
    depthTexture.destroy();
  };
}