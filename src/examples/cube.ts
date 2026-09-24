// src/examples/cube.ts
import { Mat4 } from "../utils/math";

export function runCube(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement
) {
  // 1. 立方体顶点数据 (位置 x,y,z + 颜色 r,g,b)
  // prettier-ignore
  const vertexData = new Float32Array([
    // 前面 (红)
    -1,-1, 1, 1,0,0,   1,-1, 1, 1,0,0,   1, 1, 1, 1,0,0,  -1, 1, 1, 1,0,0,
    // 后面 (绿)
    -1,-1,-1, 0,1,0,  -1, 1,-1, 0,1,0,   1, 1,-1, 0,1,0,   1,-1,-1, 0,1,0,
    // 顶面 (蓝)
    -1, 1,-1, 0,0,1,  -1, 1, 1, 0,0,1,   1, 1, 1, 0,0,1,   1, 1,-1, 0,0,1,
    // 底面 (黄)
    -1,-1,-1, 1,1,0,   1,-1,-1, 1,1,0,   1,-1, 1, 1,1,0,  -1,-1, 1, 1,1,0,
    // 右面 (青)
     1,-1,-1, 0,1,1,   1, 1,-1, 0,1,1,   1, 1, 1, 0,1,1,   1,-1, 1, 0,1,1,
    // 左面 (洋红)
    -1,-1,-1, 1,0,1,  -1,-1, 1, 1,0,1,  -1, 1, 1, 1,0,1,  -1, 1,-1, 1,0,1,
  ]);

  // 索引缓冲 (每个面由 2 个三角形组成)
  // prettier-ignore
  const indexData = new Uint16Array([
    0,1,2, 0,2,3,       4,5,6, 4,6,7,       8,9,10, 8,10,11,
    12,13,14, 12,14,15, 16,17,18, 16,18,19, 20,21,22, 20,22,23
  ]);

  const vBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vBuffer, 0, vertexData);

  const iBuffer = device.createBuffer({
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(iBuffer, 0, indexData);

  // 2. Uniform 缓冲 (存放 4x4 MVP 变换矩阵)
  const uniformBuffer = device.createBuffer({
    size: 64, // 16 floats * 4 bytes
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // 3. 深度纹理 (Depth Buffer)
  const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // 4. 着色器
  const shaderCode = `
    struct Uniforms {
      mvpMatrix: mat4x4f,
    };
    @group(0) @binding(0) var<uniform> uniforms: Uniforms;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec4f,
    };

    @vertex
    fn vs_main(@location(0) pos: vec3f, @location(1) col: vec3f) -> VertexOut {
      var out: VertexOut;
      out.pos = uniforms.mvpMatrix * vec4f(pos, 1.0);
      out.color = vec4f(col, 1.0);
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
      buffers: [{
        arrayStride: 6 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" },
        ],
      }],
    },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth24plus",
    },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // 5. 渲染循环与动画
  let animationFrameId: number;
  let rotationX = 0;
  let rotationY = 0;

  const proj = Mat4.perspective((60 * Math.PI) / 180, canvas.width / canvas.height, 0.1, 100);

  function frame() {
    rotationX += 0.8;
    rotationY += 1.2;

    const modelView = Mat4.createModelView(rotationX, rotationY, -4.5);
    const mvp = Mat4.multiply(proj, modelView);
    device.queue.writeBuffer(uniformBuffer, 0, mvp.buffer as ArrayBuffer);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.08, g: 0.08, b: 0.12, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vBuffer);
    pass.setIndexBuffer(iBuffer, "uint16");
    pass.drawIndexed(36);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animationFrameId = requestAnimationFrame(frame);
  }

  frame();

  // 清理函数：切换示例时必须取消 rAF 并释放显存
  return () => {
    cancelAnimationFrame(animationFrameId);
    vBuffer.destroy();
    iBuffer.destroy();
    uniformBuffer.destroy();
    depthTexture.destroy();
  };
}