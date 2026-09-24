// src/examples/texture3D.ts
import { Mat4 } from "../utils/math";

export function runTexture3D(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement
) {
  // 1. 立方体顶点坐标 (-1 到 1)
  // prettier-ignore
  const vertexData = new Float32Array([
    -1,-1, 1,  1,-1, 1,  1, 1, 1, -1, 1, 1,
    -1,-1,-1, -1, 1,-1,  1, 1,-1,  1,-1,-1,
    -1, 1,-1, -1, 1, 1,  1, 1, 1,  1, 1,-1,
    -1,-1,-1,  1,-1,-1,  1,-1, 1, -1,-1, 1,
     1,-1,-1,  1, 1,-1,  1, 1, 1,  1,-1, 1,
    -1,-1,-1, -1,-1, 1, -1, 1, 1, -1, 1,-1,
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

  // 2. 【核心】创建 3D 纹理并填充数据 (32x32x32 RGBA8)
  const size = 32;
  const tex3D = device.createTexture({
    size: [size, size, size],
    dimension: "3d", // 【关键】指定为 3D
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  const texData = new Uint8Array(size * size * size * 4);
  const half = size / 2;
  for (let z = 0; z < size; z++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const idx = (z * size * size + y * size + x) * 4;
        // 计算到中心的距离，生成同心球壳花纹
        const dist = Math.sqrt((x - half) ** 2 + (y - half) ** 2 + (z - half) ** 2) / half;
        const ring = Math.sin(dist * Math.PI * 5) * 0.5 + 0.5;

        texData[idx + 0] = Math.floor(ring * 255);                 // R
        texData[idx + 1] = Math.floor((1.0 - ring) * 180 + 50);    // G
        texData[idx + 2] = Math.floor((z / size) * 255);           // B: 沿 Z 轴渐变
        texData[idx + 3] = 255;                                    // A
      }
    }
  }

  // 写入 3D 纹理 (注意传 bytesPerRow 和 rowsPerImage)
  device.queue.writeTexture(
    { texture: tex3D },
    texData,
    { bytesPerRow: size * 4, rowsPerImage: size },
    [size, size, size]
  );

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  // 3. 着色器代码 (采样 3D 纹理)
  const shaderCode = `
    struct Uniforms {
      mvp: mat4x4f,
      time: f32,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var mySampler: sampler;
    @group(0) @binding(2) var myTexture3D: texture_3d<f32>; // 3D 纹理变量

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uvw: vec3f, // 三维纹理坐标
    };

    @vertex
    fn vs_main(@location(0) pos: vec3f) -> VertexOut {
      var out: VertexOut;
      out.pos = u.mvp * vec4f(pos, 1.0);
      // 将三维物体空间坐标 [-1, 1] 映射到 3D 纹理 UVW [0, 1]
      out.uvw = pos * 0.5 + 0.5;
      return out;
    }

    @fragment
    fn fs_main(@location(0) uvw: vec3f) -> @location(0) vec4f {
      // 随着时间扰动 Z 轴坐标，展现动态穿梭于 3D 纹理内部的效果
      let dynamicUVW = vec3f(uvw.xy, fract(uvw.z + u.time * 0.15));
      return textureSample(myTexture3D, mySampler, dynamicUVW);
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

  const uniformBuffer = device.createBuffer({
    size: 80, // mat4 (64) + time (4) + padding
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: tex3D.createView() },
    ],
  });

  const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  let animId: number;
  let rot = 0;
  const proj = Mat4.perspective((60 * Math.PI) / 180, canvas.width / canvas.height, 0.1, 100);

  function frame() {
    rot += 0.8;
    const modelView = Mat4.createModelView(rot * 0.6, rot, -3.5);
    const mvp = Mat4.multiply(proj, modelView);

    device.queue.writeBuffer(uniformBuffer, 0, mvp.buffer as ArrayBuffer);
    device.queue.writeBuffer(uniformBuffer, 64, new Float32Array([rot * 0.02]));

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.08, g: 0.08, b: 0.12, a: 1.0 },
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vBuffer);
    pass.setIndexBuffer(iBuffer, "uint16");
    pass.drawIndexed(36);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  frame();

  return () => {
    cancelAnimationFrame(animId);
    vBuffer.destroy();
    iBuffer.destroy();
    uniformBuffer.destroy();
    tex3D.destroy();
    depthTexture.destroy();
  };
}