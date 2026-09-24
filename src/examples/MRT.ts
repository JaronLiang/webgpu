// src/examples/mrt.ts
import { Mat4 } from "../utils/math";

export function runMRT(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement
) {
  // 1. 立方体网格数据 (Pos + Normal)
  // prettier-ignore
  const vertexData = new Float32Array([
    -1,-1, 1, 0,0,1,   1,-1, 1, 0,0,1,   1, 1, 1, 0,0,1,  -1, 1, 1, 0,0,1,
    -1,-1,-1, 0,0,-1, -1, 1,-1, 0,0,-1,  1, 1,-1, 0,0,-1,  1,-1,-1, 0,0,-1,
    -1, 1,-1, 0,1,0,  -1, 1, 1, 0,1,0,   1, 1, 1, 0,1,0,   1, 1,-1, 0,1,0,
    -1,-1,-1, 0,-1,0,  1,-1,-1, 0,-1,0,  1,-1, 1, 0,-1,0, -1,-1, 1, 0,-1,0,
     1,-1,-1, 1,0,0,   1, 1,-1, 1,0,0,   1, 1, 1, 1,0,0,   1,-1, 1, 1,0,0,
    -1,-1,-1, -1,0,0, -1,-1, 1, -1,0,0, -1, 1, 1, -1,0,0, -1, 1,-1, -1,0,0,
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

  // 2. 创建 3 个 MRT G-Buffer 纹理 (采用最广泛兼容的 rgba8unorm)
  const gBufferFormat = "rgba8unorm";
  const gAlbedo = device.createTexture({ size: [canvas.width, canvas.height], format: gBufferFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const gNormal = device.createTexture({ size: [canvas.width, canvas.height], format: gBufferFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const gPosition = device.createTexture({ size: [canvas.width, canvas.height], format: gBufferFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const depthTexture = device.createTexture({ size: [canvas.width, canvas.height], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });

  // 3. MRT 着色器与渲染管线
  const mrtShaderCode = `
    struct Uniforms { mvp: mat4x4f, model: mat4x4f };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) normal: vec3f,
      @location(1) modelPos: vec3f,
    };

    @vertex
    fn vs_main(@location(0) pos: vec3f, @location(1) norm: vec3f) -> VertexOut {
      var out: VertexOut;
      out.pos = u.mvp * vec4f(pos, 1.0);
      out.normal = (u.model * vec4f(norm, 0.0)).xyz;
      out.modelPos = pos;
      return out;
    }

    struct GBufferOut {
      @location(0) albedo: vec4f,
      @location(1) normal: vec4f,
      @location(2) position: vec4f,
    };

    @fragment
    fn fs_main(in: VertexOut) -> GBufferOut {
      var g: GBufferOut;
      g.albedo = vec4f(0.9, 0.45, 0.15, 1.0);                   // 目标 0: 固有色
      g.normal = vec4f(normalize(in.normal) * 0.5 + 0.5, 1.0); // 目标 1: 法线映射到 [0, 1]
      g.position = vec4f(in.modelPos * 0.5 + 0.5, 1.0);         // 目标 2: 模型坐标映射到 [0, 1]
      return g;
    }
  `;
  const mrtModule = device.createShaderModule({ code: mrtShaderCode });

  const mrtPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: mrtModule, entryPoint: "vs_main",
      buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x3" }] }],
    },
    fragment: {
      module: mrtModule, entryPoint: "fs_main",
      targets: [{ format: gBufferFormat }, { format: gBufferFormat }, { format: gBufferFormat }],
    },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // 4. 【核心修复】：Blit 预览着色器 (修复非统一控制流报错)
  const blitShaderCode = `
    @group(0) @binding(0) var s: sampler;
    @group(0) @binding(1) var tAlbedo: texture_2d<f32>;
    @group(0) @binding(2) var tNormal: texture_2d<f32>;
    @group(0) @binding(3) var tPos: texture_2d<f32>;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_blit(@builtin(vertex_index) id: u32) -> VertexOut {
      // 全屏 Quad (带标准 UV)
      var pos = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
      );
      var uv = array<vec2f, 6>(
        vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0),
        vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0)
      );
      var out: VertexOut;
      out.pos = vec4f(pos[id], 0.0, 1.0);
      out.uv = uv[id];
      return out;
    }

    @fragment
    fn fs_blit(in: VertexOut) -> @location(0) vec4f {
      // 【修复关键】：在无分支的最外层统一完成采样，满足 Uniform Control Flow 规则！
      let colAlbedo = textureSample(tAlbedo, s, in.uv);
      let colNormal = textureSample(tNormal, s, in.uv);
      let colPos    = textureSample(tPos, s, in.uv);

      // 三等分屏显示预览：左 = Albedo, 中 = Normal, 右 = Position
      if (in.uv.x < 0.333) {
        return colAlbedo;
      } else if (in.uv.x < 0.666) {
        return colNormal;
      } else {
        return colPos;
      }
    }
  `;
  const blitModule = device.createShaderModule({ code: blitShaderCode });

  const blitPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: blitModule, entryPoint: "vs_blit" },
    fragment: { module: blitModule, entryPoint: "fs_blit", targets: [{ format }] },
  });

  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  const blitBindGroup = device.createBindGroup({
    layout: blitPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: gAlbedo.createView() },
      { binding: 2, resource: gNormal.createView() },
      { binding: 3, resource: gPosition.createView() },
    ],
  });

  const uBuffer = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const mrtBindGroup = device.createBindGroup({
    layout: mrtPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uBuffer } }],
  });

  let animId: number;
  let rot = 0;
  const proj = Mat4.perspective((60 * Math.PI) / 180, canvas.width / canvas.height, 0.1, 100);

  function frame() {
    rot += 1.0;
    const model = Mat4.createModelView(rot, rot * 0.7, -3.5);
    const mvp = Mat4.multiply(proj, model);
    device.queue.writeBuffer(uBuffer, 0, mvp.buffer as ArrayBuffer);
    device.queue.writeBuffer(uBuffer, 64, model.buffer as ArrayBuffer);

    const encoder = device.createCommandEncoder();

    // --- Pass 1: MRT 一次性输出到 3 张 G-Buffer ---
    const mrtPass = encoder.beginRenderPass({
      colorAttachments: [
        { view: gAlbedo.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
        { view: gNormal.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
        { view: gPosition.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });
    mrtPass.setPipeline(mrtPipeline);
    mrtPass.setBindGroup(0, mrtBindGroup);
    mrtPass.setVertexBuffer(0, vBuffer);
    mrtPass.setIndexBuffer(iBuffer, "uint16");
    mrtPass.drawIndexed(36);
    mrtPass.end();

    // --- Pass 2: 合成通道，全屏三分屏展示 G-Buffer ---
    const blitPass = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }],
    });
    blitPass.setPipeline(blitPipeline);
    blitPass.setBindGroup(0, blitBindGroup);
    blitPass.draw(6);
    blitPass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  frame();

  return () => {
    cancelAnimationFrame(animId);
    vBuffer.destroy();
    iBuffer.destroy();
    uBuffer.destroy();
    gAlbedo.destroy();
    gNormal.destroy();
    gPosition.destroy();
    depthTexture.destroy();
  };
}