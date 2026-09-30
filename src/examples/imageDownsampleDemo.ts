// src/examples/imageDownsampleDemo.ts
import type { SimpleGUI } from "../utils/gui";

export async function runImageDownsampleDemo(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  const SRC_W = 1024;
  const SRC_H = 1024;
  const DST_W = 256; // 降采样到 1/4 尺寸 (支持多次降采样)
  const DST_H = 256;

  // =====================================================================
  // 1. WGSL: 13-Tap 动视高质量降采样 Filter (COD Downsample Kernel)
  // =====================================================================
  const downsampleShaderCode = `
    struct DownsampleParams {
      srcTexelSize: vec2f,
      dstResolution: vec2f,
    };

    @group(0) @binding(0) var<uniform> u: DownsampleParams;
    @group(0) @binding(1) var sLinear: sampler;
    @group(0) @binding(2) var tSource: texture_2d<f32>;
    @group(0) @binding(3) var tDest: texture_storage_2d<rgba16float, write>;

    @compute @workgroup_size(16, 16)
    fn cs_downsample(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= u32(u.dstResolution.x) || id.y >= u32(u.dstResolution.y)) { return; }

      // 目标像素的归一化中心 UV
      let uv = (vec2f(id.xy) + 0.5) / u.dstResolution;
      let x = u.srcTexelSize.x;
      let y = u.srcTexelSize.y;

      // 13-Tap 帐篷滤波采样点分布 (极大减少高频摩尔纹与几何闪烁)
      // A - B - C
      // - D - E -
      // F - G - H
      // - I - J -
      // K - L - M
      let a = textureSampleLevel(tSource, sLinear, uv + vec2f(-2.0*x,  2.0*y), 0.0).rgb;
      let b = textureSampleLevel(tSource, sLinear, uv + vec2f( 0.0,    2.0*y), 0.0).rgb;
      let c = textureSampleLevel(tSource, sLinear, uv + vec2f( 2.0*x,  2.0*y), 0.0).rgb;

      let d = textureSampleLevel(tSource, sLinear, uv + vec2f(-1.0*x,  1.0*y), 0.0).rgb;
      let e = textureSampleLevel(tSource, sLinear, uv + vec2f( 1.0*x,  1.0*y), 0.0).rgb;

      let f = textureSampleLevel(tSource, sLinear, uv + vec2f(-2.0*x,  0.0), 0.0).rgb;
      let g = textureSampleLevel(tSource, sLinear, uv, 0.0).rgb; // Center
      let h = textureSampleLevel(tSource, sLinear, uv + vec2f( 2.0*x,  0.0), 0.0).rgb;

      let i = textureSampleLevel(tSource, sLinear, uv + vec2f(-1.0*x, -1.0*y), 0.0).rgb;
      let j = textureSampleLevel(tSource, sLinear, uv + vec2f( 1.0*x, -1.0*y), 0.0).rgb;

      let k = textureSampleLevel(tSource, sLinear, uv + vec2f(-2.0*x, -2.0*y), 0.0).rgb;
      let l = textureSampleLevel(tSource, sLinear, uv + vec2f( 0.0,   -2.0*y), 0.0).rgb;
      let m = textureSampleLevel(tSource, sLinear, uv + vec2f( 2.0*x, -2.0*y), 0.0).rgb;

      // 加权平均 (保持能量守恒)
      var col = e * 0.125;
      col += (a + c + k + m) * 0.03125;
      col += (b + d + f + h + i + j + l) * 0.0625;
      col += g * 0.25;

      textureStore(tDest, id.xy, vec4f(col, 1.0));
    }
  `;

  // 呈现 Blit 着色器
  const blitShaderCode = `
    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) id: u32) -> VertexOut {
      var out: VertexOut;
      let x = f32((id << 1u) & 2u);
      let y = f32(id & 2u);
      out.uv = vec2f(x * 0.5, 1.0 - y * 0.5);
      out.pos = vec4f(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
      return out;
    }

    @group(0) @binding(0) var sLinear: sampler;
    @group(0) @binding(1) var tInput: texture_2d<f32>;

    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4f {
      return textureSample(tInput, sLinear, in.uv);
    }
  `;

  // =====================================================================
  // 2. 纹理创建与初始化高频程序化纹理（用于验证降采样防闪烁能力）
  // =====================================================================
  const sLinear = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  const srcTexture = device.createTexture({
    size: [SRC_W, SRC_H],
    format: "rgba16float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  const dstTexture = device.createTexture({
    size: [DST_W, DST_H],
    format: "rgba16float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  // 在 CPU 生成高频黑白相间摩尔纹图案写入 Source Texture
  const rawPixels = new Float32Array(SRC_W * SRC_H * 4);
  for (let y = 0; y < SRC_H; y++) {
    for (let x = 0; x < SRC_W; x++) {
      const idx = (y * SRC_W + x) * 4;
      const ring = Math.sin(Math.sqrt((x - 512) ** 2 + (y - 512) ** 2) * 0.2);
      const checker = ((x >> 2) ^ (y >> 2)) & 1 ? 1.0 : 0.0;
      rawPixels[idx + 0] = ring * 0.5 + 0.5;
      rawPixels[idx + 1] = checker;
      rawPixels[idx + 2] = 0.8;
      rawPixels[idx + 3] = 1.0;
    }
  }

  // 转换成 Float16 数组写入
  const h16 = new Uint16Array(rawPixels.length);
  for (let i = 0; i < rawPixels.length; i++) {
    // 简易 float32 -> float16 编码
    const f32 = rawPixels[i];
    const buf = new ArrayBuffer(4);
    new Float32Array(buf)[0] = f32;
    const u32 = new Uint32Array(buf)[0];
    h16[i] = ((u32 >> 16) & 0x8000) | ((((u32 & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((u32 >> 13) & 0x03ff);
  }
  device.queue.writeTexture(
    { texture: srcTexture },
    h16,
    { bytesPerRow: SRC_W * 8 },
    { width: SRC_W, height: SRC_H }
  );

  const uboBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    uboBuffer,
    0,
    new Float32Array([1.0 / SRC_W, 1.0 / SRC_H, DST_W, DST_H])
  );

  // =====================================================================
  // 3. 显式创建 PipelineLayout (无 layout: "auto")
  // =====================================================================
  const computeModule = device.createShaderModule({ code: downsampleShaderCode });
  const blitModule = device.createShaderModule({ code: blitShaderCode });

  const computeBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", minBindingSize: 16 } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float" } },
    ],
  });

  const computeLayout = device.createPipelineLayout({ bindGroupLayouts: [computeBGL] });
  const computePipeline = device.createComputePipeline({
    layout: computeLayout,
    compute: { module: computeModule, entryPoint: "cs_downsample" },
  });

  const blitBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
  });
  const blitLayout = device.createPipelineLayout({ bindGroupLayouts: [blitBGL] });
  const blitPipeline = device.createRenderPipeline({
    layout: blitLayout,
    vertex: { module: blitModule, entryPoint: "vs_main" },
    fragment: { module: blitModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const computeBG = device.createBindGroup({
    layout: computeBGL,
    entries: [
      { binding: 0, resource: { buffer: uboBuffer } },
      { binding: 1, resource: sLinear },
      { binding: 2, resource: srcTexture.createView() },
      { binding: 3, resource: dstTexture.createView() },
    ],
  });

  const blitSrcBG = device.createBindGroup({
    layout: blitBGL,
    entries: [{ binding: 0, resource: sLinear }, { binding: 1, resource: srcTexture.createView() }],
  });

  const blitDstBG = device.createBindGroup({
    layout: blitBGL,
    entries: [{ binding: 0, resource: sLinear }, { binding: 1, resource: dstTexture.createView() }],
  });

  const guiState = { showDownsampled: true };
  gui.add(guiState, "showDownsampled").name("展示降采样后画面 (256x256)");
  gui.addTextInfo("🔍 <b>13-Tap COD 防闪烁图像降采样</b><br>基于 Compute 提取的高频能量加权滤波。");

  let animId: number;
  function frame() {
    const encoder = device.createCommandEncoder();

    // 1. Compute Pass 降采样
    const cpass = encoder.beginComputePass();
    cpass.setPipeline(computePipeline);
    cpass.setBindGroup(0, computeBG);
    cpass.dispatchWorkgroups(Math.ceil(DST_W / 16), Math.ceil(DST_H / 16));
    cpass.end();

    // 2. Render Pass 呈现到屏幕
    const rpass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
        storeOp: "store",
      }],
    });
    rpass.setPipeline(blitPipeline);
    rpass.setBindGroup(0, guiState.showDownsampled ? blitDstBG : blitSrcBG);
    rpass.draw(3);
    rpass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }
  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    srcTexture.destroy();
    dstTexture.destroy();
    uboBuffer.destroy();
  };
}