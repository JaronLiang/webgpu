// src/examples/selective_bloom.ts
import type { SimpleGUI } from "../utils/gui";

export function runSelectiveBloom(
  device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat, canvas: HTMLCanvasElement, gui: SimpleGUI
) {
  const width = canvas.width || 800;
  const height = canvas.height || 600;

  // 1. 纹理准备：场景纹理、发光提取纹理、模糊双缓冲
  const texUsage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
  const sceneTex = device.createTexture({ size: [width, height], format: "rgba16float", usage: texUsage });
  const emissiveTex = device.createTexture({ size: [width, height], format: "rgba16float", usage: texUsage });
  const blurTexA = device.createTexture({ size: [width / 2, height / 2], format: "rgba16float", usage: texUsage });
  const blurTexB = device.createTexture({ size: [width / 2, height / 2], format: "rgba16float", usage: texUsage });

  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  // 全屏 Quad
  const quadData = new Float32Array([-1,-1, 1,-1, -1,1, 1,-1, 1,1, -1,1]);
  const quadBuf = device.createBuffer({ size: quadData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(quadBuf, 0, quadData);

  // Uniform
  const uniformBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // 2. 着色器 1：场景生成 (MRT)，当物体被选中发光时输出到 Target 1
  const sceneShader = `
    struct SceneParams {
      time: f32,
      glowTarget: f32, // 1: 选中小球, 2: 选中方块, 3: 全部
      bloomIntensity: f32,
      pad: f32
    };
    @group(0) @binding(0) var<uniform> p: SceneParams;

    struct MRTOut {
      @location(0) scene: vec4f,
      @location(1) emissive: vec4f,
    };

    @vertex fn vs_main(@location(0) pos: vec2f) -> @builtin(position) vec4f {
      return vec4f(pos, 0.0, 1.0);
    }

    @fragment fn fs_main(@builtin(position) fragCoord: vec4f) -> MRTOut {
      var out: MRTOut;
      let uv = (fragCoord.xy - vec2f(400.0, 300.0)) / 300.0;

      // 物体 A: 圆形 (左侧)
      let distA = length(uv - vec2f(-0.5, 0.0));
      let isBall = distA < 0.28;

      // 物体 B: 矩形方块 (右侧)
      let dBox = abs(uv - vec2f(0.5, 0.0));
      let isCube = max(dBox.x, dBox.y) < 0.25;

      var col = vec3f(0.04, 0.05, 0.07); // 暗色背景
      var em = vec3f(0.0);

      if (isBall) {
        col = vec3f(0.2, 0.7, 1.0);
        if (p.glowTarget == 1.0 || p.glowTarget == 3.0) {
          em = col * 2.0; // 强发光
        }
      } else if (isCube) {
        col = vec3f(1.0, 0.4, 0.1);
        if (p.glowTarget == 2.0 || p.glowTarget == 3.0) {
          em = col * 2.0;
        }
      }

      out.scene = vec4f(col, 1.0);
      out.emissive = vec4f(em, 1.0);
      return out;
    }
  `;

  const scenePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: sceneShader }), entryPoint: "vs_main", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: device.createShaderModule({ code: sceneShader }), entryPoint: "fs_main", targets: [{ format: "rgba16float" }, { format: "rgba16float" }] },
    primitive: { topology: "triangle-list" }
  });

  // 3. 着色器 2：高斯模糊管线 (Separable Gaussian Blur)
  const blurShader = `
    @group(0) @binding(0) var samp: sampler;
    @group(0) @binding(1) var srcTex: texture_2d<f32>;
    struct BlurUniform { dir: vec2f };
    @group(0) @binding(2) var<uniform> u: BlurUniform;

    @vertex fn vs_blur(@location(0) pos: vec2f) -> @builtin(position) vec4f {
      return vec4f(pos, 0.0, 1.0);
    }
    @fragment fn fs_blur(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      let uv = fragCoord.xy / vec2f(textureDimensions(srcTex));
      var res = textureSample(srcTex, samp, uv) * 0.227027;
      let off1 = u.dir * 1.384615;
      let off2 = u.dir * 3.230769;
      res += textureSample(srcTex, samp, uv + off1) * 0.316216;
      res += textureSample(srcTex, samp, uv - off1) * 0.316216;
      res += textureSample(srcTex, samp, uv + off2) * 0.070270;
      res += textureSample(srcTex, samp, uv - off2) * 0.070270;
      return res;
    }
  `;
  const blurPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: blurShader }), entryPoint: "vs_blur", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: device.createShaderModule({ code: blurShader }), entryPoint: "fs_blur", targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list" }
  });

  // 4. 着色器 3：合成 Bloom 结果回屏幕
  const compositeShader = `
    @group(0) @binding(0) var samp: sampler;
    @group(0) @binding(1) var baseTex: texture_2d<f32>;
    @group(0) @binding(2) var bloomTex: texture_2d<f32>;
    struct P { time: f32, glowTarget: f32, intensity: f32, pad: f32 };
    @group(0) @binding(3) var<uniform> p: P;

    @vertex fn vs_main(@location(0) pos: vec2f) -> @builtin(position) vec4f { return vec4f(pos, 0.0, 1.0); }
    @fragment fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      let uv = fragCoord.xy / vec2f(textureDimensions(baseTex));
      let scene = textureSample(baseTex, samp, uv).rgb;
      let bloom = textureSample(bloomTex, samp, uv).rgb;
      return vec4f(scene + bloom * p.intensity, 1.0);
    }
  `;
  const compositePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: compositeShader }), entryPoint: "vs_main", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: device.createShaderModule({ code: compositeShader }), entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" }
  });

  // 模糊方向 Buffer
  const blurHorizBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const blurVertBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(blurHorizBuf, 0, new Float32Array([1.0 / (width / 2), 0.0, 0, 0]));
  device.queue.writeBuffer(blurVertBuf, 0, new Float32Array([0.0, 1.0 / (height / 2), 0, 0]));

// 5. GUI 控制配置
  const settings = { glowTarget: 1, intensity: 1.5 };
  
  gui.addTextInfo(
    "<b>选择性泛光 (Selective Bloom)</b><br>" +
    "发光对象说明：<br>" +
    "0: 全部关闭 | 1: 仅蓝色小球 | 2: 仅橙色方块 | 3: 全部发光"
  );
  
  // 使用纯数值范围参数，完美适配 SimpleGUI 的 (target, prop, min, max, step) 签名
  gui.add(settings, "glowTarget", 0, 3, 1).name("发光目标(0-3)");
  gui.add(settings, "intensity", 0.0, 3.0, 0.1).name("光晕强度");
  
  let animId: number;
  function frame(time: number) {
    device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([time * 0.001, Number(settings.glowTarget), settings.intensity, 0]));

    const encoder = device.createCommandEncoder();

    // Pass 1: 渲染场景并分离出发光目标
    const pass1 = encoder.beginRenderPass({
      colorAttachments: [
        { view: sceneTex.createView(), loadOp: "clear", storeOp: "store" },
        { view: emissiveTex.createView(), loadOp: "clear", storeOp: "store" },
      ]
    });
    pass1.setPipeline(scenePipeline);
    pass1.setBindGroup(0, device.createBindGroup({
      layout: scenePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
    }));
    pass1.setVertexBuffer(0, quadBuf);
    pass1.draw(6);
    pass1.end();

    // Pass 2: 水平模糊 (Emissive -> blurTexA)
    const pass2 = encoder.beginRenderPass({
      colorAttachments: [{ view: blurTexA.createView(), loadOp: "clear", storeOp: "store" }]
    });
    pass2.setPipeline(blurPipeline);
    pass2.setBindGroup(0, device.createBindGroup({
      layout: blurPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: emissiveTex.createView() },
        { binding: 2, resource: { buffer: blurHorizBuf } }
      ]
    }));
    pass2.setVertexBuffer(0, quadBuf);
    pass2.draw(6);
    pass2.end();

    // Pass 3: 垂直模糊 (blurTexA -> blurTexB)
    const pass3 = encoder.beginRenderPass({
      colorAttachments: [{ view: blurTexB.createView(), loadOp: "clear", storeOp: "store" }]
    });
    pass3.setPipeline(blurPipeline);
    pass3.setBindGroup(0, device.createBindGroup({
      layout: blurPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: blurTexA.createView() },
        { binding: 2, resource: { buffer: blurVertBuf } }
      ]
    }));
    pass3.setVertexBuffer(0, quadBuf);
    pass3.draw(6);
    pass3.end();

    // Pass 4: 混合输出回画布
    const pass4 = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }]
    });
    pass4.setPipeline(compositePipeline);
    pass4.setBindGroup(0, device.createBindGroup({
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: sceneTex.createView() },
        { binding: 2, resource: blurTexB.createView() },
        { binding: 3, resource: { buffer: uniformBuffer } }
      ]
    }));
    pass4.setVertexBuffer(0, quadBuf);
    pass4.draw(6);
    pass4.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }
  frame(0);

  return () => {
    cancelAnimationFrame(animId);
    sceneTex.destroy(); emissiveTex.destroy(); blurTexA.destroy(); blurTexB.destroy();
    quadBuf.destroy(); uniformBuffer.destroy(); blurHorizBuf.destroy(); blurVertBuf.destroy();
  };
}