// src/examples/tonemapping.ts
import type { SimpleGUI } from "../utils/gui";

export function runToneMapping(
  device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat, canvas: HTMLCanvasElement, gui: SimpleGUI
) {
  // 全屏 Quad
  const quadData = new Float32Array([-1,-1, 1,-1, -1,1, 1,-1, 1,1, -1,1]);
  const quadBuffer = device.createBuffer({ size: quadData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(quadBuffer, 0, quadData);

  const uniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const shaderCode = `
    struct Params {
      exposure: f32,
      mode: f32,      // 0: None, 1: Reinhard, 2: ACES Filmic, 3: Hable(Uncharted2)
      gamma: f32,
      pad: f32,
    };
    @group(0) @binding(0) var<uniform> p: Params;

    @vertex fn vs_main(@location(0) pos: vec2f) -> @builtin(position) vec4f {
      return vec4f(pos, 0.0, 1.0);
    }

    // ACES Narkowicz 拟合
    fn toneMapACES(x: vec3f) -> vec3f {
      let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
      return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
    }

    // Uncharted 2 算子
    fn hableFunc(x: vec3f) -> vec3f {
      let A = 0.15; let B = 0.50; let C = 0.10; let D = 0.20; let E = 0.02; let F = 0.30;
      return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
    }

    @fragment fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      let uv = fragCoord.xy / vec2f(800.0, 600.0);
      
      // 合成一个具有超高动态范围 (HDR) 的测试色板: 光强从 0.1 飙升到 15.0
      var hdrColor = vec3f(0.0);
      let stripe = floor(uv.x * 5.0);
      let intensity = pow(2.0, uv.y * 5.0 - 1.0); // 0.5 ~ 16.0 的高亮范围

      if (stripe == 0.0) { hdrColor = vec3f(1.0, 0.2, 0.1); }
      else if (stripe == 1.0) { hdrColor = vec3f(0.2, 0.9, 0.2); }
      else if (stripe == 2.0) { hdrColor = vec3f(0.2, 0.4, 1.0); }
      else if (stripe == 3.0) { hdrColor = vec3f(1.0, 0.8, 0.2); }
      else { hdrColor = vec3f(1.0, 1.0, 1.0); }

      hdrColor *= intensity * p.exposure;

      var mapped = hdrColor;
      let m = i32(p.mode);
      if (m == 1) {
        // Reinhard
        mapped = hdrColor / (hdrColor + vec3f(1.0));
      } else if (m == 2) {
        // ACES Filmic
        mapped = toneMapACES(hdrColor);
      } else if (m == 3) {
        // Hable / Uncharted 2
        let whiteScale = 1.0 / hableFunc(vec3f(11.2));
        mapped = hableFunc(hdrColor * 2.0) * whiteScale;
      }

      // Gamma 校正
      mapped = pow(mapped, vec3f(1.0 / p.gamma));
      return vec4f(mapped, 1.0);
    }
  `;

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: shaderCode }), entryPoint: "vs_main",
      buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }],
    },
    fragment: { module: device.createShaderModule({ code: shaderCode }), entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" }
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
  });

const settings = { mode: 2, exposure: 1.0, gamma: 2.2 };
  
  gui.addTextInfo(
    "<b>色调映射 (Tone Mapping)</b><br>" +
    "算法说明：<br>" +
    "0: 无 (Clamp 过曝截断)<br>" +
    "1: Reinhard 算法<br>" +
    "2: ACES Filmic (电影级标准)<br>" +
    "3: Hable (神秘海域2算法)"
  );
  
  // 改为 (min: 0, max: 3, step: 1) 的数值滑块
  gui.add(settings, "mode", 0, 3, 1).name("算法选择(0-3)");
  gui.add(settings, "exposure", 0.1, 4.0, 0.1).name("曝光强度");
  gui.add(settings, "gamma", 1.0, 2.6, 0.1).name("Gamma值");
  
  let animId: number;
  function frame() {
    device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([settings.exposure, Number(settings.mode), settings.gamma, 0]));

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }]
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, quadBuffer);
    pass.draw(6);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }
  frame();

  return () => {
    cancelAnimationFrame(animId);
    quadBuffer.destroy(); uniformBuffer.destroy();
  };
}