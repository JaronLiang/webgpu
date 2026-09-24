// src/examples/volumeRendering.ts
export function runVolumeRendering(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement
) {
  // 1. 全屏四边形顶点 (铺满视口)
  const quadVertices = new Float32Array([
    -1, -1,   1, -1,  -1,  1,
    -1,  1,   1, -1,   1,  1,
  ]);
  const vBuffer = device.createBuffer({
    size: quadVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vBuffer, 0, quadVertices);

  // 2. Uniform Buffer: 传递分辨率与动态时间
  const uniformData = new Float32Array(4); // width, height, time, padding
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // 3. 【核心】光线投射体渲染着色器 (Raymarching Shader)
  const shaderCode = `
    struct Uniforms {
      resolution: vec2f,
      time: f32,
      pad: f32,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    // 动态 3D 标量体密度场 (3D Procedural Volume Field)
    fn getDensity(p: vec3f) -> f32 {
      // 随着时间流动的 3D 陀螺面与正弦波叠加
      let t = u.time * 0.5;
      let q = p + vec3f(sin(t * 0.7), cos(t * 0.5), t * 0.3);
      let f1 = sin(q.x * 3.0) * cos(q.y * 3.0) * sin(q.z * 3.0);
      let f2 = sin(q.x * 6.0 + t) * sin(q.y * 6.0) * cos(q.z * 6.0);
      
      // 限制在半径 1.1 的球体内
      let sphereDist = length(p) - 1.1;
      let density = f1 * 0.6 + f2 * 0.4 - sphereDist * 1.5;
      return clamp(density, 0.0, 1.0);
    }

    @vertex
    fn vs_main(@location(0) pos: vec2f) -> @builtin(position) vec4f {
      return vec4f(pos, 0.0, 1.0);
    }

    @fragment
    fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      // 归一化屏幕坐标
      let uv = (fragCoord.xy - u.resolution * 0.5) / u.resolution.y;

      // 相机设置：绕中心慢速旋转
      let angle = u.time * 0.2;
      let camDist = 2.4;
      let ro = vec3f(cos(angle) * camDist, 0.8, sin(angle) * camDist); // 光线起点
      let ta = vec3f(0.0, 0.0, 0.0);                                   // 观察目标
      
      // 相机构建
      let ww = normalize(ta - ro);
      let uu = normalize(cross(ww, vec3f(0.0, 1.0, 0.0)));
      let vv = normalize(cross(uu, ww));
      let rd = normalize(uv.x * uu + uv.y * vv + 1.2 * ww); // 光线方向

      // --- 核心体渲染步进 (Raymarching) ---
      let maxSteps = 64;
      let stepSize = 0.04;
      var t = 0.8; // 起始探测距离
      var accumColor = vec4f(0.0); // 累积颜色 (RGB + Alpha)

      for (var i = 0; i < maxSteps; i++) {
        if (accumColor.a >= 0.98) { break; } // 已完全不透明，提前退出

        let pos = ro + rd * t;
        let density = getDensity(pos);

        if (density > 0.01) {
          // 根据体密度与深度混合发光色彩 (紫色/青色云雾渐变)
          let col = mix(vec3f(0.1, 0.4, 0.9), vec3f(1.0, 0.3, 0.6), density);
          
          // 吸收系数 (Beer-Lambert Law)
          let alpha = (1.0 - accumColor.a) * density * 0.25;
          accumColor += vec4f(col * alpha, alpha);
        }
        t += stepSize;
      }

      // 背景与雾气混合
      let bgColor = vec3f(0.04, 0.04, 0.08);
      return vec4f(accumColor.rgb + bgColor * (1.0 - accumColor.a), 1.0);
    }
  `;

  const module = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vs_main",
      buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }],
    },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  let animId: number;
  let startTime = performance.now();

  function frame() {
    const elapsed = (performance.now() - startTime) * 0.001;
    uniformData[0] = canvas.width;
    uniformData[1] = canvas.height;
    uniformData[2] = elapsed;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear", storeOp: "store",
      }],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vBuffer);
    pass.draw(6);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  frame();

  return () => {
    cancelAnimationFrame(animId);
    vBuffer.destroy();
    uniformBuffer.destroy();
  };
}