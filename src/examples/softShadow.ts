// src/examples/softShadow.ts
export function runSoftShadow(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 步骤 1: 编写 WGSL 着色器
  // =========================================================================
  const shaderCode = `
    struct UniformConfig {
      resolution: vec2f,
      time: f32,
      _pad: f32,
    };

    @group(0) @binding(0) var<uniform> uConfig: UniformConfig;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) vIdx: u32) -> VertexOutput {
      var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
      var out: VertexOutput;
      out.position = vec4f(pos[vIdx], 0.0, 1.0);
      out.uv = pos[vIdx];
      return out;
    }

    // 旋转矩阵辅助
    fn rotY(a: f32) -> mat2x2<f32> {
      let c = cos(a); let s = sin(a);
      return mat2x2<f32>(c, -s, s, c);
    }

    // 立方体 SDF
    fn sdBox(p: vec3f, b: vec3f) -> f32 {
      let q = abs(p) - b;
      return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
    }

    // 场景 SDF (返回: vec2f(距离, 材质: 0=地面, 1=球体, 2=立方体))
    fn map(p: vec3f) -> vec2f {
      // 1. 地面 (y = -0.6)
      let floorDist = p.y + 0.6;
      var d = floorDist;
      var matId = 0.0;

      // 2. 悬浮浮动的球体
      let sphereDist = length(p - vec3f(-0.9, 0.1 + sin(uConfig.time * 2.0) * 0.15, 0.2)) - 0.45;
      if (sphereDist < d) {
        d = sphereDist;
        matId = 1.0;
      }

      // 3. 悬浮自转的立方体
      var boxPos = p - vec3f(0.9, 0.2, -0.2);
      let rot = rotY(uConfig.time);
      let xz = rot * boxPos.xz;
      boxPos = vec3f(xz.x, boxPos.y, xz.y);
      let boxDist = sdBox(boxPos, vec3f(0.35));
      if (boxDist < d) {
        d = boxDist;
        matId = 2.0;
      }

      return vec2f(d, matId);
    }

    fn calcNormal(p: vec3f) -> vec3f {
      let e = vec2f(0.001, 0.0);
      return normalize(vec3f(
        map(p + e.xyy).x - map(p - e.xyy).x,
        map(p + e.yxy).x - map(p - e.yxy).x,
        map(p + e.yyx).x - map(p - e.yyx).x
      ));
    }

    // 👈 核心软阴影算法 (基于半影比率累积)
    fn calcSoftShadow(ro: vec3f, rd: vec3f, mint: f32, maxt: f32, k: f32) -> f32 {
      var res = 1.0;
      var t = mint;
      for (var i = 0; i < 36; i++) {
        let h = map(ro + rd * t).x;
        if (h < 0.001) {
          return 0.0; // 完全处于本影区 (Hard shadow)
        }
        // k 是半影软度系数：离遮挡物越远，阴影越弥散
        res = min(res, k * h / t);
        t += clamp(h, 0.02, 0.15);
        if (t > maxt) { break; }
      }
      return clamp(res, 0.0, 1.0);
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      var uv = in.uv;
      uv.x *= uConfig.resolution.x / uConfig.resolution.y;

      // 观察相机
      let ro = vec3f(0.0, 2.0, -3.2);
      let ta = vec3f(0.0, -0.2, 0.0);
      let ww = normalize(ta - ro);
      let uu = normalize(cross(ww, vec3f(0.0, 1.0, 0.0)));
      let vv = normalize(cross(uu, ww));
      let rd = normalize(uv.x * uu + uv.y * vv + 1.6 * ww);

      // 主光线追踪
      var t = 0.0;
      var hitMat = 0.0;
      var hit = false;
      for (var i = 0; i < 80; i++) {
        let res = map(ro + rd * t);
        if (res.x < 0.001) {
          hit = true;
          hitMat = res.y;
          break;
        }
        t += res.x;
        if (t > 15.0) { break; }
      }

      if (!hit) {
        return vec4f(mix(vec3f(0.7, 0.8, 0.9), vec3f(0.9, 0.95, 1.0), uv.y * 0.5 + 0.5), 1.0);
      }

      let p = ro + rd * t;
      let n = calcNormal(p);

      // 动态移动旋转的高空点光源
      let lightPos = vec3f(sin(uConfig.time * 0.8) * 2.5, 2.8, cos(uConfig.time * 0.8) * 2.5);
      let lightDir = normalize(lightPos - p);
      let lightDist = length(lightPos - p);

      // 漫反射与软阴影计算
      let diff = max(dot(n, lightDir), 0.0);
      // 向光源方向步进发射软阴影射线
      let shadow = calcSoftShadow(p + n * 0.003, lightDir, 0.02, lightDist, 18.0);

      // 区分材质固有颜色
      var albedo = vec3f(0.9);
      if (hitMat == 0.0) {
        // 地面黑白棋盘格纹理，让软阴影投射更为鲜明
        let f = fract(p.x * 1.5) > 0.5;
        let g = fract(p.z * 1.5) > 0.5;
        let check = select(0.55, 0.85, (f != g));
        albedo = vec3f(check);
      } else if (hitMat == 1.0) {
        albedo = vec3f(0.9, 0.25, 0.2); // 红色球体
      } else if (hitMat == 2.0) {
        albedo = vec3f(0.2, 0.55, 0.95); // 蓝色立方体
      }

      // 高光反射
      let viewDir = -rd;
      let halfDir = normalize(lightDir + viewDir);
      let spec = pow(max(dot(n, halfDir), 0.0), 32.0) * select(0.0, 0.5, hitMat > 0.0);

      // 最终合成：直射光 * 软阴影 + 环境光 + 高光
      let ambient = vec3f(0.12, 0.14, 0.18);
      let finalColor = albedo * (diff * shadow * vec3f(1.1) + ambient) + spec * shadow;

      return vec4f(finalColor, 1.0);
    }
  `;

  // =========================================================================
  // 步骤 2: 全屏三角形
  // =========================================================================
  const vertexData = new Float32Array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
  const vertexBuffer = device.createBuffer({
    label: "SoftShadow-VertexBuffer",
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // =========================================================================
  // 步骤 3: 严格 16 字节对齐的 Uniform Buffer
  // =========================================================================
  const uniformData = new Float32Array(4); // resolution(2), time(1), pad(1)
  const uniformBuffer = device.createBuffer({
    label: "SoftShadow-UniformBuffer",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 步骤 4 & 5: 显式创建 BindGroupLayout 与 PipelineLayout
  // =========================================================================
  const bindGroupLayout = device.createBindGroupLayout({
    label: "SoftShadow-BindGroupLayout",
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "SoftShadow-PipelineLayout",
    bindGroupLayouts: [bindGroupLayout],
  });

  const bindGroup = device.createBindGroup({
    label: "SoftShadow-BindGroup",
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // =========================================================================
  // 步骤 6: 渲染管线
  // =========================================================================
  const shaderModule = device.createShaderModule({ code: shaderCode });
  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 2 * Float32Array.BYTES_PER_ELEMENT,
    stepMode: "vertex",
    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
  };

  const pipeline = device.createRenderPipeline({
    label: "SoftShadow-Pipeline",
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [vertexBufferLayout],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });

  // =========================================================================
  // 步骤 7: 动态渲染循环
  // =========================================================================
  let animId: number;
  const startTime = performance.now();

  function render() {
    const canvas = context.canvas as HTMLCanvasElement;
    uniformData[0] = canvas.width;
    uniformData[1] = canvas.height;
    uniformData[2] = (performance.now() - startTime) / 1000;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(3, 1, 0, 0);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(render);
  }
  render();

  return () => {
    cancelAnimationFrame(animId);
    vertexBuffer.destroy();
    uniformBuffer.destroy();
  };
}