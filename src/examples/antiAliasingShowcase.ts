// src/examples/antiAliasingShowcaseFixed.ts
export function runAntiAliasingShowcase(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 0. 控制面板与 UI
  // =========================================================================
  const canvas = context.canvas as HTMLCanvasElement;
  const container = canvas.parentElement || document.body;

  const panel = document.createElement("div");
  panel.style.cssText = `
    position: absolute; top: 16px; left: 16px; z-index: 1000;
    background: rgba(15, 20, 28, 0.92); backdrop-filter: blur(8px);
    padding: 14px 18px; border-radius: 8px; font-family: monospace, sans-serif;
    color: #fff; font-size: 13px; box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.18); user-select: none;
  `;
  panel.innerHTML = `
    <div style="font-weight: bold; margin-bottom: 8px; font-size: 14px; color: #58a6ff;">抗锯齿质量对比 (右下角带 5x 边缘显微镜)</div>
    <div style="display: flex; gap: 6px; margin-bottom: 8px;">
      <button id="btn-none" style="flex:1; padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer; background: #e05656; color: #fff; font-weight: bold;">关闭 (原生锯齿)</button>
      <button id="btn-fxaa" style="flex:1; padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer; background: #262c36; color: #aaa;">FXAA</button>
      <button id="btn-taa"  style="flex:1; padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer; background: #262c36; color: #aaa;">TAA (稳定收敛)</button>
      <button id="btn-msaa" style="flex:1; padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer; background: #262c36; color: #aaa;">4x MSAA</button>
    </div>
    <div id="aa-desc" style="color: #bbb; font-size: 11px; line-height: 1.4;">【关闭抗锯齿】: 原生采样。请看球体外轮廓与高光边缘，有极其明显的硬阶梯狗牙。</div>
  `;
  container.style.position = "relative";
  container.appendChild(panel);

  type AAMode = "none" | "fxaa" | "taa" | "msaa";
  let currentMode: AAMode = "none";
  let taaAccumFrames = 0;

  const btnNone = panel.querySelector("#btn-none") as HTMLButtonElement;
  const btnFXAA = panel.querySelector("#btn-fxaa") as HTMLButtonElement;
  const btnTAA = panel.querySelector("#btn-taa") as HTMLButtonElement;
  const btnMSAA = panel.querySelector("#btn-msaa") as HTMLButtonElement;
  const desc = panel.querySelector("#aa-desc") as HTMLDivElement;

  function setMode(mode: AAMode) {
    currentMode = mode;
    taaAccumFrames = 0;
    [btnNone, btnFXAA, btnTAA, btnMSAA].forEach(btn => {
      btn.style.background = "#262c36";
      btn.style.color = "#aaa";
      btn.style.fontWeight = "normal";
    });
    if (mode === "none") {
      btnNone.style.background = "#e05656"; btnNone.style.color = "#fff"; btnNone.style.fontWeight = "bold";
      desc.innerText = "【关闭抗锯齿】: 原生采样。请看球体外轮廓与高光边缘，有极其明显的硬阶梯狗牙。";
    } else if (mode === "fxaa") {
      btnFXAA.style.background = "#3b82f6"; btnFXAA.style.color = "#fff"; btnFXAA.style.fontWeight = "bold";
      desc.innerText = "【FXAA】: 屏幕空间快速近似抗锯齿，边缘梯度模糊平滑，无闪烁。";
    } else if (mode === "taa") {
      btnTAA.style.background = "#10b981"; btnTAA.style.color = "#fff"; btnTAA.style.fontWeight = "bold";
      desc.innerText = "【TAA】: 方差裁剪 + 亚像素累积，多帧静止收敛为平滑的超级采样，无任何跳动闪烁。";
    } else if (mode === "msaa") {
      btnMSAA.style.background = "#8b5cf6"; btnMSAA.style.color = "#fff"; btnMSAA.style.fontWeight = "bold";
      desc.innerText = "【4x MSAA】: 硬件几何多重采样，对多边形几何外边缘进行精准的 4 级覆盖率平滑。";
    }
  }
  btnNone.onclick = () => setMode("none");
  btnFXAA.onclick = () => setMode("fxaa");
  btnTAA.onclick = () => setMode("taa");
  btnMSAA.onclick = () => setMode("msaa");

  // =========================================================================
  // 1. 生成比例适中、大小约 0.85 的几何球体
  // =========================================================================
  const latBands = 40, lonBands = 40;
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  for (let lat = 0; lat <= latBands; lat++) {
    const theta = (lat * Math.PI) / latBands;
    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);

    for (let lon = 0; lon <= lonBands; lon++) {
      const phi = (lon * 2 * Math.PI) / lonBands;
      const x = Math.cos(phi) * sinTheta;
      const y = cosTheta;
      const z = Math.sin(phi) * sinTheta;
      positions.push(x * 0.85, y * 0.85, z * 0.85);
      normals.push(x, y, z);
    }
  }

  for (let lat = 0; lat < latBands; lat++) {
    for (let lon = 0; lon < lonBands; lon++) {
      const first = lat * (lonBands + 1) + lon;
      const second = first + lonBands + 1;
      indices.push(first, second, first + 1);
      indices.push(second, second + 1, first + 1);
    }
  }

  const vertBuffer = device.createBuffer({
    size: positions.length * 6 * 4,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  const vertArray = new Float32Array(positions.length * 2);
  for (let i = 0; i < positions.length / 3; i++) {
    vertArray[i * 6 + 0] = positions[i * 3 + 0];
    vertArray[i * 6 + 1] = positions[i * 3 + 1];
    vertArray[i * 6 + 2] = positions[i * 3 + 2];
    vertArray[i * 6 + 3] = normals[i * 3 + 0];
    vertArray[i * 6 + 4] = normals[i * 3 + 1];
    vertArray[i * 6 + 5] = normals[i * 3 + 2];
  }
  device.queue.writeBuffer(vertBuffer, 0, vertArray);

  const idxBuffer = device.createBuffer({
    size: indices.length * 2,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(idxBuffer, 0, new Uint16Array(indices));

  // =========================================================================
  // 2. 着色器代码定义 (精确 112 字节对齐)
  // =========================================================================

  // 场景着色器
  const sceneShader = `
    struct Uniforms {
      viewProj: mat4x4f,   // 0~63 (64 bytes)
      camPos: vec4f,       // 64~79 (16 bytes)
      resolution: vec2f,   // 80~87 (8 bytes)
      jitter: vec2f,       // 88~95 (8 bytes)
      taaWeight: f32,      // 96~99 (4 bytes)
      pad0: f32,           // 100~103 (4 bytes)
      pad1: f32,           // 104~107 (4 bytes)
      pad2: f32,           // 108~111 (4 bytes) -> 总计严格 112 字节
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexInput {
      @location(0) pos: vec3f,
      @location(1) norm: vec3f,
    };
    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) worldNorm: vec3f,
    };

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
      var out: VertexOutput;
      var clip = u.viewProj * vec4f(in.pos, 1.0);
      clip.x += (u.jitter.x * 2.0 / u.resolution.x) * clip.w;
      clip.y += (u.jitter.y * 2.0 / u.resolution.y) * clip.w;
      out.position = clip;
      out.worldPos = in.pos;
      out.worldNorm = in.norm;
      return out;
    }

    fn aces(c: vec3f) -> vec3f {
      let a = 2.51; let b = 0.03; let c1 = 2.43; let d = 0.59; let e = 0.14;
      return clamp((c * (a * c + b)) / (c * (c1 * c + d) + e), vec3f(0.0), vec3f(1.0));
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let N = normalize(in.worldNorm);
      let V = normalize(u.camPos.xyz - in.worldPos);
      let R = reflect(-V, N);

      let albedo = vec3f(1.0, 0.78, 0.28);
      let NdotV = max(dot(N, V), 0.001);
      let F = albedo + (vec3f(1.0) - albedo) * pow(1.0 - NdotV, 5.0);

      let lDir1 = normalize(vec3f(0.8, 0.8, 0.6));
      let s1 = pow(max(dot(R, lDir1), 0.0), 96.0) * 16.0;

      let lDir2 = normalize(vec3f(-0.9, 0.2, -0.6));
      let s2 = pow(max(dot(R, lDir2), 0.0), 32.0) * 5.0;

      let litColor = albedo * 0.18 + (vec3f(s1) * vec3f(1.0, 0.95, 0.8) + vec3f(s2) * vec3f(0.3, 0.7, 1.0)) * F;
      let finalRGB = pow(aces(litColor), vec3f(1.0 / 2.2));
      let luma = dot(finalRGB, vec3f(0.299, 0.587, 0.114));

      return vec4f(finalRGB, luma);
    }
  `;

  // Raw 原生直通着色器 (带放大镜)
  const rawBlitShader = `
    @group(0) @binding(0) var uSampler: sampler;
    @group(0) @binding(1) var uTexture: texture_2d<f32>;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
      var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
      var out: VertexOutput;
      out.position = vec4f(pos[id], 0.0, 1.0);
      out.uv = pos[id] * 0.5 + 0.5;
      out.uv.y = 1.0 - out.uv.y;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      var uv = in.uv;
      let zoomCenter = vec2f(0.64, 0.35);
      let insetPos = vec2f(0.70, 0.65);
      let insetSize = 0.28;

      if (uv.x > insetPos.x && uv.y > insetPos.y) {
        if (uv.x < insetPos.x + 0.003 || uv.y < insetPos.y + 0.004) {
          return vec4f(0.2, 0.8, 1.0, 1.0);
        }
        let localUV = (uv - insetPos) / insetSize;
        let zoomedUV = zoomCenter + (localUV - vec2f(0.5)) * (1.0 / 5.0);
        return vec4f(textureSampleLevel(uTexture, uSampler, zoomedUV, 0.0).rgb, 1.0);
      }
      return vec4f(textureSampleLevel(uTexture, uSampler, uv, 0.0).rgb, 1.0);
    }
  `;

  // FXAA 着色器
  const fxaaShader = `
    @group(0) @binding(0) var uSampler: sampler;
    @group(0) @binding(1) var uTexture: texture_2d<f32>;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
      var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
      var out: VertexOutput;
      out.position = vec4f(pos[id], 0.0, 1.0);
      out.uv = pos[id] * 0.5 + 0.5;
      out.uv.y = 1.0 - out.uv.y;
      return out;
    }

    fn sampleFXAA(uv: vec2f) -> vec3f {
      let dims = vec2f(textureDimensions(uTexture));
      let invSize = 1.0 / dims;

      let m = textureSampleLevel(uTexture, uSampler, uv, 0.0);
      let lumaM = m.a;
      let lumaNW = textureSampleLevel(uTexture, uSampler, uv + vec2f(-1.0, -1.0) * invSize, 0.0).a;
      let lumaNE = textureSampleLevel(uTexture, uSampler, uv + vec2f( 1.0, -1.0) * invSize, 0.0).a;
      let lumaSW = textureSampleLevel(uTexture, uSampler, uv + vec2f(-1.0,  1.0) * invSize, 0.0).a;
      let lumaSE = textureSampleLevel(uTexture, uSampler, uv + vec2f( 1.0,  1.0) * invSize, 0.0).a;

      let lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
      let lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
      let lumaRange = lumaMax - lumaMin;

      if (lumaRange < max(0.04, lumaMax * 0.125)) {
        return m.rgb;
      }

      var dir = vec2f(
        -((lumaNW + lumaNE) - (lumaSW + lumaSE)),
        ((lumaNW + lumaSW) - (lumaNE + lumaSE))
      );
      let dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.03125, 0.0078125);
      let rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
      dir = min(vec2f(8.0), max(vec2f(-8.0), dir * rcpDirMin)) * invSize;

      let rgbA = 0.5 * (
        textureSampleLevel(uTexture, uSampler, uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb +
        textureSampleLevel(uTexture, uSampler, uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb
      );
      let rgbB = rgbA * 0.5 + 0.25 * (
        textureSampleLevel(uTexture, uSampler, uv + dir * -0.5, 0.0).rgb +
        textureSampleLevel(uTexture, uSampler, uv + dir * 0.5, 0.0).rgb
      );

      let lumaB = dot(rgbB, vec3f(0.299, 0.587, 0.114));
      if (lumaB < lumaMin || lumaB > lumaMax) {
        return rgbA;
      }
      return rgbB;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      var uv = in.uv;
      let zoomCenter = vec2f(0.64, 0.35);
      let insetPos = vec2f(0.70, 0.65);
      let insetSize = 0.28;

      if (uv.x > insetPos.x && uv.y > insetPos.y) {
        if (uv.x < insetPos.x + 0.003 || uv.y < insetPos.y + 0.004) {
          return vec4f(0.2, 0.8, 1.0, 1.0);
        }
        let localUV = (uv - insetPos) / insetSize;
        let zoomedUV = zoomCenter + (localUV - vec2f(0.5)) * (1.0 / 5.0);
        return vec4f(sampleFXAA(zoomedUV), 1.0);
      }
      return vec4f(sampleFXAA(uv), 1.0);
    }
  `;

  // TAA 着色器 (同样使用 112 字节严格对齐)
  const taaShader = `
    struct Uniforms {
      viewProj: mat4x4f,   // 0~63
      camPos: vec4f,       // 64~79
      resolution: vec2f,   // 80~87
      jitter: vec2f,       // 88~95
      taaWeight: f32,      // 96~99
      pad0: f32,           // 100~103
      pad1: f32,           // 104~107
      pad2: f32,           // 108~111 -> 112 字节
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var uSampler: sampler;
    @group(0) @binding(2) var uCurrentTex: texture_2d<f32>;
    @group(0) @binding(3) var uHistoryTex: texture_2d<f32>;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
      var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
      var out: VertexOutput;
      out.position = vec4f(pos[id], 0.0, 1.0);
      out.uv = pos[id] * 0.5 + 0.5;
      out.uv.y = 1.0 - out.uv.y;
      return out;
    }

    fn sampleTAAResolve(uv: vec2f) -> vec3f {
      let dims = vec2f(textureDimensions(uCurrentTex));
      let invSize = 1.0 / dims;

      let cur = textureSampleLevel(uCurrentTex, uSampler, uv, 0.0).rgb;

      var m1 = vec3f(0.0);
      var m2 = vec3f(0.0);

      for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
          let s = textureSampleLevel(uCurrentTex, uSampler, uv + vec2f(f32(x), f32(y)) * invSize, 0.0).rgb;
          m1 += s;
          m2 += s * s;
        }
      }

      let mu = m1 / 9.0;
      let sigma = sqrt(abs(m2 / 9.0 - mu * mu));
      let boxMin = mu - 1.25 * sigma;
      let boxMax = mu + 1.25 * sigma;

      var hist = textureSampleLevel(uHistoryTex, uSampler, uv, 0.0).rgb;
      hist = clamp(hist, boxMin, boxMax);

      return mix(hist, cur, u.taaWeight);
    }

    struct FragmentOutput {
      @location(0) screenColor: vec4f,
      @location(1) historyColor: vec4f,
    };

    @fragment
    fn fs_main(in: VertexOutput) -> FragmentOutput {
      var uv = in.uv;
      let resolved = sampleTAAResolve(uv);

      let zoomCenter = vec2f(0.64, 0.35);
      let insetPos = vec2f(0.70, 0.65);
      let insetSize = 0.28;

      var displayColor = resolved;
      if (uv.x > insetPos.x && uv.y > insetPos.y) {
        if (uv.x < insetPos.x + 0.003 || uv.y < insetPos.y + 0.004) {
          displayColor = vec3f(0.2, 0.8, 1.0);
        } else {
          let localUV = (uv - insetPos) / insetSize;
          let zoomedUV = zoomCenter + (localUV - vec2f(0.5)) * (1.0 / 5.0);
          displayColor = sampleTAAResolve(zoomedUV);
        }
      }

      var out: FragmentOutput;
      out.screenColor = vec4f(displayColor, 1.0);
      out.historyColor = vec4f(resolved, 1.0);
      return out;
    }
  `;

  // MSAA 解析后期着色器
  const msaaPostShader = `
    @group(0) @binding(0) var uSampler: sampler;
    @group(0) @binding(1) var uTexture: texture_2d<f32>;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
      var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
      var out: VertexOutput;
      out.position = vec4f(pos[id], 0.0, 1.0);
      out.uv = pos[id] * 0.5 + 0.5;
      out.uv.y = 1.0 - out.uv.y;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      var uv = in.uv;
      let zoomCenter = vec2f(0.64, 0.35);
      let insetPos = vec2f(0.70, 0.65);
      let insetSize = 0.28;

      if (uv.x > insetPos.x && uv.y > insetPos.y) {
        if (uv.x < insetPos.x + 0.003 || uv.y < insetPos.y + 0.004) {
          return vec4f(0.2, 0.8, 1.0, 1.0);
        }
        let localUV = (uv - insetPos) / insetSize;
        let zoomedUV = zoomCenter + (localUV - vec2f(0.5)) * (1.0 / 5.0);
        return vec4f(textureSampleLevel(uTexture, uSampler, zoomedUV, 0.0).rgb, 1.0);
      }
      return vec4f(textureSampleLevel(uTexture, uSampler, uv, 0.0).rgb, 1.0);
    }
  `;

  // =========================================================================
  // 3. 构建统一的 112 字节显存缓冲与管线 (彻底消除大小报错)
  // =========================================================================
  const rawSampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });
  const linearSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  // 关键修复：统一分配 112 字节 (28 floats * 4 bytes)
  const uniformBuffer = device.createBuffer({
    size: 112,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sceneBGL = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });
  const sceneBG = device.createBindGroup({
    layout: sceneBGL,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const sceneModule = device.createShaderModule({ code: sceneShader });
  const singlePipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [sceneBGL] }),
    vertex: {
      module: sceneModule, entryPoint: "vs_main",
      buffers: [{
        arrayStride: 6 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" },
        ],
      }],
    },
    fragment: { module: sceneModule, entryPoint: "fs_main", targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
  });

  const msaaPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [sceneBGL] }),
    vertex: {
      module: sceneModule, entryPoint: "vs_main",
      buffers: [{
        arrayStride: 6 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" },
        ],
      }],
    },
    fragment: { module: sceneModule, entryPoint: "fs_main", targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    multisample: { count: 4 },
  });

  const postBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
  });

  const rawBlitPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [postBGL] }),
    vertex: { module: device.createShaderModule({ code: rawBlitShader }), entryPoint: "vs_main" },
    fragment: { module: device.createShaderModule({ code: rawBlitShader }), entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const fxaaPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [postBGL] }),
    vertex: { module: device.createShaderModule({ code: fxaaShader }), entryPoint: "vs_main" },
    fragment: { module: device.createShaderModule({ code: fxaaShader }), entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const msaaPostPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [postBGL] }),
    vertex: { module: device.createShaderModule({ code: msaaPostShader }), entryPoint: "vs_main" },
    fragment: { module: device.createShaderModule({ code: msaaPostShader }), entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const taaBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
  });

  const taaPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [taaBGL] }),
    vertex: { module: device.createShaderModule({ code: taaShader }), entryPoint: "vs_main" },
    fragment: {
      module: device.createShaderModule({ code: taaShader }),
      entryPoint: "fs_main",
      targets: [
        { format },
        { format: "rgba16float" },
      ],
    },
    primitive: { topology: "triangle-list" },
  });

  function halton(index: number, base: number): number {
    let result = 0, f = 1 / base, i = index;
    while (i > 0) {
      result += f * (i % base);
      i = Math.floor(i / base);
      f /= base;
    }
    return result;
  }
  const jitterCount = 8;
  const jitters: [number, number][] = [];
  for (let i = 1; i <= jitterCount; i++) {
    jitters.push([(halton(i, 2) - 0.5) * 0.7, (halton(i, 3) - 0.5) * 0.7]);
  }

  // =========================================================================
  // 4. 离屏纹理创建与尺寸管理
  // =========================================================================
  let sceneColorTex: GPUTexture | null = null;
  let msaaColorTex: GPUTexture | null = null;
  let msaaResolveTex: GPUTexture | null = null;
  let historyTex: [GPUTexture, GPUTexture] | null = null;

  let rawBG: GPUBindGroup | null = null;
  let fxaaBG: GPUBindGroup | null = null;
  let msaaPostBG: GPUBindGroup | null = null;
  let taaBGs: [GPUBindGroup, GPUBindGroup] | null = null;
  let frameIndex = 0;

  function resize(w: number, h: number) {
    if (sceneColorTex) sceneColorTex.destroy();
    if (msaaColorTex) msaaColorTex.destroy();
    if (msaaResolveTex) msaaResolveTex.destroy();
    if (historyTex) { historyTex[0].destroy(); historyTex[1].destroy(); }

    sceneColorTex = device.createTexture({
      size: [w, h],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    msaaColorTex = device.createTexture({
      size: [w, h],
      sampleCount: 4,
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    msaaResolveTex = device.createTexture({
      size: [w, h],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    historyTex = [
      device.createTexture({
        size: [w, h],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
      device.createTexture({
        size: [w, h],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
    ];

    rawBG = device.createBindGroup({
      layout: postBGL,
      entries: [
        { binding: 0, resource: rawSampler },
        { binding: 1, resource: sceneColorTex.createView() },
      ],
    });

    fxaaBG = device.createBindGroup({
      layout: postBGL,
      entries: [
        { binding: 0, resource: linearSampler },
        { binding: 1, resource: sceneColorTex.createView() },
      ],
    });

    msaaPostBG = device.createBindGroup({
      layout: postBGL,
      entries: [
        { binding: 0, resource: linearSampler },
        { binding: 1, resource: msaaResolveTex.createView() },
      ],
    });

    taaBGs = [
      device.createBindGroup({
        layout: taaBGL,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: linearSampler },
          { binding: 2, resource: sceneColorTex.createView() },
          { binding: 3, resource: historyTex[0].createView() },
        ],
      }),
      device.createBindGroup({
        layout: taaBGL,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: linearSampler },
          { binding: 2, resource: sceneColorTex.createView() },
          { binding: 3, resource: historyTex[1].createView() },
        ],
      }),
    ];
  }

  resize(canvas.width, canvas.height);

  function getCameraVP(width: number, height: number): { vp: Float32Array; camPos: number[] } {
    const eye = [0.0, 0.4, 2.5];
    const target = [0.0, 0.0, 0.0];
    const up = [0.0, 1.0, 0.0];

    const z = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
    const x = normalize(cross(up, z));
    const y = cross(z, x);

    const view = [
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
    ];

    const aspect = width / height;
    const fov = (42 * Math.PI) / 180;
    const f = 1.0 / Math.tan(fov / 2);
    const near = 0.1, far = 100.0;
    const rangeInv = 1.0 / (near - far);

    const proj = [
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, far * rangeInv, -1,
      0, 0, near * far * rangeInv, 0,
    ];

    const vp = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        let sum = 0;
        for (let k = 0; k < 4; k++) sum += view[i * 4 + k] * proj[k * 4 + j];
        vp[i * 4 + j] = sum;
      }
    }
    return { vp, camPos: [...eye, 1.0] };
  }

  function cross(a: number[], b: number[]): number[] {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }
  function normalize(v: number[]): number[] {
    const l = Math.hypot(...v);
    return [v[0] / l, v[1] / l, v[2] / l];
  }
  function dot(a: number[], b: number[]): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  // =========================================================================
  // 5. 渲染循环 (28 个 float 精确匹配 112 字节显存)
  // =========================================================================
  let animId: number;
  const uniformData = new Float32Array(28); // 28 * 4 = 112 bytes

  function render() {
    if (canvas.width !== sceneColorTex?.width || canvas.height !== sceneColorTex?.height) {
      resize(canvas.width, canvas.height);
    }

    const { vp, camPos } = getCameraVP(canvas.width, canvas.height);
    
    let jitter: [number, number] = [0.0, 0.0];
    let taaWeight = 0.10;

    if (currentMode === "taa") {
      jitter = jitters[frameIndex % jitterCount];
      if (taaAccumFrames === 0) {
        taaWeight = 1.0;
      }
      taaAccumFrames++;
    }

    uniformData.set(vp, 0);           // 0~15
    uniformData.set(camPos, 16);       // 16~19
    uniformData[20] = canvas.width;    // 20
    uniformData[21] = canvas.height;   // 21
    uniformData[22] = jitter[0];       // 22
    uniformData[23] = jitter[1];       // 23
    uniformData[24] = taaWeight;       // 24
    uniformData[25] = 0.0;             // 25 (pad0)
    uniformData[26] = 0.0;             // 26 (pad1)
    uniformData[27] = 0.0;             // 27 (pad2) -> 精确对齐 112 字节！
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();

    if (currentMode === "msaa") {
      const pass1 = encoder.beginRenderPass({
        colorAttachments: [{
          view: msaaColorTex!.createView(),
          resolveTarget: msaaResolveTex!.createView(),
          clearValue: { r: 0.04, g: 0.05, b: 0.07, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass1.setPipeline(msaaPipeline);
      pass1.setBindGroup(0, sceneBG);
      pass1.setVertexBuffer(0, vertBuffer);
      pass1.setIndexBuffer(idxBuffer, "uint16");
      pass1.drawIndexed(indices.length);
      pass1.end();

      const pass2 = encoder.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass2.setPipeline(msaaPostPipeline);
      pass2.setBindGroup(0, msaaPostBG!);
      pass2.draw(3);
      pass2.end();

    } else {
      const pass1 = encoder.beginRenderPass({
        colorAttachments: [{
          view: sceneColorTex!.createView(),
          clearValue: { r: 0.04, g: 0.05, b: 0.07, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass1.setPipeline(singlePipeline);
      pass1.setBindGroup(0, sceneBG);
      pass1.setVertexBuffer(0, vertBuffer);
      pass1.setIndexBuffer(idxBuffer, "uint16");
      pass1.drawIndexed(indices.length);
      pass1.end();

      if (currentMode === "none") {
        const pass2 = encoder.beginRenderPass({
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          }],
        });
        pass2.setPipeline(rawBlitPipeline);
        pass2.setBindGroup(0, rawBG!);
        pass2.draw(3);
        pass2.end();

      } else if (currentMode === "fxaa") {
        const pass2 = encoder.beginRenderPass({
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          }],
        });
        pass2.setPipeline(fxaaPipeline);
        pass2.setBindGroup(0, fxaaBG!);
        pass2.draw(3);
        pass2.end();

      } else if (currentMode === "taa") {
        const readIdx = frameIndex % 2;
        const writeIdx = 1 - readIdx;

        const pass2 = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: context.getCurrentTexture().createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
              loadOp: "clear",
              storeOp: "store",
            },
            {
              view: historyTex![writeIdx].createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
              loadOp: "clear",
              storeOp: "store",
            },
          ],
        });
        pass2.setPipeline(taaPipeline);
        pass2.setBindGroup(0, taaBGs![readIdx]);
        pass2.draw(3);
        pass2.end();
      }
    }

    device.queue.submit([encoder.finish()]);
    frameIndex++;
    animId = requestAnimationFrame(render);
  }
  render();

  return () => {
    cancelAnimationFrame(animId);
    panel.remove();
    vertBuffer.destroy();
    idxBuffer.destroy();
    uniformBuffer.destroy();
    sceneColorTex?.destroy();
    msaaColorTex?.destroy();
    msaaResolveTex?.destroy();
    historyTex?.[0].destroy();
    historyTex?.[1].destroy();
  };
}