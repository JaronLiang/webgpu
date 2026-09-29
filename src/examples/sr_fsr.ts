// src/examples/sr_fsr_advanced.ts
export function runFSR(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  const canvas = context.canvas as HTMLCanvasElement;
  const container = canvas.parentElement || document.body;

  // =========================================================================
  // 0. 控制面板与 实时性能 HUD
  // =========================================================================
  const panel = document.createElement("div");
  panel.style.cssText = `
    position: absolute; top: 16px; left: 16px; z-index: 1000;
    background: rgba(13, 17, 23, 0.94); backdrop-filter: blur(10px);
    padding: 16px 20px; border-radius: 10px; font-family: monospace, sans-serif;
    color: #fff; font-size: 13px; box-shadow: 0 12px 36px rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.18); user-select: none; width: 330px;
  `;
  panel.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
      <span style="font-weight:bold; font-size:15px; color:#ff5252;">AMD FSR 1.0 工业级空间超分</span>
      <span id="fps-counter" style="color:#00e676; font-weight:bold;">-- FPS</span>
    </div>

    <div style="background:rgba(255,255,255,0.06); padding:8px 10px; border-radius:6px; margin-bottom:12px; font-size:11px; line-height:1.5;">
      <div>着色像素负荷: <span id="pixel-load" style="color:#ffb74d;">--</span></div>
      <div>单帧着色耗时: <span id="frame-time" style="color:#00e676;">-- ms</span></div>
    </div>

    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px; margin-bottom:10px;">
      <button id="btn-fsr-uq" style="padding:7px; border-radius:4px; border:none; cursor:pointer; background:#e05656; color:#fff; font-weight:bold;">FSR 极佳 (0.77x)</button>
      <button id="btn-fsr-q"  style="padding:7px; border-radius:4px; border:none; cursor:pointer; background:#262c36; color:#aaa;">FSR 性能 (0.67x)</button>
      <button id="btn-bilinear" style="padding:7px; border-radius:4px; border:none; cursor:pointer; background:#262c36; color:#aaa;">普通双线性拉伸</button>
      <button id="btn-native" style="padding:7px; border-radius:4px; border:none; cursor:pointer; background:#262c36; color:#aaa;">原生 1.0x 全负荷</button>
    </div>

    <div style="display:flex; align-items:center; justify-content:space-between; margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.1);">
      <span style="font-size:11px; color:#ccc;">多重 GGX 复杂物理负载:</span>
      <label style="cursor:pointer; display:flex; align-items:center; gap:4px;">
        <input type="checkbox" id="stress-toggle" checked style="accent-color:#ff5252;">
        <span style="font-size:11px; color:#ffb74d; font-weight:bold;">开启</span>
      </label>
    </div>
  `;
  container.style.position = "relative";
  container.appendChild(panel);

  type FSRMode = "fsr-uq" | "fsr-q" | "bilinear" | "native";
  let currentMode: FSRMode = "fsr-uq";
  let stressLoad = true;

  const btnFSRUQ = panel.querySelector("#btn-fsr-uq") as HTMLButtonElement;
  const btnFSRQ = panel.querySelector("#btn-fsr-q") as HTMLButtonElement;
  const btnBilinear = panel.querySelector("#btn-bilinear") as HTMLButtonElement;
  const btnNative = panel.querySelector("#btn-native") as HTMLButtonElement;
  const stressToggle = panel.querySelector("#stress-toggle") as HTMLInputElement;

  const fpsElem = panel.querySelector("#fps-counter") as HTMLSpanElement;
  const loadElem = panel.querySelector("#pixel-load") as HTMLSpanElement;
  const timeElem = panel.querySelector("#frame-time") as HTMLSpanElement;

  function updateButtons() {
    [btnFSRUQ, btnFSRQ, btnBilinear, btnNative].forEach(b => {
      b.style.background = "#262c36";
      b.style.color = "#aaa";
      b.style.fontWeight = "normal";
    });
    if (currentMode === "fsr-uq") {
      btnFSRUQ.style.background = "#e05656"; btnFSRUQ.style.color = "#fff"; btnFSRUQ.style.fontWeight = "bold";
    } else if (currentMode === "fsr-q") {
      btnFSRQ.style.background = "#ff793f"; btnFSRQ.style.color = "#fff"; btnFSRQ.style.fontWeight = "bold";
    } else if (currentMode === "bilinear") {
      btnBilinear.style.background = "#3b82f6"; btnBilinear.style.color = "#fff"; btnBilinear.style.fontWeight = "bold";
    } else {
      btnNative.style.background = "#10b981"; btnNative.style.color = "#fff"; btnNative.style.fontWeight = "bold";
    }
  }

  btnFSRUQ.onclick = () => { currentMode = "fsr-uq"; updateButtons(); };
  btnFSRQ.onclick = () => { currentMode = "fsr-q"; updateButtons(); };
  btnBilinear.onclick = () => { currentMode = "bilinear"; updateButtons(); };
  btnNative.onclick = () => { currentMode = "native"; updateButtons(); };
  stressToggle.onchange = (e) => { stressLoad = (e.target as HTMLInputElement).checked; };

  // =========================================================================
  // 1. 生成精细金属球几何体
  // =========================================================================
  const latBands = 64, lonBands = 64;
  const positions: number[] = [], normals: number[] = [], indices: number[] = [];
  for (let lat = 0; lat <= latBands; lat++) {
    const theta = (lat * Math.PI) / latBands;
    for (let lon = 0; lon <= lonBands; lon++) {
      const phi = (lon * 2 * Math.PI) / lonBands;
      const x = Math.cos(phi) * Math.sin(theta);
      const y = Math.cos(theta);
      const z = Math.sin(phi) * Math.sin(theta);
      positions.push(x * 0.85, y * 0.85, z * 0.85);
      normals.push(x, y, z);
    }
  }
  for (let lat = 0; lat < latBands; lat++) {
    for (let lon = 0; lon < lonBands; lon++) {
      const first = lat * (lonBands + 1) + lon;
      const second = first + lonBands + 1;
      indices.push(first, second, first + 1, second, second + 1, first + 1);
    }
  }

  const vertBuffer = device.createBuffer({ size: positions.length * 6 * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  const vertArray = new Float32Array(positions.length * 2);
  for (let i = 0; i < positions.length / 3; i++) {
    vertArray[i * 6 + 0] = positions[i * 3 + 0]; vertArray[i * 6 + 1] = positions[i * 3 + 1]; vertArray[i * 6 + 2] = positions[i * 3 + 2];
    vertArray[i * 6 + 3] = normals[i * 3 + 0]; vertArray[i * 6 + 4] = normals[i * 3 + 1]; vertArray[i * 6 + 5] = normals[i * 3 + 2];
  }
  device.queue.writeBuffer(vertBuffer, 0, vertArray);

  const idxBuffer = device.createBuffer({ size: indices.length * 2, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(idxBuffer, 0, new Uint16Array(indices));

  // =========================================================================
  // 2. 着色器代码定义
  // =========================================================================

  // 场景着色器：带有 8 光源次世代 GGX 微表面积分（展现显著性能差异）
  const sceneShader = `
    struct Uniforms {
      viewProj: mat4x4f,   // 0~63
      camPos: vec4f,       // 64~79
      renderSize: vec2f,   // 80~87
      displaySize: vec2f,  // 88~95
      stressMode: f32,     // 96~99 (是否开启重负载)
      pad0: f32, pad1: f32, pad2: f32, // 100~111 -> 112 bytes
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexInput { @location(0) pos: vec3f, @location(1) norm: vec3f };
    struct VertexOutput { @builtin(position) position: vec4f, @location(0) worldPos: vec3f, @location(1) worldNorm: vec3f };

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
      var out: VertexOutput;
      out.position = u.viewProj * vec4f(in.pos, 1.0);
      out.worldPos = in.pos;
      out.worldNorm = in.norm;
      return out;
    }

    fn distributionGGX(N: vec3f, H: vec3f, roughness: f32) -> f32 {
      let a = roughness * roughness;
      let a2 = a * a;
      let NdotH = max(dot(N, H), 0.0);
      let NdotH2 = NdotH * NdotH;
      let denom = (NdotH2 * (a2 - 1.0) + 1.0);
      return a2 / (3.14159265 * denom * denom);
    }

    fn geometrySmith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
      let r = roughness + 1.0;
      let k = (r * r) / 8.0;
      let ggx2 = NdotV / (NdotV * (1.0 - k) + k);
      let ggx1 = NdotL / (NdotL * (1.0 - k) + k);
      return ggx1 * ggx2;
    }

    fn aces(c: vec3f) -> vec3f {
      let a = 2.51; let b = 0.03; let c1 = 2.43; let d = 0.59; let e = 0.14;
      return clamp((c * (a * c + b)) / (c * (c1 * c + d) + e), vec3f(0.0), vec3f(1.0));
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let N = normalize(in.worldNorm);
      let V = normalize(u.camPos.xyz - in.worldPos);
      let NdotV = max(dot(N, V), 0.001);

      let albedo = vec3f(1.0, 0.78, 0.32);
      let F0 = albedo;
      let roughness = 0.15;

      var totalSpecular = vec3f(0.0);

      // 8 个摄影棚光源 (开启重负载时进行完整 GGX 微表面渲染计算，充分压榨 Fillrate)
      let lightLoops = select(2, 8, u.stressMode > 0.5);

      for (var i = 0; i < 8; i++) {
        if (i >= lightLoops) { break; }
        let angle = f32(i) * 0.785398;
        let L = normalize(vec3f(cos(angle) * 1.5, sin(angle) * 0.5 + 0.8, sin(angle) * 1.5));
        let H = normalize(V + L);
        let NdotL = max(dot(N, L), 0.001);

        let D = distributionGGX(N, H, roughness);
        let G = geometrySmith(NdotV, NdotL, roughness);
        let F = F0 + (vec3f(1.0) - F0) * pow(clamp(1.0 - max(dot(H, V), 0.0), 0.0, 1.0), 5.0);

        let numerator = D * G * F;
        let denominator = 4.0 * NdotV * NdotL + 0.0001;
        let spec = (numerator / denominator) * NdotL * 4.0;
        totalSpecular += spec;
      }

      let ambient = albedo * 0.08;
      let finalColor = ambient + totalSpecular;
      return vec4f(pow(aces(finalColor), vec3f(1.0 / 2.2)), 1.0);
    }
  `;

  // 关键优化：AMD FSR 1.0 官方标准 12-Tap 椭圆 Lanczos 空间上采样 (EASU)
  const easuShader = `
    struct Uniforms {
      viewProj: mat4x4f,
      camPos: vec4f,
      renderSize: vec2f,
      displaySize: vec2f,
      stressMode: f32,
      pad0: f32, pad1: f32, pad2: f32,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var uSampler: sampler;
    @group(0) @binding(2) var uInputTex: texture_2d<f32>;

    struct VertexOutput { @builtin(position) position: vec4f, @location(0) uv: vec2f };

    @vertex
    fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
      var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
      var out: VertexOutput;
      out.position = vec4f(pos[id], 0.0, 1.0);
      out.uv = pos[id] * 0.5 + 0.5;
      out.uv.y = 1.0 - out.uv.y;
      return out;
    }

    // EASU 近似 Lanczos-2 权重核函数
    fn lanczos2Weight(d: f32) -> f32 {
      if (d <= 0.0001) { return 1.0; }
      if (d >= 2.0) { return 0.0; }
      let pi_d = 3.14159265 * d;
      return (sin(pi_d) / pi_d) * (sin(pi_d * 0.5) / (pi_d * 0.5));
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let invRender = 1.0 / u.renderSize;
      let pos = in.uv * u.renderSize - 0.5;
      let center = floor(pos);
      let f = pos - center;

      // 12-Tap 采样网格 (中央 2x2 + 交叉外延 8 个点，消除块状锯齿)
      let tc = (center + 0.5) * invRender;

      let c00 = textureSampleLevel(uInputTex, uSampler, tc + vec2f( 0.0,  0.0) * invRender, 0.0).rgb;
      let c10 = textureSampleLevel(uInputTex, uSampler, tc + vec2f( 1.0,  0.0) * invRender, 0.0).rgb;
      let c01 = textureSampleLevel(uInputTex, uSampler, tc + vec2f( 0.0,  1.0) * invRender, 0.0).rgb;
      let c11 = textureSampleLevel(uInputTex, uSampler, tc + vec2f( 1.0,  1.0) * invRender, 0.0).rgb;

      let cN0 = textureSampleLevel(uInputTex, uSampler, tc + vec2f( 0.0, -1.0) * invRender, 0.0).rgb;
      let cN1 = textureSampleLevel(uInputTex, uSampler, tc + vec2f( 1.0, -1.0) * invRender, 0.0).rgb;
      let cS0 = textureSampleLevel(uInputTex, uSampler, tc + vec2f( 0.0,  2.0) * invRender, 0.0).rgb;
      let cS1 = textureSampleLevel(uInputTex, uSampler, tc + vec2f( 1.0,  2.0) * invRender, 0.0).rgb;

      let cW0 = textureSampleLevel(uInputTex, uSampler, tc + vec2f(-1.0,  0.0) * invRender, 0.0).rgb;
      let cW1 = textureSampleLevel(uInputTex, uSampler, tc + vec2f(-1.0,  1.0) * invRender, 0.0).rgb;
      let cE0 = textureSampleLevel(uInputTex, uSampler, tc + vec2f( 2.0,  0.0) * invRender, 0.0).rgb;
      let cE1 = textureSampleLevel(uInputTex, uSampler, tc + vec2f( 2.0,  1.0) * invRender, 0.0).rgb;

      // 提取亮度感知梯度
      let l00 = dot(c00, vec3f(0.299, 0.587, 0.114));
      let l10 = dot(c10, vec3f(0.299, 0.587, 0.114));
      let l01 = dot(c01, vec3f(0.299, 0.587, 0.114));
      let l11 = dot(c11, vec3f(0.299, 0.587, 0.114));

      // 特征边缘梯度主方向判定
      let dirX = (l10 + l11) - (l00 + l01);
      let dirY = (l01 + l11) - (l00 + l10);
      let edgeLength = length(vec2f(dirX, dirY));
      let dir = select(vec2f(1.0, 0.0), normalize(vec2f(dirX, dirY)), edgeLength > 0.001);

      // 根据边缘强度拉伸椭圆旋转核
      let stretch = clamp(1.0 + edgeLength * 3.5, 1.0, 2.5);

      var colorAcc = vec3f(0.0);
      var weightAcc = 0.0;

      // 核心 12 采样点 Lanczos 积分累加
      let offsets = array<vec2f, 12>(
        vec2f( 0.0,  0.0), vec2f( 1.0,  0.0), vec2f( 0.0,  1.0), vec2f( 1.0,  1.0),
        vec2f( 0.0, -1.0), vec2f( 1.0, -1.0), vec2f( 0.0,  2.0), vec2f( 1.0,  2.0),
        vec2f(-1.0,  0.0), vec2f(-1.0,  1.0), vec2f( 2.0,  0.0), vec2f( 2.0,  1.0)
      );

      let samples = array<vec3f, 12>(c00, c10, c01, c11, cN0, cN1, cS0, cS1, cW0, cW1, cE0, cE1);

      for (var i = 0; i < 12; i++) {
        let delta = offsets[i] - f;
        let proj = dot(delta, dir);
        let perp = delta.x * dir.y - delta.y * dir.x;
        let dist = sqrt((proj * proj) / (stretch * stretch) + perp * perp);

        let w = lanczos2Weight(dist);
        colorAcc += samples[i] * w;
        weightAcc += w;
      }

      return vec4f(colorAcc / max(weightAcc, 0.0001), 1.0);
    }
  `;

  // 关键优化：AMD 官方标准 RCAS 锐化 Shader
  const rcasShader = `
    @group(0) @binding(0) var uSampler: sampler;
    @group(0) @binding(1) var uTexture: texture_2d<f32>;

    struct VertexOutput { @builtin(position) position: vec4f, @location(0) uv: vec2f };

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
      let dims = max(vec2f(textureDimensions(uTexture)), vec2f(1.0));
      let invSize = 1.0 / dims;

      let e = textureSampleLevel(uTexture, uSampler, in.uv, 0.0).rgb;
      let b = textureSampleLevel(uTexture, uSampler, in.uv + vec2f( 0.0, -1.0) * invSize, 0.0).rgb;
      let d = textureSampleLevel(uTexture, uSampler, in.uv + vec2f(-1.0,  0.0) * invSize, 0.0).rgb;
      let f = textureSampleLevel(uTexture, uSampler, in.uv + vec2f( 1.0,  0.0) * invSize, 0.0).rgb;
      let h = textureSampleLevel(uTexture, uSampler, in.uv + vec2f( 0.0,  1.0) * invSize, 0.0).rgb;

      let minRing = min(e, min(min(b, d), min(f, h)));
      let maxRing = max(e, max(max(b, d), max(f, h)));

      // RCAS 对比度自适应衰减 (杜绝高光过冲振铃伪影)
      let sharpness = 0.24;
      let peak = -1.0 / mix(8.0, 4.5, sharpness);
      let lobe = min(minRing, 1.5 - maxRing) * peak;

      let resolved = (b + d + f + h) * lobe + e;
      let den = 4.0 * lobe + 1.0;
      return vec4f(resolved / max(den, vec3f(0.001)), 1.0);
    }
  `;

  // 普通 Blit
  const blitShader = `
    @group(0) @binding(0) var uSampler: sampler;
    @group(0) @binding(1) var uTexture: texture_2d<f32>;
    struct VertexOutput { @builtin(position) position: vec4f, @location(0) uv: vec2f };
    @vertex fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
      var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
      var out: VertexOutput; out.position = vec4f(pos[id], 0.0, 1.0);
      out.uv = pos[id] * 0.5 + 0.5; out.uv.y = 1.0 - out.uv.y; return out;
    }
    @fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      return textureSampleLevel(uTexture, uSampler, in.uv, 0.0);
    }
  `;

  // =========================================================================
  // 3. 构建统一管线与资源
  // =========================================================================
  const linearSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  const uniformBuffer = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const sceneBGL = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });
  const sceneBG = device.createBindGroup({ layout: sceneBGL, entries: [{ binding: 0, resource: { buffer: uniformBuffer } }] });

  const scenePipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [sceneBGL] }),
    vertex: {
      module: device.createShaderModule({ code: sceneShader }), entryPoint: "vs_main",
      buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x3" }] }],
    },
    fragment: { module: device.createShaderModule({ code: sceneShader }), entryPoint: "fs_main", targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
  });

  const easuBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
  });
  const easuPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [easuBGL] }),
    vertex: { module: device.createShaderModule({ code: easuShader }), entryPoint: "vs_main" },
    fragment: { module: device.createShaderModule({ code: easuShader }), entryPoint: "fs_main", targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list" },
  });

  const postBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
  });

  const rcasPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [postBGL] }),
    vertex: { module: device.createShaderModule({ code: rcasShader }), entryPoint: "vs_main" },
    fragment: { module: device.createShaderModule({ code: rcasShader }), entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const blitPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [postBGL] }),
    vertex: { module: device.createShaderModule({ code: blitShader }), entryPoint: "vs_main" },
    fragment: { module: device.createShaderModule({ code: blitShader }), entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  let lowResTex: GPUTexture | null = null;
  let easuTex: GPUTexture | null = null;
  let easuBG: GPUBindGroup | null = null;
  let rcasBG: GPUBindGroup | null = null;
  let blitBG: GPUBindGroup | null = null;

  function resize(displayW: number, displayH: number) {
    let scale = 0.77;
    if (currentMode === "native") scale = 1.0;
    else if (currentMode === "fsr-q" || currentMode === "bilinear") scale = 0.67;

    const renderW = Math.max(1, Math.floor(displayW * scale));
    const renderH = Math.max(1, Math.floor(displayH * scale));

    if (lowResTex) lowResTex.destroy();
    if (easuTex) easuTex.destroy();

    lowResTex = device.createTexture({
      size: [renderW, renderH],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    easuTex = device.createTexture({
      size: [displayW, displayH],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    easuBG = device.createBindGroup({
      layout: easuBGL,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: linearSampler },
        { binding: 2, resource: lowResTex.createView() },
      ],
    });

    rcasBG = device.createBindGroup({
      layout: postBGL,
      entries: [
        { binding: 0, resource: linearSampler },
        { binding: 1, resource: easuTex.createView() },
      ],
    });

    blitBG = device.createBindGroup({
      layout: postBGL,
      entries: [
        { binding: 0, resource: linearSampler },
        { binding: 1, resource: lowResTex.createView() },
      ],
    });
  }

  function getCameraVP(width: number, height: number): { vp: Float32Array; camPos: number[] } {
    const eye = [0.0, 0.35, 2.5], target = [0.0, 0.0, 0.0], up = [0.0, 1.0, 0.0];
    const z = normalize([eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]]);
    const x = normalize(cross(up, z)); const y = cross(z, x);
    const view = [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
    const aspect = width / height, f = 1.0 / Math.tan((42 * Math.PI) / 360), near = 0.1, far = 100.0, rInv = 1.0 / (near - far);
    const proj = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * rInv, -1, 0, 0, near * far * rInv, 0];
    const vp = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      let sum = 0; for (let k = 0; k < 4; k++) sum += view[i * 4 + k] * proj[k * 4 + j];
      vp[i * 4 + j] = sum;
    }
    return { vp, camPos: [...eye, 1.0] };
  }

  function cross(a: number[], b: number[]): number[] { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
  function normalize(v: number[]): number[] { const l = Math.hypot(...v); return [v[0]/l, v[1]/l, v[2]/l]; }
  function dot(a: number[], b: number[]): number { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

  resize(Math.max(canvas.width, 1), Math.max(canvas.height, 1));
  const uniformData = new Float32Array(28);

  // =========================================================================
  // 4. 渲染循环与实时性能采样统计
  // =========================================================================
  let animId: number;
  let lastTime = performance.now();
  let frameCount = 0;
  let fpsTimer = 0;

  function render() {
    const now = performance.now();
    const delta = now - lastTime;
    lastTime = now;

    // 统计每秒 FPS 与单帧耗时
    frameCount++;
    fpsTimer += delta;
    if (fpsTimer >= 500) {
      const curFPS = Math.round((frameCount * 1000) / fpsTimer);
      fpsElem.innerText = `${curFPS} FPS`;
      timeElem.innerText = `${(fpsTimer / frameCount).toFixed(2)} ms`;
      frameCount = 0;
      fpsTimer = 0;
    }

    const displayW = Math.max(canvas.width, 1);
    const displayH = Math.max(canvas.height, 1);

    let scale = 0.77; // Ultra Quality: 视觉近乎无损黄金比例
    if (currentMode === "native") scale = 1.0;
    else if (currentMode === "fsr-q" || currentMode === "bilinear") scale = 0.67;

    const renderW = Math.max(1, Math.floor(displayW * scale));
    const renderH = Math.max(1, Math.floor(displayH * scale));

    // 更新像素负载 HUD
    const totalPixels = displayW * displayH;
    const actualPixels = renderW * renderH;
    const percent = Math.round((actualPixels / totalPixels) * 100);
    loadElem.innerText = `${(actualPixels / 1000000).toFixed(2)}M 像素 (${percent}% 算力负荷)`;

    if (lowResTex?.width !== renderW || easuTex?.width !== displayW) {
      resize(displayW, displayH);
    }

    const { vp, camPos } = getCameraVP(displayW, displayH);
    uniformData.set(vp, 0);
    uniformData.set(camPos, 16);
    uniformData[20] = renderW;
    uniformData[21] = renderH;
    uniformData[22] = displayW;
    uniformData[23] = displayH;
    uniformData[24] = stressLoad ? 1.0 : 0.0; // 是否启用 8 路物理微表面积分压力负载
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();

    // Pass 1: 低分辨率场景着色
    const pass1 = encoder.beginRenderPass({
      colorAttachments: [{
        view: lowResTex!.createView(),
        clearValue: { r: 0.04, g: 0.05, b: 0.07, a: 1.0 },
        loadOp: "clear", storeOp: "store",
      }],
    });
    pass1.setPipeline(scenePipeline);
    pass1.setBindGroup(0, sceneBG);
    pass1.setVertexBuffer(0, vertBuffer);
    pass1.setIndexBuffer(idxBuffer, "uint16");
    pass1.drawIndexed(indices.length);
    pass1.end();

    if (currentMode === "fsr-uq" || currentMode === "fsr-q") {
      // Pass 2: 12-Tap 椭圆核定向 Lanczos 空间上采样
      const pass2 = encoder.beginRenderPass({
        colorAttachments: [{ view: easuTex!.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
      });
      pass2.setPipeline(easuPipeline);
      pass2.setBindGroup(0, easuBG!);
      pass2.draw(3);
      pass2.end();

      // Pass 3: RCAS 对比度自适应锐化
      const pass3 = encoder.beginRenderPass({
        colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
      });
      pass3.setPipeline(rcasPipeline);
      pass3.setBindGroup(0, rcasBG!);
      pass3.draw(3);
      pass3.end();

    } else {
      // 普通双线性拉伸或原生 1.0x 渲染
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
      });
      pass.setPipeline(blitPipeline);
      pass.setBindGroup(0, blitBG!);
      pass.draw(3);
      pass.end();
    }

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(render);
  }
  render();

  return () => {
    cancelAnimationFrame(animId);
    panel.remove();
    vertBuffer.destroy();
    idxBuffer.destroy();
    uniformBuffer.destroy();
    lowResTex?.destroy();
    easuTex?.destroy();
  };
}