// src/examples/sr_tsr.ts
export function runTSR(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  const canvas = context.canvas as HTMLCanvasElement;
  const container = canvas.parentElement || document.body;

  // 1. UI 面板
  const panel = document.createElement("div");
  panel.style.cssText = `
    position: absolute; top: 16px; left: 16px; z-index: 1000;
    background: rgba(15, 20, 28, 0.92); backdrop-filter: blur(8px);
    padding: 14px 18px; border-radius: 8px; font-family: monospace, sans-serif;
    color: #fff; font-size: 13px; box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.18); user-select: none;
  `;
  panel.innerHTML = `
    <div style="font-weight: bold; margin-bottom: 8px; font-size: 14px; color: #10b981;">TSR (Temporal Super Resolution 时序超分)</div>
    <div style="display: flex; gap: 6px; margin-bottom: 8px;">
      <button id="btn-tsr" style="flex:1; padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer; background: #10b981; color: #fff; font-weight: bold;">开启 TSR (0.5x 渲染，25% 负荷)</button>
      <button id="btn-low" style="flex:1; padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer; background: #262c36; color: #aaa;">原生 0.5x (模糊)</button>
    </div>
    <div id="sr-desc" style="color: #bbb; font-size: 11px; line-height: 1.4;">当前模式：TSR 超分。仅以 25% 像素面积进行着色渲染，借助时序亚像素插值重建出 1.0x 高清边缘。</div>
  `;
  container.style.position = "relative";
  container.appendChild(panel);

  let enableTSR = true;
  let tsrFrameAccum = 0;

  const btnTSR = panel.querySelector("#btn-tsr") as HTMLButtonElement;
  const btnLow = panel.querySelector("#btn-low") as HTMLButtonElement;
  const desc = panel.querySelector("#sr-desc") as HTMLDivElement;

  btnTSR.onclick = () => {
    enableTSR = true;
    tsrFrameAccum = 0;
    btnTSR.style.background = "#10b981"; btnTSR.style.color = "#fff"; btnTSR.style.fontWeight = "bold";
    btnLow.style.background = "#262c36"; btnLow.style.color = "#aaa"; btnLow.style.fontWeight = "normal";
    desc.innerText = "【开启 TSR】: 0.5x 低像素极速着色，时序空间累积重构，金属轮廓锐利且完全无狗牙。";
  };
  btnLow.onclick = () => {
    enableTSR = false;
    btnLow.style.background = "#e05656"; btnLow.style.color = "#fff"; btnLow.style.fontWeight = "bold";
    btnTSR.style.background = "#262c36"; btnTSR.style.color = "#aaa"; btnTSR.style.fontWeight = "normal";
    desc.innerText = "【原生 0.5x】: 同样以 0.5x 渲染，但未经过 TSR 时域重构，画面直接放大后极度模糊。";
  };

  // 2. 几何网格
  const latBands = 40, lonBands = 40;
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

  // 3. 着色器代码定义
  const sceneShader = `
    struct Uniforms {
      viewProj: mat4x4f,
      camPos: vec4f,
      renderSize: vec2f,
      jitter: vec2f,
      tsrWeight: f32,
      pad0: f32, pad1: f32, pad2: f32,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexInput { @location(0) pos: vec3f, @location(1) norm: vec3f };
    struct VertexOutput { @builtin(position) position: vec4f, @location(0) worldPos: vec3f, @location(1) worldNorm: vec3f };

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
      var out: VertexOutput;
      var clip = u.viewProj * vec4f(in.pos, 1.0);
      clip.x += (u.jitter.x * 2.0 / u.renderSize.x) * clip.w;
      clip.y += (u.jitter.y * 2.0 / u.renderSize.y) * clip.w;
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
      return vec4f(pow(aces(litColor), vec3f(1.0 / 2.2)), 1.0);
    }
  `;

  // TSR 核心时序超分辨率重构着色器 (双输出 MRT)
  const tsrShader = `
    struct Uniforms {
      viewProj: mat4x4f,
      camPos: vec4f,
      renderSize: vec2f,
      jitter: vec2f,
      tsrWeight: f32,
      pad0: f32, pad1: f32, pad2: f32,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var uSampler: sampler;
    @group(0) @binding(2) var uLowResTex: texture_2d<f32>;
    @group(0) @binding(3) var uHistoryFullTex: texture_2d<f32>;

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

    struct FragmentOutput {
      @location(0) screenColor: vec4f,
      @location(1) historyColor: vec4f,
    };

    @fragment
    fn fs_main(in: VertexOutput) -> FragmentOutput {
      let invLow = 1.0 / u.renderSize;
      let cur = textureSampleLevel(uLowResTex, uSampler, in.uv, 0.0).rgb;

      // 提取低分辨率 3x3 邻域方差，约束超分重构范围
      var m1 = vec3f(0.0);
      var m2 = vec3f(0.0);
      for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
          let s = textureSampleLevel(uLowResTex, uSampler, in.uv + vec2f(f32(x), f32(y)) * invLow, 0.0).rgb;
          m1 += s;
          m2 += s * s;
        }
      }
      let mu = m1 / 9.0;
      let sigma = sqrt(abs(m2 / 9.0 - mu * mu));
      let boxMin = mu - 1.25 * sigma;
      let boxMax = mu + 1.25 * sigma;

      // 采样 1.0x 物理全分辨率的历史帧，并应用方差截断
      var hist = textureSampleLevel(uHistoryFullTex, uSampler, in.uv, 0.0).rgb;
      hist = clamp(hist, boxMin, boxMax);

      // 时序超采样累积融合
      let resolved = mix(hist, cur, u.tsrWeight);

      var out: FragmentOutput;
      out.screenColor = vec4f(resolved, 1.0);
      out.historyColor = vec4f(resolved, 1.0); // 存回 1.0x 物理历史纹理
      return out;
    }
  `;

  // 低分直拉着色器
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

  // 4. 管线配置 (112 字节对齐)
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

  const tsrBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
  });
  const tsrPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [tsrBGL] }),
    vertex: { module: device.createShaderModule({ code: tsrShader }), entryPoint: "vs_main" },
    fragment: { module: device.createShaderModule({ code: tsrShader }), entryPoint: "fs_main", targets: [{ format }, { format: "rgba16float" }] },
    primitive: { topology: "triangle-list" },
  });

  const blitBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
  });
  const blitPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [blitBGL] }),
    vertex: { module: device.createShaderModule({ code: blitShader }), entryPoint: "vs_main" },
    fragment: { module: device.createShaderModule({ code: blitShader }), entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // Halton(2, 3) 8 相位循环抖动
  function halton(index: number, base: number) {
    let result = 0, f = 1 / base, i = index;
    while (i > 0) { result += f * (i % base); i = Math.floor(i / base); f /= base; }
    return result;
  }
  const jitterCount = 8;
  const jitters: [number, number][] = [];
  for (let i = 1; i <= jitterCount; i++) {
    jitters.push([(halton(i, 2) - 0.5) * 0.7, (halton(i, 3) - 0.5) * 0.7]);
  }

  // 纹理分配 (0.5x 低分辨率场景纹理 + 1.0x 物理全分辨率历史纹理)
  let lowResTex: GPUTexture | null = null;
  let historyFullTex: [GPUTexture, GPUTexture] | null = null;
  let tsrBGs: [GPUBindGroup, GPUBindGroup] | null = null;
  let blitBG: GPUBindGroup | null = null;
  let frameIndex = 0;

  function resize(fullW: number, fullH: number) {
    const lowW = Math.max(1, Math.floor(fullW * 0.5));
    const lowH = Math.max(1, Math.floor(fullH * 0.5));

    if (lowResTex) lowResTex.destroy();
    if (historyFullTex) { historyFullTex[0].destroy(); historyFullTex[1].destroy(); }

    lowResTex = device.createTexture({
      size: [lowW, lowH],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    historyFullTex = [
      device.createTexture({ size: [fullW, fullH], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
      device.createTexture({ size: [fullW, fullH], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
    ];

    tsrBGs = [
      device.createBindGroup({
        layout: tsrBGL,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: linearSampler },
          { binding: 2, resource: lowResTex.createView() },
          { binding: 3, resource: historyFullTex[0].createView() },
        ],
      }),
      device.createBindGroup({
        layout: tsrBGL,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: linearSampler },
          { binding: 2, resource: lowResTex.createView() },
          { binding: 3, resource: historyFullTex[1].createView() },
        ],
      }),
    ];

    blitBG = device.createBindGroup({
      layout: blitBGL,
      entries: [
        { binding: 0, resource: linearSampler },
        { binding: 1, resource: lowResTex.createView() },
      ],
    });
  }

  function getCameraVP(w: number, h: number) {
    const eye = [0.0, 0.4, 2.5], target = [0.0, 0.0, 0.0], up = [0.0, 1.0, 0.0];
    const z = normalize([eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]]);
    const x = normalize(cross(up, z)); const y = cross(z, x);
    const view = [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
    const aspect = w / h, f = 1.0 / Math.tan((42 * Math.PI) / 360), near = 0.1, far = 100.0, rInv = 1.0 / (near - far);
    const proj = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * rInv, -1, 0, 0, near * far * rInv, 0];
    const vp = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      let sum = 0; for (let k = 0; k < 4; k++) sum += view[i * 4 + k] * proj[k * 4 + j];
      vp[i * 4 + j] = sum;
    }
    return { vp, camPos: [...eye, 1.0] };
  }
  function cross(a: number[], b: number[]) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
  function normalize(v: number[]) { const l = Math.hypot(...v); return [v[0]/l, v[1]/l, v[2]/l]; }
  function dot(a: number[], b: number[]) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

  resize(canvas.width, canvas.height);
  const uniformData = new Float32Array(28);

  let animId: number;
  function render() {
    const lowW = Math.max(1, Math.floor(canvas.width * 0.5));
    const lowH = Math.max(1, Math.floor(canvas.height * 0.5));

    if (lowResTex?.width !== lowW || historyFullTex?.[0].width !== canvas.width) {
      resize(canvas.width, canvas.height);
    }

    const { vp, camPos } = getCameraVP(canvas.width, canvas.height);
    const jitter = enableTSR ? jitters[frameIndex % jitterCount] : [0.0, 0.0];
    const tsrWeight = tsrFrameAccum === 0 ? 1.0 : 0.10;
    if (enableTSR) tsrFrameAccum++;

    uniformData.set(vp, 0);
    uniformData.set(camPos, 16);
    uniformData[20] = lowW;
    uniformData[21] = lowH;
    uniformData[22] = jitter[0];
    uniformData[23] = jitter[1];
    uniformData[24] = tsrWeight;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();

    // Pass 1: 0.5x 低分辨率场景渲染 (显存带宽消耗降低 75%)
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

    if (enableTSR) {
      // Pass 2: TSR 时序超采样重建至 1.0x 屏幕物理分辨率
      const readIdx = frameIndex % 2;
      const writeIdx = 1 - readIdx;

      const pass2 = encoder.beginRenderPass({
        colorAttachments: [
          { view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" },
          { view: historyFullTex![writeIdx].createView(), loadOp: "clear", storeOp: "store" },
        ],
      });
      pass2.setPipeline(tsrPipeline);
      pass2.setBindGroup(0, tsrBGs![readIdx]);
      pass2.draw(3);
      pass2.end();

    } else {
      // 未超分的模糊双线性对照
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }],
      });
      pass.setPipeline(blitPipeline);
      pass.setBindGroup(0, blitBG!);
      pass.draw(3);
      pass.end();
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
    lowResTex?.destroy();
    historyFullTex?.[0].destroy();
    historyFullTex?.[1].destroy();
  };
}