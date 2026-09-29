// src/examples/fsr2Demo.ts
import { Mat4 } from "../utils/math";
import type { SimpleGUI } from "../utils/gui";

// ==========================================
// 1. 低差异序列 (Halton 2,3 - 8/16 Phase)
// ==========================================
function halton(index: number, base: number): number {
  let result = 0;
  let f = 1 / base;
  let i = index;
  while (i > 0) {
    result += f * (i % base);
    i = Math.floor(i / base);
    f = f / base;
  }
  return result;
}

export async function runFSR2Demo(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  const displayWidth = canvas.width || 800;
  const displayHeight = canvas.height || 600;
  const renderScale = 0.5; // 50% 分辨率超分到 100%
  const renderWidth = Math.floor(displayWidth * renderScale);
  const renderHeight = Math.floor(displayHeight * renderScale);

  // ==========================================
  // 2. WGSL 着色器定义
  // ==========================================

  // --- Stage A: 低分辨率渲染 (颜色 + 纯几何运动矢量) ---
  const sceneShaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,          // 包含 Jitter 的当前帧变换
      nonJitterViewProj: mat4x4f, // 不含 Jitter 的当前帧变换
      prevNonJitterViewProj: mat4x4f, // 不含 Jitter 的前一帧变换
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexInput {
      @location(0) position: vec3f,
      @location(1) color: vec3f,
    };

    struct VertexOutput {
      @builtin(position) clipPos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) color: vec3f,
      @location(2) curPosClean: vec4f,
      @location(3) prevPosClean: vec4f,
    };

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
      var out: VertexOutput;
      let pos4 = vec4f(in.position, 1.0);

      // 光栅化位置：应用抗锯齿抖动 (Jitter)
      out.clipPos = u.viewProj * pos4;
      out.worldPos = in.position;
      out.color = in.color;

      // 运动矢量专用位置：严格不能包含任何 Jitter！
      out.curPosClean = u.nonJitterViewProj * pos4;
      out.prevPosClean = u.prevNonJitterViewProj * pos4;
      return out;
    }

    struct GBufferOutput {
      @location(0) color: vec4f,
      @location(1) motion: vec2f,
    };

    @fragment
    fn fs_main(in: VertexOutput) -> GBufferOutput {
      var out: GBufferOutput;
      
      // 程序化方格/条纹贴图用于测试摩尔纹与高频细节
      let coord = in.worldPos.xyz * 4.0;
      let f = abs(fract(coord - 0.5) - 0.5) / max(fwidth(coord), vec3f(0.001));
      let line = 1.0 - min(min(min(f.x, f.y), f.z), 1.0);
      let surfaceColor = mix(in.color, in.color * 0.3, line * 0.8);

      // 计算真实屏幕空间运动矢量 (UV 坐标系偏移，V 轴朝下)
      let curNDC = in.curPosClean.xy / in.curPosClean.w;
      let prevNDC = in.prevPosClean.xy / in.prevPosClean.w;

      let curUV = curNDC * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
      let prevUV = prevNDC * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);

      out.color = vec4f(surfaceColor, 1.0);
      // motion vector: 指向历史帧当前像素的位置: (curUV - prevUV)
      out.motion = curUV - prevUV;
      return out;
    }
  `;

  // --- Stage B: 工业级 FSR 2 时域超分重建 (Temporal Super Resolution) ---
  const fsr2TemporalCode = `
    struct FSR2Params {
      renderSize: vec2f,
      displaySize: vec2f,
      jitterOffset: vec2f, // 像素级抖动量 ([-0.5, 0.5])
      resetHistory: f32,
      pad: f32,
    };

    @group(0) @binding(0) var<uniform> fsr: FSR2Params;
    @group(0) @binding(1) var sLinear: sampler;
    @group(0) @binding(2) var tCurrentColor: texture_2d<f32>;
    @group(0) @binding(3) var tMotion: texture_2d<f32>;
    @group(0) @binding(4) var tCurrentDepth: texture_depth_2d;
    @group(0) @binding(5) var tHistoryColor: texture_2d<f32>;
    @group(0) @binding(6) var tHistoryDepth: texture_2d<f32>;
    @group(0) @binding(7) var tOutputColor: texture_storage_2d<rgba16float, write>;
    @group(0) @binding(8) var tOutputDepth: texture_storage_2d<r32float, write>;

    fn RGBToYCoCg(c: vec3f) -> vec3f {
      return vec3f(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b
      );
    }

    fn YCoCgToRGB(c: vec3f) -> vec3f {
      return max(vec3f(
        c.x + c.y - c.z,
        c.x + c.z,
        c.x - c.y - c.z
      ), vec3f(0.0));
    }

    // AABB 软裁剪
    fn clipToAABB(history: vec3f, boxMin: vec3f, boxMax: vec3f) -> vec3f {
      let p_clip = 0.5 * (boxMax + boxMin);
      let e_clip = 0.5 * (boxMax - boxMin) + 1e-5;
      let v_clip = history - p_clip;
      let v_unit = v_clip / e_clip;
      let a_unit = abs(v_unit);
      let ma_unit = max(max(a_unit.x, a_unit.y), a_unit.z);

      if (ma_unit > 1.0) {
        return p_clip + v_clip / ma_unit;
      }
      return history;
    }

    @compute @workgroup_size(8, 8)
    fn cs_main(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= u32(fsr.displaySize.x) || id.y >= u32(fsr.displaySize.y)) { return; }

      let displayPixel = vec2f(id.xy);
      let displayUV = (displayPixel + 0.5) / fsr.displaySize;
      let renderTexel = 1.0 / fsr.renderSize;

      // 对应的低分辨率采样 UV（去掉相机光栅化所施加的 Jitter）
      let jitterUVOffset = (fsr.jitterOffset / fsr.renderSize);
      let unjitteredUV = displayUV;
      let currentSampleUV = displayUV + jitterUVOffset;

      // 读取当前运动矢量与深度
      let motion = textureSampleLevel(tMotion, sLinear, clamp(currentSampleUV, vec2f(0.0), vec2f(1.0)), 0.0).xy;
      
      let depthCoord = vec2i(clamp(currentSampleUV * fsr.renderSize, vec2f(0.0), fsr.renderSize - 1.0));
      let rawDepth = textureLoad(tCurrentDepth, depthCoord, 0);

      // 第一帧或历史失效重置
      let centerColor = textureSampleLevel(tCurrentColor, sLinear, clamp(currentSampleUV, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
      if (fsr.resetHistory > 0.5) {
        textureStore(tOutputColor, id.xy, vec4f(centerColor, 1.0));
        textureStore(tOutputDepth, id.xy, vec4f(rawDepth, 0.0, 0.0, 0.0));
        return;
      }

      // 1. 3x3 邻域方差统计与特征提取 (YCoCg 空间)
      var m1 = vec3f(0.0);
      var m2 = vec3f(0.0);
      var cMin = vec3f(1e5);
      var cMax = vec3f(-1e5);

      for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
          let uv = currentSampleUV + vec2f(f32(x), f32(y)) * renderTexel;
          let smp = textureSampleLevel(tCurrentColor, sLinear, clamp(uv, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
          let ycocg = RGBToYCoCg(smp);
          m1 += ycocg;
          m2 += ycocg * ycocg;
          cMin = min(cMin, ycocg);
          cMax = max(cMax, ycocg);
        }
      }

      let mu = m1 / 9.0;
      let sigma = sqrt(max(m2 / 9.0 - mu * mu, vec3f(0.0)));
      
      // 动态边界系数：运动越剧烈，容差越小以防拖影
      let pixelVelocity = length(motion * fsr.displaySize);
      let gamma = mix(1.25, 0.75, clamp(pixelVelocity * 0.1, 0.0, 1.0));
      let boxMin = max(cMin, mu - gamma * sigma);
      let boxMax = min(cMax, mu + gamma * sigma);

      // 2. 重投影采样历史帧
      let historyUV = displayUV - motion;
      var historyColor = textureSampleLevel(tHistoryColor, sLinear, clamp(historyUV, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
      
      let histDepthCoord = vec2i(clamp(historyUV * fsr.displaySize, vec2f(0.0), fsr.displaySize - 1.0));
      let historyDepth = textureLoad(tHistoryDepth, histDepthCoord, 0).r;

      // 3. 深度遮挡检测 (Disocclusion)
      let depthDiff = abs(rawDepth - historyDepth);
      let isDisoccluded = depthDiff > 0.02 && rawDepth < historyDepth; // 当前比历史距离更近则判定被遮挡穿帮

      // 4. 颜色范围约束与软截断
      var historyYCoCg = RGBToYCoCg(historyColor);
      historyYCoCg = clipToAABB(historyYCoCg, boxMin, boxMax);
      historyColor = YCoCgToRGB(historyYCoCg);

      // 5. 自适应混合权重计算
      let isOffscreen = historyUV.x < 0.0 || historyUV.x > 1.0 || historyUV.y < 0.0 || historyUV.y > 1.0;
      var blendAlpha = clamp(0.05 + pixelVelocity * 0.02, 0.05, 0.8);

      if (isOffscreen || isDisoccluded) {
        blendAlpha = 1.0; // 发生解离或移出屏幕，完全丢弃历史
      }

      let finalColor = mix(historyColor, centerColor, blendAlpha);

      textureStore(tOutputColor, id.xy, vec4f(finalColor, 1.0));
      textureStore(tOutputDepth, id.xy, vec4f(rawDepth, 0.0, 0.0, 0.0));
    }
  `;

  // --- Stage C: RCAS 鲁棒对比度自适应锐化 ---
  const rcasShaderCode = `
    struct RCASParams {
      displaySize: vec2f,
      sharpness: f32,
      pad: f32,
    };
    @group(0) @binding(0) var<uniform> rcas: RCASParams;
    @group(0) @binding(1) var tInput: texture_2d<f32>;
    @group(0) @binding(2) var tOutput: texture_storage_2d<rgba16float, write>;

    @compute @workgroup_size(8, 8)
    fn cs_main(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= u32(rcas.displaySize.x) || id.y >= u32(rcas.displaySize.y)) { return; }
      let coord = vec2i(id.xy);

      let e = textureLoad(tInput, coord, 0).rgb;
      let b = textureLoad(tInput, max(coord + vec2i(0, -1), vec2i(0)), 0).rgb;
      let d = textureLoad(tInput, max(coord + vec2i(-1, 0), vec2i(0)), 0).rgb;
      let f = textureLoad(tInput, min(coord + vec2i(1, 0), vec2i(rcas.displaySize) - 1), 0).rgb;
      let h = textureLoad(tInput, min(coord + vec2i(0, 1), vec2i(rcas.displaySize) - 1), 0).rgb;

      let mn = min(min(min(d, e), min(f, b)), h);
      let mx = max(max(max(d, e), max(f, b)), h);
      
      let contrast = mx - mn;
      let amp = clamp(contrast / max(mx, vec3f(0.01)), vec3f(0.0), vec3f(1.0));
      let w = -amp * (rcas.sharpness * 0.18);

      let col = (b * w + d * w + f * w + h * w + e) / (vec3f(1.0) + 4.0 * w);
      textureStore(tOutput, id.xy, vec4f(max(col, vec3f(0.0)), 1.0));
    }
  `;

  // --- Stage D: 屏幕显示 Blit ---
  const blitShaderCode = `
    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) id: u32) -> VertexOut {
      // 修正的全屏大三角形，避免四边形剖分接缝
      var out: VertexOut;
      let x = f32((id << 1u) & 2u);
      let y = f32(id & 2u);
      out.uv = vec2f(x * 0.5, 1.0 - y * 0.5);
      out.pos = vec4f(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
      return out;
    }

    @group(0) @binding(0) var sSampler: sampler;
    @group(0) @binding(1) var tFinal: texture_2d<f32>;

    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4f {
      return textureSample(tFinal, sSampler, in.uv);
    }
  `;

  // ==========================================
  // 3. 构建 Pipelines 与资源
  // ==========================================
  const sLinear = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });

  const scenePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: sceneShaderCode }),
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
          ],
        },
      ],
    },
    fragment: {
      module: device.createShaderModule({ code: sceneShaderCode }),
      entryPoint: "fs_main",
      targets: [{ format: "rgba16float" }, { format: "rg16float" }],
    },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const fsr2TemporalPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: fsr2TemporalCode }), entryPoint: "cs_main" },
  });

  const rcasPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: rcasShaderCode }), entryPoint: "cs_main" },
  });

  const blitPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: blitShaderCode }), entryPoint: "vs_main" },
    fragment: { module: device.createShaderModule({ code: blitShaderCode }), entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  function createTextures() {
    return {
      lowResColor: device.createTexture({
        size: [renderWidth, renderHeight],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
      lowResMotion: device.createTexture({
        size: [renderWidth, renderHeight],
        format: "rg16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
      depthTexture: device.createTexture({
        size: [renderWidth, renderHeight],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
      historyColor: [
        device.createTexture({ size: [displayWidth, displayHeight], format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING }),
        device.createTexture({ size: [displayWidth, displayHeight], format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING }),
      ],
      historyDepth: [
        device.createTexture({ size: [displayWidth, displayHeight], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING }),
        device.createTexture({ size: [displayWidth, displayHeight], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING }),
      ],
      rcasOutput: device.createTexture({
        size: [displayWidth, displayHeight],
        format: "rgba16float",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      }),
    };
  }

  const tex = createTextures();

  // 立方体网格数据
  // prettier-ignore
  const vertexData = new Float32Array([
    -1, -1,  1,  1, 0.2, 0.2,   1, -1,  1,  0.2, 1, 0.2,   1,  1,  1,  0.2, 0.2, 1,  -1,  1,  1,  1, 1, 0.2,
    -1, -1, -1,  0.9, 0.2, 0.9, 1, -1, -1,  0.2, 0.9, 0.9, 1,  1, -1,  0.9, 0.9, 0.9, -1,  1, -1,  0.3, 0.3, 0.3,
    -1,  1, -1,  0.8, 0.4, 0.2, -1,  1,  1,  0.2, 0.8, 0.4, 1,  1,  1,  0.4, 0.2, 0.8,  1,  1, -1,  0.8, 0.8, 0.2,
    -1, -1, -1,  0.3, 0.7, 0.9,  1, -1, -1,  0.9, 0.3, 0.7, 1, -1,  1,  0.7, 0.9, 0.3, -1, -1,  1,  0.5, 0.5, 0.5,
     1, -1, -1,  1, 0.5, 0,     1,  1, -1,  1, 0, 0.5,     1,  1,  1,  0.5, 1, 0,      1, -1,  1,  0, 0.5, 1,
    -1, -1, -1,  0, 1, 0.5,    -1, -1,  1,  0.5, 0, 1,    -1,  1,  1,  1, 0.5, 0.5,   -1,  1, -1,  0.5, 1, 0.5,
  ]);
  // prettier-ignore
  const indexData = new Uint16Array([
    0,1,2, 0,2,3,       4,7,6, 4,6,5,       8,9,10, 8,10,11,
    12,15,14, 12,14,13, 16,17,18, 16,18,19, 20,23,22, 20,22,21
  ]);

  const vertexBuffer = device.createBuffer({ size: vertexData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);
  const indexBuffer = device.createBuffer({ size: indexData.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(indexBuffer, 0, indexData);

  // 3组 4x4 矩阵 = 48 个 float = 192 字节
  const sceneUBO = device.createBuffer({ size: 48 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const fsr2UBO = device.createBuffer({ size: 8 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const rcasUBO = device.createBuffer({ size: 4 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const sceneBindGroup = device.createBindGroup({
    layout: scenePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: sceneUBO } }],
  });

  // ==========================================
  // 4. 控制面板
  // ==========================================
  const fsrParams = { mode: 0, sharpness: 0.6, jitterEnabled: true };
  gui.add(fsrParams, "mode", 0, 1, 1).name("重建模式 (0:FSR2, 1:双线性拉伸)");
  gui.add(fsrParams, "sharpness", 0.0, 1.0, 0.05).name("RCAS 锐化强度");
  gui.add(fsrParams, "jitterEnabled").name("启用亚像素抖动");
  gui.addTextInfo("🚀 <b>修复项完成</b><br>1. 运动矢量已剔除抖动污染<br>2. 修正视口坐标系映射<br>3. 修复遮挡解离判断引起的严重拉丝");

  // ==========================================
  // 5. 渲染主循环
  // ==========================================
  let frameIndex = 0;
  let historyPingPong = 0;
  let historyValid = false;
  let prevCleanViewProj: Float32Array | null = null;
  let animId: number;

  function createLookAtMatrix(eye: number[], center: number[], up: number[]): Float32Array {
    const z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
    const lenZ = 1 / (Math.hypot(z0, z1, z2) || 1);
    const zx = z0 * lenZ, zy = z1 * lenZ, zz = z2 * lenZ;
    const x0 = up[1] * zz - up[2] * zy, x1 = up[2] * zx - up[0] * zz, x2 = up[0] * zy - up[1] * zx;
    const lenX = 1 / (Math.hypot(x0, x1, x2) || 1);
    const xx = x0 * lenX, xy = x1 * lenX, xz = x2 * lenX;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    const out = new Float32Array(16);
    out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
    out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
    out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
    out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    out[15] = 1;
    return out;
  }

  function frame() {
    frameIndex++;
    const currentHistIndex = historyPingPong;
    const nextHistIndex = 1 - historyPingPong;

    // 1. 生成抖动偏移 (Halton 序列，相位 8)
    let jitterPixelX = 0, jitterPixelY = 0;
    if (fsrParams.jitterEnabled && fsrParams.mode === 0) {
      jitterPixelX = halton((frameIndex % 8) + 1, 2) - 0.5;
      jitterPixelY = halton((frameIndex % 8) + 1, 3) - 0.5;
    }
    
    // NDC 空间下的抖动量 (注意渲染分辨率)
    const jitterNDCX = (jitterPixelX / renderWidth) * 2.0;
    const jitterNDCY = (jitterPixelY / renderHeight) * 2.0;

    // 2. 动态相机投影
    const aspect = renderWidth / renderHeight;
    const cleanProj = Mat4.perspective((45 * Math.PI) / 180, aspect, 0.1, 100);
    
    // 抖动投影矩阵（直接修改第 [8], [9] 项偏移光栅化）
    const jitteredProj = new Float32Array(cleanProj);
    jitteredProj[8] += jitterNDCX;
    jitteredProj[9] += jitterNDCY;

    const time = performance.now() * 0.0006;
    const eye = [Math.sin(time) * 4.2, 1.8, Math.cos(time) * 4.2];
    const view = createLookAtMatrix(eye, [0, 0, 0], [0, 1, 0]);

    const jitteredViewProj = Mat4.multiply(jitteredProj, view);
    const cleanViewProj = Mat4.multiply(cleanProj, view);

    if (!prevCleanViewProj) {
      prevCleanViewProj = new Float32Array(cleanViewProj);
    }

    // 3. 写入 Scene UBO
    const sceneData = new Float32Array(48);
    sceneData.set(jitteredViewProj, 0);     // 包含抖动用于顶点渲染
    sceneData.set(cleanViewProj, 16);       // 当前干净投影
    sceneData.set(prevCleanViewProj, 32);   // 上一帧干净投影
    device.queue.writeBuffer(sceneUBO, 0, sceneData);

    const encoder = device.createCommandEncoder();

    // ==========================================
    // Pass 1: G-Buffer 基础渲染
    // ==========================================
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        { view: tex.lowResColor.createView(), clearValue: { r: 0.05, g: 0.05, b: 0.07, a: 1 }, loadOp: "clear", storeOp: "store" },
        { view: tex.lowResMotion.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
      ],
      depthStencilAttachment: { view: tex.depthTexture.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    renderPass.setPipeline(scenePipeline);
    renderPass.setBindGroup(0, sceneBindGroup);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setIndexBuffer(indexBuffer, "uint16");
    renderPass.drawIndexed(indexData.length);
    renderPass.end();

    // ==========================================
    // Pass 2: FSR 2 时域重构
    // ==========================================
    if (fsrParams.mode === 0) {
      const fsrData = new Float32Array(8);
      fsrData[0] = renderWidth;
      fsrData[1] = renderHeight;
      fsrData[2] = displayWidth;
      fsrData[3] = displayHeight;
      fsrData[4] = jitterPixelX;
      fsrData[5] = jitterPixelY;
      fsrData[6] = !historyValid ? 1.0 : 0.0;
      fsrData[7] = 0.0;
      device.queue.writeBuffer(fsr2UBO, 0, fsrData);

      const fsr2ComputeBG = device.createBindGroup({
        layout: fsr2TemporalPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fsr2UBO } },
          { binding: 1, resource: sLinear },
          { binding: 2, resource: tex.lowResColor.createView() },
          { binding: 3, resource: tex.lowResMotion.createView() },
          { binding: 4, resource: tex.depthTexture.createView() },
          { binding: 5, resource: tex.historyColor[currentHistIndex].createView() },
          { binding: 6, resource: tex.historyDepth[currentHistIndex].createView() },
          { binding: 7, resource: tex.historyColor[nextHistIndex].createView() },
          { binding: 8, resource: tex.historyDepth[nextHistIndex].createView() },
        ],
      });

      const fsrPass = encoder.beginComputePass();
      fsrPass.setPipeline(fsr2TemporalPipeline);
      fsrPass.setBindGroup(0, fsr2ComputeBG);
      fsrPass.dispatchWorkgroups(Math.ceil(displayWidth / 8), Math.ceil(displayHeight / 8));
      fsrPass.end();

      // ==========================================
      // Pass 3: RCAS 锐化
      // ==========================================
      const rcasData = new Float32Array(4);
      rcasData.set([displayWidth, displayHeight, fsrParams.sharpness, 0.0]);
      device.queue.writeBuffer(rcasUBO, 0, rcasData);

      const rcasBG = device.createBindGroup({
        layout: rcasPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: rcasUBO } },
          { binding: 1, resource: tex.historyColor[nextHistIndex].createView() },
          { binding: 2, resource: tex.rcasOutput.createView() },
        ],
      });

      const rcasPass = encoder.beginComputePass();
      rcasPass.setPipeline(rcasPipeline);
      rcasPass.setBindGroup(0, rcasBG);
      rcasPass.dispatchWorkgroups(Math.ceil(displayWidth / 8), Math.ceil(displayHeight / 8));
      rcasPass.end();
    }

    // ==========================================
    // Pass 4: 屏幕 Blit (最终输出)
    // ==========================================
    const blitBG = device.createBindGroup({
      layout: blitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sLinear },
        { binding: 1, resource: fsrParams.mode === 0 ? tex.rcasOutput.createView() : tex.lowResColor.createView() },
      ],
    });

    const blitPass = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }],
    });
    blitPass.setPipeline(blitPipeline);
    blitPass.setBindGroup(0, blitBG);
    blitPass.draw(3); // 触发大三角形
    blitPass.end();

    device.queue.submit([encoder.finish()]);

    prevCleanViewProj.set(cleanViewProj);
    historyPingPong = nextHistIndex;
    historyValid = true;

    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    tex.lowResColor.destroy();
    tex.lowResMotion.destroy();
    tex.depthTexture.destroy();
    tex.historyColor[0].destroy();
    tex.historyColor[1].destroy();
    tex.historyDepth[0].destroy();
    tex.historyDepth[1].destroy();
    tex.rcasOutput.destroy();
    vertexBuffer.destroy();
    indexBuffer.destroy();
    sceneUBO.destroy();
    fsr2UBO.destroy();
    rcasUBO.destroy();
  };
}