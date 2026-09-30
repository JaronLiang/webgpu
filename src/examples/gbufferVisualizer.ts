// src/examples/gbufferVisualizer.ts

function createPerspectiveMatrix(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(fovRad / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect; out[5] = f;
  out[10] = far / (near - far); out[11] = -1;
  out[14] = (near * far) / (near - far);
  return out;
}

function createLookAtMatrix(eye: number[], center: number[], up: number[]): Float32Array {
  const z = [eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]];
  const lenZ = 1 / (Math.hypot(z[0], z[1], z[2]) || 1);
  z[0] *= lenZ; z[1] *= lenZ; z[2] *= lenZ;
  const x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]];
  const lenX = 1 / (Math.hypot(x[0], x[1], x[2]) || 1);
  x[0] *= lenX; x[1] *= lenX; x[2] *= lenX;
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const out = new Float32Array(16);
  out[0] = x[0]; out[1] = y[0]; out[2] = z[0]; out[3] = 0;
  out[4] = x[1]; out[5] = y[1]; out[6] = z[1]; out[7] = 0;
  out[8] = x[2]; out[9] = y[2]; out[10] = z[2]; out[11] = 0;
  out[12] = -(x[0]*eye[0] + x[1]*eye[1] + x[2]*eye[2]);
  out[13] = -(y[0]*eye[0] + y[1]*eye[1] + y[2]*eye[2]);
  out[14] = -(z[0]*eye[0] + z[1]*eye[1] + z[2]*eye[2]);
  out[15] = 1;
  return out;
}

// 几何生成器：生成标准 UV 球体（展示连续法线曲率）
function createSphere(radius: number, latSegments: number, lonSegments: number, center: [number, number, number], color: [number, number, number], pbr: [number, number]): number[] {
  const data: number[] = [];
  for (let y = 0; y < latSegments; y++) {
    const v0 = y / latSegments;
    const v1 = (y + 1) / latSegments;
    const theta0 = v0 * Math.PI;
    const theta1 = v1 * Math.PI;

    for (let x = 0; x < lonSegments; x++) {
      const u0 = x / lonSegments;
      const u1 = (x + 1) / lonSegments;
      const phi0 = u0 * Math.PI * 2;
      const phi1 = u1 * Math.PI * 2;

      const p0 = [center[0] + radius * Math.sin(theta0) * Math.cos(phi0), center[1] + radius * Math.cos(theta0), center[2] + radius * Math.sin(theta0) * Math.sin(phi0)];
      const p1 = [center[0] + radius * Math.sin(theta1) * Math.cos(phi0), center[1] + radius * Math.cos(theta1), center[2] + radius * Math.sin(theta1) * Math.sin(phi0)];
      const p2 = [center[0] + radius * Math.sin(theta1) * Math.cos(phi1), center[1] + radius * Math.cos(theta1), center[2] + radius * Math.sin(theta1) * Math.sin(phi1)];
      const p3 = [center[0] + radius * Math.sin(theta0) * Math.cos(phi1), center[1] + radius * Math.cos(theta0), center[2] + radius * Math.sin(theta0) * Math.sin(phi1)];

      const n0 = [(p0[0] - center[0])/radius, (p0[1] - center[1])/radius, (p0[2] - center[2])/radius];
      const n1 = [(p1[0] - center[0])/radius, (p1[1] - center[1])/radius, (p1[2] - center[2])/radius];
      const n2 = [(p2[0] - center[0])/radius, (p2[1] - center[1])/radius, (p2[2] - center[2])/radius];
      const n3 = [(p3[0] - center[0])/radius, (p3[1] - center[1])/radius, (p3[2] - center[2])/radius];

      // tri 1
      data.push(...p0, ...n0, ...color, ...pbr);
      data.push(...p1, ...n1, ...color, ...pbr);
      data.push(...p2, ...n2, ...color, ...pbr);
      // tri 2
      data.push(...p0, ...n0, ...color, ...pbr);
      data.push(...p2, ...n2, ...color, ...pbr);
      data.push(...p3, ...n3, ...color, ...pbr);
    }
  }
  return data;
}

export function runGBufferVisualizer(
  device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat, canvas: HTMLCanvasElement, gui: any
) {
  const width = canvas.width || 800;
  const height = canvas.height || 600;

  // 1. 组装复合几何体（地面 + 完整立方体 + 高精度法线球体 + 金属立柱）
  const geomData: number[] = [
    // 地板 (8x8 大平面)
    -10,0,-10,  0,1,0,  0.8,0.8,0.85, 0.4,0.05,
     10,0,-10,  0,1,0,  0.8,0.8,0.85, 0.4,0.05,
     10,0, 10,  0,1,0,  0.8,0.8,0.85, 0.4,0.05,
    -10,0,-10,  0,1,0,  0.8,0.8,0.85, 0.4,0.05,
     10,0, 10,  0,1,0,  0.8,0.8,0.85, 0.4,0.05,
    -10,0, 10,  0,1,0,  0.8,0.8,0.85, 0.4,0.05,

    // 中心红色主立方体 (完整 6 个面)
    // 前面
    -1.2,0, 1.2,  0,0,1,  0.9,0.15,0.15, 0.3,0.1,   1.2,0, 1.2,  0,0,1,  0.9,0.15,0.15, 0.3,0.1,   1.2,2.4, 1.2,  0,0,1,  0.9,0.15,0.15, 0.3,0.1,
    -1.2,0, 1.2,  0,0,1,  0.9,0.15,0.15, 0.3,0.1,   1.2,2.4, 1.2,  0,0,1,  0.9,0.15,0.15, 0.3,0.1,  -1.2,2.4, 1.2,  0,0,1,  0.9,0.15,0.15, 0.3,0.1,
    // 后面
     1.2,0,-1.2,  0,0,-1, 0.9,0.15,0.15, 0.3,0.1,  -1.2,0,-1.2,  0,0,-1, 0.9,0.15,0.15, 0.3,0.1,  -1.2,2.4,-1.2,  0,0,-1, 0.9,0.15,0.15, 0.3,0.1,
     1.2,0,-1.2,  0,0,-1, 0.9,0.15,0.15, 0.3,0.1,  -1.2,2.4,-1.2,  0,0,-1, 0.9,0.15,0.15, 0.3,0.1,   1.2,2.4,-1.2,  0,0,-1, 0.9,0.15,0.15, 0.3,0.1,
    // 顶面
    -1.2,2.4, 1.2, 0,1,0,  0.9,0.15,0.15, 0.3,0.1,   1.2,2.4, 1.2, 0,1,0,  0.9,0.15,0.15, 0.3,0.1,   1.2,2.4,-1.2, 0,1,0,  0.9,0.15,0.15, 0.3,0.1,
    -1.2,2.4, 1.2, 0,1,0,  0.9,0.15,0.15, 0.3,0.1,   1.2,2.4,-1.2, 0,1,0,  0.9,0.15,0.15, 0.3,0.1,  -1.2,2.4,-1.2, 0,1,0,  0.9,0.15,0.15, 0.3,0.1,
    // 左面
    -1.2,0,-1.2, -1,0,0,  0.15,0.85,0.3, 0.4,0.0,  -1.2,0, 1.2, -1,0,0,  0.15,0.85,0.3, 0.4,0.0,  -1.2,2.4, 1.2, -1,0,0,  0.15,0.85,0.3, 0.4,0.0,
    -1.2,0,-1.2, -1,0,0,  0.15,0.85,0.3, 0.4,0.0,  -1.2,2.4, 1.2, -1,0,0,  0.15,0.85,0.3, 0.4,0.0,  -1.2,2.4,-1.2, -1,0,0,  0.15,0.85,0.3, 0.4,0.0,
    // 右面
     1.2,0, 1.2,  1,0,0,  0.2,0.4,0.95, 0.2,0.2,   1.2,0,-1.2,  1,0,0,  0.2,0.4,0.95, 0.2,0.2,   1.2,2.4,-1.2,  1,0,0,  0.2,0.4,0.95, 0.2,0.2,
     1.2,0, 1.2,  1,0,0,  0.2,0.4,0.95, 0.2,0.2,   1.2,2.4,-1.2,  1,0,0,  0.2,0.4,0.95, 0.2,0.2,   1.2,2.4, 1.2,  1,0,0,  0.2,0.4,0.95, 0.2,0.2,

    // 右侧黄色长方立柱 (完整 6 个面)
    2.6,0, 0.6, 0,0,1, 0.95,0.82,0.2, 0.15,0.9,  3.8,0, 0.6, 0,0,1, 0.95,0.82,0.2, 0.15,0.9,  3.8,3.6, 0.6, 0,0,1, 0.95,0.82,0.2, 0.15,0.9,
    2.6,0, 0.6, 0,0,1, 0.95,0.82,0.2, 0.15,0.9,  3.8,3.6, 0.6, 0,0,1, 0.95,0.82,0.2, 0.15,0.9,  2.6,3.6, 0.6, 0,0,1, 0.95,0.82,0.2, 0.15,0.9,
    2.6,3.6, 0.6, 0,1,0, 0.95,0.82,0.2, 0.15,0.9,  3.8,3.6, 0.6, 0,1,0, 0.95,0.82,0.2, 0.15,0.9,  3.8,3.6,-0.6, 0,1,0, 0.95,0.82,0.2, 0.15,0.9,
    2.6,3.6, 0.6, 0,1,0, 0.95,0.82,0.2, 0.15,0.9,  3.8,3.6,-0.6, 0,1,0, 0.95,0.82,0.2, 0.15,0.9,  2.6,3.6,-0.6, 0,1,0, 0.95,0.82,0.2, 0.15,0.9,
    3.8,0, 0.6, 1,0,0, 0.95,0.82,0.2, 0.15,0.9,  3.8,0,-0.6, 1,0,0, 0.95,0.82,0.2, 0.15,0.9,  3.8,3.6,-0.6, 1,0,0, 0.95,0.82,0.2, 0.15,0.9,
    3.8,0, 0.6, 1,0,0, 0.95,0.82,0.2, 0.15,0.9,  3.8,3.6,-0.6, 1,0,0, 0.95,0.82,0.2, 0.15,0.9,  3.8,3.6, 0.6, 1,0,0, 0.95,0.82,0.2, 0.15,0.9,
    2.6,0,-0.6, -1,0,0, 0.95,0.82,0.2, 0.15,0.9,  2.6,0, 0.6, -1,0,0, 0.95,0.82,0.2, 0.15,0.9,  2.6,3.6, 0.6, -1,0,0, 0.95,0.82,0.2, 0.15,0.9,
    2.6,0,-0.6, -1,0,0, 0.95,0.82,0.2, 0.15,0.9,  2.6,3.6, 0.6, -1,0,0, 0.95,0.82,0.2, 0.15,0.9,  2.6,3.6,-0.6, -1,0,0, 0.95,0.82,0.2, 0.15,0.9,
  ];

  // 加入平滑法线球体（半径 1.5，位于前方偏左，最能展现法线从 [-1,1] 到 [0,1] 的连续映射渐变）
  const sphereData = createSphere(1.4, 28, 28, [-3.2, 1.4, 1.2], [0.95, 0.45, 0.1], [0.1, 0.95]);
  geomData.push(...sphereData);

  const vertices = new Float32Array(geomData);
  const vBuffer = device.createBuffer({ size: vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vBuffer, 0, vertices);

  // 全屏四边形
  const quadData = new Float32Array([-1,-1, 1,-1, -1,1, 1,-1, 1,1, -1,1]);
  const quadBuffer = device.createBuffer({ size: quadData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(quadBuffer, 0, quadData);

  // 2. G-Buffer 纹理组 (MRT 目标)
  const gAlbedo = device.createTexture({ size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const gNormal = device.createTexture({ size: [width, height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const gPosition = device.createTexture({ size: [width, height], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const gDepth = device.createTexture({ size: [width, height], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });

  const pointSampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });

  const uniformBuffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const cpuUniformData = new Float32Array(64);

  // 3. Pass 1: G-Buffer 填充 (MRT 着色器)
  const gbufferWGSL = `
    struct Uniforms {
      view: mat4x4f, proj: mat4x4f,
      eyePos: vec4f,
      settings: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VIn {
      @location(0) pos: vec3f,
      @location(1) norm: vec3f,
      @location(2) col: vec3f,
      @location(3) pbr: vec2f,
    };

    struct VOut {
      @builtin(position) posClip: vec4f,
      @location(0) posWS: vec3f,
      @location(1) normWS: vec3f,
      @location(2) col: vec3f,
      @location(3) pbr: vec2f,
    };

    @vertex fn vs(v: VIn) -> VOut {
      var o: VOut;
      o.posClip = u.proj * u.view * vec4f(v.pos, 1.0);
      o.posWS = v.pos;
      o.normWS = v.norm;
      o.col = v.col;
      o.pbr = v.pbr;
      return o;
    }

    struct GBufferOut {
      @location(0) albedo: vec4f,   // 漫反射色彩
      @location(1) normal: vec4f,   // RGB: 世界法线, A: 粗糙度
      @location(2) position: vec4f, // RGB: 世界坐标, A: 金属度
    };

    @fragment fn fs(in: VOut) -> GBufferOut {
      var g: GBufferOut;
      g.albedo = vec4f(in.col, 1.0);
      g.normal = vec4f(normalize(in.normWS), in.pbr.x);
      g.position = vec4f(in.posWS, in.pbr.y);
      return g;
    }
  `;

  const gbufferPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: gbufferWGSL }), entryPoint: "vs",
      buffers: [{
        arrayStride: 44,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" },
          { shaderLocation: 2, offset: 24, format: "float32x3" },
          { shaderLocation: 3, offset: 36, format: "float32x2" },
        ]
      }]
    },
    fragment: {
      module: device.createShaderModule({ code: gbufferWGSL }), entryPoint: "fs",
      targets: [{ format: "rgba8unorm" }, { format: "rgba16float" }, { format: "rgba32float" }]
    },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    primitive: { topology: "triangle-list", cullMode: "none" }
  });

  // 4. Pass 2: G-Buffer 实时可视化解析
  const resolveWGSL = `
    struct Uniforms {
      view: mat4x4f, proj: mat4x4f,
      eyePos: vec4f,
      settings: vec4f, // x: channel, y: near, z: far, w: pipMode (画中画)
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var samp: sampler;
    @group(0) @binding(2) var tAlbedo: texture_2d<f32>;
    @group(0) @binding(3) var tNormal: texture_2d<f32>;
    @group(0) @binding(4) var tPosition: texture_2d<f32>;
    @group(0) @binding(5) var tDepth: texture_depth_2d;

    @vertex fn vs(@location(0) pos: vec2f) -> @builtin(position) vec4f {
      return vec4f(pos, 0.0, 1.0);
    }

    fn linearizeDepth(raw: f32, near: f32, far: f32) -> f32 {
      return (near * far) / (far - raw * (far - near));
    }

    fn renderChannel(uv: vec2f, channelId: i32) -> vec4f {
      let rawZ = textureSampleLevel(tDepth, samp, uv, 0);
      if (rawZ >= 0.9999) {
        return vec4f(0.08, 0.1, 0.14, 1.0);
      }

      let albedo = textureSampleLevel(tAlbedo, samp, uv, 0).rgb;
      let normSample = textureSampleLevel(tNormal, samp, uv, 0);
      let posSample = textureSampleLevel(tPosition, samp, uv, 0);

      let worldNorm = normSample.rgb;
      let roughness = normSample.a;
      let worldPos = posSample.rgb;
      let metallic = posSample.a;

      let near = u.settings.y;
      let far = u.settings.z;
      let linearZ = linearizeDepth(rawZ, near, far);

      switch(channelId) {
        case 1: { // 世界法线 (World Space Normal: [-1,1] -> [0,1])
          return vec4f(worldNorm * 0.5 + 0.5, 1.0);
        }
        case 2: { // 视图空间法线 (View Space Normal: 随摄像机旋转联动)
          let viewNorm = normalize((u.view * vec4f(worldNorm, 0.0)).xyz);
          return vec4f(viewNorm * 0.5 + 0.5, 1.0);
        }
        case 3: { // 顶点世界坐标 (Position WS: 加取模形成空间坐标条纹)
          let grid = fract(worldPos * 0.5);
          return vec4f(grid, 1.0);
        }
        case 4: { // 漫反射基色 (Albedo)
          return vec4f(albedo, 1.0);
        }
        case 5: { // 摄像机线性深度 (Linear Depth)
          let depthVis = clamp((linearZ - near) / 22.0, 0.0, 1.0);
          return vec4f(vec3f(depthVis), 1.0);
        }
        case 6: { // 硬件非线性深度 (Raw Depth)
          return vec4f(vec3f(rawZ), 1.0);
        }
        case 7: { // 粗糙度贴图 (Roughness)
          return vec4f(vec3f(roughness), 1.0);
        }
        case 8: { // 金属度贴图 (Metallic)
          return vec4f(vec3f(metallic), 1.0);
        }
        default: { // 0: 延迟光照最终合成
          let lightPos = vec3f(4.0, 8.0, 5.0);
          let L = normalize(lightPos - worldPos);
          let V = normalize(u.eyePos.xyz - worldPos);
          let H = normalize(L + V);

          let diff = max(dot(worldNorm, L), 0.0);
          let spec = pow(max(dot(worldNorm, H), 0.0), mix(12.0, 256.0, 1.0 - roughness)) * (metallic * 0.8 + 0.2);
          let ambient = 0.15 * albedo;
          let color = albedo * diff + vec3f(spec) + ambient;
          return vec4f(color, 1.0);
        }
      }
    }

    @fragment fn fs(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      let fullDims = vec2f(textureDimensions(tAlbedo));
      let uv = fragCoord.xy / fullDims;
      let activeChannel = i32(u.settings.x);
      let enablePIP = u.settings.w > 0.5;

      // 主屏幕显示选中的主要通道
      var finalColor = renderChannel(uv, activeChannel);

      // 左下角画中画 (Picture in Picture): 4 个独立小缩略图对比
      if (enablePIP) {
        let pipWidth = 0.16;
        let pipHeight = 0.16 * (fullDims.x / fullDims.y);
        let padding = 0.015;

        // 缩略图 1: 世界法线 (WS Normal)
        let b1Min = vec2f(padding, 1.0 - (padding + pipHeight));
        let b1Max = b1Min + vec2f(pipWidth, pipHeight);
        if (uv.x >= b1Min.x && uv.x <= b1Max.x && uv.y >= b1Min.y && uv.y <= b1Max.y) {
          let innerUV = (uv - b1Min) / vec2f(pipWidth, pipHeight);
          if (innerUV.x < 0.03 || innerUV.x > 0.97 || innerUV.y < 0.03 || innerUV.y > 0.97) {
            return vec4f(0.0, 0.8, 1.0, 1.0); // 青色边框
          }
          return renderChannel(innerUV, 1);
        }

        // 缩略图 2: 顶点世界坐标 (Position WS)
        let b2Min = vec2f(padding * 2.0 + pipWidth, 1.0 - (padding + pipHeight));
        let b2Max = b2Min + vec2f(pipWidth, pipHeight);
        if (uv.x >= b2Min.x && uv.x <= b2Max.x && uv.y >= b2Min.y && uv.y <= b2Max.y) {
          let innerUV = (uv - b2Min) / vec2f(pipWidth, pipHeight);
          if (innerUV.x < 0.03 || innerUV.x > 0.97 || innerUV.y < 0.03 || innerUV.y > 0.97) {
            return vec4f(1.0, 0.4, 0.1, 1.0); // 橙色边框
          }
          return renderChannel(innerUV, 3);
        }

        // 缩略图 3: 线性深度 (Linear Depth)
        let b3Min = vec2f(padding * 3.0 + pipWidth * 2.0, 1.0 - (padding + pipHeight));
        let b3Max = b3Min + vec2f(pipWidth, pipHeight);
        if (uv.x >= b3Min.x && uv.x <= b3Max.x && uv.y >= b3Min.y && uv.y <= b3Max.y) {
          let innerUV = (uv - b3Min) / vec2f(pipWidth, pipHeight);
          if (innerUV.x < 0.03 || innerUV.x > 0.97 || innerUV.y < 0.03 || innerUV.y > 0.97) {
            return vec4f(0.8, 0.8, 0.8, 1.0); // 灰色边框
          }
          return renderChannel(innerUV, 5);
        }

        // 缩略图 4: 基色漫反射 (Albedo)
        let b4Min = vec2f(padding * 4.0 + pipWidth * 3.0, 1.0 - (padding + pipHeight));
        let b4Max = b4Min + vec2f(pipWidth, pipHeight);
        if (uv.x >= b4Min.x && uv.x <= b4Max.x && uv.y >= b4Min.y && uv.y <= b4Max.y) {
          let innerUV = (uv - b4Min) / vec2f(pipWidth, pipHeight);
          if (innerUV.x < 0.03 || innerUV.x > 0.97 || innerUV.y < 0.03 || innerUV.y > 0.97) {
            return vec4f(0.2, 1.0, 0.4, 1.0); // 绿色边框
          }
          return renderChannel(innerUV, 4);
        }
      }

      return finalColor;
    }
  `;

  const resolvePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: resolveWGSL }), entryPoint: "vs", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: device.createShaderModule({ code: resolveWGSL }), entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" }
  });

  const gbufferBindGroup = device.createBindGroup({
    layout: gbufferPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
  });

  const resolveBindGroup = device.createBindGroup({
    layout: resolvePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: pointSampler },
      { binding: 2, resource: gAlbedo.createView() },
      { binding: 3, resource: gNormal.createView() },
      { binding: 4, resource: gPosition.createView() },
      { binding: 5, resource: gDepth.createView() },
    ]
  });

  // 控制参数
  const camera = { distance: 12.0, theta: 35, phi: 24, panY: 1.4 };
  const debugOptions = {
    channel: 1, // 默认直接打开世界法线纹理（直观看到色彩渐变）
    showPIP: true, // 默认开启左下角画中画缩略对比
  };

  gui.title("G-Buffer 纹理通道检视");
  gui.add(debugOptions, "channel", {
    "1. 世界空间法线 (World Normal)": 1,
    "2. 视图空间法线 (View Normal)": 2,
    "3. 顶点世界坐标 (Position WS)": 3,
    "4. 材质漫反射基色 (Albedo)": 4,
    "5. 线性深度图 (Linear Depth)": 5,
    "6. 硬件透视深度 (Raw Depth)": 6,
    "7. 粗糙度 (Roughness)": 7,
    "8. 金属度 (Metallic)": 8,
    "0. 延迟光照最终合成 (Final Lit)": 0,
  }).name("当前全屏通道");

  gui.add(debugOptions, "showPIP").name("画中画小窗对比");

  const ctrlTheta = gui.add(camera, "theta", -180, 180, 1).name("偏航角");
  const ctrlPhi = gui.add(camera, "phi", -20, 80, 1).name("俯仰角");

  let isDragging = false, lastX = 0, lastY = 0;
  canvas.onpointerdown = (e) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); };
  canvas.onpointermove = (e) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.4;
    camera.phi = Math.max(-20, Math.min(80, camera.phi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY;
    ctrlTheta.updateDisplay();
    ctrlPhi.updateDisplay();
  };
  canvas.onpointerup = (e) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} };

  let animId: number;
  const near = 0.1, far = 60.0;

  function frame() {
    const fov = (55 * Math.PI) / 180;
    const aspect = width / height;
    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.distance * Math.cos(radPhi) * Math.sin(radTheta),
      camera.panY + camera.distance * Math.sin(radPhi),
      camera.distance * Math.cos(radPhi) * Math.cos(radTheta),
    ];

    const view = createLookAtMatrix(eye, [0, camera.panY, 0], [0, 1, 0]);
    const proj = createPerspectiveMatrix(fov, aspect, near, far);

    cpuUniformData.set(view, 0);
    cpuUniformData.set(proj, 16);
    cpuUniformData.set([eye[0], eye[1], eye[2], 1.0], 32);
    cpuUniformData.set([debugOptions.channel, near, far, debugOptions.showPIP ? 1.0 : 0.0], 36);
    device.queue.writeBuffer(uniformBuffer, 0, cpuUniformData);

    const encoder = device.createCommandEncoder();

    // Pass 1: MRT 渲染 G-Buffer
    const pass1 = encoder.beginRenderPass({
      colorAttachments: [
        { view: gAlbedo.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
        { view: gNormal.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
        { view: gPosition.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
      ],
      depthStencilAttachment: { view: gDepth.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" }
    });
    pass1.setPipeline(gbufferPipeline);
    pass1.setBindGroup(0, gbufferBindGroup);
    pass1.setVertexBuffer(0, vBuffer);
    pass1.draw(vertices.length / 11);
    pass1.end();

    // Pass 2: G-Buffer 纹理通道解析呈现
    const pass2 = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }]
    });
    pass2.setPipeline(resolvePipeline);
    pass2.setBindGroup(0, resolveBindGroup);
    pass2.setVertexBuffer(0, quadBuffer);
    pass2.draw(6);
    pass2.end();

    device.queue.submit([encoder.finish()]);

    animId = requestAnimationFrame(frame);
  }
  frame();

  return () => {
    cancelAnimationFrame(animId);
    vBuffer.destroy(); quadBuffer.destroy(); uniformBuffer.destroy();
    gAlbedo.destroy(); gNormal.destroy(); gPosition.destroy(); gDepth.destroy();
  };
}