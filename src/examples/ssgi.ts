// src/examples/ssgi.ts
// import type { SimpleGUI } from "../utils/gui";
import GUI from "lil-gui";
// ---------------------- 矩阵数学库 ----------------------
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

// 严谨相机逆视图矩阵：正交旋转矩阵转置，平移项严格对应 eye
function createInverseViewMatrix(eye: number[], center: number[], up: number[]): Float32Array {
  const z = [eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]];
  const lenZ = 1 / (Math.hypot(z[0], z[1], z[2]) || 1);
  z[0] *= lenZ; z[1] *= lenZ; z[2] *= lenZ;
  const x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]];
  const lenX = 1 / (Math.hypot(x[0], x[1], x[2]) || 1);
  x[0] *= lenX; x[1] *= lenX; x[2] *= lenX;
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const out = new Float32Array(16);
  out[0] = x[0]; out[1] = x[1]; out[2] = x[2]; out[3] = 0;
  out[4] = y[0]; out[5] = y[1]; out[6] = y[2]; out[7] = 0;
  out[8] = z[0]; out[9] = z[1]; out[10] = z[2]; out[11] = 0;
  out[12] = eye[0]; out[13] = eye[1]; out[14] = eye[2]; out[15] = 1;
  return out;
}

function multiplyMat4(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      out[i * 4 + j] =
        a[j] * b[i * 4] + a[4 + j] * b[i * 4 + 1] + a[8 + j] * b[i * 4 + 2] + a[12 + j] * b[i * 4 + 3];
    }
  }
  return out;
}

function matrixDifference(a: Float32Array, b: Float32Array): number {
  let maxDiff = 0;
  for (let i = 0; i < 16; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
  }
  return maxDiff;
}

export function runSSGI(
  device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat, canvas: HTMLCanvasElement, gui: any
) {
  const fullWidth = canvas.width || 800;
  const fullHeight = canvas.height || 600;
  const halfWidth = Math.max(1, Math.floor(fullWidth / 2));
  const halfHeight = Math.max(1, Math.floor(fullHeight / 2));

  // 1. Cornell Box 几何体（稍微旋转中心立方体约 18°，以充分展示左右红绿光漫反射截面）
  const cosA = Math.cos(0.32), sinA = Math.sin(0.32);
  const rotateY = (x: number, z: number): [number, number] => [x * cosA - z * sinA, x * sinA + z * cosA];
  
  const c0 = rotateY(-0.8, -0.8), c1 = rotateY(0.8, -0.8), c2 = rotateY(0.8, 0.8), c3 = rotateY(-0.8, 0.8);
  const nFront = rotateY(0, 1), nBack = rotateY(0, -1), nRight = rotateY(1, 0), nLeft = rotateY(-1, 0);

  const vertices = new Float32Array([
    // 地板 (哑光白灰)
    -3,0,-3,  0,1,0,  0.85,0.85,0.85,   3,0,-3,  0,1,0,  0.85,0.85,0.85,   3,0,3,  0,1,0,  0.85,0.85,0.85,
    -3,0,-3,  0,1,0,  0.85,0.85,0.85,   3,0,3,  0,1,0,  0.85,0.85,0.85,  -3,0,3,  0,1,0,  0.85,0.85,0.85,
    // 顶板 (哑光灰)
    -3,5, 3,  0,-1,0, 0.85,0.85,0.85,   3,5, 3,  0,-1,0, 0.85,0.85,0.85,   3,5,-3,  0,-1,0, 0.85,0.85,0.85,
    -3,5, 3,  0,-1,0, 0.85,0.85,0.85,   3,5,-3,  0,-1,0, 0.85,0.85,0.85,  -3,5,-3,  0,-1,0, 0.85,0.85,0.85,
    // 后墙 (白灰)
    -3,0,-3,  0,0,1,  0.85,0.85,0.85,   3,0,-3,  0,0,1,  0.85,0.85,0.85,   3,5,-3,  0,0,1,  0.85,0.85,0.85,
    -3,0,-3,  0,0,1,  0.85,0.85,0.85,   3,5,-3,  0,0,1,  0.85,0.85,0.85,  -3,5,-3,  0,0,1,  0.85,0.85,0.85,
    // 左墙 (鲜艳红 - 主溢色源)
    -3,0, 3,  1,0,0,  0.95,0.06,0.06, -3,0,-3, 1,0,0,  0.95,0.06,0.06, -3,5,-3, 1,0,0, 0.95,0.06,0.06,
    -3,0, 3,  1,0,0,  0.95,0.06,0.06, -3,5,-3, 1,0,0,  0.95,0.06,0.06, -3,5, 3, 1,0,0, 0.95,0.06,0.06,
    // 右墙 (鲜艳翠绿 - 主溢色源)
     3,0,-3, -1,0,0,  0.06,0.92,0.08,  3,0, 3, -1,0,0, 0.06,0.92,0.08,  3,5, 3, -1,0,0, 0.06,0.92,0.08,
     3,0,-3, -1,0,0,  0.06,0.92,0.08,  3,5, 3, -1,0,0, 0.06,0.92,0.08,  3,5,-3, -1,0,0, 0.06,0.92,0.08,
    // 旋转白色立方体 (顶面)
    c0[0],2,c0[1], 0,1,0, 0.92,0.92,0.92,  c1[0],2,c1[1], 0,1,0, 0.92,0.92,0.92,  c2[0],2,c2[1], 0,1,0, 0.92,0.92,0.92,
    c0[0],2,c0[1], 0,1,0, 0.92,0.92,0.92,  c2[0],2,c2[1], 0,1,0, 0.92,0.92,0.92,  c3[0],2,c3[1], 0,1,0, 0.92,0.92,0.92,
    // 立方体 (前侧面)
    c3[0],0,c3[1], nFront[0],0,nFront[1], 0.92,0.92,0.92,  c2[0],0,c2[1], nFront[0],0,nFront[1], 0.92,0.92,0.92,  c2[0],2,c2[1], nFront[0],0,nFront[1], 0.92,0.92,0.92,
    c3[0],0,c3[1], nFront[0],0,nFront[1], 0.92,0.92,0.92,  c2[0],2,c2[1], nFront[0],0,nFront[1], 0.92,0.92,0.92,  c3[0],2,c3[1], nFront[0],0,nFront[1], 0.92,0.92,0.92,
    // 立方体 (右侧面 - 面向绿墙)
    c2[0],0,c2[1], nRight[0],0,nRight[1], 0.92,0.92,0.92,  c1[0],0,c1[1], nRight[0],0,nRight[1], 0.92,0.92,0.92,  c1[0],2,c1[1], nRight[0],0,nRight[1], 0.92,0.92,0.92,
    c2[0],0,c2[1], nRight[0],0,nRight[1], 0.92,0.92,0.92,  c1[0],2,c1[1], nRight[0],0,nRight[1], 0.92,0.92,0.92,  c2[0],2,c2[1], nRight[0],0,nRight[1], 0.92,0.92,0.92,
    // 立方体 (后侧面)
    c1[0],0,c1[1], nBack[0],0,nBack[1], 0.92,0.92,0.92,   c0[0],0,c0[1], nBack[0],0,nBack[1], 0.92,0.92,0.92,   c0[0],2,c0[1], nBack[0],0,nBack[1], 0.92,0.92,0.92,
    c1[0],0,c1[1], nBack[0],0,nBack[1], 0.92,0.92,0.92,   c0[0],2,c0[1], nBack[0],0,nBack[1], 0.92,0.92,0.92,   c1[0],2,c1[1], nBack[0],0,nBack[1], 0.92,0.92,0.92,
    // 立方体 (左侧面 - 面向红墙)
    c0[0],0,c0[1], nLeft[0],0,nLeft[1], 0.92,0.92,0.92,   c3[0],0,c3[1], nLeft[0],0,nLeft[1], 0.92,0.92,0.92,   c3[0],2,c3[1], nLeft[0],0,nLeft[1], 0.92,0.92,0.92,
    c0[0],0,c0[1], nLeft[0],0,nLeft[1], 0.92,0.92,0.92,   c3[0],2,c3[1], nLeft[0],0,nLeft[1], 0.92,0.92,0.92,   c0[0],2,c0[1], nLeft[0],0,nLeft[1], 0.92,0.92,0.92,
  ]);
  const vBuffer = device.createBuffer({ size: vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vBuffer, 0, vertices);

  // 全屏 Quad
  const quadData = new Float32Array([-1,-1, 1,-1, -1,1, 1,-1, 1,1, -1,1]);
  const quadBuffer = device.createBuffer({ size: quadData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(quadBuffer, 0, quadData);

  // 统一 Uniform Buffer (384 bytes)
  const uniformBuffer = device.createBuffer({ size: 384, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const cpuUniformData = new Float32Array(384 / 4);

  // 纹理创建
  const gColor = device.createTexture({ size: [fullWidth, fullHeight], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const gNormal = device.createTexture({ size: [fullWidth, fullHeight], format: "rg16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const gDepth = device.createTexture({ size: [fullWidth, fullHeight], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  
  const currentSSGITex = device.createTexture({ size: [halfWidth, halfHeight], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const ssgiHistoryA = device.createTexture({ size: [halfWidth, halfHeight], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const ssgiHistoryB = device.createTexture({ size: [halfWidth, halfHeight], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });

  const pointSampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });
  const linearSampler = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });

  // 2. Pass 1: G-Buffer 渲染（配置强力物理顶光源，照亮彩墙以产生二次反弹光）
  // 3. Pass 1: G-Buffer
  const gbufferWGSL = `
    struct Uniforms {
      view: mat4x4f, proj: mat4x4f, invView: mat4x4f, prevViewProj: mat4x4f,
      camParams: vec4f, settings: vec4f, halfRes: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VIn { @location(0) pos: vec3f, @location(1) norm: vec3f, @location(2) col: vec3f };
    struct VOut {
      @builtin(position) pos: vec4f,
      @location(0) normalVS: vec3f,
      @location(1) directLight: vec3f
    };

    @vertex fn vs(v: VIn) -> VOut {
      var o: VOut;
      o.pos = u.proj * u.view * vec4f(v.pos, 1.0);
      o.normalVS = (u.view * vec4f(v.norm, 0.0)).xyz;

      // 经典 Cornell Box 面光源模拟：
      // 1. 顶棚偏前的主聚光（照亮彩墙、地面和立方体正面）
      let mainLightPos = vec3f(0.0, 4.3, 0.8);
      let toMain = mainLightPos - v.pos;
      let distMain = length(toMain);
      let lDir1 = toMain / distMain;
      let atten1 = 18.0 / (distMain * distMain + 1.5);
      let diff1 = max(dot(v.norm, lDir1), 0.0) * atten1;

      // 2. 补光（模拟开口处的微弱天光漫射，杜绝背光死黑）
      let fillLightDir = normalize(vec3f(0.0, 0.3, 1.0));
      let diff2 = max(dot(v.norm, fillLightDir), 0.0) * 0.25;

      let ambient = 0.15; // 柔和的环境基底光
      let totalDiff = diff1 + diff2 + ambient;

      o.directLight = v.col * totalDiff;
      return o;
    }

    fn octEncode(n: vec3f) -> vec2f {
      let l1 = dot(abs(n), vec3f(1.0));
      var p = n.xy * (1.0 / l1);
      if (n.z < 0.0) {
        p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
      }
      return p;
    }

    struct GOut { @location(0) color: vec4f, @location(1) normal: vec2f };

    @fragment fn fs(in: VOut) -> GOut {
      var g: GOut;
      g.color = vec4f(in.directLight, 1.0);
      g.normal = octEncode(normalize(in.normalVS));
      return g;
    }
  `;
  const gbufferPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: gbufferWGSL }), entryPoint: "vs",
      buffers: [{ arrayStride: 36, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x3" }
      ]}],
    },
    fragment: { module: device.createShaderModule({ code: gbufferWGSL }), entryPoint: "fs", targets: [{ format: "rgba16float" }, { format: "rg16float" }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    primitive: { topology: "triangle-list" }
  });

  // 3. Pass 2: SSGI 计算（Jitter 步进彻底消除波纹条纹，余弦加权漫反射）
  const ssgiWGSL = `
    struct Uniforms {
      view: mat4x4f, proj: mat4x4f, invView: mat4x4f, prevViewProj: mat4x4f,
      camParams: vec4f, settings: vec4f, halfRes: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var pointSamp: sampler;
    @group(0) @binding(2) var colorTex: texture_2d<f32>;
    @group(0) @binding(3) var normalTex: texture_2d<f32>;
    @group(0) @binding(4) var depthTex: texture_depth_2d;

    @vertex fn vs(@location(0) pos: vec2f) -> @builtin(position) vec4f {
      return vec4f(pos, 0.0, 1.0);
    }

    fn octDecode(e: vec2f) -> vec3f {
      var n = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
      let t = max(-n.z, 0.0);
      n.x += select(t, -t, n.x >= 0.0);
      n.y += select(t, -t, n.y >= 0.0);
      return normalize(n);
    }

    fn getLinearDepth(rawDepth: f32) -> f32 {
      let near = u.camParams.x;
      let far = u.camParams.y;
      return (near * far) / (far - rawDepth * (far - near));
    }

    fn getViewPos(uv: vec2f, linearZ: f32) -> vec3f {
      let tanHalfFov = u.camParams.z;
      let aspect = u.camParams.w;
      let x = (uv.x * 2.0 - 1.0) * tanHalfFov * aspect * linearZ;
      let y = (1.0 - uv.y * 2.0) * tanHalfFov * linearZ;
      return vec3f(x, y, -linearZ);
    }

    fn projectPosToUV(posVS: vec3f) -> vec3f {
      let clip = u.proj * vec4f(posVS, 1.0);
      let ndc = clip.xyz / clip.w;
      return vec3f(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5), -posVS.z);
    }

    fn hash12(p: vec2f) -> f32 {
      var p3 = fract(vec3f(p.xyx) * 0.1031);
      p3 += dot(p3, p3.yzx + 33.33);
      return fract((p3.x + p3.y) * p3.z);
    }

    @fragment fn fs(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      let uv = fragCoord.xy / u.halfRes.xy;
      let rawZ = textureSampleLevel(depthTex, pointSamp, uv, 0);
      if (rawZ >= 0.9999) { return vec4f(0.0); }

      let linearZ = getLinearDepth(rawZ);
      let posVS = getViewPos(uv, linearZ);
      let normalVS = octDecode(textureSampleLevel(normalTex, pointSamp, uv, 0).xy);

      let giIntensity = u.settings.x;
      let maxRadius = u.settings.y;
      let frameIndex = u.settings.z;

      // 切线正交基底
      let signZ = select(-1.0, 1.0, normalVS.z >= 0.0);
      let a = -1.0 / (signZ + normalVS.z);
      let b = normalVS.x * normalVS.y * a;
      let tangent = vec3f(1.0 + signZ * normalVS.x * normalVS.x * a, signZ * b, -signZ * normalVS.x);
      let bitangent = vec3f(b, signZ + normalVS.y * normalVS.y * a, -normalVS.y);

      let samples = 6;
      let steps = 10;
      var indirectAccum = vec3f(0.0);

      let screenJitter = hash12(floor(fragCoord.xy) + vec2f(frameIndex * 1.618));

      for (var s = 0; s < samples; s++) {
        let fSample = f32(s);
        let phi = (fSample + screenJitter) * 2.39996 + frameIndex * 0.8;
        let cosTheta = sqrt((fSample + 0.5) / f32(samples));
        let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

        // 半球 Cosine-Weighted 采样射线
        let rayDirVS = normalize(tangent * (cos(phi) * sinTheta) + bitangent * (sin(phi) * sinTheta) + normalVS * cosTheta);

        // 引入步长抖动（彻底打破等高线步进波纹假影）
        let rayJitter = hash12(vec2f(screenJitter, fSample));
        var currentDist = (0.06 + linearZ * 0.005) + (maxRadius / f32(steps)) * rayJitter;
        let stepDelta = maxRadius / f32(steps);

        for (var i = 0; i < steps; i++) {
          let marchPos = posVS + rayDirVS * currentDist;
          let proj = projectPosToUV(marchPos);

          if (proj.x < 0.01 || proj.x > 0.99 || proj.y < 0.01 || proj.y > 0.99) { break; }

          let hitRawZ = textureSampleLevel(depthTex, pointSamp, proj.xy, 0);
          if (hitRawZ >= 0.9999) {
            currentDist += stepDelta;
            continue;
          }

          let hitLinearZ = getLinearDepth(hitRawZ);
          let deltaZ = proj.z - hitLinearZ;

          // 景深厚度测试：动态厚度抑制漏光与穿透
          let thickness = max(0.08, hitLinearZ * 0.035);

          if (deltaZ > 0.015 && deltaZ < thickness) {
            let hitNormal = octDecode(textureSampleLevel(normalTex, pointSamp, proj.xy, 0).xy);
            
            // 剔除同平面自碰撞
            if (dot(normalVS, hitNormal) > 0.96 && currentDist < 0.2) {
              currentDist += stepDelta;
              continue;
            }

            let hitColor = textureSampleLevel(colorTex, pointSamp, proj.xy, 0).rgb;
            let bounceWeight = max(dot(hitNormal, -rayDirVS), 0.0);
            let distAtten = 1.0 / (1.0 + currentDist * currentDist * 0.6);

            indirectAccum += hitColor * (bounceWeight * distAtten);
            break;
          }
          currentDist += stepDelta;
        }
      }

      let avgIndirect = (indirectAccum / f32(samples)) * giIntensity;
      return vec4f(avgIndirect, 1.0);
    }
  `;
  const ssgiPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: ssgiWGSL }), entryPoint: "vs", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: device.createShaderModule({ code: ssgiWGSL }), entryPoint: "fs", targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list" }
  });

  // 4. Pass 3: 时空重投影累积（精准数学重投影 + 3x3 Color Box Clamping 防拖影）
  const temporalWGSL = `
    struct Uniforms {
      view: mat4x4f, proj: mat4x4f, invView: mat4x4f, prevViewProj: mat4x4f,
      camParams: vec4f, settings: vec4f, halfRes: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var linearSamp: sampler;
    @group(0) @binding(2) var pointSamp: sampler;
    @group(0) @binding(3) var currentTex: texture_2d<f32>;
    @group(0) @binding(4) var historyTex: texture_2d<f32>;
    @group(0) @binding(5) var fullDepthTex: texture_depth_2d;
    @group(0) @binding(6) var fullNormalTex: texture_2d<f32>;

    @vertex fn vs(@location(0) pos: vec2f) -> @builtin(position) vec4f {
      return vec4f(pos, 0.0, 1.0);
    }

    fn octDecode(e: vec2f) -> vec3f {
      var n = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
      let t = max(-n.z, 0.0);
      n.x += select(t, -t, n.x >= 0.0);
      n.y += select(t, -t, n.y >= 0.0);
      return normalize(n);
    }

    fn getLinearDepth(rawDepth: f32) -> f32 {
      let near = u.camParams.x;
      let far = u.camParams.y;
      return (near * far) / (far - rawDepth * (far - near));
    }

    fn getViewPos(uv: vec2f, linearZ: f32) -> vec3f {
      let tanHalfFov = u.camParams.z;
      let aspect = u.camParams.w;
      let x = (uv.x * 2.0 - 1.0) * tanHalfFov * aspect * linearZ;
      let y = (1.0 - uv.y * 2.0) * tanHalfFov * linearZ;
      return vec3f(x, y, -linearZ);
    }

    @fragment fn fs(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      let uv = fragCoord.xy / u.halfRes.xy;
      let currentGI = textureSampleLevel(currentTex, pointSamp, uv, 0.0).rgb;

      let rawZ = textureSampleLevel(fullDepthTex, pointSamp, uv, 0);
      if (rawZ >= 0.9999) { return vec4f(currentGI, 1.0); }

      let linearZ = getLinearDepth(rawZ);
      let posVS = getViewPos(uv, linearZ);
      let posWS = (u.invView * vec4f(posVS, 1.0)).xyz;

      let prevClip = u.prevViewProj * vec4f(posWS, 1.0);
      let prevNDC = prevClip.xyz / prevClip.w;
      let prevUV = vec2f(prevNDC.x * 0.5 + 0.5, 1.0 - (prevNDC.y * 0.5 + 0.5));

      if (prevUV.x < 0.0 || prevUV.x > 1.0 || prevUV.y < 0.0 || prevUV.y > 1.0) {
        return vec4f(currentGI, 1.0);
      }

      // 遮挡脱离检验 (Disocclusion)
      let historyDepth = getLinearDepth(textureSampleLevel(fullDepthTex, pointSamp, prevUV, 0));
      let depthInvalid = abs(linearZ - historyDepth) > max(0.12, linearZ * 0.05);

      let currentNormal = octDecode(textureSampleLevel(fullNormalTex, pointSamp, uv, 0.0).xy);
      let historyNormal = octDecode(textureSampleLevel(fullNormalTex, pointSamp, prevUV, 0.0).xy);
      let normalInvalid = dot(currentNormal, historyNormal) < 0.85;

      var alpha = u.settings.w;
      if (depthInvalid || normalInvalid) {
        alpha = 1.0;
      }

      // 3x3 邻域色彩盒约束（Color Clamping），彻底消灭拖影
      var minC = currentGI;
      var maxC = currentGI;
      let halfPixel = 1.0 / u.halfRes.xy;
      for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
          let neighbor = textureSampleLevel(currentTex, pointSamp, uv + vec2f(f32(x), f32(y)) * halfPixel, 0.0).rgb;
          minC = min(minC, neighbor);
          maxC = max(maxC, neighbor);
        }
      }

      let historyGI = textureSampleLevel(historyTex, linearSamp, prevUV, 0.0).rgb;
      let clampedHistory = clamp(historyGI, minC, maxC);

      return vec4f(mix(clampedHistory, currentGI, alpha), 1.0);
    }
  `;
  const temporalPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: temporalWGSL }), entryPoint: "vs", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: device.createShaderModule({ code: temporalWGSL }), entryPoint: "fs", targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list" }
  });

  // 5. Pass 4: 空间联合双边滤波上采样与色调合成 (Joint Bilateral Upsampling)
// 5. Pass 4: 空间联合双边滤波上采样与色调合成 (Joint Bilateral Upsampling)
  const compositeWGSL = `
    struct Uniforms {
      view: mat4x4f, proj: mat4x4f, invView: mat4x4f, prevViewProj: mat4x4f,
      camParams: vec4f, settings: vec4f, halfRes: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var linearSamp: sampler;
    @group(0) @binding(2) var pointSamp: sampler;
    @group(0) @binding(3) var fullColorTex: texture_2d<f32>;
    @group(0) @binding(4) var fullNormalTex: texture_2d<f32>;
    @group(0) @binding(5) var fullDepthTex: texture_depth_2d;
    @group(0) @binding(6) var resolvedSSGITex: texture_2d<f32>;

    @vertex fn vs(@location(0) pos: vec2f) -> @builtin(position) vec4f {
      return vec4f(pos, 0.0, 1.0);
    }

    fn octDecode(e: vec2f) -> vec3f {
      var n = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
      let t = max(-n.z, 0.0);
      n.x += select(t, -t, n.x >= 0.0);
      n.y += select(t, -t, n.y >= 0.0);
      return normalize(n);
    }

    fn getLinearDepth(rawDepth: f32) -> f32 {
      let near = u.camParams.x;
      let far = u.camParams.y;
      return (near * far) / (far - rawDepth * (far - near));
    }

    // 经典 ACES 色调映射：抑制极端过曝，提升中低对比度
    fn acesToneMapping(color: vec3f) -> vec3f {
      let a = 2.51;
      let b = 0.03;
      let c = 2.43;
      let d = 0.59;
      let e = 0.14;
      return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3f(0.0), vec3f(1.0));
    }

    @fragment fn fs(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      let fullDims = vec2f(textureDimensions(fullColorTex));
      let uv = fragCoord.xy / fullDims;
      let directColor = textureSampleLevel(fullColorTex, pointSamp, uv, 0.0).rgb;

      let rawZ = textureSampleLevel(fullDepthTex, pointSamp, uv, 0);
      if (rawZ >= 0.9999) { return vec4f(directColor, 1.0); }

      let centerDepth = getLinearDepth(rawZ);
      let centerNormal = octDecode(textureSampleLevel(fullNormalTex, pointSamp, uv, 0).xy);

      let fullPixel = 1.0 / fullDims;
      let halfPixel = 1.0 / u.halfRes.xy;

      var totalWeight = 0.0;
      var filteredGI = vec3f(0.0);

      // 3x3 空间交叉双边滤波
      for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
          let offset = vec2f(f32(x), f32(y));
          let giUV = uv + offset * halfPixel;
          let geomUV = uv + offset * fullPixel;

          let sampleGI = textureSampleLevel(resolvedSSGITex, linearSamp, giUV, 0.0).rgb;
          let sampleDepth = getLinearDepth(textureSampleLevel(fullDepthTex, pointSamp, geomUV, 0));
          let sampleNormal = octDecode(textureSampleLevel(fullNormalTex, pointSamp, geomUV, 0).xy);

          let depthDiff = abs(centerDepth - sampleDepth);
          let depthWeight = exp(-depthDiff * 14.0);
          let normalWeight = max(0.0, pow(dot(centerNormal, sampleNormal), 12.0));
          let w = depthWeight * normalWeight + 0.001;

          filteredGI += sampleGI * w;
          totalWeight += w;
        }
      }

      filteredGI = filteredGI / totalWeight;

      let renderMode = i32(u.halfRes.z);
      if (renderMode == 1) { // 仅查看 SSGI 间接溢色通道
        return vec4f(acesToneMapping(filteredGI * 1.5), 1.0);
      } else if (renderMode == 2) { // 仅查看直接光通道
        return vec4f(acesToneMapping(directColor), 1.0);
      }

      // 物理合成并施加电影级 ToneMapping
      let combined = directColor + filteredGI;
      let finalColor = acesToneMapping(combined);

      return vec4f(finalColor, 1.0);
    }
  `;
  const compositePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: compositeWGSL }), entryPoint: "vs", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: device.createShaderModule({ code: compositeWGSL }), entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" }
  });

  // 6. BindGroups
  const gbufferBindGroup = device.createBindGroup({
    layout: gbufferPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
  });

  const ssgiBindGroup = device.createBindGroup({
    layout: ssgiPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: pointSampler },
      { binding: 2, resource: gColor.createView() },
      { binding: 3, resource: gNormal.createView() },
      { binding: 4, resource: gDepth.createView() },
    ]
  });

  const temporalBindGroupAtoB = device.createBindGroup({
    layout: temporalPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: linearSampler },
      { binding: 2, resource: pointSampler },
      { binding: 3, resource: currentSSGITex.createView() },
      { binding: 4, resource: ssgiHistoryA.createView() },
      { binding: 5, resource: gDepth.createView() },
      { binding: 6, resource: gNormal.createView() },
    ]
  });

  const temporalBindGroupBtoA = device.createBindGroup({
    layout: temporalPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: linearSampler },
      { binding: 2, resource: pointSampler },
      { binding: 3, resource: currentSSGITex.createView() },
      { binding: 4, resource: ssgiHistoryB.createView() },
      { binding: 5, resource: gDepth.createView() },
      { binding: 6, resource: gNormal.createView() },
    ]
  });

  const compositeBindGroupWithA = device.createBindGroup({
    layout: compositePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: linearSampler },
      { binding: 2, resource: pointSampler },
      { binding: 3, resource: gColor.createView() },
      { binding: 4, resource: gNormal.createView() },
      { binding: 5, resource: gDepth.createView() },
      { binding: 6, resource: ssgiHistoryA.createView() },
    ]
  });

  const compositeBindGroupWithB = device.createBindGroup({
    layout: compositePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: linearSampler },
      { binding: 2, resource: pointSampler },
      { binding: 3, resource: gColor.createView() },
      { binding: 4, resource: gNormal.createView() },
      { binding: 5, resource: gDepth.createView() },
      { binding: 6, resource: ssgiHistoryB.createView() },
    ]
  });

  // 7. 控制面板与交互
  const camera = { distance: 7.2, theta: 0, phi: 12, panY: 2.3 };
  const ssgiSettings = { intensity: 2.0, radius: 2.6, renderMode: 0 };

  gui.addTextInfo("<b>工业级高保真 SSGI</b><br>时空联合降噪 + 强漫反射溢色");
  gui.add(ssgiSettings, "intensity", 0.0, 4.0, 0.1).name("GI 漫反射强度");
  gui.add(ssgiSettings, "radius", 0.5, 4.5, 0.1).name("光线弹射半径");
  gui.addSelect(ssgiSettings, "renderMode", { "完整最终合成": 0, "仅间接光溢色 (SSGI)": 1, "仅直接光": 2 }).name("渲染通道");
  gui.add(camera, "theta", -80, 80, 1).name("偏航角");
  gui.add(camera, "phi", -15, 60, 1).name("俯仰角");

  let isDragging = false, lastX = 0, lastY = 0;
  canvas.onpointerdown = (e) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); };
  canvas.onpointermove = (e) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.4;
    camera.phi = Math.max(-15, Math.min(60, camera.phi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY;
    gui.updateDisplay();
  };
  canvas.onpointerup = (e) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} };

  let animId: number;
  let frameCount = 0;

  // 初始矩阵预备（杜绝首帧为 0 导致的历史缓冲区污染）
  const initialNear = 0.1, initialFar = 50.0, initialFov = (50 * Math.PI) / 180, initialAspect = fullWidth / fullHeight;
  const initialEye = [0, camera.panY + camera.distance * Math.sin((12 * Math.PI) / 180), camera.distance * Math.cos((12 * Math.PI) / 180)];
  let prevViewMatrix = createLookAtMatrix(initialEye, [0, camera.panY, 0], [0, 1, 0]);
  let prevViewProjMatrix = multiplyMat4(createPerspectiveMatrix(initialFov, initialAspect, initialNear, initialFar), prevViewMatrix);

  function frame() {
    frameCount++;
    const near = 0.1, far = 50.0;
    const fov = (50 * Math.PI) / 180;
    const aspect = fullWidth / fullHeight;
    const tanHalfFov = Math.tan(fov / 2);

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.distance * Math.cos(radPhi) * Math.sin(radTheta),
      camera.panY + camera.distance * Math.sin(radPhi),
      camera.distance * Math.cos(radPhi) * Math.cos(radTheta),
    ];

    const view = createLookAtMatrix(eye, [0, camera.panY, 0], [0, 1, 0]);
    const proj = createPerspectiveMatrix(fov, aspect, near, far);
    const viewProj = multiplyMat4(proj, view);
    const invView = createInverseViewMatrix(eye, [0, camera.panY, 0], [0, 1, 0]);

    // 运动检测自适应累积权重
    const camDelta = matrixDifference(view, prevViewMatrix);
    const isCameraMoving = camDelta > 1e-4;
    const temporalAlpha = isCameraMoving ? 0.35 : 0.08;

    cpuUniformData.set(view, 0);
    cpuUniformData.set(proj, 16);
    cpuUniformData.set(invView, 32);
    cpuUniformData.set(prevViewProjMatrix, 48);
    cpuUniformData.set([near, far, tanHalfFov, aspect], 64);
    cpuUniformData.set([ssgiSettings.intensity, ssgiSettings.radius, frameCount & 1023, temporalAlpha], 68);
    cpuUniformData.set([halfWidth, halfHeight, Number(ssgiSettings.renderMode), 0], 72);
    device.queue.writeBuffer(uniformBuffer, 0, cpuUniformData);

    const encoder = device.createCommandEncoder();

    // 1. G-Buffer
    const pass1 = encoder.beginRenderPass({
      colorAttachments: [
        { view: gColor.createView(), clearValue: { r: 0.04, g: 0.04, b: 0.04, a: 1.0 }, loadOp: "clear", storeOp: "store" },
        { view: gNormal.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }
      ],
      depthStencilAttachment: { view: gDepth.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" }
    });
    pass1.setPipeline(gbufferPipeline);
    pass1.setBindGroup(0, gbufferBindGroup);
    pass1.setVertexBuffer(0, vBuffer);
    pass1.draw(vertices.length / 9);
    pass1.end();

    // 2. SSGI 计算
    const pass2 = encoder.beginRenderPass({
      colorAttachments: [{ view: currentSSGITex.createView(), loadOp: "clear", storeOp: "store" }]
    });
    pass2.setPipeline(ssgiPipeline);
    pass2.setBindGroup(0, ssgiBindGroup);
    pass2.setVertexBuffer(0, quadBuffer);
    pass2.draw(6);
    pass2.end();

    // 3. 时空累积 (Ping-Pong)
    const isPing = (frameCount % 2) === 0;
    const pass3 = encoder.beginRenderPass({
      colorAttachments: [{ view: (isPing ? ssgiHistoryB : ssgiHistoryA).createView(), loadOp: "clear", storeOp: "store" }]
    });
    pass3.setPipeline(temporalPipeline);
    pass3.setBindGroup(0, isPing ? temporalBindGroupAtoB : temporalBindGroupBtoA);
    pass3.setVertexBuffer(0, quadBuffer);
    pass3.draw(6);
    pass3.end();

    // 4. 双边上采样降噪合成
    const pass4 = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }]
    });
    pass4.setPipeline(compositePipeline);
    pass4.setBindGroup(0, isPing ? compositeBindGroupWithB : compositeBindGroupWithA);
    pass4.setVertexBuffer(0, quadBuffer);
    pass4.draw(6);
    pass4.end();

    device.queue.submit([encoder.finish()]);

    prevViewMatrix.set(view);
    prevViewProjMatrix.set(viewProj);

    animId = requestAnimationFrame(frame);
  }
  frame();

  return () => {
    cancelAnimationFrame(animId);
    vBuffer.destroy(); quadBuffer.destroy(); uniformBuffer.destroy();
    gColor.destroy(); gNormal.destroy(); gDepth.destroy();
    currentSSGITex.destroy(); ssgiHistoryA.destroy(); ssgiHistoryB.destroy();
  };
}