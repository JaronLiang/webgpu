// src/examples/ssgi.ts
import type { SimpleGUI } from "../utils/gui";

// 矩阵计算
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

// 4x4 矩阵求逆 (包含平移与旋转的完整求逆)
function mat4Invert(m: Float32Array): Float32Array {
  const out = new Float32Array(16);
  const b00 = m[0] * m[5] - m[1] * m[4], b01 = m[0] * m[6] - m[2] * m[4];
  const b02 = m[0] * m[7] - m[3] * m[4], b03 = m[1] * m[6] - m[2] * m[5];
  const b04 = m[1] * m[7] - m[3] * m[5], b05 = m[2] * m[7] - m[3] * m[6];
  const b06 = m[8] * m[13] - m[9] * m[12], b07 = m[8] * m[14] - m[10] * m[12];
  const b08 = m[8] * m[15] - m[11] * m[12], b09 = m[9] * m[14] - m[10] * m[13];
  const b10 = m[9] * m[15] - m[11] * m[13], b11 = m[10] * m[15] - m[11] * m[14];
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return out;
  const inv = 1.0 / det;
  out[0] = (m[5]*b11 - m[6]*b10 + m[7]*b09)*inv;
  out[1] = (-m[1]*b11 + m[2]*b10 - m[3]*b09)*inv;
  out[2] = (m[13]*b05 - m[14]*b04 + m[15]*b03)*inv;
  out[3] = (-m[9]*b05 + m[10]*b04 - m[11]*b03)*inv;
  out[4] = (-m[4]*b11 + m[6]*b08 - m[7]*b07)*inv;
  out[5] = (m[0]*b11 - m[2]*b08 + m[3]*b07)*inv;
  out[6] = (-m[12]*b05 + m[14]*b02 - m[15]*b01)*inv;
  out[7] = (m[8]*b05 - m[10]*b02 + m[11]*b01)*inv;
  out[8] = (m[4]*b10 - m[5]*b08 + m[7]*b06)*inv;
  out[9] = (-m[0]*b10 + m[1]*b08 - m[3]*b06)*inv;
  out[10] = (m[12]*b04 - m[13]*b02 + m[15]*b00)*inv;
  out[11] = (-m[8]*b04 + m[9]*b02 - m[11]*b00)*inv;
  out[12] = (-m[4]*b09 + m[5]*b07 - m[6]*b06)*inv;
  out[13] = (m[0]*b09 - m[1]*b07 + m[2]*b06)*inv;
  out[14] = (-m[12]*b03 + m[13]*b01 - m[14]*b00)*inv;
  out[15] = (m[8]*b03 - m[9]*b01 + m[10]*b00)*inv;
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
  device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat, canvas: HTMLCanvasElement, gui: SimpleGUI
) {
  const fullWidth = canvas.width || 800;
  const fullHeight = canvas.height || 600;
  const halfWidth = Math.max(1, Math.floor(fullWidth / 2));
  const halfHeight = Math.max(1, Math.floor(fullHeight / 2));

  // 1. 经典 Cornell Box 几何体
  const vertices = new Float32Array([
    // 地板 (灰色)
    -3,0,-3,  0,1,0,  0.8,0.8,0.8,   3,0,-3,  0,1,0,  0.8,0.8,0.8,   3,0,3,  0,1,0,  0.8,0.8,0.8,
    -3,0,-3,  0,1,0,  0.8,0.8,0.8,   3,0,3,  0,1,0,  0.8,0.8,0.8,  -3,0,3,  0,1,0,  0.8,0.8,0.8,
    // 顶板 (灰色)
    -3,5, 3,  0,-1,0, 0.8,0.8,0.8,   3,5, 3,  0,-1,0, 0.8,0.8,0.8,   3,5,-3,  0,-1,0, 0.8,0.8,0.8,
    -3,5, 3,  0,-1,0, 0.8,0.8,0.8,   3,5,-3,  0,-1,0, 0.8,0.8,0.8,  -3,5,-3,  0,-1,0, 0.8,0.8,0.8,
    // 后墙 (白色)
    -3,0,-3,  0,0,1,  0.8,0.8,0.8,   3,0,-3,  0,0,1,  0.8,0.8,0.8,   3,5,-3,  0,0,1,  0.8,0.8,0.8,
    -3,0,-3,  0,0,1,  0.8,0.8,0.8,   3,5,-3,  0,0,1,  0.8,0.8,0.8,  -3,5,-3,  0,0,1,  0.8,0.8,0.8,
    // 左墙 (鲜艳红 - 强溢色源)
    -3,0, 3,  1,0,0,  0.95,0.05,0.05, -3,0,-3, 1,0,0,  0.95,0.05,0.05, -3,5,-3, 1,0,0, 0.95,0.05,0.05,
    -3,0, 3,  1,0,0,  0.95,0.05,0.05, -3,5,-3, 1,0,0,  0.95,0.05,0.05, -3,5, 3, 1,0,0, 0.95,0.05,0.05,
    // 右墙 (鲜艳绿 - 强溢色源)
     3,0,-3, -1,0,0,  0.05,0.95,0.05,  3,0, 3, -1,0,0, 0.05,0.95,0.05,  3,5, 3, -1,0,0, 0.05,0.95,0.05,
     3,0,-3, -1,0,0,  0.05,0.95,0.05,  3,5, 3, -1,0,0, 0.05,0.95,0.05,  3,5,-3, -1,0,0, 0.05,0.95,0.05,
    // 中心白色立方体
    -0.8,0,0.8, 0,0,1, 0.9,0.9,0.9,   0.8,0,0.8, 0,0,1, 0.9,0.9,0.9,   0.8,2,0.8, 0,0,1, 0.9,0.9,0.9,
    -0.8,0,0.8, 0,0,1, 0.9,0.9,0.9,   0.8,2,0.8, 0,0,1, 0.9,0.9,0.9,  -0.8,2,0.8, 0,0,1, 0.9,0.9,0.9,
    -0.8,2,0.8, 0,1,0, 0.9,0.9,0.9,   0.8,2,0.8, 0,1,0, 0.9,0.9,0.9,   0.8,2,-0.8, 0,1,0, 0.9,0.9,0.9,
    -0.8,2,0.8, 0,1,0, 0.9,0.9,0.9,   0.8,2,-0.8, 0,1,0, 0.9,0.9,0.9, -0.8,2,-0.8, 0,1,0, 0.9,0.9,0.9,
    -0.8,0,-0.8, -1,0,0, 0.9,0.9,0.9, -0.8,0,0.8, -1,0,0, 0.9,0.9,0.9, -0.8,2,0.8, -1,0,0, 0.9,0.9,0.9,
    -0.8,0,-0.8, -1,0,0, 0.9,0.9,0.9, -0.8,2,0.8, -1,0,0, 0.9,0.9,0.9, -0.8,2,-0.8, -1,0,0, 0.9,0.9,0.9,
     0.8,0,0.8, 1,0,0, 0.9,0.9,0.9,   0.8,0,-0.8, 1,0,0, 0.9,0.9,0.9,  0.8,2,-0.8, 1,0,0, 0.9,0.9,0.9,
     0.8,0,0.8, 1,0,0, 0.9,0.9,0.9,   0.8,2,-0.8, 1,0,0, 0.9,0.9,0.9,  0.8,2,0.8, 1,0,0, 0.9,0.9,0.9,
  ]);
  const vBuffer = device.createBuffer({ size: vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vBuffer, 0, vertices);

  // 全屏 Quad
  const quadData = new Float32Array([-1,-1, 1,-1, -1,1, 1,-1, 1,1, -1,1]);
  const quadBuffer = device.createBuffer({ size: quadData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(quadBuffer, 0, quadData);

  // 统一 Uniform Buffer:
  // view(16), proj(16), invView(16), prevViewProj(16), camParams(4), settings(4), halfRes(4) = 76 floats -> 分配 384B
  const uniformBufferSize = 384;
  const uniformBuffer = device.createBuffer({ size: uniformBufferSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const cpuUniformData = new Float32Array(uniformBufferSize / 4);

  // 2. 纹理创建
  const gColor = device.createTexture({ size: [fullWidth, fullHeight], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const gNormal = device.createTexture({ size: [fullWidth, fullHeight], format: "rg16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const gDepth = device.createTexture({ size: [fullWidth, fullHeight], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  
  // 半分辨率纹理
  const currentSSGITex = device.createTexture({ size: [halfWidth, halfHeight], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const ssgiHistoryA = device.createTexture({ size: [halfWidth, halfHeight], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const ssgiHistoryB = device.createTexture({ size: [halfWidth, halfHeight], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });

  const pointSampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });
  const linearSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  // 3. Pass 1: G-Buffer
  const gbufferWGSL = `
    struct Uniforms {
      view: mat4x4f, proj: mat4x4f, invView: mat4x4f, prevViewProj: mat4x4f,
      camParams: vec4f, settings: vec4f, halfRes: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VIn { @location(0) pos: vec3f, @location(1) norm: vec3f, @location(2) col: vec3f };
    struct VOut { @builtin(position) pos: vec4f, @location(0) normalVS: vec3f, @location(1) color: vec3f };

    @vertex fn vs(v: VIn) -> VOut {
      var o: VOut;
      o.pos = u.proj * u.view * vec4f(v.pos, 1.0);
      o.normalVS = (u.view * vec4f(v.norm, 0.0)).xyz;
      let lightDir = normalize(vec3f(0.2, 0.9, 0.3));
      let diff = max(dot(v.norm, lightDir), 0.2);
      o.color = v.col * diff;
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
      g.color = vec4f(in.color, 1.0);
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
    fragment: { module: device.createShaderModule({ code: gbufferWGSL }), entryPoint: "fs", targets: [{ format: "rgba8unorm" }, { format: "rg16float" }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    primitive: { topology: "triangle-list" }
  });

  // 4. Pass 2: SSGI 计算着色器 (修复横条状白斑：稳健半球采样 + 正确深度剔除)
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
      let radius = u.settings.y;
      let frameIndex = u.settings.z;

      let samplesPerPixel = 4;
      var indirectSum = vec3f(0.0);
      let randSeed = hash12(floor(fragCoord.xy));

      // Duff 正交切线基底：防止极轴奇异点退化产生横斑
      let signZ = select(-1.0, 1.0, normalVS.z >= 0.0);
      let a = -1.0 / (signZ + normalVS.z);
      let b = normalVS.x * normalVS.y * a;
      let tangent = vec3f(1.0 + signZ * normalVS.x * normalVS.x * a, signZ * b, -signZ * normalVS.x);
      let bitangent = vec3f(b, signZ + normalVS.y * normalVS.y * a, -normalVS.y);

      for (var s = 0; s < samplesPerPixel; s++) {
        let fi = f32(s) + randSeed;
        let phi = fi * 2.39996 + frameIndex * 1.618;
        let cosTheta = sqrt(1.0 - (f32(s) + 0.5) / f32(samplesPerPixel));
        let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        let rayDirVS = tangent * (cos(phi) * sinTheta) + bitangent * (sin(phi) * sinTheta) + normalVS * cosTheta;

        // 起步偏移，彻底杜绝自相交引发的横斑
        var marchPos = posVS + normalVS * (0.05 + linearZ * 0.005);
        let stepDist = radius * 0.25;

        for (var step = 1; step <= 4; step++) {
          marchPos += rayDirVS * stepDist;
          let proj = projectPosToUV(marchPos);

          if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) { break; }

          let hitRawZ = textureSampleLevel(depthTex, pointSamp, proj.xy, 0);
          if (hitRawZ >= 0.9999) { continue; }

          let hitLinearZ = getLinearDepth(hitRawZ);
          let deltaZ = proj.z - hitLinearZ;

          // 核心厚度区间判定：防止把墙面/背景误认成溢色
          let thickness = max(0.04, hitLinearZ * 0.03);

          if (deltaZ > 0.01 && deltaZ < thickness) {
            let hitCol = textureSampleLevel(colorTex, pointSamp, proj.xy, 0).rgb;
            let hitNormal = octDecode(textureSampleLevel(normalTex, pointSamp, proj.xy, 0).xy);

            let receiverWeight = max(dot(normalVS, rayDirVS), 0.0);
            let bounceWeight = max(dot(hitNormal, -rayDirVS), 0.0);
            let weight = receiverWeight * bounceWeight;

            // 抑制极端过亮值 (Firefly clamp)
            let bounceLight = min(hitCol * weight, vec3f(1.5));
            indirectSum += bounceLight * (1.0 / f32(samplesPerPixel));
            break;
          }
        }
      }

      return vec4f(indirectSum * giIntensity, 1.0);
    }
  `;
  const ssgiPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: ssgiWGSL }), entryPoint: "vs", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: device.createShaderModule({ code: ssgiWGSL }), entryPoint: "fs", targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list" }
  });

  // 5. Pass 3: 【彻底修复疯狂闪烁】真·数学精准时空重投影 (使用 CPU 逆矩阵 invView)
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

      // 【核心修复】：使用严格计算的真实 invView，绝不用错位的转置
      let posWS = (u.invView * vec4f(posVS, 1.0)).xyz;

      // 准确重投影到上一帧历史 UV
      let prevClip = u.prevViewProj * vec4f(posWS, 1.0);
      let prevNDC = prevClip.xyz / prevClip.w;
      let prevUV = vec2f(prevNDC.x * 0.5 + 0.5, 1.0 - (prevNDC.y * 0.5 + 0.5));

      // 若历史 UV 出界，重置为当前帧
      if (prevUV.x < 0.0 || prevUV.x > 1.0 || prevUV.y < 0.0 || prevUV.y > 1.0) {
        return vec4f(currentGI, 1.0);
      }

      // 遮挡脱离检验 (Disocclusion Check)
      let historyDepth = getLinearDepth(textureSampleLevel(fullDepthTex, pointSamp, prevUV, 0));
      let depthDiff = abs(linearZ - historyDepth);
      let depthInvalid = depthDiff > max(0.12, linearZ * 0.06);

      let currentNormal = octDecode(textureSampleLevel(fullNormalTex, pointSamp, uv, 0.0).xy);
      let historyNormal = octDecode(textureSampleLevel(fullNormalTex, pointSamp, prevUV, 0.0).xy);
      let normalInvalid = dot(currentNormal, historyNormal) < 0.8;

      var alpha = u.settings.w;
      if (depthInvalid || normalInvalid) {
        alpha = 1.0; // 发生几何断裂或遮挡，抛弃历史
      }

      let historyGI = textureSampleLevel(historyTex, linearSamp, prevUV, 0.0).rgb;
      let accumulated = mix(historyGI, currentGI, alpha);
      return vec4f(accumulated, 1.0);
    }
  `;
  const temporalPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: temporalWGSL }), entryPoint: "vs", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: device.createShaderModule({ code: temporalWGSL }), entryPoint: "fs", targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list" }
  });

  // 6. Pass 4: 空间双边联合上采样降噪合成 (Joint Bilateral Upsampling)
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

    @fragment fn fs(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      let uv = fragCoord.xy / vec2f(textureDimensions(fullColorTex));
      let directColor = textureSampleLevel(fullColorTex, pointSamp, uv, 0.0).rgb;

      let rawZ = textureSampleLevel(fullDepthTex, pointSamp, uv, 0);
      if (rawZ >= 0.9999) { return vec4f(directColor, 1.0); }

      let centerDepth = getLinearDepth(rawZ);
      let centerNormal = octDecode(textureSampleLevel(fullNormalTex, pointSamp, uv, 0).xy);

      let fullPixel = 1.0 / vec2f(textureDimensions(fullColorTex));
      let halfPixel = 1.0 / u.halfRes.xy;

      var totalWeight = 0.0;
      var filteredGI = vec3f(0.0);

      // 3x3 空间交叉双边滤波：彻底消除残留噪斑
      for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
          let offset = vec2f(f32(x), f32(y));
          let giUV = uv + offset * halfPixel;
          let geomUV = uv + offset * fullPixel;

          let sampleGI = textureSampleLevel(resolvedSSGITex, linearSamp, giUV, 0.0).rgb;
          let sampleDepth = getLinearDepth(textureSampleLevel(fullDepthTex, pointSamp, geomUV, 0));
          let sampleNormal = octDecode(textureSampleLevel(fullNormalTex, pointSamp, geomUV, 0).xy);

          let depthWeight = exp(-abs(centerDepth - sampleDepth) * 20.0);
          let normalWeight = max(0.0, pow(dot(centerNormal, sampleNormal), 16.0));
          let w = depthWeight * normalWeight;

          filteredGI += sampleGI * w;
          totalWeight += w;
        }
      }

      filteredGI = filteredGI / max(totalWeight, 0.0001);
      return vec4f(directColor + filteredGI, 1.0);
    }
  `;
  const compositePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: compositeWGSL }), entryPoint: "vs", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: device.createShaderModule({ code: compositeWGSL }), entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" }
  });

  // 7. 预创建并缓存 BindGroups
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

  // 8. 控制交互
  const camera = { distance: 7.0, theta: 0, phi: 12, panY: 2.2 };
  const ssgiSettings = { intensity: 1.6, radius: 1.5 };

  gui.addTextInfo("<b>工业级高保真 SSGI</b><br>精准时空重投影 + 双边联合降噪");
  gui.add(ssgiSettings, "intensity", 0.0, 3.0, 0.1).name("GI 强度");
  gui.add(ssgiSettings, "radius", 0.3, 3.0, 0.1).name("光线弹射半径");
  gui.add(camera, "theta", -90, 90, 1).name("偏航角");
  gui.add(camera, "phi", -20, 60, 1).name("俯仰角");

  let isDragging = false, lastX = 0, lastY = 0;
  canvas.onpointerdown = (e) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); };
  canvas.onpointermove = (e) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.4;
    camera.phi = Math.max(-20, Math.min(60, camera.phi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY;
    gui.updateDisplay();
  };
  canvas.onpointerup = (e) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} };

  let animId: number;
  let frameCount = 0;
  let prevViewMatrix = new Float32Array(16);
  let prevViewProjMatrix = new Float32Array(16);

  function frame() {
    frameCount++;
    const near = 0.1;
    const far = 50.0;
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
    const invView = mat4Invert(view); // 真实逆视图矩阵！

    // 检测相机位移
    const camDelta = matrixDifference(view, prevViewMatrix);
    const isCameraMoving = camDelta > 1e-4;
    const temporalAlpha = isCameraMoving ? 0.35 : 0.05;

    // 单次写入连续缓冲
    cpuUniformData.set(view, 0);
    cpuUniformData.set(proj, 16);
    cpuUniformData.set(invView, 32);
    cpuUniformData.set(prevViewProjMatrix, 48);
    cpuUniformData.set([near, far, tanHalfFov, aspect], 64);
    cpuUniformData.set([ssgiSettings.intensity, ssgiSettings.radius, frameCount & 1023, temporalAlpha], 68);
    cpuUniformData.set([halfWidth, halfHeight, 0, 0], 72);
    device.queue.writeBuffer(uniformBuffer, 0, cpuUniformData);

    const encoder = device.createCommandEncoder();

    // Pass 1: G-Buffer
    const pass1 = encoder.beginRenderPass({
      colorAttachments: [
        { view: gColor.createView(), clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 }, loadOp: "clear", storeOp: "store" },
        { view: gNormal.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }
      ],
      depthStencilAttachment: { view: gDepth.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" }
    });
    pass1.setPipeline(gbufferPipeline);
    pass1.setBindGroup(0, gbufferBindGroup);
    pass1.setVertexBuffer(0, vBuffer);
    pass1.draw(vertices.length / 9);
    pass1.end();

    // Pass 2: SSGI 计算
    const pass2 = encoder.beginRenderPass({
      colorAttachments: [{ view: currentSSGITex.createView(), loadOp: "clear", storeOp: "store" }]
    });
    pass2.setPipeline(ssgiPipeline);
    pass2.setBindGroup(0, ssgiBindGroup);
    pass2.setVertexBuffer(0, quadBuffer);
    pass2.draw(6);
    pass2.end();

    // Pass 3: 时空重投影累积
    const isPing = (frameCount % 2) === 0;
    const pass3 = encoder.beginRenderPass({
      colorAttachments: [{ view: (isPing ? ssgiHistoryB : ssgiHistoryA).createView(), loadOp: "clear", storeOp: "store" }]
    });
    pass3.setPipeline(temporalPipeline);
    pass3.setBindGroup(0, isPing ? temporalBindGroupAtoB : temporalBindGroupBtoA);
    pass3.setVertexBuffer(0, quadBuffer);
    pass3.draw(6);
    pass3.end();

    // Pass 4: 空间联合双边滤波合成
    const pass4 = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }]
    });
    pass4.setPipeline(compositePipeline);
    pass4.setBindGroup(0, isPing ? compositeBindGroupWithB : compositeBindGroupWithA);
    pass4.setVertexBuffer(0, quadBuffer);
    pass4.draw(6);
    pass4.end();

    device.queue.submit([encoder.finish()]);

    // 更新历史矩阵
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