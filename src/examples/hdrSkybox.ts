// src/examples/hdrSkybox.ts
import { Mat4 } from "../utils/math";
import type { SimpleGUI } from "../utils/gui";

// 视角观察矩阵
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

// 单精度浮点数转半精度浮点数 (Float32 -> Float16 Uint16 bitpack)
function floatToHalf(val: number): number {
  const f32 = new Float32Array([val]);
  const u32 = new Uint32Array(f32.buffer)[0];
  const sign = (u32 >> 16) & 0x8000;
  let exp = ((u32 >> 23) & 0xff) - 127 + 15;
  const mant = (u32 >> 13) & 0x3ff;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | mant;
}

// 4x4 矩阵求逆矩阵
function invertMat4(m: Float32Array): Float32Array {
  const out = new Float32Array(16);
  const m00 = m[0], m01 = m[1], m02 = m[2], m03 = m[3];
  const m10 = m[4], m11 = m[5], m12 = m[6], m13 = m[7];
  const m20 = m[8], m21 = m[9], m22 = m[10], m23 = m[11];
  const m30 = m[12], m31 = m[13], m32 = m[14], m33 = m[15];

  const b00 = m00 * m11 - m01 * m10;
  const b01 = m00 * m12 - m02 * m10;
  const b02 = m00 * m13 - m03 * m10;
  const b03 = m01 * m12 - m02 * m11;
  const b04 = m01 * m13 - m03 * m11;
  const b05 = m02 * m13 - m03 * m12;
  const b06 = m20 * m31 - m21 * m30;
  const b07 = m20 * m32 - m22 * m30;
  const b08 = m20 * m33 - m23 * m30;
  const b09 = m21 * m32 - m22 * m31;
  const b10 = m21 * m33 - m23 * m31;
  const b11 = m22 * m33 - m23 * m32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return out;
  det = 1.0 / det;

  out[0] = (m11 * b11 - m12 * b10 + m13 * b09) * det;
  out[1] = (m02 * b10 - m01 * b11 - m03 * b09) * det;
  out[2] = (m31 * b05 - m32 * b04 + m33 * b03) * det;
  out[3] = (m22 * b04 - m21 * b05 - m23 * b03) * det;
  out[4] = (m12 * b08 - m10 * b11 - m13 * b07) * det;
  out[5] = (m00 * b11 - m02 * b08 + m03 * b07) * det;
  out[6] = (m32 * b02 - m30 * b05 - m33 * b01) * det;
  out[7] = (m20 * b05 - m22 * b02 + m23 * b01) * det;
  out[8] = (m10 * b10 - m11 * b08 + m13 * b06) * det;
  out[9] = (m01 * b08 - m00 * b10 - m03 * b06) * det;
  out[10] = (m30 * b04 - m31 * b02 + m33 * b00) * det;
  out[11] = (m21 * b02 - m20 * b04 - m23 * b00) * det;
  out[12] = (m11 * b07 - m10 * b09 - m12 * b06) * det;
  out[13] = (m00 * b09 - m01 * b07 + m02 * b06) * det;
  out[14] = (m31 * b01 - m30 * b03 - m32 * b00) * det;
  out[15] = (m20 * b03 - m21 * b01 + m22 * b00) * det;
  return out;
}

export function runHdrSkybox(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  const faceSize = 128;

  // 1. 创建 HDR (rgba16float) 和 LDR (rgba8unorm) 两套立方体贴图
  const hdrTexture = device.createTexture({
    size: [faceSize, faceSize, 6],
    format: "rgba16float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  const ldrTexture = device.createTexture({
    size: [faceSize, faceSize, 6],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  // 太阳方向
  const sunDir = [0.45, 0.45, 0.77];
  const sunLen = Math.hypot(...sunDir);
  const sx = sunDir[0] / sunLen, sy = sunDir[1] / sunLen, sz = sunDir[2] / sunLen;

  // 生成 6 个立方体面：+X, -X, +Y, -Y, +Z, -Z
  for (let face = 0; face < 6; face++) {
    const hdrData = new Uint16Array(faceSize * faceSize * 4);
    const ldrData = new Uint8Array(faceSize * faceSize * 4);

    for (let y = 0; y < faceSize; y++) {
      for (let x = 0; x < faceSize; x++) {
        const u = (x + 0.5) / faceSize * 2 - 1;
        const v = -((y + 0.5) / faceSize * 2 - 1);

        let dx = 0, dy = 0, dz = 0;
        if (face === 0) { dx = 1;  dy = v;  dz = -u; }
        else if (face === 1) { dx = -1; dy = v;  dz = u;  }
        else if (face === 2) { dx = u;  dy = 1;  dz = -v; }
        else if (face === 3) { dx = u;  dy = -1; dz = v;  }
        else if (face === 4) { dx = u;  dy = v;  dz = 1;  }
        else if (face === 5) { dx = -u; dy = v;  dz = -1; }

        const len = Math.hypot(dx, dy, dz) || 1;
        const nx = dx / len, ny = dy / len, nz = dz / len;

        // 大气渐变
        let r = 0.15, g = 0.35, b = 0.75;
        if (ny > 0) {
          const horizonBlend = Math.pow(1.0 - ny, 3.0);
          r = 0.12 * (1 - horizonBlend) + 0.65 * horizonBlend;
          g = 0.35 * (1 - horizonBlend) + 0.75 * horizonBlend;
          b = 0.85 * (1 - horizonBlend) + 0.95 * horizonBlend;
        } else {
          const ground = Math.min(1.0, -ny * 2.0);
          r = 0.25 * (1 - ground) + 0.08 * ground;
          g = 0.22 * (1 - ground) + 0.07 * ground;
          b = 0.20 * (1 - ground) + 0.06 * ground;
        }

        // 计算太阳高光
        const dotSun = Math.max(0, nx * sx + ny * sy + nz * sz);
        if (dotSun > 0.0) {
          const corona = Math.pow(dotSun, 64) * 6.0;
          const sunCore = Math.pow(dotSun, 512) * 35.0;
          const totalSun = corona + sunCore;

          r += totalSun * 1.0;
          g += totalSun * 0.9;
          b += totalSun * 0.7;
        }

        const idx = (y * faceSize + x) * 4;

        // HDR 数据 (Float16)
        hdrData[idx] = floatToHalf(r);
        hdrData[idx + 1] = floatToHalf(g);
        hdrData[idx + 2] = floatToHalf(b);
        hdrData[idx + 3] = floatToHalf(1.0);

        // LDR 数据 (截断封顶在 255)
        ldrData[idx] = Math.min(255, Math.max(0, Math.round(r * 255)));
        ldrData[idx + 1] = Math.min(255, Math.max(0, Math.round(g * 255)));
        ldrData[idx + 2] = Math.min(255, Math.max(0, Math.round(b * 255)));
        ldrData[idx + 3] = 255;
      }
    }

    device.queue.writeTexture(
      { texture: hdrTexture, origin: [0, 0, face] },
      hdrData,
      { bytesPerRow: faceSize * 8 },
      [faceSize, faceSize, 1]
    );

    device.queue.writeTexture(
      { texture: ldrTexture, origin: [0, 0, face] },
      ldrData,
      { bytesPerRow: faceSize * 4 },
      [faceSize, faceSize, 1]
    );
  }

  const hdrView = hdrTexture.createView({ dimension: "cube" });
  const ldrView = ldrTexture.createView({ dimension: "cube" });
  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  // 2. Uniform Buffer: 分配 128 字节保证超越最小绑定对齐限制
  const uniformBuffer = device.createBuffer({
    size: 128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // 3. 全屏着色器（pad0/1/2 替代 vec3f 避免跳跃对齐）
  const shaderCode = `
    struct Uniforms {
      invViewProj: mat4x4f,
      resolution: vec2f,
      exposure: f32,
      displayMode: f32,
      toneMapping: f32,
      pad0: f32,
      pad1: f32,
      pad2: f32,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var skySampler: sampler;
    @group(0) @binding(2) var hdrCube: texture_cube<f32>;
    @group(0) @binding(3) var ldrCube: texture_cube<f32>;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
      var p = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0)
      );
      var out: VertexOut;
      out.pos = vec4f(p[vid], 1.0, 1.0);
      out.uv = p[vid] * 0.5 + 0.5;
      return out;
    }

    fn acesFilmic(x: vec3f) -> vec3f {
      let a = 2.51;
      let b = 0.03;
      let c = 2.43;
      let d = 0.59;
      let e = 0.14;
      return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
    }

    fn applyTonemap(c: vec3f, tm: f32) -> vec3f {
      var mapped = c;
      if (tm > 1.5) {
        mapped = acesFilmic(c);
      } else if (tm > 0.5) {
        mapped = c / (c + vec3f(1.0));
      }
      return pow(mapped, vec3f(1.0 / 2.2));
    }

    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4f {
      let clip = vec4f(in.uv * 2.0 - 1.0, 1.0, 1.0);
      let unproj = u.invViewProj * clip;
      let viewRay = normalize(unproj.xyz / unproj.w);

      let rawHdr = textureSample(hdrCube, skySampler, viewRay).rgb;
      let rawLdr = textureSample(ldrCube, skySampler, viewRay).rgb;

      let screenX = in.pos.x / u.resolution.x;
      var finalColor = vec3f(0.0);

      let splitDist = abs(screenX - 0.5) * u.resolution.x;
      if (u.displayMode < 0.5 && splitDist < 1.5) {
        return vec4f(1.0, 1.0, 1.0, 1.0);
      }

      var isHdr = false;
      if (u.displayMode < 0.5) {
        isHdr = screenX > 0.5;
      } else if (u.displayMode < 1.5) {
        isHdr = true;
      } else {
        isHdr = false;
      }

      if (isHdr) {
        let exposed = rawHdr * u.exposure;
        finalColor = applyTonemap(exposed, u.toneMapping);
      } else {
        let exposed = rawLdr * u.exposure;
        finalColor = applyTonemap(exposed, u.toneMapping);
      }

      return vec4f(finalColor, 1.0);
    }
  `;

  const module = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs_main" },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: hdrView },
      { binding: 3, resource: ldrView },
    ],
  });

  // 4. GUI 控制
  const params = {
    displayMode: 0,
    exposure: 1.0,
    toneMapping: 2,
    pitch: 15,
    yaw: 40,
  };

  gui.add(params, "displayMode", 0, 2, 1).name("对比模式 (0:分屏 1:HDR 2:LDR)");
  gui.add(params, "exposure", 0.05, 5.0, 0.05).name("曝光系数 (Exposure)");
  gui.add(params, "toneMapping", 0, 2, 1).name("色调映射 (0:None 1:Rein 2:ACES)");

  gui.addButton("一键对比: 极低曝光(0.15) 观察太阳高光", () => {
    params.exposure = 0.15;
    gui.updateDisplay();
  });
  gui.addButton("一键对比: 正常曝光(1.00)", () => {
    params.exposure = 1.0;
    gui.updateDisplay();
  });
  gui.addButton("循环切换对比模式", () => {
    params.displayMode = (params.displayMode + 1) % 3;
    gui.updateDisplay();
  });

  gui.addTextInfo(
    "<b>【LDR vs HDR 核心实验指南】：</b><br>" +
    "1. <b>向左拖动查看太阳：</b> 在默认分屏模式下将太阳转到中间分割线附近。<br>" +
    "2. <b>将 Exposure 调低至 0.15~0.25：</b><br>" +
    "   • <b>左侧 (LDR):</b> 太阳变成一块死气沉沉的<b>灰色平顶圆盘</b>（信息在上传前已被剪切在 1.0）。<br>" +
    "   • <b>右侧 (HDR):</b> 太阳呈现<b>极高亮度的白热核心与层次分明的渐变光晕</b>（保留 35.0+ 浮点能量）。<br>" +
    "3. <b>切换色调映射：</b> 观察 ACES 对高光与反差色彩的平滑电影感压缩。"
  );

  // 5. 事件交互
  let isDragging = false;
  let lastX = 0, lastY = 0;

  const onPointerDown = (e: PointerEvent) => {
    isDragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    params.yaw -= dx * 0.3;
    params.pitch = Math.max(-85, Math.min(85, params.pitch + dy * 0.3));
    gui.updateDisplay();
  };
  const onPointerUp = (e: PointerEvent) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  };

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);

  // 6. 渲染循环（32 个 float = 128 字节）
  let animId: number;
  const uniformData = new Float32Array(32);

  function frame() {
    const width = canvas.width || 800;
    const height = canvas.height || 600;

    const radYaw = (params.yaw * Math.PI) / 180;
    const radPitch = (params.pitch * Math.PI) / 180;

    const dirX = Math.cos(radPitch) * Math.sin(radYaw);
    const dirY = Math.sin(radPitch);
    const dirZ = Math.cos(radPitch) * Math.cos(radYaw);

    const viewMatrix = createLookAtMatrix([0, 0, 0], [dirX, dirY, dirZ], [0, 1, 0]);
    const projMatrix = Mat4.perspective((65 * Math.PI) / 180, width / height, 0.1, 100);
    const viewProj = Mat4.multiply(projMatrix, viewMatrix);
    const invViewProj = invertMat4(viewProj);

    uniformData.set(invViewProj, 0);
    uniformData[16] = width;
    uniformData[17] = height;
    uniformData[18] = params.exposure;
    uniformData[19] = params.displayMode;
    uniformData[20] = params.toneMapping;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  frame();

  // 7. 清理
  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    uniformBuffer.destroy();
    hdrTexture.destroy();
    ldrTexture.destroy();
  };
}