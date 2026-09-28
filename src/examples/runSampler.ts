// src/examples/runSampler.ts
import { Mat4 } from "../utils/math";
import type { SimpleGUI } from "../utils/gui";

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

export function runSampler(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // 1. 创建水平大平面网格 (pos vec3 + uv vec2)
  // prettier-ignore
  const vertexData = new Float32Array([
    // pos(x,y,z)        uv(u,v)
    -3.0, 0.0, -3.0,    0.0, 0.0,
     3.0, 0.0, -3.0,    1.0, 0.0,
     3.0, 0.0,  3.0,    1.0, 1.0,
    -3.0, 0.0,  3.0,    0.0, 1.0,
  ]);
  const indexData = new Uint16Array([0, 1, 2, 0, 2, 3]);

  const vBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vBuffer, 0, vertexData);

  const iBuffer = device.createBuffer({
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(iBuffer, 0, indexData);

  // 2. 创建 8x8 高对比度特征测试纹理（非对称色彩）
  const texSize = 8;
  const texPixels = new Uint8Array(texSize * texSize * 4);
  for (let y = 0; y < texSize; y++) {
    for (let x = 0; x < texSize; x++) {
      const idx = (y * texSize + x) * 4;
      if (x === 0 || y === 0) {
        // 顶部与左侧边缘：黄色（便于直观辨认 Clamp-to-edge 边缘拉伸效果）
        texPixels[idx] = 255; texPixels[idx + 1] = 220; texPixels[idx + 2] = 0; texPixels[idx + 3] = 255;
      } else if (x < 4 && y < 4) {
        // 左上象限：红色
        texPixels[idx] = 235; texPixels[idx + 1] = 60; texPixels[idx + 2] = 60; texPixels[idx + 3] = 255;
      } else if (x >= 4 && y < 4) {
        // 右上象限：蓝色
        texPixels[idx] = 60; texPixels[idx + 1] = 130; texPixels[idx + 2] = 245; texPixels[idx + 3] = 255;
      } else if (x < 4 && y >= 4) {
        // 左下象限：绿色
        texPixels[idx] = 50; texPixels[idx + 1] = 200; texPixels[idx + 2] = 90; texPixels[idx + 3] = 255;
      } else {
        // 右下象限：紫色
        texPixels[idx] = 180; texPixels[idx + 1] = 70; texPixels[idx + 2] = 230; texPixels[idx + 3] = 255;
      }
    }
  }

  const texture = device.createTexture({
    size: [texSize, texSize, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture },
    texPixels,
    { bytesPerRow: texSize * 4 },
    [texSize, texSize, 1]
  );
  const textureView = texture.createView();

  // Uniform Buffer：mvp (64 bytes) + uvScale (8 bytes) + padding (8 bytes) = 80 bytes
  const uniformBuffer = device.createBuffer({
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  let depthTexture = device.createTexture({
    size: [canvas.width || 800, canvas.height || 600],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // 3. WGSL 着色器
  const shaderCode = `
    struct Uniforms {
      mvp: mat4x4f,
      uvScale: vec2f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var mySampler: sampler;
    @group(0) @binding(2) var myTexture: texture_2d<f32>;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@location(0) pos: vec3f, @location(1) uv: vec2f) -> VertexOut {
      var out: VertexOut;
      out.pos = u.mvp * vec4f(pos, 1.0);
      // 以中心 (0.5, 0.5) 扩展 UV，向边缘延伸出超出 [0, 1] 的区间
      out.uv = (uv - 0.5) * u.uvScale + 0.5;
      return out;
    }

    @fragment
    fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
      return textureSample(myTexture, mySampler, uv);
    }
  `;

  const module = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 20,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x2" },
          ],
        },
      ],
    },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // 4. 模式映射表（通过数值下标索引避免 GUI 类型限制）
  const addressModes: GPUAddressMode[] = ["clamp-to-edge", "repeat", "mirror-repeat"];
  const filterModes: GPUFilterMode[] = ["nearest", "linear"];

  const samplerConfig = {
    addressModeU: 1, // 0: clamp-to-edge, 1: repeat, 2: mirror-repeat
    addressModeV: 1,
    magFilter: 0,    // 0: nearest, 1: linear
    minFilter: 0,
    uvScale: 3.0,
  };

  let currentSampler: GPUSampler | null = null;
  let bindGroup: GPUBindGroup | null = null;
  let lastSamplerKey = "";

  function rebuildSamplerAndBindGroup() {
    const uMode = addressModes[Math.round(samplerConfig.addressModeU)] || "repeat";
    const vMode = addressModes[Math.round(samplerConfig.addressModeV)] || "repeat";
    const mag = filterModes[Math.round(samplerConfig.magFilter)] || "nearest";
    const min = filterModes[Math.round(samplerConfig.minFilter)] || "nearest";

    const key = `${uMode}_${vMode}_${mag}_${min}`;
    if (key === lastSamplerKey && bindGroup) return;
    lastSamplerKey = key;

    currentSampler = device.createSampler({
      addressModeU: uMode,
      addressModeV: vMode,
      magFilter: mag,
      minFilter: min,
    });

    bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: currentSampler },
        { binding: 2, resource: textureView },
      ],
    });
  }

  rebuildSamplerAndBindGroup();

  // 5. GUI 控件注册（完全兼容数值滑块）
  gui.add(samplerConfig, "magFilter", 0, 1, 1).name("放大过滤 (0:近邻, 1:线性)");
  gui.add(samplerConfig, "minFilter", 0, 1, 1).name("缩小过滤 (0:近邻, 1:线性)");
  gui.add(samplerConfig, "addressModeU", 0, 2, 1).name("U 轴寻址 (0:拉伸, 1:重复, 2:镜像)");
  gui.add(samplerConfig, "addressModeV", 0, 2, 1).name("V 轴寻址 (0:拉伸, 1:重复, 2:镜像)");
  gui.add(samplerConfig, "uvScale", 0.5, 6.0, 0.1).name("UV 缩放 (UV Scale)");

  // 快捷切换按钮
  gui.addButton("切换过滤模式 (nearest ⇋ linear)", () => {
    const next = samplerConfig.magFilter === 0 ? 1 : 0;
    samplerConfig.magFilter = next;
    samplerConfig.minFilter = next;
    gui.updateDisplay();
  });
  gui.addButton("循环寻址模式 (clamp ➔ repeat ➔ mirror)", () => {
    const next = (samplerConfig.addressModeU + 1) % 3;
    samplerConfig.addressModeU = next;
    samplerConfig.addressModeV = next;
    gui.updateDisplay();
  });

  // 相机参数
  const defaultCamera = { distance: 5.5, theta: 25, phi: 40, panX: 0.0, panY: 0.0, fov: 60 };
  const camera = { ...defaultCamera };

  gui.add(camera, "distance", 1.5, 15.0, 0.1).name("相机距离 (Dist)");
  gui.add(camera, "phi", 5, 85, 1).name("相机仰角 (Pitch)");
  gui.addButton("重置视图与参数", () => {
    Object.assign(camera, defaultCamera);
    samplerConfig.addressModeU = 1;
    samplerConfig.addressModeV = 1;
    samplerConfig.magFilter = 0;
    samplerConfig.minFilter = 0;
    samplerConfig.uvScale = 3.0;
    gui.updateDisplay();
  });

  gui.addTextInfo(
    "<b>模式代号说明：</b><br>" +
    "• <b>过滤:</b> 0 = nearest (马赛克锐利), 1 = linear (平滑插值)<br>" +
    "• <b>寻址:</b> 0 = clamp-to-edge (边缘黄色拉伸), 1 = repeat (重复平铺), 2 = mirror-repeat (镜像翻转)"
  );

  // 6. 相机交互
  let isDragging = false;
  let dragButton = 0;
  let lastX = 0, lastY = 0;

  const onPointerDown = (e: PointerEvent) => {
    isDragging = true;
    dragButton = e.shiftKey ? 2 : e.button;
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

    if (dragButton === 0) {
      camera.theta -= dx * 0.5;
      camera.phi = Math.max(5, Math.min(85, camera.phi + dy * 0.5));
    } else if (dragButton === 2) {
      const factor = camera.distance * 0.0018;
      camera.panX -= dx * factor;
      camera.panY += dy * factor;
    }
    gui.updateDisplay();
  };

  const onPointerUp = (e: PointerEvent) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  };

  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    camera.distance = Math.max(1.5, Math.min(15.0, camera.distance + e.deltaY * 0.005));
    gui.updateDisplay();
  };

  const onContextMenu = (e: MouseEvent) => e.preventDefault();

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("contextmenu", onContextMenu);

  // 7. 渲染循环
  let animId: number;
  const uniformData = new Float32Array(20);

  function frame() {
    rebuildSamplerAndBindGroup();

    if (depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;

    const eyeX = camera.panX + camera.distance * Math.cos(radPhi) * Math.sin(radTheta);
    const eyeY = camera.panY + camera.distance * Math.sin(radPhi);
    const eyeZ = camera.distance * Math.cos(radPhi) * Math.cos(radTheta);

    const viewMatrix = createLookAtMatrix([eyeX, eyeY, eyeZ], [camera.panX, camera.panY, 0], [0, 1, 0]);
    const aspect = (canvas.width || 800) / (canvas.height || 600);
    const projMatrix = Mat4.perspective((camera.fov * Math.PI) / 180, aspect, 0.1, 100);
    const mvpMatrix = Mat4.multiply(projMatrix, viewMatrix);

    uniformData.set(mvpMatrix, 0);
    uniformData[16] = samplerConfig.uvScale;
    uniformData[17] = samplerConfig.uvScale;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.12, g: 0.12, b: 0.16, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup!);
    pass.setVertexBuffer(0, vBuffer);
    pass.setIndexBuffer(iBuffer, "uint16");
    pass.drawIndexed(6);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  frame();

  // 8. 资源清理
  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);
    canvas.removeEventListener("contextmenu", onContextMenu);
    vBuffer.destroy();
    iBuffer.destroy();
    uniformBuffer.destroy();
    depthTexture.destroy();
    texture.destroy();
  };
}