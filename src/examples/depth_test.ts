// src/examples/depth_test.ts
import { Mat4 } from "../utils/math";
import type { SimpleGUI } from "../utils/gui";

// 视图矩阵辅助函数
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

// 4x4 平移矩阵
function createTranslationMatrix(x: number, y: number, z: number): Float32Array {
  // prettier-ignore
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1
  ]);
}

export function runDepthTest(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // 1. 六色立方体顶点数据 (位置 3 + 颜色 3)
  // prettier-ignore
  const vertexData = new Float32Array([
    -1,-1, 1, 1,0.2,0.2,   1,-1, 1, 1,0.2,0.2,   1, 1, 1, 1,0.2,0.2,  -1, 1, 1, 1,0.2,0.2,
    -1,-1,-1, 0.2,0.8,0.2, -1, 1,-1, 0.2,0.8,0.2,  1, 1,-1, 0.2,0.8,0.2,  1,-1,-1, 0.2,0.8,0.2,
    -1, 1,-1, 0.2,0.4,1,  -1, 1, 1, 0.2,0.4,1,   1, 1, 1, 0.2,0.4,1,   1, 1,-1, 0.2,0.4,1,
    -1,-1,-1, 1,0.9,0.1,   1,-1,-1, 1,0.9,0.1,   1,-1, 1, 1,0.9,0.1,  -1,-1, 1, 1,0.9,0.1,
     1,-1,-1, 0.1,0.8,0.9,  1, 1,-1, 0.1,0.8,0.9,  1, 1, 1, 0.1,0.8,0.9,  1,-1, 1, 0.1,0.8,0.9,
    -1,-1,-1, 0.9,0.2,0.8, -1,-1, 1, 0.9,0.2,0.8, -1, 1, 1, 0.9,0.2,0.8, -1, 1,-1, 0.9,0.2,0.8,
  ]);
  // prettier-ignore
  const indexData = new Uint16Array([
    0,1,2, 0,2,3,       4,5,6, 4,6,7,       8,9,10, 8,10,11,
    12,13,14, 12,14,15, 16,17,18, 16,18,19, 20,21,22, 20,22,23
  ]);

  const vBuffer = device.createBuffer({ size: vertexData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vBuffer, 0, vertexData);
  const iBuffer = device.createBuffer({ size: indexData.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(iBuffer, 0, indexData);

  // 为近处、远处两个立方体创建独立的 Uniform 缓冲
  const uniformBufferNear = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const uniformBufferFar = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  let depthTexture = device.createTexture({
    size: [canvas.width || 800, canvas.height || 600],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const shaderCode = `
    struct Uniforms { mvp: mat4x4f };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    struct VertexOut { @builtin(position) pos: vec4f, @location(0) col: vec3f };

    @vertex fn vs_main(@location(0) pos: vec3f, @location(1) col: vec3f) -> VertexOut {
      var out: VertexOut;
      out.pos = u.mvp * vec4f(pos, 1.0);
      out.col = col;
      return out;
    }
    @fragment fn fs_main(@location(0) col: vec3f) -> @location(0) vec4f {
      return vec4f(col, 1.0);
    }
  `;
  const module = device.createShaderModule({ code: shaderCode });

  // ---------------------------------------------------------------------------
  // 核心修复点：显式创建 BindGroupLayout 和 PipelineLayout，解决跨管线 BindGroup 不兼容错误
  // ---------------------------------------------------------------------------
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const bindGroupNear = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBufferNear } }],
  });
  const bindGroupFar = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBufferFar } }],
  });

  // 2. 深度测试与剔除模式配置
  const depthCompareList: GPUCompareFunction[] = [
    "less",
    "less-equal",
    "always",
    "greater",
    "greater-equal",
    "equal",
    "not-equal",
    "never",
  ];
  let compareIndex = 0; // 默认 "less"

  const cullList: GPUCullMode[] = ["back", "none", "front"];
  let cullIndex = 0; // 默认 "back"

  const depthSettings = {
    depthWriteEnabled: true,
    depthCompare: depthCompareList[compareIndex],
    cullMode: cullList[cullIndex],
    nearFirst: true,
  };

  // 3. Pipeline 缓存池（统一使用 pipelineLayout）
  const pipelineCache = new Map<string, GPURenderPipeline>();

  function getPipeline(): GPURenderPipeline {
    const key = `${depthSettings.depthWriteEnabled}_${depthSettings.depthCompare}_${depthSettings.cullMode}`;
    let p = pipelineCache.get(key);
    if (!p) {
      p = device.createRenderPipeline({
        layout: pipelineLayout, // 必须显式指定布局，保证与 bindGroup 兼容
        vertex: {
          module,
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
        fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
        primitive: { topology: "triangle-list", cullMode: depthSettings.cullMode },
        depthStencil: {
          depthWriteEnabled: depthSettings.depthWriteEnabled,
          depthCompare: depthSettings.depthCompare,
          format: "depth24plus",
        },
      });
      pipelineCache.set(key, p);
    }
    return p;
  }
// 4. GUI 控制器
  // 原生 HTMLButtonElement 通过修改 textContent 更新按钮显示文本
  gui.addButton(`深度写入: ${depthSettings.depthWriteEnabled ? "开启(ON)" : "关闭(OFF)"}`, function (this: HTMLButtonElement) {
    depthSettings.depthWriteEnabled = !depthSettings.depthWriteEnabled;
    this.textContent = `深度写入: ${depthSettings.depthWriteEnabled ? "开启(ON)" : "关闭(OFF)"}`;
  });

  gui.addButton(`深度比较: [${depthSettings.depthCompare}]`, function (this: HTMLButtonElement) {
    compareIndex = (compareIndex + 1) % depthCompareList.length;
    depthSettings.depthCompare = depthCompareList[compareIndex];
    this.textContent = `深度比较: [${depthSettings.depthCompare}]`;
  });

  gui.addButton(`面剔除: [${depthSettings.cullMode}]`, function (this: HTMLButtonElement) {
    cullIndex = (cullIndex + 1) % cullList.length;
    depthSettings.cullMode = cullList[cullIndex];
    this.textContent = `面剔除: [${depthSettings.cullMode}]`;
  });

  gui.addButton(`绘制次序: ${depthSettings.nearFirst ? "先近后远" : "先远后近"}`, function (this: HTMLButtonElement) {
    depthSettings.nearFirst = !depthSettings.nearFirst;
    this.textContent = `绘制次序: ${depthSettings.nearFirst ? "先近后远" : "先远后近"}`;
  });

  gui.addTextInfo(
    "<b>测试技巧：</b><br>" +
    "1. 将深度比较切为 <code>[always]</code> 或关闭深度写入。<br>" +
    "2. 将绘制次序保持在 <code>先近后远</code>，会看到后绘制的远方立方体错误地遮挡了近处立方体。<br>" +
    "3. 调回 <code>[less]</code> 并开启写入即可恢复正常的纵深遮挡关系。"
  );

  // 5. 相机控制
  const defaultCamera = { distance: 5.5, theta: 40, phi: 20, panX: 0.0, panY: 0.0, fov: 55 };
  const camera = { ...defaultCamera };

  gui.add(camera, "distance", 1.5, 15.0, 0.1).name("相机距离");
  gui.add(camera, "fov", 30, 100, 1).name("视场 FOV");
  gui.addButton("重置相机", () => {
    Object.assign(camera, defaultCamera);
    gui.updateDisplay();
  });

  let isDragging = false, dragButton = 0, lastX = 0, lastY = 0;
  const onPointerDown = (e: PointerEvent) => {
    isDragging = true;
    dragButton = e.shiftKey ? 2 : e.button;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    if (dragButton === 0) {
      camera.theta -= dx * 0.5;
      camera.phi = Math.max(-85, Math.min(85, camera.phi + dy * 0.5));
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
    camera.distance = Math.max(1.5, Math.min(20.0, camera.distance + e.deltaY * 0.005));
    gui.updateDisplay();
  };
  const onContextMenu = (e: MouseEvent) => e.preventDefault();

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("contextmenu", onContextMenu);

  // 6. 渲染循环
  let animId: number;
  function frame() {
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
    const vpMatrix = Mat4.multiply(projMatrix, viewMatrix);

    // 近处立方体 (稍向左前)
    const modelNear = createTranslationMatrix(-0.5, -0.2, 0.6);
    const mvpNear = Mat4.multiply(vpMatrix, modelNear);
    device.queue.writeBuffer(uniformBufferNear, 0, mvpNear.buffer as ArrayBuffer);

    // 远处立方体 (稍向右后)
    const modelFar = createTranslationMatrix(0.5, 0.2, -0.6);
    const mvpFar = Mat4.multiply(vpMatrix, modelFar);
    device.queue.writeBuffer(uniformBufferFar, 0, mvpFar.buffer as ArrayBuffer);

    const currentPipeline = getPipeline();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.1, g: 0.1, b: 0.14, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    pass.setPipeline(currentPipeline);
    pass.setVertexBuffer(0, vBuffer);
    pass.setIndexBuffer(iBuffer, "uint16");

    if (depthSettings.nearFirst) {
      // 先近后远
      pass.setBindGroup(0, bindGroupNear);
      pass.drawIndexed(36);
      pass.setBindGroup(0, bindGroupFar);
      pass.drawIndexed(36);
    } else {
      // 先远后近
      pass.setBindGroup(0, bindGroupFar);
      pass.drawIndexed(36);
      pass.setBindGroup(0, bindGroupNear);
      pass.drawIndexed(36);
    }

    pass.end();
    device.queue.submit([encoder.finish()]);

    animId = requestAnimationFrame(frame);
  }

  frame();

  // 7. 销毁与解绑
  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);
    canvas.removeEventListener("contextmenu", onContextMenu);
    vBuffer.destroy();
    iBuffer.destroy();
    uniformBufferNear.destroy();
    uniformBufferFar.destroy();
    depthTexture.destroy();
  };
}