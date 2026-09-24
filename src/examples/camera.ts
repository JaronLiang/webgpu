// src/examples/camera.ts
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

export function runCamera(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI // 注入 GUI
) {
  // 1. 六色立方体网格
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

  const uniformBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let depthTexture = device.createTexture({
    size: [canvas.width || 800, canvas.height || 600], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const shaderCode = `
    struct Uniforms { mvp: mat4x4f };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    struct VertexOut { @builtin(position) pos: vec4f, @location(0) col: vec3f };
    @vertex fn vs_main(@location(0) pos: vec3f, @location(1) col: vec3f) -> VertexOut {
      var out: VertexOut; out.pos = u.mvp * vec4f(pos, 1.0); out.col = col; return out;
    }
    @fragment fn fs_main(@location(0) col: vec3f) -> @location(0) vec4f { return vec4f(col, 1.0); }
  `;
  const module = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module, entryPoint: "vs_main",
      buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x3" }] }],
    },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // 2. 相机参数
  const defaultCamera = { distance: 4.5, theta: 45, phi: 25, panX: 0.0, panY: 0.0, fov: 60 };
  const camera = { ...defaultCamera };

  // ==========================================
  // 3. 【极简接入】：直接注册相机系数到 GUI
  // ==========================================
  gui.add(camera, "distance", 1.5, 15.0, 0.1).name("缩放距离 (Dist)");
  gui.add(camera, "phi", -85, 85, 1).name("俯仰角 (Pitch)");
  gui.add(camera, "theta", -180, 180, 1).name("偏航角 (Yaw)");
  gui.add(camera, "panX", -3.0, 3.0, 0.1).name("平移 X");
  gui.add(camera, "panY", -3.0, 3.0, 0.1).name("平移 Y");
  gui.add(camera, "fov", 30, 100, 1).name("视场 (FOV)");
  gui.addButton("重置相机 (Reset)", () => {
    Object.assign(camera, defaultCamera);
    gui.updateDisplay();
  });
  gui.addTextInfo("• 左键拖动: 旋转视角<br>• 右键/Shift+左键: 平移<br>• 滚轮: 缩放远近");

  // 4. 鼠标拖拽事件监听
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
      camera.phi = Math.max(-85, Math.min(85, camera.phi + dy * 0.5));
    } else if (dragButton === 2) {
      const factor = camera.distance * 0.0018;
      camera.panX -= dx * factor;
      camera.panY += dy * factor;
    }
    // 同步滑块位置
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

  // 5. 渲染循环
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
    const mvpMatrix = Mat4.multiply(projMatrix, viewMatrix);

    device.queue.writeBuffer(uniformBuffer, 0, mvpMatrix.buffer as ArrayBuffer);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.1, b: 0.14, a: 1.0 },
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vBuffer);
    pass.setIndexBuffer(iBuffer, "uint16");
    pass.drawIndexed(36);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  frame();

  // 6. 清理：只需解绑自身事件和 GPU Buffer（GUI 清理完全交给 App.vue）
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
  };
}