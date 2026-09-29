// src/examples/gaussian_splatting.ts
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

// 【优化点 1】: 生成结构更清晰的数学模型，降低点数防止拥挤
async function fetchFeedForwardGaussians(type: "human" | "object"): Promise<Float32Array> {
  return new Promise((resolve) => {
    setTimeout(() => {
      const numGaussians = 30000; // 降低点数，让缝隙透气
      const data = new Float32Array(numGaussians * 8);
      
      for (let i = 0; i < numGaussians; i++) {
        const offset = i * 8;
        if (type === "object") {
          // 生成一个清晰的彩色甜甜圈 (Torus)
          const u = Math.random() * Math.PI * 2;
          const v = Math.random() * Math.PI * 2;
          const R = 2.0, r = 0.8;
          data[offset + 0] = (R + r * Math.cos(v)) * Math.cos(u);
          data[offset + 1] = r * Math.sin(v); // Y轴朝上
          data[offset + 2] = (R + r * Math.cos(v)) * Math.sin(u);
          data[offset + 3] = 0.8; // 透明度提高，变成实体
          data[offset + 4] = Math.abs(Math.cos(u));     // R
          data[offset + 5] = Math.abs(Math.sin(v));     // G
          data[offset + 6] = Math.abs(Math.sin(u));     // B
          data[offset + 7] = Math.random() * 0.08 + 0.02; // 缩小缩放系数
        } else {
          // 生成一个清晰的人形 (雪人模型：头 + 身体)
          const part = Math.random();
          if (part < 0.25) {
            // 头部 (球体)
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const radius = 0.7;
            data[offset + 0] = radius * Math.sin(phi) * Math.cos(theta);
            data[offset + 1] = 2.2 + radius * Math.cos(phi); // 头部较高
            data[offset + 2] = radius * Math.sin(phi) * Math.sin(theta);
            data[offset + 3] = 0.9;
            data[offset + 4] = 0.9; data[offset + 5] = 0.7; data[offset + 6] = 0.6; // 肤色
          } else {
            // 身体 (圆台/衣服)
            const h = Math.random() * 3.5 - 1.5; // 从 -1.5 到 2.0
            const angle = Math.random() * Math.PI * 2;
            // 越往下越宽
            const radius = 1.2 - (h * 0.2); 
            data[offset + 0] = Math.cos(angle) * radius;
            data[offset + 1] = h;
            data[offset + 2] = Math.sin(angle) * radius;
            data[offset + 3] = 0.9;
            data[offset + 4] = 0.2; data[offset + 5] = 0.5; data[offset + 6] = 0.8; // 蓝色衣服
          }
          data[offset + 7] = Math.random() * 0.06 + 0.02; // 缩放系数
        }
      }
      resolve(data);
    }, 500);
  });
}

export function runFeedForward3DGS(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  const quadVertices = new Float32Array([
    -1.0, -1.0,   1.0, -1.0,  -1.0,  1.0,
     1.0, -1.0,   1.0,  1.0,  -1.0,  1.0,
  ]);
  const vBuffer = device.createBuffer({
    size: quadVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vBuffer, 0, quadVertices);

  const uniformBuffer = device.createBuffer({
    size: 64 + 16 + 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  let instanceBuffer: GPUBuffer | null = null;
  let numInstances = 0;

  // 【优化点 2】: 调整 Fragment Shader，移除颜色的 Alpha 预乘
  const shaderCode = `
    struct Uniforms {
      mvp: mat4x4f,
      camRight: vec4f,
      camUp: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec3f,
      @location(1) uv: vec2f,
      @location(2) opacity: f32
    };

    @vertex fn vs_main(
      @location(0) quadPos: vec2f,
      @location(1) instPosAlpha: vec4f,
      @location(2) instColScale: vec4f
    ) -> VertexOut {
      var out: VertexOut;
      let center = instPosAlpha.xyz;
      let scale = instColScale.w;
      
      let worldPos = center 
                   + u.camRight.xyz * quadPos.x * scale 
                   + u.camUp.xyz * quadPos.y * scale;

      out.pos = u.mvp * vec4f(worldPos, 1.0);
      out.color = instColScale.xyz;
      out.uv = quadPos;
      out.opacity = instPosAlpha.w;
      return out;
    }

    @fragment fn fs_main(in: VertexOut) -> @location(0) vec4f {
      let d = dot(in.uv, in.uv);
      if (d > 1.0) { discard; }
      
      // 衰减公式，5.0 让高斯球边缘更锐利一点
      let power = exp(-d * 5.0); 
      let finalAlpha = power * in.opacity;
      
      // 直接输出颜色和 Alpha，交由管线的 Blend 状态处理
      return vec4f(in.color, finalAlpha); 
    }
  `;

  const module = device.createShaderModule({ code: shaderCode });

  // 【优化点 3】: 将加法混合改为 Alpha 混合，渲染出坚实的物体表面
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 8, stepMode: "vertex",
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
        },
        {
          arrayStride: 32, stepMode: "instance",
          attributes: [
            { shaderLocation: 1, offset: 0, format: "float32x4" }, 
            { shaderLocation: 2, offset: 16, format: "float32x4" },
          ],
        }
      ],
    },
    fragment: {
      module,
      entryPoint: "fs_main",
      targets: [{
        format,
        // 改为标准的 Alpha Blend：src-alpha + one-minus-src-alpha
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        }
      }],
    },
    primitive: { topology: "triangle-list" },
    // 依然关闭深度写入，避免不透明方块互相切割，实现无排序的伪体积渲染
    depthStencil: { depthWriteEnabled: false, depthCompare: "less", format: "depth24plus" },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  let depthTexture = device.createTexture({
    size: [canvas.width || 800, canvas.height || 600], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const defaultCamera = { distance: 8.0, theta: 45, phi: 15, panX: 0.0, panY: 0.0, fov: 60 };
  const camera = { ...defaultCamera };
  
  const splatState = { loading: false };

  const loadNetworkData = async (type: "human" | "object") => {
    if (splatState.loading) return;
    splatState.loading = true;
    
    const splatData = await fetchFeedForwardGaussians(type);
    
    if (instanceBuffer) instanceBuffer.destroy();
    numInstances = splatData.length / 8;
    instanceBuffer = device.createBuffer({
      size: splatData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(instanceBuffer, 0, splatData as any);
    splatState.loading = false;
  };

  gui.addTextInfo("<b>AI 前馈 3DGS 模拟</b><br>模拟从图像直接输出3D高斯场");
  gui.addButton("🍩 运行网络推理: 物体重建", () => loadNetworkData("object"));
  gui.addButton("🧍 运行网络推理: 人像重建", () => loadNetworkData("human"));
  gui.add(camera, "distance", 2.0, 20.0, 0.1).name("相机距离");
  gui.add(camera, "phi", -85, 85, 1).name("俯仰角");
  gui.add(camera, "theta", -180, 180, 1).name("偏航角");
  
  loadNetworkData("object");

  let isDragging = false; let dragButton = 0; let lastX = 0, lastY = 0;

  const onPointerDown = (e: PointerEvent) => {
    isDragging = true; dragButton = e.shiftKey ? 2 : e.button;
    lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (dragButton === 0) {
      camera.theta -= dx * 0.5; camera.phi = Math.max(-85, Math.min(85, camera.phi + dy * 0.5));
    } else if (dragButton === 2) {
      const factor = camera.distance * 0.0018;
      camera.panX -= dx * factor; camera.panY += dy * factor;
    }
    gui.updateDisplay();
  };
  const onPointerUp = (e: PointerEvent) => {
    isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {}
  };
  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    camera.distance = Math.max(2.0, Math.min(20.0, camera.distance + e.deltaY * 0.01));
    gui.updateDisplay();
  };

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("contextmenu", e => e.preventDefault());

  let animId: number;
  function frame() {
    if (depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT,
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

    const camRight = new Float32Array([viewMatrix[0], viewMatrix[4], viewMatrix[8], 0]);
    const camUp = new Float32Array([viewMatrix[1], viewMatrix[5], viewMatrix[9], 0]);

    device.queue.writeBuffer(uniformBuffer, 0, mvpMatrix.buffer as ArrayBuffer);
    device.queue.writeBuffer(uniformBuffer, 64, camRight);
    device.queue.writeBuffer(uniformBuffer, 80, camUp);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.1, b: 0.12, a: 1.0 }, // 稍微提亮一点背景色
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });

    if (instanceBuffer && numInstances > 0) {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.setVertexBuffer(0, vBuffer);
      pass.setVertexBuffer(1, instanceBuffer); 
      pass.draw(6, numInstances);
    }

    pass.end();
    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  frame();

  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);
    vBuffer.destroy();
    uniformBuffer.destroy();
    if (instanceBuffer) instanceBuffer.destroy();
    depthTexture.destroy();
  };
}