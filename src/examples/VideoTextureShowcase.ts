import type { SimpleGUI } from "../utils/gui";

// ==========================================
// 1. 基础 3D 矩阵数学 (免第三方依赖)
// ==========================================
namespace Math3D {
  export function lookAt(eye: number[], center: number[], up: number[]): Float32Array {
    const z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
    const lenZ = 1 / (Math.hypot(z0, z1, z2) || 1);
    const zx = z0 * lenZ, zy = z1 * lenZ, zz = z2 * lenZ;

    const x0 = up[1] * zz - up[2] * zy, x1 = up[2] * zx - up[0] * zz, x2 = up[0] * zy - up[1] * zx;
    const lenX = 1 / (Math.hypot(x0, x1, x2) || 1);
    const xx = x0 * lenX, xy = x1 * lenX, xz = x2 * lenX;

    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;

    const out = new Float32Array(16);
    out[0] = xx;  out[1] = yx;  out[2] = zx;  out[3] = 0;
    out[4] = xy;  out[5] = yy;  out[6] = zy;  out[7] = 0;
    out[8] = xz;  out[9] = yz;  out[10] = zz; out[11] = 0;
    out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    out[15] = 1;
    return out;
  }

  export function perspective(fovRad: number, aspect: number, near: number, far: number): Float32Array {
    const f = 1.0 / Math.tan(fovRad / 2.0);
    const nf = 1 / (near - far);
    const out = new Float32Array(16);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = far * nf;
    out[11] = -1.0;
    out[14] = near * far * nf;
    return out;
  }

  export function ortho(left: number, right: number, bottom: number, top: number, near: number, far: number): Float32Array {
    const lr = 1 / (left - right);
    const bt = 1 / (bottom - top);
    const nf = 1 / (near - far);
    const out = new Float32Array(16);
    out[0] = -2 * lr;
    out[5] = -2 * bt;
    out[10] = nf;
    out[12] = (left + right) * lr;
    out[13] = (top + bottom) * bt;
    out[14] = near * nf;
    out[15] = 1;
    return out;
  }

  export function multiply(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      const ai0 = a[i], ai1 = a[i + 4], ai2 = a[i + 8], ai3 = a[i + 12];
      out[i]      = ai0 * b[0]  + ai1 * b[1]  + ai2 * b[2]  + ai3 * b[3];
      out[i + 4]  = ai0 * b[4]  + ai1 * b[5]  + ai2 * b[6]  + ai3 * b[7];
      out[i + 8]  = ai0 * b[8]  + ai1 * b[9]  + ai2 * b[10] + ai3 * b[11];
      out[i + 12] = ai0 * b[12] + ai1 * b[13] + ai2 * b[14] + ai3 * b[15];
    }
    return out;
  }
}



// ============================================================================
// 功能 2：3D 视频材质 (彻底修复 readyState 抛错，支持多视角观察)
// ============================================================================
export function runVideoTextureShowcase(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // 1. 初始化视频元素并妥善规避浏览器自动播放限制
  const video = document.createElement("video");
  video.loop = true;
  video.muted = true;
  video.playsInline = true;
  video.crossOrigin = "anonymous";
  video.src = "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.webm";
  // 捕获播放 Promise，杜绝未处理异常
  video.play().catch(() => {});

  // 2. 构建带 UV 的 3D 立方体展示台 (Pos:3, UV:2, Normal:3)
  const vertices: number[] = [];
  function pushFace(
    p1: number[], p2: number[], p3: number[], p4: number[],
    n: number[]
  ) {
    vertices.push(
      ...p1, 0, 1, ...n,
      ...p2, 1, 1, ...n,
      ...p3, 1, 0, ...n,
      ...p1, 0, 1, ...n,
      ...p3, 1, 0, ...n,
      ...p4, 0, 0, ...n
    );
  }

  const s = 2.5;
  pushFace([-s,-s, s], [ s,-s, s], [ s, s, s], [-s, s, s], [ 0,  0,  1]); // 前
  pushFace([ s,-s,-s], [-s,-s,-s], [-s, s,-s], [ s, s,-s], [ 0,  0, -1]); // 后
  pushFace([-s, s, s], [ s, s, s], [ s, s,-s], [-s, s,-s], [ 0,  1,  0]); // 上
  pushFace([-s,-s,-s], [ s,-s,-s], [ s,-s, s], [-s,-s, s], [ 0, -1,  0]); // 下
  pushFace([-s,-s,-s], [-s,-s, s], [-s, s, s], [-s, s,-s], [-1,  0,  0]); // 左
  pushFace([ s,-s, s], [ s,-s,-s], [ s, s,-s], [ s, s, s], [ 1,  0,  0]); // 右

  const vBuffer = device.createBuffer({
    size: vertices.length * 4,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vBuffer, 0, new Float32Array(vertices));
  const vertexCount = vertices.length / 8;

  // 3. 着色器: 使用 texture_external 和 textureSampleBaseClampToEdge
  const videoShaderWGSL = `
    struct Uniforms {
      viewProj: mat4x4f,
      camPos: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var s: sampler;
    @group(0) @binding(2) var videoTex: texture_external;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
      @location(1) normal: vec3f,
      @location(2) worldPos: vec3f,
    };

    @vertex
    fn vs_main(@location(0) pos: vec3f, @location(1) uv: vec2f, @location(2) normal: vec3f) -> VertexOut {
      var out: VertexOut;
      out.pos = u.viewProj * vec4f(pos, 1.0);
      out.uv = uv;
      out.normal = normal;
      out.worldPos = pos;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4f {
      // 视频纹理标准采样方式
      let videoCol = textureSampleBaseClampToEdge(videoTex, s, in.uv);
      
      // 随相机视角计算光照高光
      let N = normalize(in.normal);
      let V = normalize(u.camPos.xyz - in.worldPos);
      let L = normalize(vec3f(1.0, 2.0, 1.5));
      let diff = max(dot(N, L), 0.2);

      let H = normalize(L + V);
      let spec = pow(max(dot(N, H), 0.0), 32.0) * 0.25;

      let col = videoCol.rgb * diff + spec;
      return vec4f(col, 1.0);
    }
  `;

  const videoModule = device.createShaderModule({ code: videoShaderWGSL });

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: videoModule,
      entryPoint: "vs_main",
      buffers: [{
        arrayStride: 8 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x2" },
          { shaderLocation: 2, offset: 20, format: "float32x3" },
        ],
      }],
    },
    fragment: {
      module: videoModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const uniformBuffer = device.createBuffer({
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  const state = {
    autoRotateCamera: 1,
    rotationSpeed: 0.8,
  };

  gui.add(state, "autoRotateCamera", 0, 1, 1).name("相机自动环绕");
  gui.add(state, "rotationSpeed", 0.1, 3.0, 0.1).name("旋转速度");
  gui.addButton("播放/暂停视频", () => {
    if (video.paused) video.play();
    else video.pause();
  });

  const camera = { distance: 8.5, phi: 20, theta: 0 };
  let isDragging = false, lastX = 0, lastY = 0;

  const onPointerDown = (e: PointerEvent) => {
    isDragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.5;
    camera.phi = Math.max(-85, Math.min(85, camera.phi + (e.clientY - lastY) * 0.5));
    lastX = e.clientX; lastY = e.clientY;
  };
  const onPointerUp = (e: PointerEvent) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  };
  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    camera.distance = Math.max(4.0, Math.min(20.0, camera.distance + e.deltaY * 0.01));
  };

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  let animId: number;
  let depthTexture: GPUTexture | null = null;
  const uniformData = new Float32Array(20);

  function frame() {
    // 关键防御：如果视频尚未加载到第一帧数据，跳过当帧导入以防 DOMException 抛出
    if (video.readyState < 2) {
      animId = requestAnimationFrame(frame);
      return;
    }

    if (!depthTexture || depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    if (state.autoRotateCamera) {
      camera.theta += state.rotationSpeed * 0.5;
    }

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.distance * Math.cos(radPhi) * Math.sin(radTheta),
      camera.distance * Math.sin(radPhi),
      camera.distance * Math.cos(radPhi) * Math.cos(radTheta),
    ];

    const camView = Math3D.lookAt(eye, [0, 0, 0], [0, 1, 0]);
    const camProj = Math3D.perspective((45 * Math.PI) / 180, (canvas.width || 800) / (canvas.height || 600), 0.1, 50.0);
    const camViewProj = Math3D.multiply(camProj, camView);

    uniformData.set(camViewProj, 0);
    uniformData.set([eye[0], eye[1], eye[2], 1.0], 16);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // 每一帧安全导入视频纹理
    const externalTexture = device.importExternalTexture({ source: video });

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: externalTexture },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.05, g: 0.06, b: 0.08, a: 1.0 },
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
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vBuffer);
    pass.draw(vertexCount);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);
    video.pause();
    video.src = "";
    video.load();
    vBuffer.destroy();
    uniformBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}