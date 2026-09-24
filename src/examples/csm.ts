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
// 功能 1：级联阴影 (CSM) 示例 (支持动态光源、地面积木与层级调试可视化)
// ============================================================================
export function runCSMShowcase(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // 1. 构建场景几何体：宽阔地面 + 多层级立方体 (Pos:3, Normal:3, Color:3)
  const vertices: number[] = [];
  function pushQuad(p1: number[], p2: number[], p3: number[], p4: number[], n: number[], col: number[]) {
    vertices.push(
      ...p1, ...n, ...col, ...p2, ...n, ...col, ...p3, ...n, ...col,
      ...p1, ...n, ...col, ...p3, ...n, ...col, ...p4, ...n, ...col
    );
  }
  function addBox(cx: number, cy: number, cz: number, sx: number, sy: number, sz: number, col: number[]) {
    const x0 = cx - sx / 2, x1 = cx + sx / 2;
    const y0 = cy - sy / 2, y1 = cy + sy / 2;
    const z0 = cz - sz / 2, z1 = cz + sz / 2;
    pushQuad([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], [ 0,  1,  0], col);
    pushQuad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], [ 0, -1,  0], col);
    pushQuad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [ 0,  0,  1], col);
    pushQuad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [ 0,  0, -1], col);
    pushQuad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [-1,  0,  0], col);
    pushQuad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [ 1,  0,  0], col);
  }

  // 大地板
  addBox(0, -0.2, 0, 80, 0.4, 80, [0.85, 0.85, 0.88]);
  // 中心主立方体
  addBox(0.0, 1.5, 0.0, 3.0, 3.0, 3.0, [0.95, 0.35, 0.2]);
  // 环绕不同距离的立方体柱子（体现近、中、远 CSM 分层）
  addBox(-5.0, 2.0, -3.0, 1.2, 4.0, 1.2, [0.2, 0.7, 0.9]);
  addBox(6.0, 1.5, 4.0, 1.5, 3.0, 1.5, [0.3, 0.85, 0.4]);
  addBox(-12.0, 3.0, 8.0, 2.0, 6.0, 2.0, [0.9, 0.75, 0.2]);
  addBox(15.0, 2.5, -12.0, 2.5, 5.0, 2.5, [0.7, 0.3, 0.85]);
  addBox(0.0, 4.0, -18.0, 3.0, 8.0, 3.0, [0.2, 0.8, 0.7]);

  const vBuffer = device.createBuffer({
    size: vertices.length * 4,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vBuffer, 0, new Float32Array(vertices));
  const vertexCount = vertices.length / 9;

  // 2. CSM 2D Texture Array (3 层深度贴图)
  const cascadeCount = 3;
  const shadowMapSize = 2048;
  const csmDepthTexture = device.createTexture({
    size: [shadowMapSize, shadowMapSize, cascadeCount],
    format: "depth32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  const shadowSampler = device.createSampler({
    compare: "less",
    magFilter: "linear",
    minFilter: "linear",
  });

  // 3. Uniform Buffers
  // 布局: mat4x4(64B) * 3(Light) + mat4x4(64B)(Cam) + vec4(16B)(CamPos) + vec4(16B)(Splits) + vec4(16B)(Params) = 304 Bytes
  const uniformBuffer = device.createBuffer({
    size: 320,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // 单独给 Shadow Pass 用的 3 个矩阵 Buffer，避免数组动态寻址兼容问题
  const shadowUniformBuffers = [
    device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
  ];

  // 4. 着色器实现
  const csmShaderWGSL = `
    struct ShadowData {
      lightViewProj: mat4x4f,
    };
    @group(0) @binding(0) var<uniform> uShadow: ShadowData;

    @vertex
    fn vs_shadow(@location(0) pos: vec3f) -> @builtin(position) vec4f {
      return uShadow.lightViewProj * vec4f(pos, 1.0);
    }

    struct MainUniforms {
      lightVP0: mat4x4f,
      lightVP1: mat4x4f,
      lightVP2: mat4x4f,
      camViewProj: mat4x4f,
      camPos: vec4f,
      csmSplits: vec4f,       // x: Split 0, y: Split 1, z: Debug (1:彩色层级, 0:正常)
      lightDir_Intensity: vec4f, // xyz: 光源方向, w: 强度
    };

    @group(0) @binding(0) var<uniform> u: MainUniforms;
    @group(0) @binding(1) var shadowMapArray: texture_depth_2d_array;
    @group(0) @binding(2) var shadowSampler: sampler_comparison;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) color: vec3f,
    };

    @vertex
    fn vs_main(@location(0) pos: vec3f, @location(1) normal: vec3f, @location(2) color: vec3f) -> VertexOut {
      var out: VertexOut;
      out.worldPos = pos;
      out.normal = normal;
      out.color = color;
      out.pos = u.camViewProj * vec4f(pos, 1.0);
      return out;
    }

    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4f {
      let dist = length(u.camPos.xyz - in.worldPos);
      
      // 动态选择级联层级
      var cascadeIdx = 0u;
      var lightVP = u.lightVP0;
      var debugColor = vec3f(1.0, 0.4, 0.4); // 0: 近景 (红)

      if (dist > u.csmSplits.y) {
        cascadeIdx = 2u;
        lightVP = u.lightVP2;
        debugColor = vec3f(0.4, 0.5, 1.0); // 2: 远景 (蓝)
      } else if (dist > u.csmSplits.x) {
        cascadeIdx = 1u;
        lightVP = u.lightVP1;
        debugColor = vec3f(0.4, 1.0, 0.4); // 1: 中景 (绿)
      }

      // 计算对应层的投影坐标
      let lightProj = lightVP * vec4f(in.worldPos, 1.0);
      let proj = lightProj.xyz / lightProj.w;
      let uv = proj.xy * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
      let curDepth = proj.z;

      var shadow = 1.0;
      let bias = 0.0025;
      
      // 在有效范围内执行 PCF 软滤波采样
      if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 && curDepth <= 1.0) {
        var s = 0.0;
        let texel = 1.0 / 2048.0;
        for (var x = -1; x <= 1; x++) {
          for (var y = -1; y <= 1; y++) {
            let offset = vec2f(f32(x), f32(y)) * texel;
            s += textureSampleCompareLevel(shadowMapArray, shadowSampler, uv + offset, cascadeIdx, curDepth - bias);
          }
        }
        shadow = s / 9.0;
      }

      let N = normalize(in.normal);
      let L = normalize(u.lightDir_Intensity.xyz);
      let diff = max(dot(N, L), 0.0);
      let diffuse = diff * in.color * u.lightDir_Intensity.w;
      let ambient = in.color * 0.15;

      var finalColor = ambient + diffuse * shadow;
      if (u.csmSplits.z > 0.5) {
        finalColor *= debugColor; // 叠加层级调试颜色
      }

      return vec4f(pow(finalColor, vec3f(1.0 / 2.2)), 1.0);
    }
  `;

  const csmModule = device.createShaderModule({ code: csmShaderWGSL });

  const vertexLayout: GPUVertexBufferLayout = {
    arrayStride: 9 * 4,
    attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" },
      { shaderLocation: 1, offset: 12, format: "float32x3" },
      { shaderLocation: 2, offset: 24, format: "float32x3" },
    ],
  };

  const shadowPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: csmModule, entryPoint: "vs_shadow", buffers: [vertexLayout] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth32float" },
    primitive: { cullMode: "front" },
  });

  const mainPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: csmModule, entryPoint: "vs_main", buffers: [vertexLayout] },
    fragment: { module: csmModule, entryPoint: "fs_main", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    primitive: { cullMode: "back" },
  });

  const shadowBindGroups = shadowUniformBuffers.map((buf) =>
    device.createBindGroup({
      layout: shadowPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: buf } }],
    })
  );

  const mainBindGroup = device.createBindGroup({
    layout: mainPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: csmDepthTexture.createView({ dimension: "2d-array" }) },
      { binding: 2, resource: shadowSampler },
    ],
  });

  // 5. 状态与控制
  const csmState = {
    autoRotateLight: 1,
    lightHeight: 14.0,
    debugCascades: 1, // 是否可视化红/绿/蓝级联区域
    intensity: 1.2,
  };

  gui.add(csmState, "autoRotateLight", 0, 1, 1).name("光源自动旋转");
  gui.add(csmState, "debugCascades", 0, 1, 1).name("CSM 层级着色调试");
  gui.add(csmState, "lightHeight", 8.0, 25.0, 0.5).name("光源高度");
  gui.add(csmState, "intensity", 0.5, 3.0, 0.1).name("光照强度");

  const camera = { distance: 26.0, phi: 32, theta: 45, panX: 0, panY: 0 };
  let isDragging = false, dragButton = 0, lastX = 0, lastY = 0;

  const onPointerDown = (e: PointerEvent) => {
    isDragging = true; dragButton = e.button; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (dragButton === 0) {
      camera.theta -= dx * 0.4;
      camera.phi = Math.max(-85, Math.min(85, camera.phi + dy * 0.4));
    }
  };
  const onPointerUp = (e: PointerEvent) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  };
  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    camera.distance = Math.max(5.0, Math.min(60.0, camera.distance + e.deltaY * 0.02));
  };
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  // 6. 渲染循环
  let animId: number;
  let depthTexture: GPUTexture | null = null;
  const uniformData = new Float32Array(80);

  function frame(timestamp: number) {
    if (!depthTexture || depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    // 动态光源方向
    const angle = csmState.autoRotateLight ? timestamp * 0.0006 : 0.8;
    const lightDist = 18.0;
    const lightPos = [Math.cos(angle) * lightDist, csmState.lightHeight, Math.sin(angle) * lightDist];
    const lightView = Math3D.lookAt(lightPos, [0, 0, 0], [0, 1, 0]);

    // 级联视锥设计：近层高精细，中层覆盖，远层宽广
    const ortho0 = Math3D.ortho(-8, 8, -8, 8, 1.0, 45.0);
    const ortho1 = Math3D.ortho(-20, 20, -20, 20, 1.0, 55.0);
    const ortho2 = Math3D.ortho(-45, 45, -45, 45, 1.0, 75.0);

    const lvp0 = Math3D.multiply(ortho0, lightView);
    const lvp1 = Math3D.multiply(ortho1, lightView);
    const lvp2 = Math3D.multiply(ortho2, lightView);

    // 上传 Shadow Pass 单独 Buffer
    device.queue.writeBuffer(shadowUniformBuffers[0], 0, lvp0 as any);
    device.queue.writeBuffer(shadowUniformBuffers[1], 0, lvp1 as any);
    device.queue.writeBuffer(shadowUniformBuffers[2], 0, lvp2 as any);

    // 主相机
    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.distance * Math.cos(radPhi) * Math.sin(radTheta),
      camera.distance * Math.sin(radPhi),
      camera.distance * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const camView = Math3D.lookAt(eye, [0, 0, 0], [0, 1, 0]);
    const camProj = Math3D.perspective((50 * Math.PI) / 180, (canvas.width || 800) / (canvas.height || 600), 0.1, 150.0);
    const camViewProj = Math3D.multiply(camProj, camView);

    // 组装主着色器 Uniforms
    uniformData.set(lvp0, 0);
    uniformData.set(lvp1, 16);
    uniformData.set(lvp2, 32);
    uniformData.set(camViewProj, 48);
    uniformData.set([eye[0], eye[1], eye[2], 1.0], 64);
    uniformData.set([15.0, 32.0, csmState.debugCascades, 0.0], 68); // Splits 距离及 Debug
    uniformData.set([lightPos[0], lightPos[1], lightPos[2], csmState.intensity], 72);

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();

    // Pass 1: 渲染 3 层级联深度贴图
    for (let i = 0; i < cascadeCount; i++) {
      const shadowPass = encoder.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: csmDepthTexture.createView({
            dimension: "2d",
            baseArrayLayer: i,
            arrayLayerCount: 1,
          }),
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      shadowPass.setPipeline(shadowPipeline);
      shadowPass.setBindGroup(0, shadowBindGroups[i]);
      shadowPass.setVertexBuffer(0, vBuffer);
      shadowPass.draw(vertexCount);
      shadowPass.end();
    }

    // Pass 2: 主渲染 Pass
    const mainPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.12, b: 0.15, a: 1.0 },
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
    mainPass.setPipeline(mainPipeline);
    mainPass.setBindGroup(0, mainBindGroup);
    mainPass.setVertexBuffer(0, vBuffer);
    mainPass.draw(vertexCount);
    mainPass.end();

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
    vBuffer.destroy();
    uniformBuffer.destroy();
    shadowUniformBuffers.forEach((b) => b.destroy());
    csmDepthTexture.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}

