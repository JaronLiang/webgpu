// src/examples/ssr.ts
import { Mat4 } from "../utils/math";
import type { SimpleGUI } from "../utils/gui";

// 4x4 矩阵求逆实现 (完全解耦，不依赖外部库)
function mat4Invert(m: Float32Array): Float32Array {
  const out = new Float32Array(16);
  const m00 = m[0],  m01 = m[1],  m02 = m[2],  m03 = m[3];
  const m10 = m[4],  m11 = m[5],  m12 = m[6],  m13 = m[7];
  const m20 = m[8],  m21 = m[9],  m22 = m[10], m23 = m[11];
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
  if (!det) { return out; }
  const invDet = 1.0 / det;

  out[0] = (m11 * b11 - m12 * b10 + m13 * b09) * invDet;
  out[1] = (-m01 * b11 + m02 * b10 - m03 * b09) * invDet;
  out[2] = (m31 * b05 - m32 * b04 + m33 * b03) * invDet;
  out[3] = (-m21 * b05 + m22 * b04 - m23 * b03) * invDet;
  out[4] = (-m10 * b11 + m12 * b08 - m13 * b07) * invDet;
  out[5] = (m00 * b11 - m02 * b08 + m03 * b07) * invDet;
  out[6] = (-m30 * b05 + m32 * b02 - m33 * b01) * invDet;
  out[7] = (m20 * b05 - m22 * b02 + m23 * b01) * invDet;
  out[8] = (m10 * b10 - m11 * b08 + m13 * b06) * invDet;
  out[9] = (-m00 * b10 + m01 * b08 - m03 * b06) * invDet;
  out[10] = (m30 * b04 - m31 * b02 + m33 * b00) * invDet;
  out[11] = (-m20 * b04 + m21 * b02 - m23 * b00) * invDet;
  out[12] = (-m10 * b09 + m11 * b07 - m12 * b06) * invDet;
  out[13] = (m00 * b09 - m01 * b07 + m02 * b06) * invDet;
  out[14] = (-m30 * b03 + m31 * b01 - m32 * b00) * invDet;
  out[15] = (m20 * b03 - m21 * b01 + m22 * b00) * invDet;

  return out;
}

// 视图矩阵
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

export function runSSR(
  device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat, canvas: HTMLCanvasElement, gui: SimpleGUI
) {
  // 1. 构造场景几何：光滑地板 (6顶点) + 悬空发光立方体 (36顶点)
  const vertices = new Float32Array([
    // 地面 (Y=0, 6个顶点)
    -6, 0, -6,  0, 1, 0,    6, 0, -6,  0, 1, 0,    6, 0,  6,  0, 1, 0,
    -6, 0, -6,  0, 1, 0,    6, 0,  6,  0, 1, 0,   -6, 0,  6,  0, 1, 0,

    // 立方体 (中心在 Y=2，边长 1.5, 36个顶点)
    // 前面 (Z = 0.75)
    -0.75, 1.25,  0.75,  0,0,1,   0.75, 1.25,  0.75,  0,0,1,   0.75, 2.75,  0.75,  0,0,1,
    -0.75, 1.25,  0.75,  0,0,1,   0.75, 2.75,  0.75,  0,0,1,  -0.75, 2.75,  0.75,  0,0,1,
    // 后面 (Z = -0.75)
     0.75, 1.25, -0.75,  0,0,-1, -0.75, 1.25, -0.75,  0,0,-1, -0.75, 2.75, -0.75,  0,0,-1,
     0.75, 1.25, -0.75,  0,0,-1, -0.75, 2.75, -0.75,  0,0,-1,  0.75, 2.75, -0.75,  0,0,-1,
    // 顶面 (Y = 2.75)
    -0.75, 2.75,  0.75,  0,1,0,   0.75, 2.75,  0.75,  0,1,0,   0.75, 2.75, -0.75,  0,1,0,
    -0.75, 2.75,  0.75,  0,1,0,   0.75, 2.75, -0.75,  0,1,0,  -0.75, 2.75, -0.75,  0,1,0,
    // 底面 (Y = 1.25)
    -0.75, 1.25, -0.75,  0,-1,0,  0.75, 1.25, -0.75,  0,-1,0,  0.75, 1.25,  0.75,  0,-1,0,
    -0.75, 1.25, -0.75,  0,-1,0,  0.75, 1.25,  0.75,  0,-1,0, -0.75, 1.25,  0.75,  0,-1,0,
    // 右面 (X = 0.75)
     0.75, 1.25,  0.75,  1,0,0,   0.75, 1.25, -0.75,  1,0,0,   0.75, 2.75, -0.75,  1,0,0,
     0.75, 1.25,  0.75,  1,0,0,   0.75, 2.75, -0.75,  1,0,0,   0.75, 2.75,  0.75,  1,0,0,
    // 左面 (X = -0.75)
    -0.75, 1.25, -0.75, -1,0,0,  -0.75, 1.25,  0.75, -1,0,0,  -0.75, 2.75,  0.75, -1,0,0,
    -0.75, 1.25, -0.75, -1,0,0,  -0.75, 2.75,  0.75, -1,0,0,  -0.75, 2.75, -0.75, -1,0,0,
  ]);
  const totalVertexCount = 42;

  const vBuffer = device.createBuffer({ size: vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vBuffer, 0, vertices);

  // 全屏 Quad
  const quadData = new Float32Array([-1,-1, 1,-1, -1,1, 1,-1, 1,1, -1,1]);
  const quadBuffer = device.createBuffer({ size: quadData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(quadBuffer, 0, quadData);

  // Uniform: view(64) + proj(64) + invProj(64) + params(16) = 208 字节 -> 分配 256
  const uniformBuffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // 2. G-Buffer 纹理
  const texWidth = canvas.width || 800;
  const texHeight = canvas.height || 600;
  let normalTex = device.createTexture({ size: [texWidth, texHeight], usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING, format: "rgba16float" });
  let colorTex  = device.createTexture({ size: [texWidth, texHeight], usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING, format: format });
  let depthTex  = device.createTexture({ size: [texWidth, texHeight], usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING, format: "depth24plus" });

  const nearestSampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });

  // 3. Pass 1: G-Buffer 渲染
  const gbufferShader = `
    struct Uniforms { 
      view: mat4x4f,
      proj: mat4x4f,
      invProj: mat4x4f,
      params: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    
    struct VOut { 
      @builtin(position) pos: vec4f, 
      @location(0) normalVS: vec3f, 
      @location(1) wPos: vec3f 
    };
    
    @vertex fn vs_main(@location(0) pos: vec3f, @location(1) normal: vec3f) -> VOut {
      var out: VOut; 
      out.pos = u.proj * u.view * vec4f(pos, 1.0); 
      out.normalVS = (u.view * vec4f(normal, 0.0)).xyz; 
      out.wPos = pos; 
      return out;
    }
    
    struct GBufferOut { 
      @location(0) color: vec4f, 
      @location(1) normalVS: vec4f 
    };
    
    @fragment fn fs_main(in: VOut) -> GBufferOut {
      var out: GBufferOut;
      out.normalVS = vec4f(normalize(in.normalVS), 1.0);
      
      if (in.wPos.y < 0.1) {
        // 地板：深色镜面反射材质 + 极细网格
        let grid = step(0.96, fract(in.wPos.x)) + step(0.96, fract(in.wPos.z));
        let col = mix(vec3f(0.08, 0.09, 0.12), vec3f(0.18, 0.2, 0.23), grid);
        out.color = vec4f(col, 1.0);
      } else {
        // 立方体：金橙色明亮立体
        let lightDir = normalize(vec3f(0.5, 1.0, 0.8));
        let diff = max(dot(normalize(in.normalVS), (u.view * vec4f(lightDir, 0.0)).xyz), 0.25);
        out.color = vec4f(vec3f(1.0, 0.6, 0.1) * (diff * 0.75 + 0.25), 1.0);
      }
      return out;
    }
  `;
  const gbufferPipeline = device.createRenderPipeline({
    layout: "auto", 
    vertex: { 
      module: device.createShaderModule({ code: gbufferShader }), 
      entryPoint: "vs_main", 
      buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x3" }] }] 
    },
    fragment: { 
      module: device.createShaderModule({ code: gbufferShader }), 
      entryPoint: "fs_main", 
      targets: [{ format }, { format: "rgba16float" }] 
    },
    primitive: { topology: "triangle-list" }, 
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // 4. Pass 2: SSR 视图空间光线步进着色器
  const ssrShader = `
    struct Uniforms { 
      view: mat4x4f,
      proj: mat4x4f,
      invProj: mat4x4f,
      params: vec4f, // x: stepSize, y: maxSteps, z: intensity, w: thickness
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var samp: sampler;
    @group(0) @binding(2) var colorTex: texture_2d<f32>;
    @group(0) @binding(3) var normalTex: texture_2d<f32>;
    @group(0) @binding(4) var depthTex: texture_depth_2d;

    @vertex fn vs_main(@location(0) pos: vec2f) -> @builtin(position) vec4f { 
      return vec4f(pos, 0.0, 1.0); 
    }

    fn reconstructViewPos(uv: vec2f, depth: f32) -> vec3f {
      let clip = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
      let viewPos = u.invProj * clip;
      return viewPos.xyz / viewPos.w;
    }

    fn projectToUV(posVS: vec3f) -> vec3f {
      let clip = u.proj * vec4f(posVS, 1.0);
      let ndc = clip.xyz / clip.w;
      return vec3f(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5), ndc.z);
    }

    @fragment fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      let dims = vec2f(textureDimensions(colorTex));
      let uv = fragCoord.xy / dims;

      let baseColor = textureSampleLevel(colorTex, samp, uv, 0.0);
      let depth = textureSampleLevel(depthTex, samp, uv, 0);

      // 天空背景跳过
      if (depth >= 1.0) {
        return baseColor;
      }

      let normalVS = normalize(textureSampleLevel(normalTex, samp, uv, 0.0).xyz);
      let startVS = reconstructViewPos(uv, depth);
      let viewDirVS = normalize(startVS);

      // 地板法线判断：视图空间中只有表面面向视线的水平面才进行倒影计算
      let reflDirVS = normalize(reflect(viewDirVS, normalVS));

      let stepSize = u.params.x;
      let maxSteps = i32(u.params.y);
      let intensity = u.params.z;
      let thickness = u.params.w;

      var currentPosVS = startVS + reflDirVS * 0.1;
      var hitColor = vec3f(0.0);
      var hitWeight = 0.0;

      for (var i = 0; i < maxSteps; i++) {
        currentPosVS += reflDirVS * stepSize;
        if (currentPosVS.z > 0.0) { break; } // 超出近剪裁面

        let sampleUV = projectToUV(currentPosVS);

        // 超出屏幕视口边界
        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) {
          break;
        }

        let sceneDepth = textureSampleLevel(depthTex, samp, sampleUV.xy, 0);
        if (sceneDepth >= 1.0) { continue; }

        let scenePosVS = reconstructViewPos(sampleUV.xy, sceneDepth);

        // 深度厚度相交判定 (解决倒影拉长成柱子的核心)
        let depthDelta = currentPosVS.z - scenePosVS.z;
        if (depthDelta <= 0.0 && depthDelta > -thickness) {
          let sampledCol = textureSampleLevel(colorTex, samp, sampleUV.xy, 0.0).rgb;
          let edgeFade = min(1.0, 10.0 * min(min(sampleUV.x, 1.0 - sampleUV.x), min(sampleUV.y, 1.0 - sampleUV.y)));
          hitColor = sampledCol;
          hitWeight = edgeFade;
          break;
        }
      }

      // 菲涅尔渐变：增加镜面玻璃质感
      let fresnel = 0.2 + 0.8 * pow(1.0 - max(dot(normalVS, -viewDirVS), 0.0), 4.0);
      let finalColor = mix(baseColor.rgb, hitColor, hitWeight * intensity * fresnel);
      return vec4f(finalColor, 1.0);
    }
  `;

  const ssrPipeline = device.createRenderPipeline({
    layout: "auto", 
    vertex: { 
      module: device.createShaderModule({ code: ssrShader }), 
      entryPoint: "vs_main", 
      buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] 
    },
    fragment: { 
      module: device.createShaderModule({ code: ssrShader }), 
      entryPoint: "fs_main", 
      targets: [{ format }] 
    },
    primitive: { topology: "triangle-list" }
  });

  // 5. GUI 与控制
  const camera = { distance: 7.5, theta: 35, phi: 24, panX: 0.0, panY: 1.2, fov: 50 };
  const ssrSettings = { stepSize: 0.12, maxSteps: 60, intensity: 0.85, thickness: 0.6 };

  gui.addTextInfo("<b>屏幕空间镜面反射 (SSR)</b><br>基于 View-Space Raymarching");
  gui.add(ssrSettings, "intensity", 0.0, 1.0, 0.05).name("镜面反射度");
  gui.add(ssrSettings, "stepSize", 0.02, 0.3, 0.01).name("步长精度");
  gui.add(ssrSettings, "maxSteps", 20, 120, 1).name("步进次数");
  gui.add(ssrSettings, "thickness", 0.1, 2.0, 0.05).name("厚度容差");
  gui.add(camera, "theta", -180, 180, 1).name("偏航角");
  gui.add(camera, "phi", 5, 85, 1).name("俯仰角");

  let isDragging = false;
  let lastX = 0, lastY = 0;
  const onPointerDown = (e: PointerEvent) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.5;
    camera.phi = Math.max(5, Math.min(85, camera.phi + (e.clientY - lastY) * 0.5));
    lastX = e.clientX; lastY = e.clientY;
    gui.updateDisplay();
  };
  const onPointerUp = (e: PointerEvent) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} };
  const onWheel = (e: WheelEvent) => { e.preventDefault(); camera.distance = Math.max(3.0, camera.distance + e.deltaY * 0.01); gui.updateDisplay(); };

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  // 6. 渲染循环
  let animId: number;
  function frame() {
    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eyeX = camera.panX + camera.distance * Math.cos(radPhi) * Math.sin(radTheta);
    const eyeY = camera.panY + camera.distance * Math.sin(radPhi);
    const eyeZ = camera.distance * Math.cos(radPhi) * Math.cos(radTheta);

    const aspect = (canvas.width || 800) / (canvas.height || 600);
    const viewMatrix = createLookAtMatrix([eyeX, eyeY, eyeZ], [camera.panX, camera.panY, 0], [0, 1, 0]);
    const projMatrix = Mat4.perspective((camera.fov * Math.PI) / 180, aspect, 0.1, 100);
    // 使用内置的矩阵求逆，避免调用未定义的 Mat4.invert
    const invProjMatrix = mat4Invert(projMatrix);

    // 写入统一缓冲
    device.queue.writeBuffer(uniformBuffer, 0, viewMatrix.buffer as ArrayBuffer);
    device.queue.writeBuffer(uniformBuffer, 64, projMatrix.buffer as ArrayBuffer);
    device.queue.writeBuffer(uniformBuffer, 128, invProjMatrix.buffer as ArrayBuffer);
    device.queue.writeBuffer(uniformBuffer, 192, new Float32Array([
      ssrSettings.stepSize,
      ssrSettings.maxSteps,
      ssrSettings.intensity,
      ssrSettings.thickness
    ]));

    const encoder = device.createCommandEncoder();

    // Pass 1: G-Buffer
    const gbufferPass = encoder.beginRenderPass({
      colorAttachments: [
        { view: colorTex.createView(), clearValue: { r: 0.02, g: 0.02, b: 0.03, a: 1.0 }, loadOp: "clear", storeOp: "store" },
        { view: normalTex.createView(), clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }, loadOp: "clear", storeOp: "store" }
      ],
      depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    gbufferPass.setPipeline(gbufferPipeline);
    gbufferPass.setBindGroup(0, device.createBindGroup({
      layout: gbufferPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
    }));
    gbufferPass.setVertexBuffer(0, vBuffer);
    gbufferPass.draw(totalVertexCount);
    gbufferPass.end();

    // Pass 2: SSR 合成
    const ssrBindGroup = device.createBindGroup({
      layout: ssrPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: nearestSampler },
        { binding: 2, resource: colorTex.createView() },
        { binding: 3, resource: normalTex.createView() },
        { binding: 4, resource: depthTex.createView() }
      ]
    });

    const ssrPass = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }],
    });
    ssrPass.setPipeline(ssrPipeline);
    ssrPass.setBindGroup(0, ssrBindGroup);
    ssrPass.setVertexBuffer(0, quadBuffer);
    ssrPass.draw(6);
    ssrPass.end();

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
    vBuffer.destroy(); quadBuffer.destroy(); uniformBuffer.destroy();
    normalTex.destroy(); colorTex.destroy(); depthTex.destroy();
  };
}