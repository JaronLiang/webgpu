// src/examples/ssao.ts
import type { SimpleGUI } from "../utils/gui";

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

// 生成用于 SSAO 半球采样的核心数组 (Kernel)
function generateSSAOKernel(samples: number): Float32Array {
  const kernel = new Float32Array(samples * 4); // vec4f 对齐
  const lerp = (a: number, b: number, f: number) => a + f * (b - a);
  for (let i = 0; i < samples; i++) {
    let x = Math.random() * 2.0 - 1.0;
    let y = Math.random() * 2.0 - 1.0;
    let z = Math.random(); // Z为正，意味着朝向半球的法线方向
    let len = Math.hypot(x, y, z) || 1.0;
    x /= len; y /= len; z /= len;

    // 让采样点更集中于靠近中心的地方 (而不是全部在边缘)
    let scale = i / samples;
    scale = lerp(0.1, 1.0, scale * scale);
    x *= scale; y *= scale; z *= scale;

    kernel[i * 4 + 0] = x;
    kernel[i * 4 + 1] = y;
    kernel[i * 4 + 2] = z;
    kernel[i * 4 + 3] = 0.0;
  }
  return kernel;
}

export function runSSAO(
  device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat, canvas: HTMLCanvasElement, gui: SimpleGUI
) {
  // 1. 场景几何：底座 + 台阶 + 立柱 (pos:3, normal:3, color:3)
  const vertices = new Float32Array([
    -7,0,-7, 0,1,0, 0.82,0.82,0.85,   7,0,-7, 0,1,0, 0.82,0.82,0.85,   7,0,7, 0,1,0, 0.82,0.82,0.85,
    -7,0,-7, 0,1,0, 0.82,0.82,0.85,   7,0,7, 0,1,0, 0.82,0.82,0.85,  -7,0,7, 0,1,0, 0.82,0.82,0.85,
    -3.5,0,-3.5, 0,1,0, 0.9,0.85,0.75,   3.5,0,-3.5, 0,1,0, 0.9,0.85,0.75,   3.5,0.8,3.5, 0,1,0, 0.9,0.85,0.75,
    -3.5,0,-3.5, 0,1,0, 0.9,0.85,0.75,   3.5,0.8,3.5, 0,1,0, 0.9,0.85,0.75, -3.5,0.8,3.5, 0,1,0, 0.9,0.85,0.75,
    -1.2,0.8,-1.2, 0,0,-1, 0.7,0.8,0.85,  1.2,0.8,-1.2, 0,0,-1, 0.7,0.8,0.85,  1.2,3.5,-1.2, 0,0,-1, 0.7,0.8,0.85,
    -1.2,0.8,-1.2, 0,0,-1, 0.7,0.8,0.85,  1.2,3.5,-1.2, 0,0,-1, 0.7,0.8,0.85, -1.2,3.5,-1.2, 0,0,-1, 0.7,0.8,0.85,
    -1.2,3.5,-1.2, 0,1,0, 0.7,0.8,0.85,   1.2,3.5,-1.2, 0,1,0, 0.7,0.8,0.85,   1.2,3.5, 1.2, 0,1,0, 0.7,0.8,0.85,
    -1.2,3.5,-1.2, 0,1,0, 0.7,0.8,0.85,   1.2,3.5, 1.2, 0,1,0, 0.7,0.8,0.85,  -1.2,3.5, 1.2, 0,1,0, 0.7,0.8,0.85,
  ]);
  const vBuffer = device.createBuffer({ size: vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vBuffer, 0, vertices);

  const quadData = new Float32Array([-1,-1, 1,-1, -1,1, 1,-1, 1,1, -1,1]);
  const quadBuffer = device.createBuffer({ size: quadData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(quadBuffer, 0, quadData);

  const uniformBuffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const pointSampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });

  // 生成 64 个半球采样点 (通过单独的 Uniform 传递)
  const NUM_SAMPLES = 64;
  const ssaoKernelData = generateSSAOKernel(NUM_SAMPLES);
  const kernelBuffer = device.createBuffer({ size: ssaoKernelData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(kernelBuffer, 0, ssaoKernelData as any);

  // 2. Pass 1: G-Buffer 阶段
  const gbufferShader = `
    struct Uniforms {
      viewProj: mat4x4f, view: mat4x4f, proj: mat4x4f,
      camParams: vec4f, ssaoParams: vec4f
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VIn {
      @location(0) pos: vec3f,
      @location(1) norm: vec3f,
      @location(2) col: vec3f,
    };
    struct VOut {
      @builtin(position) pos: vec4f,
      @location(0) normalVS: vec3f,
      @location(1) color: vec3f,
    };

    @vertex fn vs(v: VIn) -> VOut {
      var o: VOut;
      o.pos = u.viewProj * vec4f(v.pos, 1.0);
      o.normalVS = (u.view * vec4f(v.norm, 0.0)).xyz;
      
      let sunDir = normalize(vec3f(0.3, 0.9, 0.4));
      let diff = max(dot(v.norm, sunDir), 0.25);
      o.color = v.col * diff;
      return o;
    }

    struct GOut {
      @location(0) color: vec4f,
      @location(1) normalVS: vec4f,
    };

    @fragment fn fs(in: VOut) -> GOut {
      var g: GOut;
      g.color = vec4f(in.color, 1.0);
      g.normalVS = vec4f(normalize(in.normalVS), 1.0);
      return g;
    }
  `;
  const gbufferPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: gbufferShader }), entryPoint: "vs",
      buffers: [{ arrayStride: 36, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x3" }
      ]}],
    },
    fragment: { module: device.createShaderModule({ code: gbufferShader }), entryPoint: "fs", targets: [{ format: "rgba16float" }, { format: "rgba16float" }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    primitive: { topology: "triangle-list" }
  });

  // 3. Pass 2: SSAO 计算着色器 (符合 WGSL 强类型规范)
  const ssaoShader = `
    struct Uniforms {
      viewProj: mat4x4f, view: mat4x4f, proj: mat4x4f,
      camParams: vec4f,   // x: near, y: far, z: tanHalfFov, w: aspect
      ssaoParams: vec4f,  // x: radius, y: bias, z: intensity, w: displayMode
    };
    struct KernelArray {
      samples: array<vec4f, 64>
    };

    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var samp: sampler;
    @group(0) @binding(2) var colorTex: texture_2d<f32>;
    @group(0) @binding(3) var normalTex: texture_2d<f32>;
    @group(0) @binding(4) var depthTex: texture_depth_2d;
    @group(0) @binding(5) var<uniform> kernelData: KernelArray;

    @vertex fn vs(@location(0) pos: vec2f) -> @builtin(position) vec4f {
      return vec4f(pos, 0.0, 1.0);
    }

    fn getLinearDepth(rawDepth: f32) -> f32 {
      let near = u.camParams.x; 
      let far = u.camParams.y;
      return (near * far) / (far - rawDepth * (far - near));
    }

    // 从深度重建视图空间坐标 (因为摄像机向 -Z 方向看，因此返回负Z)
    fn getViewPos(uv: vec2f, linearZ: f32) -> vec3f {
      let x = (uv.x * 2.0 - 1.0) * u.camParams.z * u.camParams.w * linearZ;
      let y = (1.0 - uv.y * 2.0) * u.camParams.z * linearZ;
      return vec3f(x, y, -linearZ); 
    }

    // 简易屏幕空间白噪声 Hash，用来随机旋转法线半球
    fn hash12(p: vec2f) -> f32 {
      var p3 = fract(vec3f(p.xyx) * 0.1031);
      p3 += dot(p3, p3.yzx + vec3f(33.33)); // 严格类型：标量常量包装成 vec3f
      return fract((p3.x + p3.y) * p3.z);
    }

    @fragment fn fs(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      let uv = fragCoord.xy / vec2f(textureDimensions(depthTex));
      let rawZ = textureSampleLevel(depthTex, samp, uv, 0);
      let sceneColor = textureSampleLevel(colorTex, samp, uv, 0.0).rgb;

      // 天空盒/背景不计算 SSAO
      if (rawZ >= 0.99999) {
        return vec4f(sceneColor, 1.0);
      }

      let linearZ = getLinearDepth(rawZ);
      let P = getViewPos(uv, linearZ); // 当前像素点的视图空间坐标
      let N = normalize(textureSampleLevel(normalTex, samp, uv, 0).xyz); // 视图空间法线

      // 使用噪声生成一个随机向量，构建 TBN 矩阵用于旋转半球采样点
      let randomAngle = hash12(fragCoord.xy) * 6.2831853;
      let randVec = vec3f(cos(randomAngle), sin(randomAngle), 0.001); // 加上极小的Z避免共线
      
      let tangent = normalize(randVec - N * dot(randVec, N));
      let bitangent = cross(N, tangent);
      let tbn = mat3x3f(tangent, bitangent, N); // TBN 矩阵：将切线空间转换到视图空间

      let radius = u.ssaoParams.x;
      let bias = u.ssaoParams.y;
      let intensity = u.ssaoParams.z;
      let displayMode = i32(u.ssaoParams.w);
      let numSamples = 64;
      
      var occlusion = 0.0;

      for (var i = 0; i < numSamples; i++) {
        // 获取预计算的样本点，并应用 TBN 矩阵旋转到该像素的法线方向
        let sampleLocal = kernelData.samples[i].xyz;
        let sampleView = tbn * sampleLocal;
        let samplePos = P + sampleView * radius; // 采样点在视图空间的位置

        // 将采样点投影回屏幕空间，以读取对应位置的深度缓冲
        var offset = u.proj * vec4f(samplePos, 1.0);
        offset = vec4f(offset.xyz / offset.w, 1.0);
        // WebGPU 裁剪空间：Y朝上，UV：Y朝下。
        let sampleUV = vec2f(offset.x, -offset.y) * 0.5 + vec2f(0.5);

        // 读取深度，并防止越界
        if (sampleUV.x >= 0.0 && sampleUV.x <= 1.0 && sampleUV.y >= 0.0 && sampleUV.y <= 1.0) {
          let sampleDepthRaw = textureSampleLevel(depthTex, samp, sampleUV, 0);
          let sampleLinearZ = getLinearDepth(sampleDepthRaw);
          
          // 范围检查：如果采样的物体离我们在Z轴上太远（比如背景），则它不应该遮挡当前像素
          // abs((-linearZ) - (-sampleLinearZ))
          let depthDiff = abs(linearZ - sampleLinearZ);
          // WGSL smoothstep 强类型：前后必须保持同类型
          let rangeCheck = smoothstep(0.0, 1.0, radius / depthDiff);

          // 遮挡判断：如果采样点屏幕位置上存储的深度 更靠近 摄像机 (距离更小)，则产生遮挡！
          // 注意：视图空间中摄像机前方的 Z 坐标均为负数。
          // -sampleLinearZ 是纹理存储的真实 Z 坐标，samplePos.z 是我们探针点的 Z 坐标。
          if (-sampleLinearZ >= samplePos.z + bias) {
            occlusion += 1.0 * rangeCheck;
          }
        }
      }

      // 计算最终遮蔽率并附加强度
      var ao = 1.0 - (occlusion / f32(numSamples));
      ao = pow(ao, intensity);

      // WGSL类型严谨：ao是标量，必须用 vec3f(ao) 包装成三维向量参与颜色乘法
      let finalAO = clamp(vec3f(ao), vec3f(0.0), vec3f(1.0));

      if (displayMode == 1) {
        return vec4f(finalAO, 1.0); // 纯AO
      } else if (displayMode == 2) {
        return vec4f(sceneColor, 1.0); // 无AO
      }
      return vec4f(sceneColor * finalAO, 1.0); // 混合叠加
    }
  `;

  const ssaoPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: ssaoShader }), entryPoint: "vs", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: device.createShaderModule({ code: ssaoShader }), entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" }
  });

  // 4. 贴图尺寸自适应
  let curWidth = 0, curHeight = 0;
  let gColorTex: GPUTexture, gNormalTex: GPUTexture, gDepthTex: GPUTexture;
  let gbufferBindGroup: GPUBindGroup, ssaoBindGroup: GPUBindGroup;

  function ensureTextures() {
    const w = canvas.width || 800;
    const h = canvas.height || 600;
    if (w === curWidth && h === curHeight && gColorTex) return;
    curWidth = w; curHeight = h;

    if (gColorTex) gColorTex.destroy();
    if (gNormalTex) gNormalTex.destroy();
    if (gDepthTex) gDepthTex.destroy();

    gColorTex = device.createTexture({ size: [w, h], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    gNormalTex = device.createTexture({ size: [w, h], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    gDepthTex = device.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });

    gbufferBindGroup = device.createBindGroup({
      layout: gbufferPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
    });

    ssaoBindGroup = device.createBindGroup({
      layout: ssaoPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: pointSampler },
        { binding: 2, resource: gColorTex.createView() },
        { binding: 3, resource: gNormalTex.createView() },
        { binding: 4, resource: gDepthTex.createView() },
        { binding: 5, resource: { buffer: kernelBuffer } },
      ]
    });
  }

  // 5. GUI 参数面板
  const camera = { distance: 9.0, theta: 40, phi: 28, panY: 1.2 };
  const ssaoParams = { radius: 1.0, bias: 0.025, intensity: 2.0, displayMode: 0 };

  gui.addTextInfo("<b>屏幕空间环境光遮蔽 (SSAO)</b><br>0: 混合叠加<br>1: 纯AO(带白噪声)<br>2: 无AO");
  gui.add(ssaoParams, "displayMode", 0, 2, 1).name("视图模式(0-2)");
  gui.add(ssaoParams, "intensity", 0.0, 5.0, 0.1).name("AO 遮蔽强度");
  gui.add(ssaoParams, "radius", 0.1, 5.0, 0.1).name("半球采样半径");
  gui.add(ssaoParams, "bias", 0.001, 0.1, 0.001).name("深度偏移(Bias)");
  gui.add(camera, "theta", -180, 180, 1).name("偏航角");
  gui.add(camera, "phi", 5, 85, 1).name("俯仰角");

  let isDragging = false, lastX = 0, lastY = 0;
  canvas.onpointerdown = (e) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); };
  canvas.onpointermove = (e) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.4;
    camera.phi = Math.max(5, Math.min(85, camera.phi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY;
    gui.updateDisplay();
  };
  canvas.onpointerup = (e) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} };

  const cpuData = new Float32Array(256 / 4);
  let animId: number;

  function frame() {
    ensureTextures();
    const aspect = curWidth / curHeight;
    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.distance * Math.cos(radPhi) * Math.sin(radTheta),
      camera.panY + camera.distance * Math.sin(radPhi),
      camera.distance * Math.cos(radPhi) * Math.cos(radTheta),
    ];

    const near = 0.1, far = 50.0;
    const fov = (50 * Math.PI) / 180;
    const tanHalfFov = Math.tan(fov / 2);

    const view = createLookAtMatrix(eye, [0, camera.panY, 0], [0, 1, 0]);
    const proj = createPerspectiveMatrix(fov, aspect, near, far);
    const viewProj = multiplyMat4(proj, view);

    // 写入主要 Uniforms
    cpuData.set(viewProj, 0);
    cpuData.set(view, 16);
    cpuData.set(proj, 32);
    cpuData.set([near, far, tanHalfFov, aspect], 48);
    cpuData.set([ssaoParams.radius, ssaoParams.bias, ssaoParams.intensity, ssaoParams.displayMode], 52);
    device.queue.writeBuffer(uniformBuffer, 0, cpuData);

    const encoder = device.createCommandEncoder();

    // Pass 1: G-Buffer
    const pass1 = encoder.beginRenderPass({
      colorAttachments: [
        { view: gColorTex.createView(), clearValue: { r: 0.1, g: 0.12, b: 0.15, a: 1.0 }, loadOp: "clear", storeOp: "store" },
        { view: gNormalTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }
      ],
      depthStencilAttachment: { view: gDepthTex.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" }
    });
    pass1.setPipeline(gbufferPipeline);
    pass1.setBindGroup(0, gbufferBindGroup);
    pass1.setVertexBuffer(0, vBuffer);
    pass1.draw(vertices.length / 9);
    pass1.end();

    // Pass 2: SSAO 计算与合成输出
    const pass2 = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }]
    });
    pass2.setPipeline(ssaoPipeline);
    pass2.setBindGroup(0, ssaoBindGroup);
    pass2.setVertexBuffer(0, quadBuffer);
    pass2.draw(6); // 绘制全屏 Quad
    pass2.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }
  frame();

  return () => {
    cancelAnimationFrame(animId);
    vBuffer.destroy(); quadBuffer.destroy(); uniformBuffer.destroy(); kernelBuffer.destroy();
    if (gColorTex) gColorTex.destroy();
    if (gNormalTex) gNormalTex.destroy();
    if (gDepthTex) gDepthTex.destroy();
  };
}