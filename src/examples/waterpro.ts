// src/examples/water_pro.ts
import type { SimpleGUI } from "../utils/gui";

// 矩阵计算辅助函数
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

// 细分网格平面生成 (用于物理波浪顶点置换)
function createPlaneGrid(size: number, segments: number) {
  const vertices: number[] = [];
  const indices: number[] = [];
  const halfSize = size * 0.5;

  for (let z = 0; z <= segments; z++) {
    const vz = (z / segments) * size - halfSize;
    for (let x = 0; x <= segments; x++) {
      const vx = (x / segments) * size - halfSize;
      vertices.push(vx, 0, vz);
    }
  }

  for (let z = 0; z < segments; z++) {
    for (let x = 0; x < segments; x++) {
      const row1 = z * (segments + 1);
      const row2 = (z + 1) * (segments + 1);
      indices.push(row1 + x, row2 + x, row1 + x + 1);
      indices.push(row1 + x + 1, row2 + x, row2 + x + 1);
    }
  }

  return {
    vertices: new Float32Array(vertices),
    indices: new Uint32Array(indices),
  };
}

export function runWaterPro(
  device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat, canvas: HTMLCanvasElement, gui: SimpleGUI
) {
  // 1. 水面网格
  const waterGrid = createPlaneGrid(40.0, 160);
  const waterVBuffer = device.createBuffer({ size: waterGrid.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(waterVBuffer, 0, waterGrid.vertices);
  const waterIBuffer = device.createBuffer({ size: waterGrid.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(waterIBuffer, 0, waterGrid.indices);

  // 海底地形顶点 (pos:3, normal:3, color:3)
  const terrainData = new Float32Array([
    // 沙滩斜坡
    -20,-6,-20, 0,1,0, 0.76,0.70,0.50,   20,-6,-20, 0,1,0, 0.76,0.70,0.50,   20, 1, 20, 0,1,0, 0.92,0.85,0.65,
    -20,-6,-20, 0,1,0, 0.76,0.70,0.50,   20, 1, 20, 0,1,0, 0.92,0.85,0.65,  -20, 1, 20, 0,1,0, 0.92,0.85,0.65,
    // 暗色水下礁石
    -3,-2,-2, 0,1,0, 0.3,0.3,0.35,   0,-2,-4, 0,1,0, 0.3,0.3,0.35,  -1, 0.4,-2, 0,1,0, 0.4,0.4,0.42,
     2,-2, 0, 0,1,0, 0.3,0.3,0.35,   4,-2,-2, 0,1,0, 0.3,0.3,0.35,   3, 0.3,-1, 0,1,0, 0.4,0.4,0.42,
  ]);
  const terrainVBuffer = device.createBuffer({ size: terrainData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(terrainVBuffer, 0, terrainData);

  // 2. Uniform Buffer
  const uniformBuffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // 3. 【修复 1】：深度使用精确的最近邻(NonFiltering)采样器，彩色使用线性(Filtering)采样器
  const linearSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  const pointSampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });

  // 4. Pass 1: 渲染水下世界与海底 (输出到 SceneColor 与 SceneDepth)
  const terrainShader = `
    struct Uniforms {
      viewProj: mat4x4f, view: mat4x4f,
      eyePos: vec4f, params: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec3f,
      @location(1) wPos: vec3f,
    };

    @vertex fn vs(@location(0) pos: vec3f, @location(1) norm: vec3f, @location(2) col: vec3f) -> VOut {
      var o: VOut;
      o.pos = u.viewProj * vec4f(pos, 1.0);
      o.wPos = pos;
      let sunDir = normalize(vec3f(0.4, 0.8, -0.5));
      let diff = max(dot(norm, sunDir), 0.25);
      o.color = col * diff;
      return o;
    }

    @fragment fn fs(in: VOut) -> @location(0) vec4f {
      let t = u.params.x * 2.0;
      let c1 = sin(in.wPos.x * 3.0 + t) * sin(in.wPos.z * 3.0 + t);
      let c2 = sin(in.wPos.x * 2.0 - t * 1.3) * sin(in.wPos.z * 2.0 - t * 1.3);
      let caustics = max(0.0, c1 + c2) * 0.15;
      return vec4f(in.color + vec3f(caustics), 1.0);
    }
  `;
  const terrainPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: terrainShader }), entryPoint: "vs",
      buffers: [{ arrayStride: 36, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x3" }
      ]}],
    },
    fragment: { module: device.createShaderModule({ code: terrainShader }), entryPoint: "fs", targets: [{ format: "rgba16float" }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    primitive: { topology: "triangle-list" }
  });

  // 5. Pass 2: Water Pro 高级物理水体管线
  const waterShader = `
    struct Uniforms {
      viewProj: mat4x4f, view: mat4x4f,
      eyePos: vec4f,
      params: vec4f, // x: time, y: waveHeight, z: speed, w: transparency
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var linearSamp: sampler;
    @group(0) @binding(2) var pointSamp: sampler;
    @group(0) @binding(3) var sceneColor: texture_2d<f32>;
    @group(0) @binding(4) var sceneDepth: texture_depth_2d;

    struct WaterVOut {
      @builtin(position) pos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) crestFoam: f32,
    };

    fn gerstnerWave(
      pos: vec3f, dir: vec2f, steepness: f32, wavelength: f32, time: f32,
      disp: ptr<function, vec3f>, norm: ptr<function, vec3f>
    ) -> f32 {
      let k = 6.28318 / wavelength;
      let c = sqrt(9.8 / k);
      let d = normalize(dir);
      let f = k * (dot(d, pos.xz) - c * time);
      let a = steepness / k;

      (*disp).x += d.x * (a * cos(f));
      (*disp).y += a * sin(f);
      (*disp).z += d.y * (a * cos(f));

      let wa = a * k;
      (*norm).x -= d.x * wa * cos(f);
      (*norm).y -= steepness * sin(f);
      (*norm).z -= d.y * wa * cos(f);

      return sin(f);
    }

    @vertex fn vs(@location(0) pos: vec3f) -> WaterVOut {
      var o: WaterVOut;
      let t = u.params.x * u.params.z;
      let h = u.params.y;

      var displaced = pos;
      var normalOffset = vec3f(0.0, 1.0, 0.0);

      var crest = 0.0;
      crest += gerstnerWave(pos, vec2f(1.0, 0.2), 0.25 * h, 7.0, t, &displaced, &normalOffset);
      crest += gerstnerWave(pos, vec2f(-0.7, 0.7), 0.18 * h, 3.8, t * 1.2, &displaced, &normalOffset);
      crest += gerstnerWave(pos, vec2f(0.3, -1.0), 0.12 * h, 1.9, t * 1.5, &displaced, &normalOffset);
      crest += gerstnerWave(pos, vec2f(-0.2, -0.6), 0.08 * h, 1.0, t * 2.0, &displaced, &normalOffset);

      o.worldPos = displaced;
      o.normal = normalize(normalOffset);
      o.pos = u.viewProj * vec4f(displaced, 1.0);
      o.crestFoam = smoothstep(1.8, 2.5, crest);
      return o;
    }

    fn linearizeDepth(depth: f32) -> f32 {
      let near = 0.1; let far = 100.0;
      return (near * far) / (far - depth * (far - near));
    }

    @fragment fn fs(in: WaterVOut) -> @location(0) vec4f {
      let screenUV = in.pos.xy / vec2f(textureDimensions(sceneColor));

      // 1. 双向流动微波细节
      let t = u.params.x * 0.4;
      let uv1 = in.worldPos.xz * 0.4 + vec2f(t * 0.04, t * 0.02);
      let uv2 = in.worldPos.xz * 0.8 - vec2f(t * 0.03, t * 0.05);
      let waveNoise = sin(uv1.x * 12.0) * cos(uv1.y * 12.0) + sin(uv2.x * 24.0) * cos(uv2.y * 24.0);
      let detailNormal = normalize(in.normal + vec3f(waveNoise * 0.08, 0.0, waveNoise * 0.08));

      // 2. 折射采样
      let distortUV = screenUV + detailNormal.xz * 0.02;
      let underwaterColor = textureSample(sceneColor, linearSamp, distortUV).rgb;

      // 3. 【修复】：使用 pointSamp 采样深度纹理，避免类型冲突
      let sceneRawDepth = textureSampleLevel(sceneDepth, pointSamp, screenUV, 0);
      let waterSurfaceDepth = in.pos.z;
      let waterDepth = max(0.0, linearizeDepth(sceneRawDepth) - linearizeDepth(waterSurfaceDepth));

      // 4. 水深吸收 (Beer-Lambert Law)
      let shallowColor = vec3f(0.12, 0.78, 0.72);
      let deepColor = vec3f(0.02, 0.12, 0.35);
      let depthFactor = 1.0 - exp(-waterDepth * 0.8);
      let waterBodyColor = mix(shallowColor, deepColor, depthFactor);

      let trans = u.params.w;
      var finalColor = mix(underwaterColor, waterBodyColor, clamp(depthFactor * trans, 0.2, 0.95));

      // 5. 岸边浪花与浪尖白沫
      let shoreFoam = smoothstep(0.35, 0.02, waterDepth);
      let foamNoise = sin(in.worldPos.x * 15.0 + t * 4.0) * cos(in.worldPos.z * 15.0 + t * 4.0);
      let totalFoam = clamp(shoreFoam * (0.6 + 0.4 * foamNoise) + in.crestFoam, 0.0, 1.0);
      finalColor = mix(finalColor, vec3f(0.96, 0.98, 1.0), totalFoam * 0.85);

      // 6. 菲涅尔与镜面太阳高光
      let V = normalize(u.eyePos.xyz - in.worldPos);
      let fresnel = 0.04 + 0.96 * pow(1.0 - max(dot(V, detailNormal), 0.0), 5.0);
      let skyReflection = mix(vec3f(0.4, 0.6, 0.9), vec3f(0.8, 0.9, 1.0), detailNormal.y);

      let sunDir = normalize(vec3f(0.4, 0.8, -0.5));
      let H = normalize(sunDir + V);
      let spec = pow(max(dot(detailNormal, H), 0.0), 128.0) * 2.5;
      let sunColor = vec3f(1.0, 0.92, 0.75);

      finalColor = mix(finalColor, skyReflection, fresnel * 0.65) + sunColor * spec;
      return vec4f(finalColor, 1.0);
    }
  `;

  const waterPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: waterShader }), entryPoint: "vs",
      buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }],
    },
    fragment: { module: device.createShaderModule({ code: waterShader }), entryPoint: "fs", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    primitive: { topology: "triangle-list" }
  });

  const terrainBindGroup = device.createBindGroup({
    layout: terrainPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
  });

  // 6. 纹理管理器 (支持 Canvas 窗口动态 Resize，彻底消除 Attachment Mismatch 报错)
  let curWidth = 0;
  let curHeight = 0;
  let sceneColorTex: GPUTexture;
  let sceneDepthTex: GPUTexture;
  let screenDepthTex: GPUTexture;
  let waterBindGroup: GPUBindGroup;

  function ensureTextures() {
    const w = canvas.width || 800;
    const h = canvas.height || 600;
    if (w === curWidth && h === curHeight && sceneColorTex) return;

    curWidth = w;
    curHeight = h;

    if (sceneColorTex) sceneColorTex.destroy();
    if (sceneDepthTex) sceneDepthTex.destroy();
    if (screenDepthTex) screenDepthTex.destroy();

    sceneColorTex = device.createTexture({
      size: [w, h],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    });
    sceneDepthTex = device.createTexture({
      size: [w, h],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    });
    // 专门为最终画布 Pass 使用的独立屏幕深度纹理，确保尺寸 100% 匹配
    screenDepthTex = device.createTexture({
      size: [w, h],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });

    waterBindGroup = device.createBindGroup({
      layout: waterPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: linearSampler },
        { binding: 2, resource: pointSampler },
        { binding: 3, resource: sceneColorTex.createView() },
        { binding: 4, resource: sceneDepthTex.createView() },
      ]
    });
  }

  // 7. GUI 控制
  const camera = { distance: 16.0, theta: 25, phi: 24, panY: 2.0 };
  const waterSettings = { waveHeight: 0.85, speed: 1.2, transparency: 0.85 };

  gui.addTextInfo("<b>Water Pro 拟真水体系统</b><br>Gerstner物理波浪 + 岸边浪花泡沫 + 水深吸收 + 菲涅尔高光");
  gui.add(waterSettings, "waveHeight", 0.0, 2.5, 0.05).name("波浪高度");
  gui.add(waterSettings, "speed", 0.2, 3.0, 0.1).name("波浪流速");
  gui.add(waterSettings, "transparency", 0.2, 1.0, 0.05).name("海水通透度");
  gui.add(camera, "theta", -180, 180, 1).name("偏航角");
  gui.add(camera, "phi", 5, 80, 1).name("俯仰角");

  let isDragging = false, lastX = 0, lastY = 0;
  canvas.onpointerdown = (e) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); };
  canvas.onpointermove = (e) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.4;
    camera.phi = Math.max(5, Math.min(80, camera.phi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY;
    gui.updateDisplay();
  };
  canvas.onpointerup = (e) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} };

  const cpuUniformData = new Float32Array(256 / 4);
  let animId: number;
  let startTime = performance.now();

  function frame() {
    // 确保所有渲染附件尺寸严格一致
    ensureTextures();

    const elapsed = (performance.now() - startTime) * 0.001;
    const aspect = curWidth / curHeight;

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.distance * Math.cos(radPhi) * Math.sin(radTheta),
      camera.panY + camera.distance * Math.sin(radPhi),
      camera.distance * Math.cos(radPhi) * Math.cos(radTheta),
    ];

    const view = createLookAtMatrix(eye, [0, 0, 0], [0, 1, 0]);
    const proj = createPerspectiveMatrix((50 * Math.PI) / 180, aspect, 0.1, 100);
    const viewProj = multiplyMat4(proj, view);

    cpuUniformData.set(viewProj, 0);
    cpuUniformData.set(view, 16);
    cpuUniformData.set([eye[0], eye[1], eye[2], 1.0], 32);
    cpuUniformData.set([elapsed, waterSettings.waveHeight, waterSettings.speed, waterSettings.transparency], 48);
    device.queue.writeBuffer(uniformBuffer, 0, cpuUniformData);

    const encoder = device.createCommandEncoder();

    // Pass 1: 渲染水下世界与海底 (输出到 sceneColorTex 与 sceneDepthTex)
    const pass1 = encoder.beginRenderPass({
      colorAttachments: [{ view: sceneColorTex.createView(), clearValue: { r: 0.1, g: 0.4, b: 0.65, a: 1.0 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: sceneDepthTex.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    pass1.setPipeline(terrainPipeline);
    pass1.setBindGroup(0, terrainBindGroup);
    pass1.setVertexBuffer(0, terrainVBuffer);
    pass1.draw(12);
    pass1.end();

    // Pass 2: 渲染动态水面并直接合成输出到屏幕
    const pass2 = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: screenDepthTex.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    pass2.setPipeline(waterPipeline);
    pass2.setBindGroup(0, waterBindGroup);
    pass2.setVertexBuffer(0, waterVBuffer);
    pass2.setIndexBuffer(waterIBuffer, "uint32");
    pass2.drawIndexed(waterGrid.indices.length);
    pass2.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }
  frame();

  return () => {
    cancelAnimationFrame(animId);
    waterVBuffer.destroy(); waterIBuffer.destroy(); terrainVBuffer.destroy();
    uniformBuffer.destroy();
    if (sceneColorTex) sceneColorTex.destroy();
    if (sceneDepthTex) sceneDepthTex.destroy();
    if (screenDepthTex) screenDepthTex.destroy();
  };
}