// src/examples/opaqueAndTransparent.ts
import GUI from "lil-gui";

// =========================================================================
// 1. 标准 3D 数学库 (坚如磐石的实现)
// =========================================================================
function mat4Perspective(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const out = new Float32Array(16);
  const f = 1.0 / Math.tan(fovRad / 2);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far / (near - far);
  out[11] = -1;
  out[14] = (near * far) / (near - far);
  out[15] = 0;
  return out;
}

function mat4LookAt(eye: number[], center: number[], up: number[]): Float32Array {
  const out = new Float32Array(16);
  let x0, x1, x2, y0, y1, y2, z0, z1, z2, len;
  let eyex = eye[0], eyey = eye[1], eyez = eye[2];
  let centerx = center[0], centery = center[1], centerz = center[2];
  let upx = up[0], upy = up[1], upz = up[2];

  z0 = eyex - centerx; z1 = eyey - centery; z2 = eyez - centerz;
  len = 1 / Math.hypot(z0, z1, z2);
  z0 *= len; z1 *= len; z2 *= len;

  x0 = upy * z2 - upz * z1; x1 = upz * z0 - upx * z2; x2 = upx * z1 - upy * z0;
  len = 1 / Math.hypot(x0, x1, x2);
  x0 *= len; x1 *= len; x2 *= len;

  y0 = z1 * x2 - z2 * x1; y1 = z2 * x0 - z0 * x2; y2 = z0 * x1 - z1 * x0;

  out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
  out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
  out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
  out[12] = -(x0 * eyex + x1 * eyey + x2 * eyez);
  out[13] = -(y0 * eyex + y1 * eyey + y2 * eyez);
  out[14] = -(z0 * eyex + z1 * eyey + z2 * eyez);
  out[15] = 1;
  return out;
}

function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) { // col
    for (let j = 0; j < 4; j++) { // row
      out[i * 4 + j] =
        a[0 * 4 + j] * b[i * 4 + 0] +
        a[1 * 4 + j] * b[i * 4 + 1] +
        a[2 * 4 + j] * b[i * 4 + 2] +
        a[3 * 4 + j] * b[i * 4 + 3];
    }
  }
  return out;
}

// O(N) 高速基数排序
function radixSort(depths: Float32Array, indices: Uint32Array, count: number): any {
  const depthBuffer = new Uint32Array(depths.buffer, depths.byteOffset, count);
  let tempIndices: any = new Uint32Array(count);
  let currentIndices: any = indices;
  for (let i = 0; i < count; i++) {
    depthBuffer[i] ^= (depthBuffer[i] & 0x80000000) ? 0xffffffff : 0x80000000;
  }
  for (let byte = 0; byte < 4; byte++) {
    const shift = byte * 8;
    const counts = new Uint32Array(256);
    for (let i = 0; i < count; i++) counts[(depthBuffer[currentIndices[i]] >> shift) & 0xFF]++;
    let sum = 0;
    for (let i = 0; i < 256; i++) { const c = counts[i]; counts[i] = sum; sum += c; }
    for (let i = 0; i < count; i++) {
      const id = currentIndices[i];
      const val = (depthBuffer[id] >> shift) & 0xFF;
      tempIndices[counts[val]++] = id;
    }
    const t = currentIndices; currentIndices = tempIndices; tempIndices = t;
  }
  return currentIndices;
}

// =========================================================================
// 2. 主渲染程序
// =========================================================================
export function runOpaqueAndTransparent(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: any
) {
  const OPAQUE_COUNT = 10000;      // 核心不透明物体数量
  const TRANSPARENT_COUNT = 10000; // 外围半透明光晕数量

  // 简单的 3D 立方体几何数据
  const cubePositions = new Float32Array([
    -1,-1,1,  1,-1,1,  1,1,1,  -1,1,1,  // Front
    -1,-1,-1, -1,1,-1, 1,1,-1, 1,-1,-1, // Back
    -1,1,-1,  -1,1,1,  1,1,1,  1,1,-1,  // Top
    -1,-1,-1, 1,-1,-1, 1,-1,1, -1,-1,1, // Bottom
    1,-1,-1,  1,1,-1,  1,1,1,  1,-1,1,  // Right
    -1,-1,-1, -1,-1,1, -1,1,1, -1,1,-1, // Left
  ]);
  const cubeIndices = new Uint16Array([
    0,1,2, 0,2,3,       4,5,6, 4,6,7,
    8,9,10, 8,10,11,    12,13,14, 12,14,15,
    16,17,18, 16,18,19, 20,21,22, 20,22,23
  ]);

  const vertexBuffer = device.createBuffer({ size: cubePositions.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vertexBuffer, 0, cubePositions);
  const indexBuffer = device.createBuffer({ size: cubeIndices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(indexBuffer, 0, cubeIndices);

  // 初始化大量实例数据 [X, Y, Z, Scale, R, G, B, A]
  const opaqueData = new Float32Array(OPAQUE_COUNT * 8);
  for (let i = 0; i < OPAQUE_COUNT; i++) {
    const r = Math.pow(Math.random(), 0.5) * 15;
    const theta = Math.random() * Math.PI * 2;
    const phi = (Math.random() - 0.5) * Math.PI;
    opaqueData[i*8+0] = r * Math.cos(phi) * Math.cos(theta);
    opaqueData[i*8+1] = r * Math.sin(phi);
    opaqueData[i*8+2] = r * Math.cos(phi) * Math.sin(theta);
    opaqueData[i*8+3] = 0.2 + Math.random() * 0.3; // Scale
    opaqueData[i*8+4] = 0.6 + Math.random() * 0.4; // R (偏红)
    opaqueData[i*8+5] = 0.2 + Math.random() * 0.2; // G
    opaqueData[i*8+6] = 0.1 + Math.random() * 0.2; // B
    opaqueData[i*8+7] = 1.0;                       // Alpha 100%
  }

  const transparentData = new Float32Array(TRANSPARENT_COUNT * 8);
  for (let i = 0; i < TRANSPARENT_COUNT; i++) {
    const r = 10 + Math.random() * 25; 
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1) - Math.PI / 2;
    transparentData[i*8+0] = r * Math.cos(phi) * Math.cos(theta);
    transparentData[i*8+1] = r * Math.sin(phi);
    transparentData[i*8+2] = r * Math.cos(phi) * Math.sin(theta);
    transparentData[i*8+3] = 0.4 + Math.random() * 0.6; // Scale
    transparentData[i*8+4] = 0.1 + Math.random() * 0.3; // R (偏蓝绿)
    transparentData[i*8+5] = 0.4 + Math.random() * 0.5; // G
    transparentData[i*8+6] = 0.8 + Math.random() * 0.2; // B
    transparentData[i*8+7] = 0.1 + Math.random() * 0.3; // Alpha (半透明)
  }

  const opaqueStorage = device.createBuffer({ size: opaqueData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(opaqueStorage, 0, opaqueData);

  const transparentStorage = device.createBuffer({ size: transparentData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(transparentStorage, 0, transparentData);

  let sortedIndicesArray = new Uint32Array(TRANSPARENT_COUNT);
  const sortDepthBuffer = new Float32Array(TRANSPARENT_COUNT);
  for(let i=0; i<TRANSPARENT_COUNT; i++) sortedIndicesArray[i] = i;
  
  const sortedIndexStorage = device.createBuffer({ size: sortedIndicesArray.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(sortedIndexStorage, 0, sortedIndicesArray);

  const uniformBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // WGSL Shader (移除了复杂的透明剔除，确保绝对显示)
  const shaderCode = `
    struct Camera { viewProj: mat4x4f };
    @group(0) @binding(0) var<uniform> camera: Camera;

    struct Instance {
      transform: vec4f,
      color: vec4f,
    };
    @group(0) @binding(1) var<storage, read> instances: array<Instance>;
    @group(0) @binding(2) var<storage, read> sortedIndices: array<u32>;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) color: vec4f,
    };

    @vertex
    fn vs_opaque(@location(0) pos: vec3f, @builtin(instance_index) instIdx: u32) -> VertexOutput {
      let instance = instances[instIdx];
      return processVertex(pos, instance);
    }

    @vertex
    fn vs_transparent(@location(0) pos: vec3f, @builtin(instance_index) instIdx: u32) -> VertexOutput {
      let realIdx = sortedIndices[instIdx];
      let instance = instances[realIdx];
      return processVertex(pos, instance);
    }

    fn processVertex(pos: vec3f, instance: Instance) -> VertexOutput {
      var out: VertexOutput;
      let worldPos = pos * instance.transform.w + instance.transform.xyz;
      out.position = camera.viewProj * vec4f(worldPos, 1.0);
      out.color = instance.color;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      // 直接输出颜色，不干预 Alpha，避免 Canvas 复合模式将其镂空
      return in.color;
    }
  `;

  const module = device.createShaderModule({ code: shaderCode });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ]
  });

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  // 第一管线：不透明物体 (写深度)
  const opaquePipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module, entryPoint: "vs_opaque", buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }]}] },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    primitive: { topology: "triangle-list", cullMode: "back" },
  });

  // 第二管线：半透明物体 (关闭写深度，开启混合)
  const transparentPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module, entryPoint: "vs_transparent", buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }]}] },
    fragment: { 
      module, entryPoint: "fs_main", 
      targets: [{ 
        format, 
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" }
        } 
      }] 
    },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less", format: "depth24plus" },
    primitive: { topology: "triangle-list", cullMode: "none" },
  });

  const opaqueBindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: opaqueStorage } },
      { binding: 2, resource: { buffer: sortedIndexStorage } }
    ]
  });

  const transparentBindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: transparentStorage } },
      { binding: 2, resource: { buffer: sortedIndexStorage } }
    ]
  });

  // 相机与控制逻辑
  const camera = { target: [0, 0, 0], radius: 70.0, theta: 45.0, phi: 20.0 };
  let isDragging = false, dragButton = 0, lastX = 0, lastY = 0;
  
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    isDragging = true; dragButton = e.shiftKey ? 2 : e.button;
    lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!isDragging) return;
    const dx = e.clientX - lastX; const dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (dragButton === 0) {
      camera.theta -= dx * 0.35;
      camera.phi = Math.max(-88, Math.min(88, camera.phi + dy * 0.35));
    } else if (dragButton === 2) {
      const radTheta = (camera.theta * Math.PI) / 180;
      const pan = camera.radius * 0.0015;
      camera.target[0] -= Math.cos(radTheta) * dx * pan;
      camera.target[2] -= -Math.sin(radTheta) * dx * pan;
      camera.target[1] += dy * pan;
    }
  });
  canvas.addEventListener("pointerup", (e) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); camera.radius = Math.max(5, camera.radius * Math.exp(e.deltaY * 0.001)); }, { passive: false });

  let depthTexture: GPUTexture | null = null;
  let animId: number;

  const settings = { autoRotate: true, opaqueCount: OPAQUE_COUNT, transCount: TRANSPARENT_COUNT };
  gui.add(settings, 'autoRotate').name("自动旋转");
  gui.add(settings, 'opaqueCount').name("不透明方块数").disable();
  gui.add(settings, 'transCount').name("半透明光晕数").disable();

  function frame() {
    if (settings.autoRotate && !isDragging) camera.theta += 0.2;

    const dpr = window.devicePixelRatio || 1;
    const renderWidth = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const renderHeight = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    
    // 自动重置深度缓冲
    if (!depthTexture || depthTexture.width !== renderWidth || depthTexture.height !== renderHeight) {
      canvas.width = renderWidth; 
      canvas.height = renderHeight;
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({ 
        size: [renderWidth, renderHeight], 
        format: "depth24plus", 
        usage: GPUTextureUsage.RENDER_ATTACHMENT 
      });
    }

    // 相机矩阵运算
    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.target[0] + camera.radius * Math.cos(radPhi) * Math.sin(radTheta),
      camera.target[1] + camera.radius * Math.sin(radPhi),
      camera.target[2] + camera.radius * Math.cos(radPhi) * Math.cos(radTheta),
    ];

    const proj = mat4Perspective((45 * Math.PI) / 180, renderWidth / renderHeight, 0.1, 1000.0);
    const view = mat4LookAt(eye, camera.target, [0, 1, 0]);
    const viewProj = mat4Multiply(proj, view); // P * V
    device.queue.writeBuffer(uniformBuffer, 0, viewProj as any);

    // 动态深度排序 (针对透明物体)
    for (let i = 0; i < TRANSPARENT_COUNT; i++) {
      const dx = transparentData[i*8+0] - eye[0];
      const dy = transparentData[i*8+1] - eye[1];
      const dz = transparentData[i*8+2] - eye[2];
      sortDepthBuffer[i] = -(dx*dx + dy*dy + dz*dz); 
    }
    sortedIndicesArray = radixSort(sortDepthBuffer, sortedIndicesArray, TRANSPARENT_COUNT);
    device.queue.writeBuffer(sortedIndexStorage, 0, sortedIndicesArray);

    // 录制指令序列
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear", clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1.0 }, storeOp: "store"
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthLoadOp: "clear", depthClearValue: 1.0, depthStoreOp: "store"
      }
    });

    pass.setVertexBuffer(0, vertexBuffer);
    pass.setIndexBuffer(indexBuffer, "uint16");

    // 第 1 步：绘制不透明方块
    pass.setPipeline(opaquePipeline);
    pass.setBindGroup(0, opaqueBindGroup);
    pass.drawIndexed(36, OPAQUE_COUNT, 0, 0, 0);

    // 第 2 步：绘制半透明发光体
    pass.setPipeline(transparentPipeline);
    pass.setBindGroup(0, transparentBindGroup);
    pass.drawIndexed(36, TRANSPARENT_COUNT, 0, 0, 0);

    pass.end();
    device.queue.submit([encoder.finish()]);

    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    vertexBuffer.destroy(); indexBuffer.destroy();
    opaqueStorage.destroy(); transparentStorage.destroy();
    sortedIndexStorage.destroy(); uniformBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}