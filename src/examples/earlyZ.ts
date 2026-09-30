// src/examples/earlyZ.ts
import GUI from "lil-gui";

// =========================================================================
// 1. 标准 3D 数学库 (透视与相机计算)
// =========================================================================
function mat4Perspective(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const out = new Float32Array(16);
  const f = 1.0 / Math.tan(fovRad / 2);
  out[0] = f / aspect; out[5] = f;
  out[10] = far / (near - far); out[11] = -1; out[14] = (near * far) / (near - far);
  return out;
}

function mat4LookAt(eye: number[], center: number[], up: number[]): Float32Array {
  const out = new Float32Array(16);
  let x0, x1, x2, y0, y1, y2, z0, z1, z2, len;
  z0 = eye[0] - center[0]; z1 = eye[1] - center[1]; z2 = eye[2] - center[2];
  len = 1 / Math.hypot(z0, z1, z2); z0 *= len; z1 *= len; z2 *= len;
  x0 = up[1] * z2 - up[2] * z1; x1 = up[2] * z0 - up[0] * z2; x2 = up[0] * z1 - up[1] * z0;
  len = 1 / Math.hypot(x0, x1, x2); x0 *= len; x1 *= len; x2 *= len;
  y0 = z1 * x2 - z2 * x1; y1 = z2 * x0 - z0 * x2; y2 = z0 * x1 - z1 * x0;
  out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
  out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
  out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
  out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
  out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
  out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]); out[15] = 1;
  return out;
}

function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] = a[0 * 4 + r] * b[c * 4 + 0] + a[1 * 4 + r] * b[c * 4 + 1] + 
                       a[2 * 4 + r] * b[c * 4 + 2] + a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

// =========================================================================
// 2. 主程序
// =========================================================================
export function runEarlyZ(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: any
) {
  // 实例数量：用 10,000 个紧密排列的立方体构成一堵“墙”或“小行星带”，制造大量遮挡
  const INSTANCE_COUNT = 10000;

  // 带法线的标准 3D 立方体顶点 (用于光照)，36 个顶点
  // 格式: X, Y, Z,  NX, NY, NZ
  const cubeData = new Float32Array([
    // Front
    -1,-1,1, 0,0,1,   1,-1,1, 0,0,1,   1,1,1, 0,0,1,
    -1,-1,1, 0,0,1,   1,1,1, 0,0,1,   -1,1,1, 0,0,1,
    // Back
    1,-1,-1, 0,0,-1, -1,-1,-1, 0,0,-1, -1,1,-1, 0,0,-1,
    1,-1,-1, 0,0,-1, -1,1,-1, 0,0,-1,  1,1,-1, 0,0,-1,
    // Top
    -1,1,1, 0,1,0,    1,1,1, 0,1,0,    1,1,-1, 0,1,0,
    -1,1,1, 0,1,0,    1,1,-1, 0,1,0,  -1,1,-1, 0,1,0,
    // Bottom
    -1,-1,-1, 0,-1,0, 1,-1,-1, 0,-1,0, 1,-1,1, 0,-1,0,
    -1,-1,-1, 0,-1,0, 1,-1,1, 0,-1,0, -1,-1,1, 0,-1,0,
    // Right
    1,-1,1, 1,0,0,    1,-1,-1, 1,0,0,  1,1,-1, 1,0,0,
    1,-1,1, 1,0,0,    1,1,-1, 1,0,0,   1,1,1, 1,0,0,
    // Left
    -1,-1,-1, -1,0,0, -1,-1,1, -1,0,0, -1,1,1, -1,0,0,
    -1,-1,-1, -1,0,0, -1,1,1, -1,0,0,  -1,1,-1, -1,0,0,
  ]);

  const vertexBuffer = device.createBuffer({ size: cubeData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vertexBuffer, 0, cubeData);

  // 生成大量立方体的位置和颜色数据 [PosX, PosY, PosZ, Scale, R, G, B, 1.0]
  const instanceData = new Float32Array(INSTANCE_COUNT * 8);
  const instancePositions: [number, number, number][] = []; // 备份一份用于 CPU 排序

  for (let i = 0; i < INSTANCE_COUNT; i++) {
    // 集中在一个深邃的隧道状/厚墙状空间里
    const x = (Math.random() - 0.5) * 10;
    const y = (Math.random() - 0.5) * 10;
    const z = (Math.random() - 0.5) * 60; // Z轴方向拉长，产生极端的遮挡纵深
    
    instancePositions.push([x, y, z]);

    instanceData[i*8+0] = x;
    instanceData[i*8+1] = y;
    instanceData[i*8+2] = z;
    instanceData[i*8+3] = 0.5 + Math.random() * 0.5; // Scale
    
    // 颜色按 Z 轴深度渐变，方便肉眼分辨远近
    instanceData[i*8+4] = (z + 30) / 60; // R
    instanceData[i*8+5] = 0.5;           // G
    instanceData[i*8+6] = 1.0 - (z + 30) / 60; // B
    instanceData[i*8+7] = 1.0;
  }

  const instanceStorage = device.createBuffer({ size: instanceData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(instanceStorage, 0, instanceData);

  // 排序索引 Buffer
  let sortedIndices = new Uint32Array(INSTANCE_COUNT);
  for (let i=0; i<INSTANCE_COUNT; i++) sortedIndices[i] = i;
  const indexStorage = device.createBuffer({ size: sortedIndices.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  const uniformBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // =========================================================================
  // 3. WGSL 着色器代码 (内含 Early-Z 模拟重载)
  // =========================================================================
  const shaderCode = `
    struct Camera { viewProj: mat4x4f };
    @group(0) @binding(0) var<uniform> camera: Camera;

    struct Instance {
      transform: vec4f,
      color: vec4f,
    };
    @group(0) @binding(1) var<storage, read> instances: array<Instance>;
    @group(0) @binding(2) var<storage, read> indices: array<u32>;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec3f,
      @location(1) normal: vec3f,
    };

    @vertex
    fn vs_main(
      @location(0) position: vec3f, 
      @location(1) normal: vec3f,
      @builtin(instance_index) instIdx: u32
    ) -> VertexOut {
      // 核心：通过间接索引读取实例，从而由 CPU 控制渲染顺序！
      let realIdx = indices[instIdx];
      let instance = instances[realIdx];

      let worldPos = position * instance.transform.w + instance.transform.xyz;
      
      var out: VertexOut;
      out.pos = camera.viewProj * vec4f(worldPos, 1.0);
      out.color = instance.color.rgb;
      out.normal = normal;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4f {
      // 【关键点】：人工模拟极其昂贵的片元着色器计算 (例如百步光线步进、复杂的毛发高光等)
      // 如果 Early-Z 生效，被遮挡的像素在进入这里之前就被硬件扔掉了，所以完全不卡！
      // 如果 Early-Z 被逆序绘制破坏，所有的无效像素都会执行这 300 次无意义的循环计算，导致 FPS 暴跌。
      
      var fakeWorkload = 0.0;
      for (var i = 0; i < 300; i++) {
         fakeWorkload += sin(in.pos.x * f32(i)) * cos(in.pos.y * f32(i));
      }

      // 简单的 3D 方向光 (兰伯特光照)
      let lightDir = normalize(vec3f(1.0, 1.0, 1.0));
      let diff = max(dot(normalize(in.normal), lightDir), 0.2);
      
      // 把无意义的负载乘一个极小数(几乎为0)加到颜色上，防止编译器把 for 循环优化删掉
      let finalColor = in.color * diff + vec3f(fakeWorkload * 0.000001);

      return vec4f(finalColor, 1.0);
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
  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module, entryPoint: "vs_main",
      buffers: [{
        arrayStride: 6 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" }
        ],
      }],
    },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    // 只要开启以下两项，现代 GPU 的 Early-Z 就会自动运作
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: instanceStorage } },
      { binding: 2, resource: { buffer: indexStorage } },
    ]
  });

  // =========================================================================
  // 4. GUI 控制与交互
  // =========================================================================
  const settings = {
    drawOrder: "从前向后 (利用Early-Z, 流畅)",
    autoRotate: true,
  };
  gui.title("Early-Z 硬件深度剔除揭秘");
  gui.add(settings, 'drawOrder', [
    "从前向后 (利用Early-Z, 流畅)", 
    "从后向前 (性能灾难/超绘, 卡顿)"
  ]).name("CPU提交渲染顺序");
  gui.add(settings, 'autoRotate').name("自动旋转");

  const camera = { target: [0, 0, 0], radius: 40.0, theta: 25.0, phi: 15.0 };
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
      camera.theta -= dx * 0.35; camera.phi = Math.max(-88, Math.min(88, camera.phi + dy * 0.35));
    } else if (dragButton === 2) {
      const radTheta = (camera.theta * Math.PI) / 180; const pan = camera.radius * 0.0015;
      camera.target[0] -= Math.cos(radTheta) * dx * pan; camera.target[2] -= -Math.sin(radTheta) * dx * pan; camera.target[1] += dy * pan;
    }
  });
  canvas.addEventListener("pointerup", (e) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); camera.radius = Math.max(5, camera.radius * Math.exp(e.deltaY * 0.001)); }, { passive: false });

  let depthTexture: GPUTexture | null = null;
  let animId: number;

  function frame() {
    if (settings.autoRotate && !isDragging) camera.theta += 0.2;

    const dpr = window.devicePixelRatio || 1;
    const renderWidth = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const renderHeight = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    
    if (!depthTexture || depthTexture.width !== renderWidth || depthTexture.height !== renderHeight) {
      canvas.width = renderWidth; canvas.height = renderHeight;
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({ size: [renderWidth, renderHeight], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.target[0] + camera.radius * Math.cos(radPhi) * Math.sin(radTheta),
      camera.target[1] + camera.radius * Math.sin(radPhi),
      camera.target[2] + camera.radius * Math.cos(radPhi) * Math.cos(radTheta),
    ];

    const proj = mat4Perspective((45 * Math.PI) / 180, renderWidth / renderHeight, 0.1, 1000.0);
    const view = mat4LookAt(eye, camera.target, [0, 1, 0]);
    const viewProj = mat4Multiply(proj, view);
    device.queue.writeBuffer(uniformBuffer, 0, viewProj as any);

    // =========================================================================
    // 关键逻辑：根据 GUI 设置，利用 CPU 为提交给 GPU 的物体排序
    // =========================================================================
    const distances = new Float32Array(INSTANCE_COUNT);
    for (let i = 0; i < INSTANCE_COUNT; i++) {
      const pos = instancePositions[i];
      distances[i] = (pos[0]-eye[0])**2 + (pos[1]-eye[1])**2 + (pos[2]-eye[2])**2;
    }

    // 这里使用 JS 原生 Sort 即可 (因为每次切换模式或者相机移动才需要重排，这里为了代码简洁直接每帧排)
    // 数组中存储的是 0 ~ 9999 的索引
    const indices = Array.from({ length: INSTANCE_COUNT }, (_, i) => i);
    
    if (settings.drawOrder.includes("从前向后")) {
      // 升序：近处的距离小，排在前面，优先被 draw。完美利用 Early-Z。
      indices.sort((a, b) => distances[a] - distances[b]);
    } else {
      // 降序：远处的排在前面。先画远处的，再画近处的，导致严重的 Overdraw，破坏 Early-Z。
      indices.sort((a, b) => distances[b] - distances[a]);
    }
    
    // 更新排序后的索引到 GPU
    device.queue.writeBuffer(indexStorage, 0, new Uint32Array(indices));

    // =========================================================================
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear", clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 }, storeOp: "store"
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthLoadOp: "clear", depthClearValue: 1.0, depthStoreOp: "store"
      }
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    
    // 一次性按照我们排好的诡异/优秀顺序，绘制 10000 个立方体
    pass.draw(36, INSTANCE_COUNT, 0, 0); 

    pass.end();
    device.queue.submit([encoder.finish()]);

    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    vertexBuffer.destroy(); instanceStorage.destroy();
    indexStorage.destroy(); uniformBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}