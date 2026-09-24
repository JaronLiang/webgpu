// src/examples/skybox.ts

// 简易 4x4 矩阵计算工具
function createPerspectiveMatrix(fovY: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(fovY / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far / (near - far);
  out[11] = -1.0;
  out[14] = (far * near) / (near - far);
  return out;
}

function createViewMatrix(yaw: number, pitch: number): Float32Array {
  const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
  const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
  const out = new Float32Array(16);
  // 仅包含旋转，剔除平移分量，使得天空盒永远相对观察者居中
  out[0] = cosY;
  out[1] = sinY * sinP;
  out[2] = -sinY * cosP;
  out[4] = 0;
  out[5] = cosP;
  out[6] = sinP;
  out[8] = sinY;
  out[9] = -cosY * sinP;
  out[10] = cosY * cosP;
  out[15] = 1.0;
  return out;
}

function multiplyMatrices(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[k * 4 + r] * b[c * 4 + k];
      }
      out[c * 4 + r] = sum;
    }
  }
  return out;
}

// 默认提供的经典自然风光天空盒 6 面图源 (Three.js 官方静态资产)
const DEFAULT_SKYBOX_URLS = [
  'https://threejs.org/examples/textures/cube/Bridge2/posx.jpg', // +X (右)
  'https://threejs.org/examples/textures/cube/Bridge2/negx.jpg', // -X (左)
  'https://threejs.org/examples/textures/cube/Bridge2/posy.jpg', // +Y (上)
  'https://threejs.org/examples/textures/cube/Bridge2/negy.jpg', // -Y (下)
  'https://threejs.org/examples/textures/cube/Bridge2/posz.jpg', // +Z (前)
  'https://threejs.org/examples/textures/cube/Bridge2/negz.jpg', // -Z (后)
];

export async function runSkybox(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  imageUrls: string[] = DEFAULT_SKYBOX_URLS // 允许传入自定义 6 面图片
) {
  // -------------------------------------------------------------
  // 1. 标准立方体几何数据 (36 个顶点)
  // -------------------------------------------------------------
  const cubeVertices = new Float32Array([
    // 正面 (+Z)
    -1, -1,  1,   1, -1,  1,   1,  1,  1,  -1, -1,  1,   1,  1,  1,  -1,  1,  1,
    // 背面 (-Z)
    -1, -1, -1,  -1,  1, -1,   1,  1, -1,  -1, -1, -1,   1,  1, -1,   1, -1, -1,
    // 顶面 (+Y)
    -1,  1, -1,  -1,  1,  1,   1,  1,  1,  -1,  1, -1,   1,  1,  1,   1,  1, -1,
    // 底面 (-Y)
    -1, -1, -1,   1, -1, -1,   1, -1,  1,  -1, -1, -1,   1, -1,  1,  -1, -1,  1,
    // 右面 (+X)
     1, -1, -1,   1,  1, -1,   1,  1,  1,   1, -1, -1,   1,  1,  1,   1, -1,  1,
    // 左面 (-X)
    -1, -1, -1,  -1, -1,  1,  -1,  1,  1,  -1, -1, -1,  -1,  1,  1,  -1,  1, -1,
  ]);

  const vertexBuffer = device.createBuffer({
    size: cubeVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, cubeVertices);

  // -------------------------------------------------------------
  // 2. 加载外部真实天空盒贴图 (Cubemap)
  // -------------------------------------------------------------
  // 先创建一个 1x1 的占位贴图，防止图片异步下载期间报错
  let cubemapTexture = device.createTexture({
    size: [1, 1, 6],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  const matrixBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: "filtering" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: "float", viewDimension: "cube" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform", minBindingSize: 64 },
      },
    ],
  });

  let bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: cubemapTexture.createView({ dimension: "cube" }) },
      { binding: 2, resource: { buffer: matrixBuffer } },
    ],
  });

  // 异步加载外部 6 张图片
  async function loadSkyboxImages(urls: string[]) {
    try {
      const bitmaps = await Promise.all(
        urls.map(async (url) => {
          const res = await fetch(url);
          const blob = await res.blob();
          return await createImageBitmap(blob, { colorSpaceConversion: "none" });
        })
      );

      const size = bitmaps[0].width;
      
      // 创建正式的高清立方体纹理
      const newCubemapTexture = device.createTexture({
        size: [size, size, 6],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });

      // 将 6 个 ImageBitmap 依次拷入 6 个 face 深度层
      for (let face = 0; face < 6; face++) {
        device.queue.copyExternalImageToTexture(
          { source: bitmaps[face] },
          { texture: newCubemapTexture, origin: [0, 0, face] },
          [size, size]
        );
      }

      cubemapTexture.destroy();
      cubemapTexture = newCubemapTexture;

      // 重新生成 BindGroup 绑定最新贴图
      bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: sampler },
          { binding: 1, resource: cubemapTexture.createView({ dimension: "cube" }) },
          { binding: 2, resource: { buffer: matrixBuffer } },
        ],
      });
    } catch (err) {
      console.warn("加载外部天空盒失败，回退到占位模式:", err);
    }
  }

  loadSkyboxImages(imageUrls);

  // -------------------------------------------------------------
  // 3. WGSL 着色器
  // -------------------------------------------------------------
  const shaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
    };

    @group(0) @binding(0) var skySampler: sampler;
    @group(0) @binding(1) var skyTexture: texture_cube<f32>;
    @group(0) @binding(2) var<uniform> uniforms: Uniforms;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) dir: vec3f,
    };

    @vertex
    fn vs_main(@location(0) position: vec3f) -> VertexOut {
      var out: VertexOut;
      out.dir = position;
      let clipPos = uniforms.viewProj * vec4f(position, 1.0);
      out.pos = clipPos.xyww; // 技巧：将 z 置为 w，确保深度值在 1.0 远平面
      return out;
    }

    @fragment
    fn fs_main(@location(0) dir: vec3f) -> @location(0) vec4f {
      return textureSample(skyTexture, skySampler, dir);
    }
  `;

  // -------------------------------------------------------------
  // 4. 创建渲染管线
  // -------------------------------------------------------------
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module: device.createShaderModule({ code: shaderCode }),
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 3 * 4,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
        },
      ],
    },
    fragment: {
      module: device.createShaderModule({ code: shaderCode }),
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "none",
    },
  });

  // -------------------------------------------------------------
  // 5. 交互式相机控制 (鼠标拖拽与滚轮缩放)
  // -------------------------------------------------------------
  let yaw = 0;               // 水平偏航角 (左右)
  let pitch = 0;             // 垂直俯仰角 (上下)
  let targetYaw = 0;
  let targetPitch = 0;
  let fovDegrees = 65;       // 视场角 (视野缩放)
  let targetFov = 65;

  let isDragging = false;
  let lastMouseX = 0;
  let lastMouseY = 0;

  const onMouseDown = (e: MouseEvent) => {
    isDragging = true;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  };

  const onMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;
    const deltaX = e.clientX - lastMouseX;
    const deltaY = e.clientY - lastMouseY;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;

    const sensitivity = 0.005;
    targetYaw += deltaX * sensitivity;
    targetPitch += deltaY * sensitivity;

    // 限制垂直俯仰角度在 [-89°, 89°]，防止视角颠覆
    const maxPitch = Math.PI / 2 - 0.01;
    targetPitch = Math.max(-maxPitch, Math.min(maxPitch, targetPitch));
  };

  const onMouseUp = () => {
    isDragging = false;
  };

  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    targetFov += e.deltaY * 0.05;
    // 限制 FOV 在 25° 到 100° 之间
    targetFov = Math.max(25, Math.min(100, targetFov));
  };

  // 挂载 DOM 事件
  canvas.addEventListener("mousedown", onMouseDown);
  window.addEventListener("mousemove", onMouseMove);
  window.addEventListener("mouseup", onMouseUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  // -------------------------------------------------------------
  // 6. 渲染循环
  // -------------------------------------------------------------
  let animId: number;

  function frame() {
    // 缓动平滑过渡 (Lerp)，获得平滑阻尼手感
    yaw += (targetYaw - yaw) * 0.15;
    pitch += (targetPitch - pitch) * 0.15;
    fovDegrees += (targetFov - fovDegrees) * 0.15;

    const aspect = canvas.width / canvas.height;
    const proj = createPerspectiveMatrix((fovDegrees * Math.PI) / 180, aspect, 0.1, 1000.0);
    const view = createViewMatrix(yaw, pitch);
    const viewProj = multiplyMatrices(proj, view);

    // 写入矩阵
    device.queue.writeBuffer(matrixBuffer, 0, viewProj as any);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setBindGroup(0, bindGroup);
    pass.draw(36);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  // -------------------------------------------------------------
  // 7. 清理资源与解绑事件
  // -------------------------------------------------------------
  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("mousedown", onMouseDown);
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseup", onMouseUp);
    canvas.removeEventListener("wheel", onWheel);

    cubemapTexture.destroy();
    vertexBuffer.destroy();
    matrixBuffer.destroy();
  };
}