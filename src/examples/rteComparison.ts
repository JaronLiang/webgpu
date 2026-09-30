// src/examples/rteComparison.ts
import GUI from "lil-gui";

// =========================================================================
// 1. 标准 3D 数学库 (支持 64位 CPU 坐标计算)
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
  let z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
  let len = 1 / Math.hypot(z0, z1, z2); z0 *= len; z1 *= len; z2 *= len;
  let x0 = up[1] * z2 - up[2] * z1, x1 = up[2] * z0 - up[0] * z2, x2 = up[0] * z1 - up[1] * z0;
  len = 1 / Math.hypot(x0, x1, x2); x0 *= len; x1 *= len; x2 *= len;
  let y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;
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
export function runRTEComparison(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: any
) {
  // 生成精密细密的矢量线段几何（同心圆与雷达十字矢量网格，对坐标抖动极度敏感）
  const vectorVertices: number[] = [];

  // 1. 同心圆环矢量线
  const RINGS = 12;
  const SEGMENTS = 128;
  for (let r = 1; r <= RINGS; r++) {
    const radius = r * 0.7;
    for (let s = 0; s < SEGMENTS; s++) {
      const a1 = (s / SEGMENTS) * Math.PI * 2;
      const a2 = ((s + 1) / SEGMENTS) * Math.PI * 2;
      // 点1
      vectorVertices.push(Math.cos(a1) * radius, 0.0, Math.sin(a1) * radius, 0.0, 0.8, 1.0);
      // 点2
      vectorVertices.push(Math.cos(a2) * radius, 0.0, Math.sin(a2) * radius, 0.0, 0.8, 1.0);
    }
  }

  // 2. 细密十字网格线 (Grid)
  const GRID_SIZE = 10;
  const STEP = 0.5;
  for (let x = -GRID_SIZE; x <= GRID_SIZE; x += STEP) {
    vectorVertices.push(x, 0.0, -GRID_SIZE, 0.3, 0.4, 0.5);
    vectorVertices.push(x, 0.0,  GRID_SIZE, 0.3, 0.4, 0.5);
  }
  for (let z = -GRID_SIZE; z <= GRID_SIZE; z += STEP) {
    vectorVertices.push(-GRID_SIZE, 0.0, z, 0.3, 0.4, 0.5);
    vectorVertices.push( GRID_SIZE, 0.0, z, 0.3, 0.4, 0.5);
  }

  const vertexData = new Float32Array(vectorVertices);
  const vertexCount = vectorVertices.length / 6;

  const vertexBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // 统一变量缓冲 (Uniform Buffer)
  // [0..15]: cameraViewProj (64B)
  // [16..19]: cameraProjOnly (64B)
  // [20..35]: cameraViewRotationOnly (64B)
  // [36..39]: modelRelativeToEye (16B)
  // [40..43]: hugeWorldOrigin (16B)
  // [44]: useRTE (4B), [45..47]: padding
  const uniformBuffer = device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 3. WGSL 核心着色器：对比 RTE 与非 RTE 计算链路
  // =========================================================================
  const shaderCode = `
    struct Uniforms {
      cameraViewProj: mat4x4f,          // 绝对世界坐标矩阵 (包含超大平移)
      cameraProj: mat4x4f,              // 纯投影矩阵
      viewRotationOnly: mat4x4f,        // 纯旋转视图矩阵 (无位移)
      relativeToEye: vec4f,             // CPU 64位双精度减算出的相对位移 (数值极小)
      hugeWorldOrigin: vec4f,           // 超大世界坐标 (例如 1,000,000.0)
      useRTE: u32,
      pad1: u32,
      pad2: u32,
      pad3: u32,
    };

    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexInput {
      @location(0) position: vec3f,
      @location(1) color: vec3f,
    };

    struct VertexOutput {
      @builtin(position) clipPos: vec4f,
      @location(0) color: vec3f,
    };

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
      var out: VertexOutput;

      if (u.useRTE == 0u) {
        // -------------------------------------------------------------
        // 【非 RTE 传统模式 (严重抖动)】
        // 顶点坐标直接在 GPU 32位单精度下加上数百万级的大坐标：
        // 此时 float32 尾数被大整数完全吃满，微小的毫米/厘米级小数直接被截断舍入！
        // 摄像机稍微转动，顶点坐标就会在相邻的离散浮点数之间来回震荡跳动！
        // -------------------------------------------------------------
        let absoluteWorldPos = in.position + u.hugeWorldOrigin.xyz;
        out.clipPos = u.cameraViewProj * vec4f(absoluteWorldPos, 1.0);
      } else {
        // -------------------------------------------------------------
        // 【RTE (Relative to Eye) 模式 (丝滑稳定)】
        // 在 CPU 上使用 64位双精度完成: RelativeOffset = WorldPos64 - EyePos64
        // 传入 GPU 的数值极其微小（接近0），完全处于单精度浮点数最高精度区间！
        // 顶点直接乘以纯旋转矩阵和投影矩阵，彻底消除截断误差！
        // -------------------------------------------------------------
        let localEyePos = in.position + u.relativeToEye.xyz;
        let viewPos = u.viewRotationOnly * vec4f(localEyePos, 1.0);
        out.clipPos = u.cameraProj * viewPos;
      }

      out.color = in.color;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      return vec4f(in.color, 1.0);
    }
  `;

  const shaderModule = device.createShaderModule({ code: shaderCode });

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [{
        arrayStride: 6 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" },
        ],
      }],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: {
      topology: "line-list", // 矢量线段图元
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth24plus",
    },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // =========================================================================
  // 4. GUI 控制与超大坐标设置
  // =========================================================================
  const settings = {
    enableRTE: true,
    worldCoordinateDistance: 1000000.0, // 距离世界原点 100 万米（典型 GIS 场景）
    cameraVibration: true,             // 模拟相机细微巡航晃动，放大对比抖动
    zoom: 15.0,
  };

  gui.title("WebGPU RTE 相对视点坐标渲染");
  gui.add(settings, "enableRTE").name("开启 RTE (防抖)");
  gui.add(settings, "worldCoordinateDistance", 0, 5000000, 10000).name("世界坐标距离(米)");
  gui.add(settings, "cameraVibration").name("微小视点扰动(凸显抖动)");
  gui.add(settings, "zoom", 3.0, 30.0, 0.5).name("观察距离");

  let depthTexture: GPUTexture | null = null;
  let animId: number;
  let time = 0;

  function frame() {
    time += 0.02;

    const dpr = window.devicePixelRatio || 1;
    const renderWidth = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const renderHeight = Math.max(1, Math.floor(canvas.clientHeight * dpr));

    if (!depthTexture || depthTexture.width !== renderWidth || depthTexture.height !== renderHeight) {
      canvas.width = renderWidth; canvas.height = renderHeight;
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [renderWidth, renderHeight],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    // =========================================================================
    // 关键核心：在 CPU 侧用 JS 64 位原生高精度浮点计算绝对坐标与相对坐标
    // =========================================================================
    // 1. 物体的实际物理大坐标 (64-bit IEEE 754)
    const objectWorldX = settings.worldCoordinateDistance;
    const objectWorldY = 0.0;
    const objectWorldZ = settings.worldCoordinateDistance;

    // 2. 摄像机物理大坐标
    const vib = settings.cameraVibration ? Math.sin(time * 3) * 0.005 : 0;
    const eyeWorldX = objectWorldX + Math.cos(time * 0.3) * settings.zoom + vib;
    const eyeWorldY = objectWorldY + settings.zoom * 0.7;
    const eyeWorldZ = objectWorldZ + Math.sin(time * 0.3) * settings.zoom;

    // 3. RTE 核心：在 CPU 端用 64 位精度计算相对向量 (Double Precision Subtraction)
    // 结果只有 ~15 米，放入 Float32 完全保留毫米级的所有精度！
    const relX = objectWorldX - eyeWorldX;
    const relY = objectWorldY - eyeWorldY;
    const relZ = objectWorldZ - eyeWorldZ;

    // 投影矩阵
    const proj = mat4Perspective((45 * Math.PI) / 180, renderWidth / renderHeight, 0.1, 500.0);

    // 传统绝对视口矩阵 (32 位浮点下计算，平移量过大)
    const viewAbsolute = mat4LookAt([eyeWorldX, eyeWorldY, eyeWorldZ], [objectWorldX, objectWorldY, objectWorldZ], [0, 1, 0]);
    const viewProjAbsolute = mat4Multiply(proj, viewAbsolute);

    // RTE 纯旋转矩阵 (将相机平移清零)
    const viewRotationOnly = mat4LookAt([0, 0, 0], [relX, relY, relZ], [0, 1, 0]);

    // 打包 Uniform 数据
    const uniformData = new ArrayBuffer(256);
    const f32 = new Float32Array(uniformData);
    const u32 = new Uint32Array(uniformData);

    f32.set(viewProjAbsolute, 0);     // 0..15 (64B)
    f32.set(proj, 16);                 // 16..31 (64B)
    f32.set(viewRotationOnly, 32);     // 32..47 (64B)

    // modelRelativeToEye (vec4f)
    f32[48] = relX; f32[49] = relY; f32[50] = relZ; f32[51] = 0.0;

    // hugeWorldOrigin (vec4f)
    f32[52] = objectWorldX; f32[53] = objectWorldY; f32[54] = objectWorldZ; f32[55] = 0.0;

    // 开关标志位
    u32[56] = settings.enableRTE ? 1 : 0;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // =========================================================================
    // 渲染管线提交
    // =========================================================================
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.08, g: 0.09, b: 0.12, a: 1.0 },
        loadOp: "clear", storeOp: "store"
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store"
      }
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(vertexCount, 1, 0, 0);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    vertexBuffer.destroy();
    uniformBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}