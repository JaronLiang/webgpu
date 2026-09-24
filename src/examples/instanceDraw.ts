// src/examples/instanceDraw.ts
import { Mat4 } from "../utils/math";
import type { SimpleGUI } from "../utils/gui";

// 简易 LookAt 视图矩阵生成器
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

export function runInstanceDraw(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // ==========================================
  // 1. 基础单体立方体几何 (包含 Pos + Normal)
  // ==========================================
  // prettier-ignore
  const vertexData = new Float32Array([
    // 前面
    -1,-1, 1, 0,0,1,   1,-1, 1, 0,0,1,   1, 1, 1, 0,0,1,  -1, 1, 1, 0,0,1,
    // 后面
    -1,-1,-1, 0,0,-1, -1, 1,-1, 0,0,-1,  1, 1,-1, 0,0,-1,  1,-1,-1, 0,0,-1,
    // 顶面
    -1, 1,-1, 0,1,0,  -1, 1, 1, 0,1,0,   1, 1, 1, 0,1,0,   1, 1,-1, 0,1,0,
    // 底面
    -1,-1,-1, 0,-1,0,  1,-1,-1, 0,-1,0,  1,-1, 1, 0,-1,0, -1,-1, 1, 0,-1,0,
    // 右面
     1,-1,-1, 1,0,0,   1, 1,-1, 1,0,0,   1, 1, 1, 1,0,0,   1,-1, 1, 1,0,0,
    // 左面
    -1,-1,-1, -1,0,0, -1,-1, 1, -1,0,0, -1, 1, 1, -1,0,0, -1, 1,-1, -1,0,0,
  ]);
  // prettier-ignore
  const indexData = new Uint16Array([
    0,1,2, 0,2,3,       4,5,6, 4,6,7,       8,9,10, 8,10,11,
    12,13,14, 12,14,15, 16,17,18, 16,18,19, 20,21,22, 20,22,23
  ]);

  const vBuffer = device.createBuffer({
    size: vertexData.byteLength, 
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vBuffer, 0, vertexData);
  
  const iBuffer = device.createBuffer({ 
    size: indexData.byteLength, 
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });

  device.queue.writeBuffer(iBuffer, 0, indexData);

  // ==========================================
  // 2. 生成 10,000 个实例数据 (Instance Buffer)
  // 每个实例 8 个 Float32 = 32 字节:
  // pos (vec3f), scale (f32), color (vec3f), rotSpeed (f32)
  // ==========================================
  const maxInstances = 10000;
  const instanceData = new Float32Array(maxInstances * 8);

  // 在 3D 空间中分布成 22 x 22 x 21 约 10,000 个网格晶格
  const gridSize = 22;
  const spacing = 2.4;
  let idx = 0;

  for (let z = 0; z < gridSize; z++) {
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        if (idx >= maxInstances) break;
        const offset = idx * 8;

        // 1. 位置 (居中排布)
        instanceData[offset + 0] = (x - gridSize / 2) * spacing;
        instanceData[offset + 1] = (y - gridSize / 2) * spacing;
        instanceData[offset + 2] = (z - gridSize / 2) * spacing;

        // 2. 个体缩放尺寸
        instanceData[offset + 3] = 0.45 + Math.random() * 0.35;

        // 3. 颜色 (基于三维坐标形成平滑霓虹渐变色)
        instanceData[offset + 4] = 0.2 + (x / gridSize) * 0.8;
        instanceData[offset + 5] = 0.3 + (y / gridSize) * 0.7;
        instanceData[offset + 6] = 0.4 + (z / gridSize) * 0.6;

        // 4. 各自独立的自转角速度
        instanceData[offset + 7] = (Math.random() - 0.5) * 3.0;

        idx++;
      }
    }
  }

  const instanceBuffer = device.createBuffer({
    size: instanceData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(instanceBuffer, 0, instanceData);

  // ==========================================
  // 3. Uniform 缓冲与深度贴图
  // ==========================================
  const uniformData = new Float32Array(20); // 16(mat4) + time(1) + baseScale(1) + pad(2)
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  let depthTexture = device.createTexture({
    size: [canvas.width || 800, canvas.height || 600],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // ==========================================
  // 4. 着色器代码 (硬件实例化读取)
  // ==========================================
  const shaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
      time: f32,
      baseScale: f32,
      pad1: f32,
      pad2: f32,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexInput {
      // 槽位 0: 顶点属性 (逐顶点)
      @location(0) pos: vec3f,
      @location(1) normal: vec3f,

      // 槽位 1: 实例属性 (逐正方体，stepMode: "instance")
      @location(2) instPos: vec3f,
      @location(3) instScale: f32,
      @location(4) instColor: vec3f,
      @location(5) instRotSpeed: f32,
    };

    struct VertexOutput {
      @builtin(position) clipPos: vec4f,
      @location(0) color: vec3f,
      @location(1) normal: vec3f,
    };

    // 沿 Y 轴与 X 轴旋转
    fn rotateY(v: vec3f, a: f32) -> vec3f {
      let c = cos(a); let s = sin(a);
      return vec3f(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
    }
    fn rotateX(v: vec3f, a: f32) -> vec3f {
      let c = cos(a); let s = sin(a);
      return vec3f(v.x, v.y * c - v.z * s, v.y * s + v.z * c);
    }

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
      var out: VertexOutput;

      // 1. 各正方体根据各自速度在 GPU 端高并发自转
      let angle = u.time * in.instRotSpeed;
      var rotatedPos = rotateY(in.pos, angle);
      rotatedPos = rotateX(rotatedPos, angle * 0.7);

      let rotatedNormal = rotateX(rotateY(in.normal, angle), angle * 0.7);

      // 2. 局部缩放并偏移到世界空间实例位置
      let finalScale = in.instScale * u.baseScale;
      let worldPos = rotatedPos * finalScale + in.instPos;

      // 3. MVP 投影
      out.clipPos = u.viewProj * vec4f(worldPos, 1.0);
      out.color = in.instColor;
      out.normal = rotatedNormal;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      // 定向光漫反射
      let lightDir = normalize(vec3f(0.6, 0.8, 0.5));
      let diff = max(dot(normalize(in.normal), lightDir), 0.0);
      let ambient = 0.25;
      let litColor = in.color * (ambient + diff * 0.75);
      return vec4f(litColor, 1.0);
    }
  `;
  const module = device.createShaderModule({ code: shaderCode });

  // ==========================================
  // 5. 渲染管线：配置 双顶点缓冲区 (Vertex + Instance)
  // ==========================================
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vs_main",
      buffers: [
        // 缓冲 0: 几何体单体顶点数据 (stepMode: "vertex")
        {
          arrayStride: 24,
          stepMode: "vertex",
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },  // pos
            { shaderLocation: 1, offset: 12, format: "float32x3" }, // normal
          ],
        },
        // 缓冲 1: 【核心】每个实例的数据 (stepMode: "instance")
        {
          arrayStride: 32,
          stepMode: "instance", // 告诉 GPU 每画完一个正方体才步进一次！
          attributes: [
            { shaderLocation: 2, offset: 0, format: "float32x3" },  // instPos
            { shaderLocation: 3, offset: 12, format: "float32" },   // instScale
            { shaderLocation: 4, offset: 16, format: "float32x3" }, // instColor
            { shaderLocation: 5, offset: 28, format: "float32" },   // instRotSpeed
          ],
        },
      ],
    },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // ==========================================
  // 6. GUI 参数注入与控制
  // ==========================================
  const params = {
    instanceCount: 10000, // 当前加载/绘制数量
    baseScale: 1.0,       // 全局正方体大小
    camSpeed: 0.3,        // 镜头环绕旋转速度
  };

  gui.add(params, "instanceCount", 100, 10000, 100).name("方块数量 (Count)");
  gui.add(params, "baseScale", 0.2, 2.0, 0.05).name("方块缩放 (Scale)");
  gui.add(params, "camSpeed", 0.0, 2.0, 0.1).name("相机公转 (Speed)");
  gui.addTextInfo("💡 纯硬件实例化渲染<br>单次 Draw Call 绘制 10000 个独立自转实体！");

  // ==========================================
  // 7. 动态渲染循环
  // ==========================================
  let animId: number;
  let camAngle = 0;

  function frame(timestamp: number) {
    if (depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    const t = timestamp * 0.001;
    camAngle += params.camSpeed * 0.01;

    // 相机绕外围以椭圆轨道漫游俯视
    const camDist = 55.0;
    const eyeX = Math.cos(camAngle) * camDist;
    const eyeY = 28.0;
    const eyeZ = Math.sin(camAngle) * camDist;

    const view = createLookAtMatrix([eyeX, eyeY, eyeZ], [0, 0, 0], [0, 1, 0]);
    const aspect = (canvas.width || 800) / (canvas.height || 600);
    const proj = Mat4.perspective((60 * Math.PI) / 180, aspect, 0.1, 300);
    const viewProj = Mat4.multiply(proj, view);

    // 填充 Uniform 矩阵与参数
    uniformData.set(viewProj, 0);
    uniformData[16] = t;                 // time
    uniformData[17] = params.baseScale;  // baseScale
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.06, g: 0.06, b: 0.09, a: 1.0 },
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vBuffer);        // 单体几何网格
    pass.setVertexBuffer(1, instanceBuffer); // 【关键】：实例属性缓冲
    pass.setIndexBuffer(iBuffer, "uint16");

    // 【核心绘制命令】：第 2 个参数即为动态绘制的实例数！
    // 拖动 GUI 滑块直接无缝改变此数值，无需重建任何 Buffer
    pass.drawIndexed(36, params.instanceCount, 0, 0, 0);

    pass.end();
    device.queue.submit([encoder.finish()]);

    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  // 释放资源
  return () => {
    cancelAnimationFrame(animId);
    vBuffer.destroy();
    iBuffer.destroy();
    instanceBuffer.destroy();
    uniformBuffer.destroy();
    depthTexture.destroy();
  };
}