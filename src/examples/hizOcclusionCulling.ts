// src/examples/hizOcclusionCulling.ts
import { Mat4 } from "../utils/math";
import type { SimpleGUI } from "../utils/gui";

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

function updateFrustumPlanes(m: Float32Array, out: Float32Array): void {
  const r0 = [m[0], m[4], m[8], m[12]];
  const r1 = [m[1], m[5], m[9], m[13]];
  const r2 = [m[2], m[6], m[10], m[14]];
  const r3 = [m[3], m[7], m[11], m[15]];

  const raw: number[][] = [
    [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]], // Left
    [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]], // Right
    [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]], // Bottom
    [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]], // Top
    [r2[0],         r2[1],         r2[2],         r2[3]        ], // Near
    [r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]], // Far
  ];

  for (let i = 0; i < 6; i++) {
    const p = raw[i];
    const len = Math.hypot(p[0], p[1], p[2]) || 1;
    out[i * 4 + 0] = p[0] / len;
    out[i * 4 + 1] = p[1] / len;
    out[i * 4 + 2] = p[2] / len;
    out[i * 4 + 3] = p[3] / len;
  }
}

export function runHizOcclusionCulling(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  const TOTAL_OBJECTS = 24000;
  const HIZ_SIZE = 256;      // Hi-Z 贴图分辨率
  const HIZ_MIP_LEVELS = 9;  // 256 -> 1x1 共 9 级

  // 1. 标准立方体网格数据 (24 顶点，36 索引)
  // prettier-ignore
  const cubeVertices = new Float32Array([
    -0.5,-0.5, 0.5, 0,0,1,   0.5,-0.5, 0.5, 0,0,1,   0.5, 0.5, 0.5, 0,0,1,  -0.5, 0.5, 0.5, 0,0,1,
    -0.5,-0.5,-0.5, 0,0,-1, -0.5, 0.5,-0.5, 0,0,-1,  0.5, 0.5,-0.5, 0,0,-1,  0.5,-0.5,-0.5, 0,0,-1,
    -0.5, 0.5,-0.5, 0,1,0,  -0.5, 0.5, 0.5, 0,1,0,   0.5, 0.5, 0.5, 0,1,0,   0.5, 0.5,-0.5, 0,1,0,
    -0.5,-0.5,-0.5, 0,-1,0,  0.5,-0.5,-0.5, 0,-1,0,  0.5,-0.5, 0.5, 0,-1,0, -0.5,-0.5, 0.5, 0,-1,0,
     0.5,-0.5,-0.5, 1,0,0,   0.5, 0.5,-0.5, 1,0,0,   0.5, 0.5, 0.5, 1,0,0,   0.5,-0.5, 0.5, 1,0,0,
    -0.5,-0.5,-0.5,-1,0,0,  -0.5,-0.5, 0.5,-1,0,0,  -0.5, 0.5, 0.5,-1,0,0,  -0.5, 0.5,-0.5,-1,0,0,
  ]);
  // prettier-ignore
  const cubeIndices = new Uint16Array([
    0,1,2, 0,2,3, 4,5,6, 4,6,7, 8,9,10, 8,10,11,
    12,13,14, 12,14,15, 16,17,18, 16,18,19, 20,21,22, 20,22,23
  ]);

  const cubeVBuffer = device.createBuffer({
    size: cubeVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(cubeVBuffer, 0, cubeVertices);

  const cubeIBuffer = device.createBuffer({
    size: cubeIndices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(cubeIBuffer, 0, cubeIndices);

  // 2. 前景左右两面巨型遮挡黑墙（中央留下一个宽度为 6 的门洞缝隙）
  // prettier-ignore
  const wallVertices = new Float32Array([
    // 左墙：X 从 -35 到 -3，高 22，Z 在 -5
    -35, -2, -5,  0,0,1,    -3, -2, -5,  0,0,1,    -3, 20, -5,  0,0,1,   -35, 20, -5,  0,0,1,
    // 右墙：X 从 3 到 35，高 22，Z 在 -5
      3, -2, -5,  0,0,1,    35, -2, -5,  0,0,1,    35, 20, -5,  0,0,1,     3, 20, -5,  0,0,1,
  ]);
  const wallIndices = new Uint16Array([
    0, 1, 2,  0, 2, 3,
    4, 5, 6,  4, 6, 7
  ]);

  const wallVBuffer = device.createBuffer({
    size: wallVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(wallVBuffer, 0, wallVertices);

  const wallIBuffer = device.createBuffer({
    size: wallIndices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(wallIBuffer, 0, wallIndices);

  // 3. 生成 24,000 个待剔除方块物体数据 (全部分布在 Z = -8 到 -90 的大墙后方)
  const objectData = new Float32Array(TOTAL_OBJECTS * 8);
  for (let i = 0; i < TOTAL_OBJECTS; i++) {
    const idx = i * 8;
    const x = (Math.random() - 0.5) * 68.0;
    const y = Math.random() * 18.0 - 1.0;
    const z = -8.0 - Math.random() * 80.0; // 紧随大墙后方
    const scale = 0.35 + Math.random() * 0.65;
    const radius = scale * 0.866;

    objectData[idx + 0] = x;
    objectData[idx + 1] = y;
    objectData[idx + 2] = z;
    objectData[idx + 3] = radius;
    // 彩色
    objectData[idx + 4] = 0.2 + Math.random() * 0.8;
    objectData[idx + 5] = 0.2 + Math.random() * 0.8;
    objectData[idx + 6] = 0.2 + Math.random() * 0.8;
    objectData[idx + 7] = scale;
  }

  const objectsBuffer = device.createBuffer({
    size: objectData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(objectsBuffer, 0, objectData);

  // 4. 间接参数与可见列表
  const visibleIndicesBuffer = device.createBuffer({
    size: TOTAL_OBJECTS * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  const indirectDrawBuffer = device.createBuffer({
    size: 20,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const countReadbackBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // 5. Hi-Z 纹理与深度缓冲
  const hizTexture = device.createTexture({
    size: [HIZ_SIZE, HIZ_SIZE, 1],
    mipLevelCount: HIZ_MIP_LEVELS,
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
  });

  const occluderDepthTexture = device.createTexture({
    size: [HIZ_SIZE, HIZ_SIZE, 1],
    format: "depth32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const occluderDepthView = occluderDepthTexture.createView();

  let mainDepthTexture = device.createTexture({
    size: [canvas.width || 800, canvas.height || 600],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Uniform Buffer:
  // viewProj (64) + 6 planes (96) + camPos (12) + totalCount (4) + cullMode (4) + pad (12) = 192 bytes
  const uniformBuffer = device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ========================================================
  // 6. Compute Pipeline A: 分离两个独立 WGSL 模块，杜绝绑定冲突
  // ========================================================
  // 6.1 首层 Mip 拷贝模块
  const initMip0Shader = `
    @group(0) @binding(0) var depthTex: texture_depth_2d;
    @group(0) @binding(1) var outMip0: texture_storage_2d<r32float, write>;

    @compute @workgroup_size(8, 8)
    fn cs_init_mip0(@builtin(global_invocation_id) gid: vec3u) {
      if (gid.x >= 256u || gid.y >= 256u) { return; }
      let coord = vec2i(gid.xy);
      let d = textureLoad(depthTex, coord, 0);
      textureStore(outMip0, coord, vec4f(d, 0.0, 0.0, 0.0));
    }
  `;
  const initMip0Module = device.createShaderModule({ code: initMip0Shader });
  const initMip0Pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: initMip0Module, entryPoint: "cs_init_mip0" },
  });

  const hizMip0View = hizTexture.createView({ baseMipLevel: 0, mipLevelCount: 1 });
  const initMip0BindGroup = device.createBindGroup({
    layout: initMip0Pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: occluderDepthView },
      { binding: 1, resource: hizMip0View },
    ],
  });

  // 6.2 逐级下采样模块
  const downsampleShader = `
    @group(0) @binding(0) var prevMip: texture_2d<f32>;
    @group(0) @binding(1) var nextMip: texture_storage_2d<r32float, write>;

    @compute @workgroup_size(8, 8)
    fn cs_downsample(@builtin(global_invocation_id) gid: vec3u) {
      let dstCoord = vec2i(gid.xy);
      let srcCoord = dstCoord * 2;

      let d0 = textureLoad(prevMip, srcCoord + vec2i(0, 0), 0).r;
      let d1 = textureLoad(prevMip, srcCoord + vec2i(1, 0), 0).r;
      let d2 = textureLoad(prevMip, srcCoord + vec2i(0, 1), 0).r;
      let d3 = textureLoad(prevMip, srcCoord + vec2i(1, 1), 0).r;

      // 保守遮挡检测：取 2x2 像素中的最大深度 (Max Depth)
      let maxDepth = max(max(d0, d1), max(d2, d3));
      textureStore(nextMip, dstCoord, vec4f(maxDepth, 0.0, 0.0, 0.0));
    }
  `;
  const downsampleModule = device.createShaderModule({ code: downsampleShader });
  const downsamplePipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: downsampleModule, entryPoint: "cs_downsample" },
  });

  const downsampleBindGroups: GPUBindGroup[] = [];
  for (let m = 0; m < HIZ_MIP_LEVELS - 1; m++) {
    const srcView = hizTexture.createView({ baseMipLevel: m, mipLevelCount: 1 });
    const dstView = hizTexture.createView({ baseMipLevel: m + 1, mipLevelCount: 1 });
    downsampleBindGroups.push(
      device.createBindGroup({
        layout: downsamplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: srcView },
          { binding: 1, resource: dstView },
        ],
      })
    );
  }

  // ========================================================
  // 7. Compute Pipeline B: 精确表面深度的 Hi-Z 遮挡剔除
  // ========================================================
  const cullingShader = `
    struct Uniforms {
      viewProj: mat4x4f,
      planes: array<vec4f, 6>,
      camPos: vec3f,
      totalCount: u32,
      cullMode: u32, // 0: 关, 1: 仅视锥, 2: 视锥 + Hi-Z
      pad0: u32,
      pad1: u32,
      pad2: u32,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct ObjectData {
      pos: vec3f, radius: f32, col: vec3f, scale: f32,
    };
    @group(0) @binding(1) var<storage, read> objects: array<ObjectData>;

    struct DrawIndirectArgs {
      indexCount: u32,
      instanceCount: atomic<u32>,
      firstIndex: u32,
      baseVertex: u32,
      firstInstance: u32,
    };
    @group(0) @binding(2) var<storage, read_write> drawArgs: DrawIndirectArgs;
    @group(0) @binding(3) var<storage, read_write> visibleIndices: array<u32>;
    @group(0) @binding(4) var hizTex: texture_2d<f32>;

    @compute @workgroup_size(64)
    fn cs_cull(@builtin(global_invocation_id) gid: vec3u) {
      let index = gid.x;
      if (index >= u.totalCount) { return; }

      let obj = objects[index];
      var isVisible = true;

      // 阶段 1：视锥体六平面剔除
      if (u.cullMode >= 1u) {
        for (var i = 0u; i < 6u; i = i + 1u) {
          let plane = u.planes[i];
          let dist = dot(plane.xyz, obj.pos) + plane.w;
          if (dist < -obj.radius) {
            isVisible = false;
            break;
          }
        }
      }

      // 阶段 2：Hi-Z 遮挡检测
      if (isVisible && u.cullMode == 2u) {
        let toCam = normalize(u.camPos - obj.pos);
        // 物体表面距离相机最近的点
        let nearPt = obj.pos + toCam * obj.radius;
        let clipNear = u.viewProj * vec4f(nearPt, 1.0);

        if (clipNear.w > 0.05) {
          let ndcCenter = clipNear.xyz / clipNear.w;

          // 计算包围盒屏幕空间半径与尺寸
          let screenRadius = (obj.radius / clipNear.w) * 1.2;
          let minUv = clamp(vec2f(ndcCenter.x - screenRadius, -ndcCenter.y - screenRadius) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));
          let maxUv = clamp(vec2f(ndcCenter.x + screenRadius, -ndcCenter.y + screenRadius) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));

          let bboxPixelSize = max(maxUv.x - minUv.x, maxUv.y - minUv.y) * 256.0;
          // 自适应挑选覆盖度为 1~2 个像素的 Mip 等级
          let mip = clamp(i32(floor(log2(max(bboxPixelSize, 1.0)))), 0, 4);
          let mipDim = max(256 >> u32(mip), 1);

          let c0 = clamp(vec2i(minUv * f32(mipDim)), vec2i(0), vec2i(mipDim - 1));
          let c1 = clamp(vec2i(maxUv * f32(mipDim)), vec2i(0), vec2i(mipDim - 1));

          // 采样该物体屏幕区域覆盖的 4 个顶角深度
          let d0 = textureLoad(hizTex, vec2i(c0.x, c0.y), mip).r;
          let d1 = textureLoad(hizTex, vec2i(c1.x, c0.y), mip).r;
          let d2 = textureLoad(hizTex, vec2i(c0.x, c1.y), mip).r;
          let d3 = textureLoad(hizTex, vec2i(c1.x, c1.y), mip).r;
          let occluderMaxDepth = max(max(d0, d1), max(d2, d3));

          let objectDepth = ndcCenter.z;
          // 若物体最近深度比遮挡物最远还要深，判定为完全被遮挡
          if (objectDepth > occluderMaxDepth + 0.001) {
            isVisible = false;
          }
        }
      }

      if (isVisible) {
        let slot = atomicAdd(&drawArgs.instanceCount, 1u);
        visibleIndices[slot] = index;
      }
    }
  `;

  const cullingModule = device.createShaderModule({ code: cullingShader });
  const cCullPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: cullingModule, entryPoint: "cs_cull" },
  });

  const fullHizView = hizTexture.createView();
  const cCullBindGroup = device.createBindGroup({
    layout: cCullPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: objectsBuffer } },
      { binding: 2, resource: { buffer: indirectDrawBuffer } },
      { binding: 3, resource: { buffer: visibleIndicesBuffer } },
      { binding: 4, resource: fullHizView },
    ],
  });

  // ========================================================
  // 8. Render Pipelines: 深度预处理 + 主场景渲染
  // ========================================================
  const sceneShader = `
    struct Uniforms {
      viewProj: mat4x4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct ObjectData {
      pos: vec3f, radius: f32, col: vec3f, scale: f32,
    };
    @group(0) @binding(1) var<storage, read> objects: array<ObjectData>;
    @group(0) @binding(2) var<storage, read> visibleIndices: array<u32>;

    struct VertexInput {
      @location(0) pos: vec3f,
      @location(1) norm: vec3f,
    };
    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) col: vec3f,
      @location(1) norm: vec3f,
    };

    @vertex
    fn vs_cube(in: VertexInput, @builtin(instance_index) instanceIdx: u32) -> VertexOut {
      let objIdx = visibleIndices[instanceIdx];
      let obj = objects[objIdx];
      let worldPos = in.pos * obj.scale + obj.pos;
      var out: VertexOut;
      out.pos = u.viewProj * vec4f(worldPos, 1.0);
      out.col = obj.col;
      out.norm = in.norm;
      return out;
    }

    @vertex
    fn vs_wall(in: VertexInput) -> VertexOut {
      var out: VertexOut;
      out.pos = u.viewProj * vec4f(in.pos, 1.0);
      out.col = vec3f(0.12, 0.14, 0.18);
      out.norm = in.norm;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4f {
      let lightDir = normalize(vec3f(0.5, 0.9, 0.4));
      let diff = max(dot(in.norm, lightDir), 0.2);
      return vec4f(in.col * diff, 1.0);
    }
  `;

  const sceneModule = device.createShaderModule({ code: sceneShader });

  const wallDepthPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: sceneModule,
      entryPoint: "vs_wall",
      buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x3" }] }],
    },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth32float" },
  });

  const cubeRenderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: sceneModule,
      entryPoint: "vs_cube",
      buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x3" }] }],
    },
    fragment: { module: sceneModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const wallRenderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: sceneModule,
      entryPoint: "vs_wall",
      buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x3" }] }],
    },
    fragment: { module: sceneModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const wallDepthBindGroup = device.createBindGroup({
    layout: wallDepthPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const wallRenderBindGroup = device.createBindGroup({
    layout: wallRenderPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const cubeRenderBindGroup = device.createBindGroup({
    layout: cubeRenderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: objectsBuffer } },
      { binding: 2, resource: { buffer: visibleIndicesBuffer } },
    ],
  });

  // 9. GUI 控制
  const params = {
    cullMode: 2,
    freezeFrustum: 0,
  };
  const statsObj = { visibleCount: TOTAL_OBJECTS };

  const defaultCamera = { distance: 36.0, theta: 0, phi: 6, panX: 0.0, panY: 6.0, fov: 60 };
  const camera = { ...defaultCamera };

  gui.add(params, "cullMode", 0, 2, 1).name("剔除模式 (0:关 1:视锥 2:HiZ遮挡)");
  gui.add(params, "freezeFrustum", 0, 1, 1).name("冻结剔除测试 (0:实时 1:冻结)");
  gui.add(statsObj, "visibleCount", 0, TOTAL_OBJECTS, 1).name("最终绘制实例数");

  gui.addButton("切换为: 仅视锥剔除 (无遮挡剔除)", () => {
    params.cullMode = 1;
    gui.updateDisplay();
  });
  gui.addButton("切换为: 开启 Hi-Z 遮挡剔除 (暴降绘制数)", () => {
    params.cullMode = 2;
    gui.updateDisplay();
  });
  gui.addButton("冻结视锥并升起相机俯瞰 (验证墙后镂空)", () => {
    params.freezeFrustum = 1;
    camera.phi = 58;
    camera.distance = 65;
    gui.updateDisplay();
  });
  gui.addButton("恢复正视视角", () => {
    params.freezeFrustum = 0;
    Object.assign(camera, defaultCamera);
    gui.updateDisplay();
  });

  gui.add(camera, "distance", 10.0, 180.0, 1.0).name("相机距离 (Dist)");
  gui.add(camera, "phi", -80, 80, 1).name("仰角 (Pitch)");
  gui.add(camera, "theta", -180, 180, 1).name("偏航角 (Yaw)");

  gui.addTextInfo(
    "<b>【Hi-Z 核心效果验证】：</b><br>" +
    "• <b>模式 1 (仅视锥)：</b> 绘制数高达 <b>17,000 ~ 20,000</b> 个（大墙后的方块全在被盲画）。<br>" +
    "• <b>模式 2 (Hi-Z 遮挡)：</b> 绘制数骤降至 <b>1,500 ~ 3,000</b> 个！仅正中间门洞可见的方块被送入渲染！<br>" +
    "• <b>点击『冻结并升起相机俯瞰』：</b> 视角拉高后，可直观看到大墙后方被挖出了极其壮观的<b>三角形空洞阴影</b>！"
  );

  // 10. 交互
  let isDragging = false;
  let dragButton = 0;
  let lastX = 0, lastY = 0;

  const onPointerDown = (e: PointerEvent) => {
    isDragging = true;
    dragButton = e.shiftKey ? 2 : e.button;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    if (dragButton === 0) {
      camera.theta -= dx * 0.4;
      camera.phi = Math.max(-80, Math.min(80, camera.phi + dy * 0.4));
    } else if (dragButton === 2) {
      const f = camera.distance * 0.0015;
      camera.panX -= dx * f;
      camera.panY += dy * f;
    }
    gui.updateDisplay();
  };
  const onPointerUp = (e: PointerEvent) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  };
  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    camera.distance = Math.max(10.0, Math.min(180.0, camera.distance + e.deltaY * 0.05));
    gui.updateDisplay();
  };

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  // 11. 渲染主循环
  let animId: number;
  const uniformData = new Float32Array(64);
  const frozenPlanes = new Float32Array(24);
  const frozenCamPos = new Float32Array(3);
  let isFrustumInitialized = false;

  const resetArgs = new Uint32Array([36, 0, 0, 0, 0]);

  // 【核心修复】：互斥标志，彻底杜绝 mapAsync 与 submit 冲突导致的闪烁
  let isReading = false;
  let canCopyReadback = true;

  async function pollVisibleCount() {
    if (isReading || countReadbackBuffer.mapState !== "unmapped") return;
    isReading = true;
    canCopyReadback = false; // 映射期间严禁向该 Buffer 写入 copy 指令
    try {
      await countReadbackBuffer.mapAsync(GPUMapMode.READ);
      const arr = new Uint32Array(countReadbackBuffer.getMappedRange());
      statsObj.visibleCount = arr[0];
      countReadbackBuffer.unmap();
      gui.updateDisplay();
    } catch {}
    canCopyReadback = true;
    isReading = false;
  }

  let frameCount = 0;

  function frame() {
    if (mainDepthTexture.width !== canvas.width || mainDepthTexture.height !== canvas.height) {
      mainDepthTexture.destroy();
      mainDepthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eyeX = camera.panX + camera.distance * Math.cos(radPhi) * Math.sin(radTheta);
    const eyeY = camera.panY + camera.distance * Math.sin(radPhi);
    const eyeZ = camera.distance * Math.cos(radPhi) * Math.cos(radTheta);

    const viewMatrix = createLookAtMatrix([eyeX, eyeY, eyeZ], [camera.panX, camera.panY, -20.0], [0, 1, 0]);
    const aspect = (canvas.width || 800) / (canvas.height || 600);
    const projMatrix = Mat4.perspective((camera.fov * Math.PI) / 180, aspect, 0.1, 300);
    const viewProj = Mat4.multiply(projMatrix, viewMatrix);

    if (params.freezeFrustum === 0 || !isFrustumInitialized) {
      updateFrustumPlanes(viewProj, frozenPlanes);
      frozenCamPos[0] = eyeX;
      frozenCamPos[1] = eyeY;
      frozenCamPos[2] = eyeZ;
      isFrustumInitialized = true;
    }

    // 填充 Uniform 变量
    uniformData.set(viewProj, 0);
    uniformData.set(frozenPlanes, 16);
    uniformData[40] = frozenCamPos[0];
    uniformData[41] = frozenCamPos[1];
    uniformData[42] = frozenCamPos[2];

    const u32View = new Uint32Array(uniformData.buffer);
    u32View[43] = TOTAL_OBJECTS;
    u32View[44] = params.cullMode;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    device.queue.writeBuffer(indirectDrawBuffer, 0, resetArgs);

    const encoder = device.createCommandEncoder();

    // ========================================================
    // 阶段 1: 渲染大墙 Depth 到 256x256 深度贴图
    // ========================================================
    const depthPass = encoder.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: occluderDepthView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    depthPass.setPipeline(wallDepthPipeline);
    depthPass.setBindGroup(0, wallDepthBindGroup);
    depthPass.setVertexBuffer(0, wallVBuffer);
    depthPass.setIndexBuffer(wallIBuffer, "uint16");
    depthPass.drawIndexed(12); // 两面大墙
    depthPass.end();

    // ========================================================
    // 阶段 2: 构建 8 级 Hi-Z 深度金字塔
    // ========================================================
    const hizPass = encoder.beginComputePass();
    hizPass.setPipeline(initMip0Pipeline);
    hizPass.setBindGroup(0, initMip0BindGroup);
    hizPass.dispatchWorkgroups(HIZ_SIZE / 8, HIZ_SIZE / 8);

    hizPass.setPipeline(downsamplePipeline);
    for (let m = 0; m < HIZ_MIP_LEVELS - 1; m++) {
      const mipDim = Math.max(HIZ_SIZE >> (m + 1), 1);
      hizPass.setBindGroup(0, downsampleBindGroups[m]);
      hizPass.dispatchWorkgroups(Math.max(Math.ceil(mipDim / 8), 1), Math.max(Math.ceil(mipDim / 8), 1));
    }
    hizPass.end();

    // ========================================================
    // 阶段 3: 执行 GPU 视锥 + Hi-Z 遮挡剔除 Compute Pass
    // ========================================================
    const cPass = encoder.beginComputePass();
    cPass.setPipeline(cCullPipeline);
    cPass.setBindGroup(0, cCullBindGroup);
    cPass.dispatchWorkgroups(Math.ceil(TOTAL_OBJECTS / 64));
    cPass.end();

    // 仅在缓冲区未被异步读取时拷贝，杜绝冲突
    if (canCopyReadback) {
      encoder.copyBufferToBuffer(indirectDrawBuffer, 4, countReadbackBuffer, 0, 4);
    }

    // ========================================================
    // 阶段 4: 主场景渲染
    // ========================================================
    const rPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.08, g: 0.09, b: 0.12, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: mainDepthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    // 4.1 绘制前景遮挡墙
    rPass.setPipeline(wallRenderPipeline);
    rPass.setBindGroup(0, wallRenderBindGroup);
    rPass.setVertexBuffer(0, wallVBuffer);
    rPass.setIndexBuffer(wallIBuffer, "uint16");
    rPass.drawIndexed(12);

    // 4.2 间接绘制所有可见方块
    rPass.setPipeline(cubeRenderPipeline);
    rPass.setBindGroup(0, cubeRenderBindGroup);
    rPass.setVertexBuffer(0, cubeVBuffer);
    rPass.setIndexBuffer(cubeIBuffer, "uint16");
    rPass.drawIndexedIndirect(indirectDrawBuffer, 0);

    rPass.end();

    device.queue.submit([encoder.finish()]);

    frameCount++;
    if (frameCount % 12 === 0) {
      pollVisibleCount();
    }

    animId = requestAnimationFrame(frame);
  }

  frame();

  // 12. 资源清理
  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);
    cubeVBuffer.destroy();
    cubeIBuffer.destroy();
    wallVBuffer.destroy();
    wallIBuffer.destroy();
    objectsBuffer.destroy();
    visibleIndicesBuffer.destroy();
    indirectDrawBuffer.destroy();
    countReadbackBuffer.destroy();
    hizTexture.destroy();
    occluderDepthTexture.destroy();
    mainDepthTexture.destroy();
    uniformBuffer.destroy();
  };
}