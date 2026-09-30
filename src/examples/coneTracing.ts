/// <reference types="@webgpu/types" />
import { SimpleGUI } from "../utils/gui";

export async function runConeTracing(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui?: SimpleGUI
) {
  // 1. 3D 体素网格分辨率设定 (64x64x64, 6 级 Mipmap)
  const VOXEL_RES = 64;
  const MIP_LEVELS = 6;

  // 创建 3D 浮点体积纹理
  const voxelTexture = device.createTexture({
    size: [VOXEL_RES, VOXEL_RES, VOXEL_RES],
    dimension: "3d",
    format: "rgba16float",
    mipLevelCount: MIP_LEVELS,
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST,
  });

  const voxelSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    mipmapFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
    addressModeW: "clamp-to-edge",
  });

  // 2. 交互控制参数
  const params = {
    roughness: 0.12,          // 反射锥体张角（粗糙度）
    lightRadius: 0.15,        // 面积光源半径（控制软阴影半影宽度）
    shadowIntensity: 0.85,    // 软阴影强度
    aoIntensity: 1.0,         // 环境遮蔽强度
    cameraTheta: 0.45,
    cameraPhi: 0.35,
    cameraDistance: 4.2,
  };

  if (gui) {
    const g = gui as any;
    if (typeof g.add === "function") {
      g.add(params, "roughness", 0.01, 1.0, 0.01).name?.("反射粗糙度 (Cone张角)");
      g.add(params, "lightRadius", 0.01, 0.4, 0.01).name?.("面积光源半径 (软阴影半影)");
      g.add(params, "shadowIntensity", 0.0, 1.0, 0.05).name?.("阴影强度");
      g.add(params, "aoIntensity", 0.0, 2.0, 0.1).name?.("体素遮蔽 (AO)");
    }
  }

  const uniformBufferSize = 256;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // Pass 1: 体素化着色器 (生成地面、多个几何立柱与中央反射球的 3D 体素)
  // =========================================================================
  const voxelizeShader = `
    @group(0) @binding(0) var voxelGrid: texture_storage_3d<rgba16float, write>;

    @compute @workgroup_size(4, 4, 4)
    fn main(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= 64u || id.y >= 64u || id.z >= 64u) { return; }

      let p = (vec3f(id) + 0.5) / 64.0 * 2.0 - 1.0;
      var color = vec3f(0.0);
      var opacity = 0.0;

      // 遮挡体 1: 青色高立柱 (注入阻挡不透明度)
      let c1Min = vec3f(-0.7, -1.0, -0.6);
      let c1Max = vec3f(-0.35, 0.1, -0.25);
      if (all(p >= c1Min) && all(p <= c1Max)) {
        color = vec3f(0.1, 0.8, 0.8);
        opacity = 1.0;
      }

      // 遮挡体 2: 橙色矮立柱 (注入阻挡不透明度)
      let c2Min = vec3f(0.35, -1.0, 0.2);
      let c2Max = vec3f(0.7, -0.2, 0.55);
      if (all(p >= c2Min) && all(p <= c2Max)) {
        color = vec3f(0.95, 0.55, 0.1);
        opacity = 1.0;
      }

      // 遮挡体 3: 中央高光球体 (注入阻挡不透明度)
      let sCenter = vec3f(0.0, -0.47, -0.1);
      if (distance(p, sCenter) < 0.45) {
        color = vec3f(0.95, 0.95, 0.98);
        opacity = 1.0;
      }

      // 地面仅在最底层保留微弱底色反射，不产生空中遮挡
      if (p.y < -0.96) {
        color = vec3f(0.8, 0.8, 0.85);
        opacity = 0.0; // 设为 0，防止地面自己遮挡光线
      }

      textureStore(voxelGrid, id, vec4f(color, opacity));
    }
  `;

  // =========================================================================
  // Pass 2: 3D 体素金字塔下采样 (构建 Mipmaps)
  // =========================================================================
  const downsampleShader = `
    @group(0) @binding(0) var srcMip: texture_3d<f32>;
    @group(0) @binding(1) var dstMip: texture_storage_3d<rgba16float, write>;

    @compute @workgroup_size(4, 4, 4)
    fn main(@builtin(global_invocation_id) id: vec3u) {
      let dstDim = textureDimensions(dstMip);
      if (id.x >= dstDim.x || id.y >= dstDim.y || id.z >= dstDim.z) { return; }

      let srcCoord = id * 2u;
      var sum = vec4f(0.0);
      for (var z = 0u; z < 2u; z++) {
        for (var y = 0u; y < 2u; y++) {
          for (var x = 0u; x < 2u; x++) {
            sum += textureLoad(srcMip, srcCoord + vec3u(x, y, z), 0);
          }
        }
      }
      textureStore(dstMip, id, sum / 8.0);
    }
  `;

  // =========================================================================
  // Pass 3: 渲染着色器 (执行锥体软阴影、锥体粗糙度反射与体素 AO)
  // =========================================================================
  const renderShader = `
    struct Uniforms {
      viewProj: mat4x4f,
      cameraPos: vec4f,
      lightPos: vec4f,
      params: vec4f, // x: roughness, y: lightRadius, z: shadowIntensity, w: aoIntensity
    };

    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var voxelTex: texture_3d<f32>;
    @group(0) @binding(2) var voxelSmp: sampler;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) albedo: vec3f,
      @location(3) matType: f32, // 0: 漫反射, 1: 镜面球体
    };

    @vertex
    fn vsMain(
      @location(0) pos: vec3f,
      @location(1) norm: vec3f,
      @location(2) col: vec3f,
      @location(3) mat: f32
    ) -> VertexOut {
      var out: VertexOut;
      out.worldPos = pos;
      out.normal = normalize(norm);
      out.albedo = col;
      out.matType = mat;
      out.pos = u.viewProj * vec4f(pos, 1.0);
      return out;
    }

    // 🌟【核心算法】：三维体素锥体步进追踪函数
    fn traceCone(origin: vec3f, dir: vec3f, tanHalfAngle: f32, maxDist: f32) -> vec4f {
      var accum = vec4f(0.0);
      let voxelSize = 2.0 / 64.0;
      var dist = voxelSize * 2.5; // 自相交安全偏移

      while (dist < maxDist && accum.a < 0.98) {
        let p = origin + dir * dist;
        let uvw = p * 0.5 + 0.5;

        // 超出场景空间退出
        if (any(uvw < vec3f(0.01)) || any(uvw > vec3f(0.99))) { break; }

        // 计算当前步长下的锥形圆截面直径
        let diameter = 2.0 * tanHalfAngle * dist;
        // 动态推导并采样对应的 3D Mipmap 层级
        let mipLevel = max(0.0, log2(max(diameter / voxelSize, 1.0)));

        let sampleVal = textureSampleLevel(voxelTex, voxelSmp, uvw, mipLevel);
        let a = sampleVal.a;

        // 前向透明度混合累加
        accum += (1.0 - accum.a) * vec4f(sampleVal.rgb * a, a);
        dist += max(voxelSize * 0.5, diameter * 0.35);
      }
      return accum;
    }

    // 稳健的正交基底生成函数
    fn getOrthoBasis(n: vec3f) -> mat3x3f {
      let up = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.y) > 0.85);
      let tangent = normalize(cross(up, n));
      let bitangent = cross(n, tangent);
      return mat3x3f(tangent, bitangent, n);
    }

   @fragment
    fn fsMain(in: VertexOut) -> @location(0) vec4f {
      let N = normalize(in.normal);
      let V = normalize(u.cameraPos.xyz - in.worldPos);
      let L = normalize(u.lightPos.xyz - in.worldPos);
      let distToLight = length(u.lightPos.xyz - in.worldPos);

      // 🌟【关键修复 1】：光追标准 Normal Bias，彻底消除自身体素相交产生的黑框
      let voxelSize = 2.0 / 64.0;
      let surfacePos = in.worldPos + N * (voxelSize * 1.6);

      // -------------------------------------------------------------
      // 1. 锥体软阴影 (从抬高后的 surfacePos 发射)
      // -------------------------------------------------------------
      let shadowConeAngle = u.params.y;
      let shadowTrace = traceCone(surfacePos, L, shadowConeAngle, distToLight);
      let shadow = clamp(1.0 - shadowTrace.a * u.params.z, 0.08, 1.0);

      let NdotL = max(dot(N, L), 0.0);
      let directLight = in.albedo * vec3f(3.5) * NdotL * shadow;

      // -------------------------------------------------------------
      // 2. 锥体体素环境遮蔽 (AO)
      // -------------------------------------------------------------
      let TBN = getOrthoBasis(N);
      let aoDirs = array<vec3f, 4>(
        vec3f(0.0, 0.0, 1.0),
        vec3f(0.707, 0.0, 0.707),
        vec3f(-0.707, 0.0, 0.707),
        vec3f(0.0, 0.707, 0.707)
      );
      var occ = 0.0;
      for (var i = 0; i < 4; i++) {
        let dir = normalize(TBN * aoDirs[i]);
        occ += traceCone(surfacePos, dir, 0.577, 1.2).a;
      }
      let ao = clamp(1.0 - (occ / 4.0) * u.params.w, 0.05, 1.0);

      // -------------------------------------------------------------
      // 3. 粗糙度镜面反射
      // -------------------------------------------------------------
      var specularColor = vec3f(0.0);
      if (in.matType > 0.5) {
        let R = reflect(-V, N);
        let specTan = tan(max(u.params.x, 0.02) * 0.5);
        let specTrace = traceCone(surfacePos, R, specTan, 3.5);
        specularColor = specTrace.rgb * 1.5;
      }

      var finalColor = (directLight * ao) + specularColor;
      finalColor = finalColor / (finalColor + vec3f(1.0));
      return vec4f(finalColor, 1.0);
    }
  `;

  // =========================================================================
  // 4. 构建场景网格几何体 (地面 + 两个立柱 + 中央平滑球体)
  // =========================================================================
  const vertices: number[] = [];

  function addQuad(p1: number[], p2: number[], p3: number[], p4: number[], normal: number[], col: number[], mat = 0) {
    const pushV = (p: number[]) => {
      vertices.push(p[0], p[1], p[2], normal[0], normal[1], normal[2], col[0], col[1], col[2], mat);
    };
    pushV(p1); pushV(p2); pushV(p3);
    pushV(p1); pushV(p3); pushV(p4);
  }

  function addBox(min: number[], max: number[], col: number[], mat = 0) {
    addQuad([min[0],min[1],max[2]], [max[0],min[1],max[2]], [max[0],max[1],max[2]], [min[0],max[1],max[2]], [0,0,1], col, mat);
    addQuad([max[0],min[1],min[2]], [min[0],min[1],min[2]], [min[0],max[1],min[2]], [max[0],max[1],min[2]], [0,0,-1], col, mat);
    addQuad([min[0],min[1],min[2]], [min[0],min[1],max[2]], [min[0],max[1],max[2]], [min[0],max[1],min[2]], [-1,0,0], col, mat);
    addQuad([max[0],min[1],max[2]], [max[0],min[1],min[2]], [max[0],max[1],min[2]], [max[0],max[1],max[2]], [1,0,0], col, mat);
    addQuad([min[0],max[1],min[2]], [min[0],max[1],max[2]], [max[0],max[1],max[2]], [max[0],max[1],min[2]], [0,1,0], col, mat);
    addQuad([min[0],min[1],max[2]], [min[0],min[1],min[2]], [max[0],min[1],min[2]], [max[0],min[1],max[2]], [0,-1,0], col, mat);
  }

  function addSphere(center: number[], radius: number, seg: number, col: number[], mat = 1) {
    for (let i = 0; i < seg; i++) {
      const lat0 = Math.PI * (-0.5 + i / seg);
      const lat1 = Math.PI * (-0.5 + (i + 1) / seg);
      const z0 = Math.sin(lat0); const r0 = Math.cos(lat0);
      const z1 = Math.sin(lat1); const r1 = Math.cos(lat1);

      for (let j = 0; j < seg * 2; j++) {
        const lng0 = (2 * Math.PI * j) / (seg * 2);
        const lng1 = (2 * Math.PI * (j + 1)) / (seg * 2);

        const p1 = [center[0] + radius * r0 * Math.cos(lng0), center[1] + radius * z0, center[2] + radius * r0 * Math.sin(lng0)];
        const p2 = [center[0] + radius * r0 * Math.cos(lng1), center[1] + radius * z0, center[2] + radius * r0 * Math.sin(lng1)];
        const p3 = [center[0] + radius * r1 * Math.cos(lng1), center[1] + radius * z1, center[2] + radius * r1 * Math.sin(lng1)];
        const p4 = [center[0] + radius * r1 * Math.cos(lng0), center[1] + radius * z1, center[2] + radius * r1 * Math.sin(lng0)];

        const norm = (p: number[]) => {
          const l = Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]);
          return [(p[0] - center[0]) / l, (p[1] - center[1]) / l, (p[2] - center[2]) / l];
        };

        const pushVert = (p: number[]) => {
          const n = norm(p);
          vertices.push(p[0], p[1], p[2], n[0], n[1], n[2], col[0], col[1], col[2], mat);
        };
        pushVert(p1); pushVert(p2); pushVert(p3);
        pushVert(p1); pushVert(p3); pushVert(p4);
      }
    }
  }

  // 1) 宽阔地面
  addQuad([-1.8, -0.92, -1.8], [1.8, -0.92, -1.8], [1.8, -0.92, 1.8], [-1.8, -0.92, 1.8], [0, 1, 0], [0.85, 0.85, 0.9]);
  // 2) 两个不同材质的立柱 (青色高柱与橙色矮柱)
  addBox([-0.7, -0.92, -0.6], [-0.35, 0.1, -0.25], [0.1, 0.8, 0.8], 0);
  addBox([0.35, -0.92, 0.2], [0.7, -0.2, 0.55], [0.95, 0.55, 0.1], 0);
  // 3) 中央金属镜面反射球体
  addSphere([0.0, -0.47, -0.1], 0.45, 24, [0.95, 0.95, 0.98], 1);

  const vertexData = new Float32Array(vertices);
  const vertexBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // 5. 编译与管线初始化
  const voxelizeModule = device.createShaderModule({ code: voxelizeShader });
  const downsampleModule = device.createShaderModule({ code: downsampleShader });
  const renderModule = device.createShaderModule({ code: renderShader });

  const voxelizePipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: voxelizeModule, entryPoint: "main" },
  });

  const downsamplePipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: downsampleModule, entryPoint: "main" },
  });

  const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: renderModule,
      entryPoint: "vsMain",
      buffers: [
        {
          arrayStride: 10 * 4,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 3 * 4, format: "float32x3" },
            { shaderLocation: 2, offset: 6 * 4, format: "float32x3" },
            { shaderLocation: 3, offset: 9 * 4, format: "float32" },
          ],
        },
      ],
    },
    fragment: {
      module: renderModule,
      entryPoint: "fsMain",
      targets: [{ format }],
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth24plus",
    },
    primitive: { topology: "triangle-list", cullMode: "none" },
  });

  // 6. BindGroup 初始化
  const voxelizeBindGroup = device.createBindGroup({
    layout: voxelizePipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: voxelTexture.createView({ baseMipLevel: 0, mipLevelCount: 1, dimension: "3d" }),
      },
    ],
  });

  const downsampleBindGroups: GPUBindGroup[] = [];
  for (let i = 0; i < MIP_LEVELS - 1; i++) {
    downsampleBindGroups.push(
      device.createBindGroup({
        layout: downsamplePipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: voxelTexture.createView({ baseMipLevel: i, mipLevelCount: 1, dimension: "3d" }),
          },
          {
            binding: 1,
            resource: voxelTexture.createView({ baseMipLevel: i + 1, mipLevelCount: 1, dimension: "3d" }),
          },
        ],
      })
    );
  }

  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      {
        binding: 1,
        resource: voxelTexture.createView({ baseMipLevel: 0, mipLevelCount: MIP_LEVELS, dimension: "3d" }),
      },
      { binding: 2, resource: voxelSampler },
    ],
  });

  let depthTexture: GPUTexture | null = null;
  function updateDepthTexture(w: number, h: number) {
    if (depthTexture) depthTexture.destroy();
    depthTexture = device.createTexture({
      size: [w, h, 1],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  // 7. 相机控制器
  let isDragging = false;
  let lastX = 0, lastY = 0;
  const onMouseDown = (e: MouseEvent) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; };
  const onMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;
    params.cameraTheta += (e.clientX - lastX) * 0.005;
    params.cameraPhi = Math.max(-0.6, Math.min(0.65, params.cameraPhi + (e.clientY - lastY) * 0.005));
    lastX = e.clientX;
    lastY = e.clientY;
  };
  const onMouseUp = () => { isDragging = false; };
  canvas.addEventListener("mousedown", onMouseDown);
  window.addEventListener("mousemove", onMouseMove);
  window.addEventListener("mouseup", onMouseUp);

  // 矩阵辅助数学运算
  function lookAt(eye: number[], center: number[], up: number[]) {
    let z = [eye[0]-center[0], eye[1]-center[1], eye[2]-center[2]];
    let lz = Math.hypot(z[0], z[1], z[2]); z = [z[0]/lz, z[1]/lz, z[2]/lz];
    let x = [up[1]*z[2] - up[2]*z[1], up[2]*z[0] - up[0]*z[2], up[0]*z[1] - up[1]*z[0]];
    let lx = Math.hypot(x[0], x[1], x[2]); x = [x[0]/lx, x[1]/lx, x[2]/lx];
    let y = [z[1]*x[2] - z[2]*x[1], z[2]*x[0] - z[0]*x[2], z[0]*x[1] - z[1]*x[0]];
    return [
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]),
      -(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]),
      -(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]), 1
    ];
  }

  function perspective(fovRad: number, aspect: number, near: number, far: number) {
    const f = 1.0 / Math.tan(fovRad / 2);
    const nf = 1 / (near - far);
    return [
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, far * nf, -1,
      0, 0, far * near * nf, 0
    ];
  }

  function multiply(a: number[], b: number[]) {
    const out = new Array(16).fill(0);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
      }
    }
    return out;
  }

  // 8. 主渲染循环
  let isRunning = true;
  let startTime = performance.now();

  function frame() {
    if (!isRunning) return;

    const width = Math.max(canvas.width, 1);
    const height = Math.max(canvas.height, 1);
    if (!depthTexture || depthTexture.width !== width || depthTexture.height !== height) {
      updateDepthTexture(width, height);
    }

    // 面积光源位置 (微动态绕顶部旋转)
    const time = (performance.now() - startTime) * 0.001;
    const lightX = Math.sin(time * 0.8) * 0.4;
    const lightZ = Math.cos(time * 0.8) * 0.4;
    const lightPos = [lightX, 1.2, lightZ, 1.0];

    // 计算相机
    const camX = params.cameraDistance * Math.sin(params.cameraTheta) * Math.cos(params.cameraPhi);
    const camY = params.cameraDistance * Math.sin(params.cameraPhi);
    const camZ = params.cameraDistance * Math.cos(params.cameraTheta) * Math.cos(params.cameraPhi);
    const eye = [camX, camY, camZ];
    const view = lookAt(eye, [0, -0.3, 0], [0, 1, 0]);
    const proj = perspective((45 * Math.PI) / 180, width / height, 0.1, 20.0);
    const viewProj = multiply(proj, view);

    // 写入 Uniform
    const uniformData = new Float32Array(uniformBufferSize / 4);
    uniformData.set(viewProj, 0);
    uniformData.set([eye[0], eye[1], eye[2], 1.0], 16);
    uniformData.set(lightPos, 20);
    uniformData.set([
      params.roughness,
      params.lightRadius,
      params.shadowIntensity,
      params.aoIntensity
    ], 24);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();

    // 1) 体素化计算
    const voxelPass = encoder.beginComputePass();
    voxelPass.setPipeline(voxelizePipeline);
    voxelPass.setBindGroup(0, voxelizeBindGroup);
    voxelPass.dispatchWorkgroups(VOXEL_RES / 4, VOXEL_RES / 4, VOXEL_RES / 4);
    voxelPass.end();

    // 2) 3D Mipmap 下采样金字塔
    const downPass = encoder.beginComputePass();
    downPass.setPipeline(downsamplePipeline);
    let curRes = VOXEL_RES / 2;
    for (let i = 0; i < MIP_LEVELS - 1; i++) {
      downPass.setBindGroup(0, downsampleBindGroups[i]);
      downPass.dispatchWorkgroups(Math.ceil(curRes / 4), Math.ceil(curRes / 4), Math.ceil(curRes / 4));
      curRes = Math.floor(curRes / 2);
    }
    downPass.end();

    // 3) 场景光栅化与锥体追踪渲染
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0.05, g: 0.06, b: 0.08, a: 1.0 },
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthTexture!.createView(),
        depthLoadOp: "clear",
        depthClearValue: 1.0,
        depthStoreOp: "discard",
      },
    });

    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.draw(vertexData.length / 10, 1, 0, 0);
    renderPass.end();

    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);

  // 9. 清理资源
  return () => {
    isRunning = false;
    canvas.removeEventListener("mousedown", onMouseDown);
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseup", onMouseUp);
    voxelTexture.destroy();
    uniformBuffer.destroy();
    vertexBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}