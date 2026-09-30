/// <reference types="@webgpu/types" />
import { SimpleGUI } from "../utils/gui";

export async function runVXGI(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui?: SimpleGUI
) {
  // 1. 体素网格分辨率设定 (64x64x64，Mip 级别共 6 级)
  const VOXEL_RES = 64;
  const MIP_LEVELS = 6;

  // 2. 创建 3D 体素 Radiance 纹理
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

  // 3. 参数控制
  const config = {
    indirectIntensity: 1.6,
    aoStrength: 1.0,
    roughness: 0.08,
    cameraTheta: 0.15,
    cameraPhi: 0.22,
    cameraDistance: 3.8,
  };

  if (gui) {
    const g = gui as any;
    if (typeof g.add === "function") {
      g.add(config, "indirectIntensity", 0.0, 4.0, 0.1).name?.("间接光强度");
      g.add(config, "aoStrength", 0.0, 3.0, 0.1).name?.("体素遮蔽(AO)");
      g.add(config, "roughness", 0.0, 1.0, 0.05).name?.("球体粗糙度");
    }
  }

  const uniformBufferSize = 256;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // Pass 1: 场景体素化计算着色器 (注入墙壁、球体、盒子与发光顶灯)
  // =========================================================================
  const voxelizeShaderCode = `
    @group(0) @binding(0) var voxelGrid: texture_storage_3d<rgba16float, write>;

    @compute @workgroup_size(4, 4, 4)
    fn main(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= 64u || id.y >= 64u || id.z >= 64u) { return; }

      let p = (vec3f(id) + 0.5) / 64.0 * 2.0 - 1.0;
      var radiance = vec3f(0.0);
      var opacity = 0.0;

      // 1. 顶部发光光源
      if (p.y > 0.92 && abs(p.x) < 0.35 && abs(p.z) < 0.35) {
        radiance = vec3f(15.0, 13.0, 9.0);
        opacity = 1.0;
      }
      // 2. 左墙 (红色)
      else if (p.x < -0.95) {
        radiance = vec3f(0.9, 0.05, 0.05);
        opacity = 1.0;
      }
      // 3. 右墙 (绿色)
      else if (p.x > 0.95) {
        radiance = vec3f(0.05, 0.9, 0.05);
        opacity = 1.0;
      }
      // 4. 地面、后墙、天花板 (纯白灰)
      else if (p.y < -0.95 || p.y > 0.95 || p.z < -0.95) {
        radiance = vec3f(0.8, 0.8, 0.8);
        opacity = 1.0;
      }
      // 5. 左侧黄色立方体
      let bMin = vec3f(-0.65, -1.0, -0.2);
      let bMax = vec3f(-0.15, -0.35,  0.3);
      if (all(p >= bMin) && all(p <= bMax)) {
        radiance = vec3f(0.85, 0.75, 0.15);
        opacity = 1.0;
      }
      // 6. 右侧镜面球体
      let sphereCenter = vec3f(0.42, -0.48, -0.1);
      if (distance(p, sphereCenter) < 0.42) {
        radiance = vec3f(0.95, 0.95, 0.98);
        opacity = 1.0;
      }

      textureStore(voxelGrid, id, vec4f(radiance, opacity));
    }
  `;

  // =========================================================================
  // Pass 2: 3D 体素金字塔下采样
  // =========================================================================
  const downsampleShaderCode = `
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
  // Pass 3: 场景光栅化与体素锥体追踪 (VCT)
  // =========================================================================
  const renderShaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
      invView: mat4x4f,
      cameraPos: vec4f,
      lightPos: vec4f,
      params: vec4f, // x: indirectIntensity, y: aoStrength, z: roughness, w: time
    };

    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var voxelTex: texture_3d<f32>;
    @group(0) @binding(2) var voxelSmp: sampler;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) albedo: vec3f,
      @location(3) materialId: f32, // 0: 漫反射, 1: 镜面球
    };

    @vertex
    fn vsMain(
      @location(0) position: vec3f,
      @location(1) normal: vec3f,
      @location(2) albedo: vec3f,
      @location(3) matId: f32
    ) -> VertexOut {
      var out: VertexOut;
      out.worldPos = position;
      out.normal = normalize(normal);
      out.albedo = albedo;
      out.materialId = matId;
      out.pos = u.viewProj * vec4f(position, 1.0);
      return out;
    }

    // 体素锥体步进追踪核心函数
    fn traceCone(origin: vec3f, direction: vec3f, tanHalfAngle: f32, maxDist: f32) -> vec4f {
      var accum = vec4f(0.0);
      let voxelSize = 2.0 / 64.0;
      // 适度偏移避开自身表面产生自遮挡
      var dist = voxelSize * 3.0;

      while (dist < maxDist && accum.a < 0.95) {
        let currentPos = origin + direction * dist;
        let uvw = currentPos * 0.5 + 0.5;

        if (any(uvw < vec3f(0.01)) || any(uvw > vec3f(0.99))) { break; }

        let diameter = 2.0 * tanHalfAngle * dist;
        let mipLevel = max(0.0, log2(max(diameter / voxelSize, 1.0)));

        let voxelSample = textureSampleLevel(voxelTex, voxelSmp, uvw, mipLevel);
        let alpha = voxelSample.a;

        // 体积累加合成
        accum += (1.0 - accum.a) * vec4f(voxelSample.rgb * alpha, alpha);
        dist += max(voxelSize * 0.6, diameter * 0.35);
      }
      return accum;
    }

// 稳健正交基：彻底解决法线垂直于地面/天花板时的叉乘奇异退化
    fn getOrthoBasis(n: vec3f) -> mat3x3f {
      // 当法线平行于 Y 轴（如地面、天花板）时，up 切换为 Z 轴，永不平行
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

      // 1. 直接光照
      let NdotL = max(dot(N, L), 0.0);
      let atten = 1.0 / (1.0 + 0.4 * distToLight + 0.3 * distToLight * distToLight);
      let directLight = in.albedo * vec3f(4.5) * NdotL * atten;

      // 2. 漫反射光锥追踪 (6 个正交余弦锥体)
      // 2. 漫反射光锥追踪 (正法线 1 个 + 环绕四周对称分布 4 个)
      let TBN = getOrthoBasis(N);
      
      // 1 个主法线锥体 (权重 0.28) + 4 个 60° 侧向对称光锥 (权重各 0.18)
      let coneDirs = array<vec3f, 5>(
        vec3f( 0.0,    0.0,    1.0),   // 正法线主轴
        vec3f( 0.866,  0.0,    0.5),   // +X 偏角
        vec3f(-0.866,  0.0,    0.5),   // -X 偏角
        vec3f( 0.0,    0.866,  0.5),   // +Y 偏角
        vec3f( 0.0,   -0.866,  0.5)    // -Y 偏角
      );
      let coneWeights = array<f32, 5>(0.28, 0.18, 0.18, 0.18, 0.18);

      var indirectColor = vec3f(0.0);
      var occlusion = 0.0;

      for (var i = 0; i < 5; i++) {
        let dir = normalize(TBN * coneDirs[i]);
        let traceRes = traceCone(in.worldPos, dir, 0.577, 2.5);
        indirectColor += traceRes.rgb * coneWeights[i];
        occlusion += traceRes.a * coneWeights[i];
      }

      indirectColor = indirectColor * u.params.x * in.albedo;
      let ao = clamp(1.0 - occlusion * u.params.y, 0.05, 1.0);

      // 3. 镜面反射光锥追踪 (高光球体)
      var specularColor = vec3f(0.0);
      if (in.materialId > 0.5) {
        let R = reflect(-V, N);
        let specTan = tan(max(u.params.z, 0.02) * 0.5);
        let specTrace = traceCone(in.worldPos, R, specTan, 3.0);
        specularColor = specTrace.rgb * 1.6;
      }

      // 4. 合成最终色彩
      var finalColor = (directLight * ao + indirectColor) + specularColor;
      // ACES Tone mapping
      finalColor = finalColor / (finalColor + vec3f(1.0));
      return vec4f(finalColor, 1.0);
    }
  `;

  // =========================================================================
  // 4. 几何体生成函数 (墙体 + 黄色箱体 + 镜面球体)
  // =========================================================================
  const vertices: number[] = [];

  function addQuad(
    p1: number[], p2: number[], p3: number[], p4: number[],
    normal: number[], color: number[], matId = 0
  ) {
    const pushV = (p: number[]) => {
      vertices.push(p[0], p[1], p[2], normal[0], normal[1], normal[2], color[0], color[1], color[2], matId);
    };
    pushV(p1); pushV(p2); pushV(p3);
    pushV(p1); pushV(p3); pushV(p4);
  }

  function addBox(min: number[], max: number[], color: number[], matId = 0) {
    // 前后左右上下六个面
    addQuad([min[0],min[1],max[2]], [max[0],min[1],max[2]], [max[0],max[1],max[2]], [min[0],max[1],max[2]], [0,0,1], color, matId);
    addQuad([max[0],min[1],min[2]], [min[0],min[1],min[2]], [min[0],max[1],min[2]], [max[0],max[1],min[2]], [0,0,-1], color, matId);
    addQuad([min[0],min[1],min[2]], [min[0],min[1],max[2]], [min[0],max[1],max[2]], [min[0],max[1],min[2]], [-1,0,0], color, matId);
    addQuad([max[0],min[1],max[2]], [max[0],min[1],min[2]], [max[0],max[1],min[2]], [max[0],max[1],max[2]], [1,0,0], color, matId);
    addQuad([min[0],max[1],min[2]], [min[0],max[1],max[2]], [max[0],max[1],max[2]], [max[0],max[1],min[2]], [0,1,0], color, matId);
    addQuad([min[0],min[1],max[2]], [min[0],min[1],min[2]], [max[0],min[1],min[2]], [max[0],min[1],max[2]], [0,-1,0], color, matId);
  }

  function addSphere(center: number[], radius: number, segments: number, color: number[], matId = 1) {
    for (let i = 0; i < segments; i++) {
      const lat0 = Math.PI * (-0.5 + i / segments);
      const lat1 = Math.PI * (-0.5 + (i + 1) / segments);
      const z0 = Math.sin(lat0); const zr0 = Math.cos(lat0);
      const z1 = Math.sin(lat1); const zr1 = Math.cos(lat1);

      for (let j = 0; j < segments * 2; j++) {
        const lng0 = (2 * Math.PI * j) / (segments * 2);
        const lng1 = (2 * Math.PI * (j + 1)) / (segments * 2);
        const p1 = [center[0] + radius * zr0 * Math.cos(lng0), center[1] + radius * z0, center[2] + radius * zr0 * Math.sin(lng0)];
        const p2 = [center[0] + radius * zr0 * Math.cos(lng1), center[1] + radius * z0, center[2] + radius * zr0 * Math.sin(lng1)];
        const p3 = [center[0] + radius * zr1 * Math.cos(lng1), center[1] + radius * z1, center[2] + radius * zr1 * Math.sin(lng1)];
        const p4 = [center[0] + radius * zr1 * Math.cos(lng0), center[1] + radius * z1, center[2] + radius * zr1 * Math.sin(lng0)];

        const n = (p: number[]) => {
          const l = Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]);
          return [(p[0] - center[0]) / l, (p[1] - center[1]) / l, (p[2] - center[2]) / l];
        };

        const pushVert = (p: number[]) => {
          const norm = n(p);
          vertices.push(p[0], p[1], p[2], norm[0], norm[1], norm[2], color[0], color[1], color[2], matId);
        };
        pushVert(p1); pushVert(p2); pushVert(p3);
        pushVert(p1); pushVert(p3); pushVert(p4);
      }
    }
  }

  // 1) 房间五面墙体
// 1) 房间五面墙体 (法线全部朝向房间内部)
  addQuad([-1,-1,-1], [ 1,-1,-1], [ 1,-1, 1], [-1,-1, 1], [0, 1, 0], [0.8, 0.8, 0.8]); // 地面
  addQuad([-1, 1,-1], [ 1, 1,-1], [ 1, 1, 1], [-1, 1, 1], [0,-1, 0], [0.8, 0.8, 0.8]); // 天花板
  addQuad([-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1], [0, 0, 1], [0.8, 0.8, 0.8]); // 后墙
  addQuad([-1,-1, 1], [-1,-1,-1], [-1, 1,-1], [-1, 1, 1], [1, 0, 0], [0.9, 0.08, 0.08]); // 左红墙
  addQuad([ 1,-1,-1], [ 1,-1, 1], [ 1, 1, 1], [ 1, 1,-1], [-1,0, 0], [0.08, 0.9, 0.08]); // 右绿墙

  // 顶部加一个小白色发光灯面，使光源更立体
  addQuad([-0.3, 0.99, -0.3], [0.3, 0.99, -0.3], [0.3, 0.99, 0.3], [-0.3, 0.99, 0.3], [0, -1, 0], [3.0, 3.0, 2.5]);
  // 2) 内部物体：左侧黄色漫反射箱体
  addBox([-0.65, -1.0, -0.2], [-0.15, -0.35, 0.3], [0.85, 0.75, 0.15], 0);

  // 3) 内部物体：右侧金属高光镜面球
  addSphere([0.42, -0.55, -0.1], 0.42, 24, [0.95, 0.95, 0.98], 1);

  const vertexData = new Float32Array(vertices);
  const vertexBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // 5. 管线构建
  const voxelizeModule = device.createShaderModule({ code: voxelizeShaderCode });
  const downsampleModule = device.createShaderModule({ code: downsampleShaderCode });
  const renderModule = device.createShaderModule({ code: renderShaderCode });

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

  // 6. 绑定组创建
  const voxelizeBindGroup = device.createBindGroup({
    layout: voxelizePipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: voxelTexture.createView({
          baseMipLevel: 0,
          mipLevelCount: 1,
          dimension: "3d",
        }),
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
    config.cameraTheta += (e.clientX - lastX) * 0.005;
    config.cameraPhi = Math.max(-0.55, Math.min(0.55, config.cameraPhi + (e.clientY - lastY) * 0.005));
    lastX = e.clientX;
    lastY = e.clientY;
  };
  const onMouseUp = () => { isDragging = false; };
  canvas.addEventListener("mousedown", onMouseDown);
  window.addEventListener("mousemove", onMouseMove);
  window.addEventListener("mouseup", onMouseUp);

  // 矩阵辅助函数
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

  // 8. 动画主循环
  let isRunning = true;
  const startTime = performance.now();

  function frame() {
    if (!isRunning) return;

    const width = Math.max(canvas.width, 1);
    const height = Math.max(canvas.height, 1);
    if (!depthTexture || depthTexture.width !== width || depthTexture.height !== height) {
      updateDepthTexture(width, height);
    }

    const camX = config.cameraDistance * Math.sin(config.cameraTheta) * Math.cos(config.cameraPhi);
    const camY = config.cameraDistance * Math.sin(config.cameraPhi);
    const camZ = config.cameraDistance * Math.cos(config.cameraTheta) * Math.cos(config.cameraPhi);
    const eye = [camX, camY, camZ];
    const view = lookAt(eye, [0, -0.1, 0], [0, 1, 0]);
    const proj = perspective((45 * Math.PI) / 180, width / height, 0.1, 20.0);
    const viewProj = multiply(proj, view);

    const uniformData = new Float32Array(uniformBufferSize / 4);
    uniformData.set(viewProj, 0);
    uniformData.set([eye[0], eye[1], eye[2], 1.0], 32);
    uniformData.set([0.0, 0.85, 0.0, 1.0], 36);
    uniformData.set([
      config.indirectIntensity,
      config.aoStrength,
      config.roughness,
      (performance.now() - startTime) * 0.001
    ], 40);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();

    // 1) 场景体素化 (64/4 = 16)
    const voxelPass = encoder.beginComputePass();
    voxelPass.setPipeline(voxelizePipeline);
    voxelPass.setBindGroup(0, voxelizeBindGroup);
    voxelPass.dispatchWorkgroups(VOXEL_RES / 4, VOXEL_RES / 4, VOXEL_RES / 4);
    voxelPass.end();

    // 2) 3D 体素金字塔下采样
    const downPass = encoder.beginComputePass();
    downPass.setPipeline(downsamplePipeline);
    let curRes = VOXEL_RES / 2;
    for (let i = 0; i < MIP_LEVELS - 1; i++) {
      downPass.setBindGroup(0, downsampleBindGroups[i]);
      downPass.dispatchWorkgroups(Math.ceil(curRes / 4), Math.ceil(curRes / 4), Math.ceil(curRes / 4));
      curRes = Math.floor(curRes / 2);
    }
    downPass.end();

    // 3) 场景光栅化与体素追踪
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0.04, g: 0.04, b: 0.05, a: 1.0 },
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

  // 9. 销毁逻辑
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