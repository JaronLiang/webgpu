// src/examples/virtualShadowMap.ts
import GUI from "lil-gui";

// =========================================================================
// 1. 标准 3D 数学库 (针对 WebGPU 0..1 深度定制的标准矩阵计算)
// =========================================================================
function mat4Perspective(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const out = new Float32Array(16);
  const f = 1.0 / Math.tan(fovRad / 2);
  out[0] = f / aspect; out[5] = f;
  out[10] = far / (near - far); out[11] = -1; out[14] = (near * far) / (near - far);
  return out;
}

function mat4Ortho(left: number, right: number, bottom: number, top: number, near: number, far: number): Float32Array {
  const out = new Float32Array(16);
  const w = 1.0 / (right - left);
  const h = 1.0 / (top - bottom);
  const p = 1.0 / (far - near);
  out[0] = 2.0 * w;
  out[5] = 2.0 * h;
  out[10] = -p;
  out[12] = -(right + left) * w;
  out[13] = -(top + bottom) * h;
  out[14] = -near * p;
  out[15] = 1.0;
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

function mat4Inverse(m: Float32Array): Float32Array {
  const out = new Float32Array(16);
  const m00 = m[0], m01 = m[1], m02 = m[2], m03 = m[3];
  const m10 = m[4], m11 = m[5], m12 = m[6], m13 = m[7];
  const m20 = m[8], m21 = m[9], m22 = m[10], m23 = m[11];
  const m30 = m[12], m31 = m[13], m32 = m[14], m33 = m[15];

  const b00 = m00 * m11 - m01 * m10, b01 = m00 * m12 - m02 * m10, b02 = m00 * m13 - m03 * m10;
  const b03 = m01 * m12 - m02 * m11, b04 = m01 * m13 - m03 * m11, b05 = m02 * m13 - m03 * m12;
  const b06 = m20 * m31 - m21 * m30, b07 = m20 * m32 - m22 * m30, b08 = m20 * m33 - m23 * m30;
  const b09 = m21 * m32 - m22 * m31, b10 = m21 * m33 - m23 * m31, b11 = m22 * m33 - m23 * m32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return out;
  det = 1.0 / det;

  out[0] = (m11 * b11 - m12 * b10 + m13 * b09) * det;
  out[1] = (m02 * b10 - m01 * b11 - m03 * b09) * det;
  out[2] = (m31 * b05 - m32 * b04 + m33 * b03) * det;
  out[3] = (m22 * b04 - m21 * b05 - m23 * b03) * det;
  out[4] = (m12 * b08 - m10 * b11 - m13 * b07) * det;
  out[5] = (m00 * b11 - m02 * b08 + m03 * b07) * det;
  out[6] = (m32 * b02 - m30 * b05 - m33 * b01) * det;
  out[7] = (m20 * b05 - m22 * b02 + m23 * b01) * det;
  out[8] = (m10 * b10 - m11 * b08 + m13 * b06) * det;
  out[9] = (m01 * b08 - m00 * b10 - m03 * b06) * det;
  out[10] = (m30 * b04 - m31 * b02 + m33 * b00) * det;
  out[11] = (m21 * b02 - m20 * b04 - m23 * b00) * det;
  out[12] = (m11 * b07 - m10 * b09 - m12 * b06) * det;
  out[13] = (m00 * b09 - m01 * b07 + m02 * b06) * det;
  out[14] = (m31 * b01 - m30 * b03 - m32 * b00) * det;
  out[15] = (m20 * b03 - m21 * b01 + m22 * b00) * det;
  return out;
}

function transformPoint(p: number[], m: Float32Array): number[] {
  const x = p[0], y = p[1], z = p[2];
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
  ];
}

// =========================================================================
// 2. 主程序
// =========================================================================
export function runVirtualShadowMap(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: any
) {
  // VSM 规格定义
  const PAGE_SIZE = 64;              // 每个物理页 64x64 像素
  const VIRTUAL_AXIS = 32;           // 虚拟网格 32x32 -> 等效 2048x2048 虚拟阴影贴图
  const PHYSICAL_AXIS = 32;          // 物理图集扩展至 32x32 (1024 槽)，完全覆盖远景视锥
  const PHYSICAL_TEXTURE_SIZE = PHYSICAL_AXIS * PAGE_SIZE; // 2048x2048
  const TOTAL_PHYSICAL_PAGES = PHYSICAL_AXIS * PHYSICAL_AXIS; // 1024 个物理页槽

  // 全局正交视锥覆盖范围 (扩大至 54.0，彻底消除地面四角被截断的八角形黑边)
  const SCENE_BOUNDS = 54.0;

  // 1. 标准立方体几何
  const cubeData = new Float32Array([
    // Front (Z+)
    -1,-1, 1, 0,0,1,   1,-1, 1, 0,0,1,   1, 1, 1, 0,0,1,
    -1,-1, 1, 0,0,1,   1, 1, 1, 0,0,1,  -1, 1, 1, 0,0,1,
    // Back (Z-)
     1,-1,-1, 0,0,-1, -1,-1,-1, 0,0,-1, -1, 1,-1, 0,0,-1,
     1,-1,-1, 0,0,-1, -1, 1,-1, 0,0,-1,  1, 1,-1, 0,0,-1,
    // Top (Y+)
    -1, 1, 1, 0,1,0,   1, 1, 1, 0,1,0,   1, 1,-1, 0,1,0,
    -1, 1, 1, 0,1,0,   1, 1,-1, 0,1,0,  -1, 1,-1, 0,1,0,
    // Bottom (Y-)
    -1,-1,-1, 0,-1,0,  1,-1,-1, 0,-1,0,  1,-1, 1, 0,-1,0,
    -1,-1,-1, 0,-1,0,  1,-1, 1, 0,-1,0, -1,-1, 1, 0,-1,0,
    // Right (X+)
     1,-1, 1, 1,0,0,   1,-1,-1, 1,0,0,   1, 1,-1, 1,0,0,
     1,-1, 1, 1,0,0,   1, 1,-1, 1,0,0,   1, 1, 1, 1,0,0,
    // Left (X-)
    -1,-1,-1,-1,0,0,  -1,-1, 1,-1,0,0,  -1, 1, 1,-1,0,0,
    -1,-1,-1,-1,0,0,  -1, 1, 1,-1,0,0,  -1, 1,-1,-1,0,0,
  ]);
  const vertexBuffer = device.createBuffer({ size: cubeData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vertexBuffer, 0, cubeData);

  // 2. 场景实例（1 个大基座地面 + 120 根立体彩色柱体）
  const INSTANCE_COUNT = 120;
  const instanceData = new Float32Array(INSTANCE_COUNT * 16);

  // 实例 0: 地面 (半宽 36，完整包含在 54 正交包围圈内)
  instanceData[0] = 0.0; instanceData[1] = -0.5; instanceData[2] = 0.0; instanceData[3] = 0.0;
  instanceData[4] = 36.0; instanceData[5] = 0.5; instanceData[6] = 36.0; instanceData[7] = 0.0;
  instanceData[8] = 0.35; instanceData[9] = 0.38; instanceData[10] = 0.42; instanceData[11] = 1.0;

  for (let i = 1; i < INSTANCE_COUNT; i++) {
    const angle = (i / (INSTANCE_COUNT - 1)) * Math.PI * 6.5;
    const dist = 4.0 + Math.random() * 25.0;
    const x = Math.cos(angle) * dist;
    const z = Math.sin(angle) * dist;
    const h = 1.2 + Math.random() * 4.5;
    const w = 0.7 + Math.random() * 0.7;

    instanceData[i * 16 + 0] = x;
    instanceData[i * 16 + 1] = h;
    instanceData[i * 16 + 2] = z;

    instanceData[i * 16 + 4] = w;
    instanceData[i * 16 + 5] = h;
    instanceData[i * 16 + 6] = w;

    instanceData[i * 16 + 8] = 0.4 + Math.random() * 0.5;
    instanceData[i * 16 + 9] = 0.5 + Math.random() * 0.4;
    instanceData[i * 16 + 10] = 0.6 + Math.random() * 0.4;
    instanceData[i * 16 + 11] = 1.0;
  }

  const instanceBuffer = device.createBuffer({
    size: instanceData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(instanceBuffer, 0, instanceData);

  // 3. 显存核心资源
  const physicalAtlasTexture = device.createTexture({
    size: [PHYSICAL_TEXTURE_SIZE, PHYSICAL_TEXTURE_SIZE],
    format: "depth32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
  });

  const pageTableBuffer = device.createBuffer({
    size: VIRTUAL_AXIS * VIRTUAL_AXIS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  const shadowSampler = device.createSampler({
    compare: "less-equal",
    magFilter: "linear",
    minFilter: "linear"
  });

  const shadowPassUniformBuffer = device.createBuffer({
    size: TOTAL_PHYSICAL_PAGES * 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  const sceneUniformBuffer = device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  // =========================================================================
  // 4. 着色器实现
  // =========================================================================
  const shadowShaderCode = `
    struct Uniforms { pageViewProj: mat4x4f };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct Instance {
      pos: vec4f,
      scale: vec4f,
      color: vec4f,
      pad: vec4f,
    };
    @group(0) @binding(1) var<storage, read> instances: array<Instance>;

    @vertex
    fn vs_main(@location(0) pos: vec3f, @builtin(instance_index) instIdx: u32) -> @builtin(position) vec4f {
      let inst = instances[instIdx];
      let worldPos = pos * inst.scale.xyz + inst.pos.xyz;
      return u.pageViewProj * vec4f(worldPos, 1.0);
    }
  `;

  const sceneShaderCode = `
    struct SceneUniforms {
      cameraViewProj: mat4x4f,
      lightViewProj: mat4x4f,
      lightDir: vec4f,
      debugMode: u32,
      bias: f32,
      virtualAxis: u32,
      physicalAxis: u32,
    };

    @group(0) @binding(0) var<uniform> u: SceneUniforms;

    struct Instance {
      pos: vec4f,
      scale: vec4f,
      color: vec4f,
      pad: vec4f,
    };
    @group(0) @binding(1) var<storage, read> instances: array<Instance>;
    @group(0) @binding(2) var<storage, read> pageTable: array<u32>;
    @group(0) @binding(3) var shadowAtlas: texture_depth_2d;
    @group(0) @binding(4) var shadowSampler: sampler_comparison;

    struct VertexOut {
      @builtin(position) clipPos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) shadowPos: vec4f,
      @location(3) color: vec3f,
    };

    @vertex
    fn vs_main(@location(0) pos: vec3f, @location(1) normal: vec3f, @builtin(instance_index) instIdx: u32) -> VertexOut {
      var out: VertexOut;
      let inst = instances[instIdx];
      let worldPos = pos * inst.scale.xyz + inst.pos.xyz;

      out.clipPos = u.cameraViewProj * vec4f(worldPos, 1.0);
      out.worldPos = worldPos;
      out.normal = normal;
      out.shadowPos = u.lightViewProj * vec4f(worldPos, 1.0);
      out.color = inst.color.rgb;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4f {
      let N = normalize(in.normal);
      let L = normalize(u.lightDir.xyz);
      let diff = max(dot(N, L), 0.0);

      let projCoords = in.shadowPos.xyz / in.shadowPos.w;
      let virtualUV = projCoords.xy * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
      let currentDepth = projCoords.z;

      var shadowFactor = 1.0;
      var debugColor = vec3f(0.0);

      if (virtualUV.x >= 0.0 && virtualUV.x <= 1.0 && virtualUV.y >= 0.0 && virtualUV.y <= 1.0 && currentDepth <= 1.0) {
        let vAxis = f32(u.virtualAxis);
        let vPage = vec2u(clamp(virtualUV * vAxis, vec2f(0.0), vec2f(vAxis - 1.0)));
        let pageIdx = vPage.y * u.virtualAxis + vPage.x;

        let pageEntry = pageTable[pageIdx];
        let isValid = (pageEntry & 0x80000000u) != 0u;

        if (isValid) {
          let physId = pageEntry & 0xFFFFu;
          let physX = f32(physId % u.physicalAxis);
          let physY = f32(physId / u.physicalAxis);

          // 【关键修复点】：Clamp 物理页内 UV，防止双线性滤波跨页缝隙产生黑边伪影
          let intraUV = fract(virtualUV * vAxis);
          let halfTexel = 0.5 / 64.0;
          let clampedIntraUV = clamp(intraUV, vec2f(halfTexel), vec2f(1.0 - halfTexel));
          let physicalUV = (vec2f(physX, physY) + clampedIntraUV) / f32(u.physicalAxis);

          // 自适应法线斜率 Bias，消灭柱体表面的痤疮伪影
          let slopeBias = (1.0 - diff) * 0.001;
          let biasVal = u.bias + slopeBias;
          shadowFactor = textureSampleCompareLevel(shadowAtlas, shadowSampler, physicalUV, currentDepth - biasVal);

          if (u.debugMode == 1u) {
            let r = f32((physId * 43u + 17u) % 255u) / 255.0;
            let g = f32((physId * 89u + 53u) % 255u) / 255.0;
            let b = f32((physId * 173u + 97u) % 255u) / 255.0;
            debugColor = vec3f(r, g, b);
          } else if (u.debugMode == 2u) {
            debugColor = vec3f(0.15, 0.85, 0.25); // 已分配有效页呈现清新亮绿色
          }
        } else {
          // 【修复点】：未分配区域不应设为 0.3 造成死黑，而在热力图模式下标红提示
          shadowFactor = 1.0;
          if (u.debugMode == 2u) { debugColor = vec3f(0.8, 0.15, 0.15); }
        }
      }

      if (u.debugMode != 0u) {
        return vec4f(debugColor * (diff * 0.6 + 0.4), 1.0);
      }

      let ambient = 0.22;
      let lightTerm = ambient + (1.0 - ambient) * diff * shadowFactor;
      return vec4f(in.color * lightTerm, 1.0);
    }
  `;

  // =========================================================================
  // 5. 管线配置
  // =========================================================================
  const shadowModule = device.createShaderModule({ code: shadowShaderCode });
  const shadowBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 64 } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }
    ]
  });

  const shadowPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [shadowBindGroupLayout] }),
    vertex: {
      module: shadowModule, entryPoint: "vs_main",
      buffers: [{ arrayStride: 6 * 4, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }]
    },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth32float" }
  });

  const shadowBindGroup = device.createBindGroup({
    layout: shadowBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: shadowPassUniformBuffer, size: 64 } },
      { binding: 1, resource: { buffer: instanceBuffer } }
    ]
  });

  const sceneModule = device.createShaderModule({ code: sceneShaderCode });
  const scenePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: sceneModule, entryPoint: "vs_main",
      buffers: [{
        arrayStride: 6 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" }
        ]
      }]
    },
    fragment: { module: sceneModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" }
  });

  const sceneBindGroup = device.createBindGroup({
    layout: scenePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sceneUniformBuffer } },
      { binding: 1, resource: { buffer: instanceBuffer } },
      { binding: 2, resource: { buffer: pageTableBuffer } },
      { binding: 3, resource: physicalAtlasTexture.createView() },
      { binding: 4, resource: shadowSampler },
    ]
  });

  // =========================================================================
  // 6. 交互配置与帧循环
  // =========================================================================
  const settings = {
    viewMode: "VSM 虚拟阴影 (高清物理图)",
    autoRotate: true,
    shadowBias: 0.0015,
    lightElevation: 50.0,
  };

  gui.title("WebGPU 虚拟阴影贴图 (VSM) - 修复完善版");
  gui.add(settings, "viewMode", [
    "VSM 虚拟阴影 (高清物理图)",
    "物理页槽着色 (Physical Pages)",
    "虚拟页表分配热力图 (Page Table)"
  ]).name("视图模式");
  gui.add(settings, "shadowBias", 0.0005, 0.006, 0.0005).name("阴影偏移 (Bias)");
  gui.add(settings, "lightElevation", 20.0, 80.0, 1.0).name("太阳高度角");
  gui.add(settings, "autoRotate").name("相机旋转");

  const camera = { target: [0, 1.5, 0], radius: 46.0, theta: 45.0, phi: 30.0 };
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
      camera.theta -= dx * 0.35; camera.phi = Math.max(8, Math.min(85, camera.phi + dy * 0.35));
    } else if (dragButton === 2) {
      const radTheta = (camera.theta * Math.PI) / 180; const pan = camera.radius * 0.0015;
      camera.target[0] -= Math.cos(radTheta) * dx * pan; camera.target[2] -= -Math.sin(radTheta) * dx * pan; camera.target[1] += dy * pan;
    }
  });
  canvas.addEventListener("pointerup", (e) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); camera.radius = Math.max(8, camera.radius * Math.exp(e.deltaY * 0.001)); }, { passive: false });

  let depthTexture: GPUTexture | null = null;
  let animId: number;

  const cpuPageTable = new Uint32Array(VIRTUAL_AXIS * VIRTUAL_AXIS);
  const shadowUniformFloats = new Float32Array(TOTAL_PHYSICAL_PAGES * 64);

  function frame() {
    if (settings.autoRotate && !isDragging) camera.theta += 0.15;

    const dpr = window.devicePixelRatio || 1;
    const renderWidth = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const renderHeight = Math.max(1, Math.floor(canvas.clientHeight * dpr));

    if (!depthTexture || depthTexture.width !== renderWidth || depthTexture.height !== renderHeight) {
      canvas.width = renderWidth; canvas.height = renderHeight;
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({ size: [renderWidth, renderHeight], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }

    // 1. 相机矩阵计算（限制合理的视锥远平面，避免无意义的过大投影）
    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.target[0] + camera.radius * Math.cos(radPhi) * Math.sin(radTheta),
      camera.target[1] + camera.radius * Math.sin(radPhi),
      camera.target[2] + camera.radius * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const cameraProj = mat4Perspective((45 * Math.PI) / 180, renderWidth / renderHeight, 0.5, 120.0);
    const cameraView = mat4LookAt(eye, camera.target, [0, 1, 0]);
    const cameraViewProj = mat4Multiply(cameraProj, cameraView);
    const cameraInvViewProj = mat4Inverse(cameraViewProj);

    // 2. 光源全局虚拟视锥计算
    const lightRad = (settings.lightElevation * Math.PI) / 180;
    const lightDir = [Math.cos(lightRad) * 0.65, Math.sin(lightRad), Math.cos(lightRad) * 0.65];
    const lightView = mat4LookAt([lightDir[0] * 50, lightDir[1] * 50, lightDir[2] * 50], [0, 0, 0], [0, 1, 0]);
    const lightGlobalProj = mat4Ortho(-SCENE_BOUNDS, SCENE_BOUNDS, -SCENE_BOUNDS, SCENE_BOUNDS, -30, 150);
    const lightViewProj = mat4Multiply(lightGlobalProj, lightView);

    // =========================================================================
    // 3. 【核心修复】：基于视锥射线与地面精确交点提取光源空间活跃页
    // =========================================================================
    cpuPageTable.fill(0);
    let nextPhysId = 0;

    const testPoints: number[][] = [];
    testPoints.push([camera.target[0], camera.target[1], camera.target[2]]);

    // 视锥近平面与投射到地面/建筑顶部的射线采样
    const corners2D = [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0]];
    for (const [cx, cy] of corners2D) {
      const pNear = transformPoint([cx, cy, 0.0], cameraInvViewProj);
      const pFar  = transformPoint([cx, cy, 1.0], cameraInvViewProj);
      testPoints.push(pNear);

      const dx = pFar[0] - pNear[0];
      const dy = pFar[1] - pNear[1];
      const dz = pFar[2] - pNear[2];

      if (Math.abs(dy) > 1e-4) {
        const t0 = (0.0 - pNear[1]) / dy; // 与地面 Y=0 的交点
        if (t0 > 0 && t0 < 200.0) {
          testPoints.push([pNear[0] + dx * t0, 0.0, pNear[2] + dz * t0]);
        }
        const t6 = (6.0 - pNear[1]) / dy; // 与柱顶 Y=6 的交点
        if (t6 > 0 && t6 < 200.0) {
          testPoints.push([pNear[0] + dx * t6, 6.0, pNear[2] + dz * t6]);
        }
      }
    }

    let minU = 1.0, maxU = 0.0, minV = 1.0, maxV = 0.0;
    for (const pt of testPoints) {
      const lightPos = transformPoint(pt, lightViewProj);
      const u = lightPos[0] * 0.5 + 0.5;
      const v = lightPos[1] * -0.5 + 0.5;
      minU = Math.min(minU, u); maxU = Math.max(maxU, u);
      minV = Math.min(minV, v); maxV = Math.max(maxV, v);
    }

    minU = Math.max(0.0, Math.min(1.0, minU));
    maxU = Math.max(0.0, Math.min(1.0, maxU));
    minV = Math.max(0.0, Math.min(1.0, minV));
    maxV = Math.max(0.0, Math.min(1.0, maxV));

    // 边界安全扩充 2 个物理页缓冲区防漏影
    const startPX = Math.max(0, Math.min(VIRTUAL_AXIS - 1, Math.floor(minU * VIRTUAL_AXIS) - 2));
    const endPX   = Math.max(0, Math.min(VIRTUAL_AXIS - 1, Math.ceil(maxU * VIRTUAL_AXIS) + 2));
    const startPY = Math.max(0, Math.min(VIRTUAL_AXIS - 1, Math.floor(minV * VIRTUAL_AXIS) - 2));
    const endPY   = Math.max(0, Math.min(VIRTUAL_AXIS - 1, Math.ceil(maxV * VIRTUAL_AXIS) + 2));

    interface ActivePage { vx: number; vy: number; px: number; py: number; slot: number }
    const activePages: ActivePage[] = [];

    const tileW = (SCENE_BOUNDS * 2.0) / VIRTUAL_AXIS;
    const tileH = (SCENE_BOUNDS * 2.0) / VIRTUAL_AXIS;

    for (let vy = startPY; vy <= endPY; vy++) {
      for (let vx = startPX; vx <= endPX; vx++) {
        if (nextPhysId < TOTAL_PHYSICAL_PAGES) {
          const slot = nextPhysId++;
          const px = slot % PHYSICAL_AXIS;
          const py = Math.floor(slot / PHYSICAL_AXIS);

          cpuPageTable[vy * VIRTUAL_AXIS + vx] = 0x80000000 | slot;
          activePages.push({ vx, vy, px, py, slot });

          const left = -SCENE_BOUNDS + vx * tileW;
          const right = left + tileW;
          const top = SCENE_BOUNDS - vy * tileH;
          const bottom = top - tileH;

          const tileProj = mat4Ortho(left, right, bottom, top, -30, 150);
          const pageViewProj = mat4Multiply(tileProj, lightView);

          shadowUniformFloats.set(pageViewProj, slot * 64);
        }
      }
    }

    device.queue.writeBuffer(pageTableBuffer, 0, cpuPageTable);
    device.queue.writeBuffer(shadowPassUniformBuffer, 0, shadowUniformFloats);

    // 主场景 Uniform
    const sceneUniformData = new ArrayBuffer(256);
    const f32View = new Float32Array(sceneUniformData);
    const u32View = new Uint32Array(sceneUniformData);

    f32View.set(cameraViewProj, 0);
    f32View.set(lightViewProj, 16);
    f32View[32] = lightDir[0]; f32View[33] = lightDir[1]; f32View[34] = lightDir[2]; f32View[35] = 0.0;

    let modeVal = 0;
    if (settings.viewMode.includes("物理页槽")) modeVal = 1;
    if (settings.viewMode.includes("热力图")) modeVal = 2;
    u32View[36] = modeVal;
    f32View[37] = settings.shadowBias;
    u32View[38] = VIRTUAL_AXIS;
    u32View[39] = PHYSICAL_AXIS;

    device.queue.writeBuffer(sceneUniformBuffer, 0, sceneUniformData);

    // =========================================================================
    // 4. 提交渲染指令
    // =========================================================================
    const encoder = device.createCommandEncoder();

    // 阶段 1: 物理图集渲染
    const shadowPass = encoder.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: physicalAtlasTexture.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store"
      }
    });

    shadowPass.setPipeline(shadowPipeline);
    shadowPass.setVertexBuffer(0, vertexBuffer);

    for (const page of activePages) {
      const vx = page.px * PAGE_SIZE;
      const vy = page.py * PAGE_SIZE;
      shadowPass.setViewport(vx, vy, PAGE_SIZE, PAGE_SIZE, 0.0, 1.0);
      shadowPass.setScissorRect(vx, vy, PAGE_SIZE, PAGE_SIZE);

      shadowPass.setBindGroup(0, shadowBindGroup, [page.slot * 256]);
      shadowPass.draw(36, INSTANCE_COUNT, 0, 0);
    }
    shadowPass.end();

    // 阶段 2: 主场景绘制
    const scenePass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.14, g: 0.16, b: 0.20, a: 1.0 }, loadOp: "clear", storeOp: "store"
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store"
      }
    });

    scenePass.setPipeline(scenePipeline);
    scenePass.setBindGroup(0, sceneBindGroup);
    scenePass.setVertexBuffer(0, vertexBuffer);
    scenePass.draw(36, INSTANCE_COUNT, 0, 0);
    scenePass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    vertexBuffer.destroy();
    instanceBuffer.destroy();
    physicalAtlasTexture.destroy();
    pageTableBuffer.destroy();
    sceneUniformBuffer.destroy();
    shadowPassUniformBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}