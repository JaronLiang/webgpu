// src/examples/gaussianSplattingGPU.ts

export async function initWebGPUDevice(): Promise<{ device: GPUDevice; adapter: GPUAdapter }> {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU not supported on this browser.");

  const requiredLimits: Record<string, number> = {};
  if (adapter.limits.maxStorageBufferBindingSize) {
    requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
  }
  if (adapter.limits.maxBufferSize) {
    requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
  }

  const device = await adapter.requestDevice({ requiredLimits });
  return { device, adapter };
}

export function runGaussianSplatting(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 1. WGSL: 核心着色器
  // =========================================================================

  // 1.1 Compute Culling & Projection
  const cullingShaderCode = /* wgsl */ `
    struct CameraUniform {
      view: mat4x4f,
      proj: mat4x4f,
      camPos: vec3f,
      _pad0: f32,
      viewport: vec2f,
      focal: vec2f,
    };

    struct SplatSource {
      pos: vec3f,
      colorPacked: u32,
      scale: vec3f,
      pad: f32,
      rot: vec4f,
    };

    struct ProjectedSplat {
      xy: vec2f,
      conic: vec3f,
      radius: f32,
      color: vec4f,
      depth: f32,
      clipZ: f32,
    };

    struct DrawIndirectArgs {
      vertexCount: u32,
      instanceCount: atomic<u32>,
      firstVertex: u32,
      firstInstance: u32,
    };

    @group(0) @binding(0) var<uniform> uCam: CameraUniform;
    @group(0) @binding(1) var<storage, read> splatsIn: array<SplatSource>;
    @group(0) @binding(2) var<storage, read_write> splatsOut: array<ProjectedSplat>;
    @group(0) @binding(3) var<storage, read_write> drawArgs: DrawIndirectArgs;
    @group(0) @binding(4) var<storage, read_write> sortKeys: array<u32>;
    @group(0) @binding(5) var<storage, read_write> sortValues: array<u32>;

    fn quatToMat(q: vec4f) -> mat3x3f {
      let w = q.x; let x = q.y; let y = q.z; let z = q.w;
      return mat3x3f(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y),
        2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x),
        2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y)
      );
    }

    @compute @workgroup_size(256, 1)
    fn cs_cull_and_project(
      @builtin(workgroup_id) wid: vec3u,
      @builtin(local_invocation_id) lid: vec3u,
      @builtin(num_workgroups) nwid: vec3u
    ) {
      let idx = (wid.y * nwid.x + wid.x) * 256u + lid.x;
      if (idx >= arrayLength(&splatsIn)) { return; }

      let splat = splatsIn[idx];
      let viewPos4 = uCam.view * vec4f(splat.pos, 1.0);
      let viewPos = viewPos4.xyz;

      // 【核心修复 1】：将近裁减面退后到 -0.3，避免贴近镜头时导致 Jacobian 除法溢出
      if (viewPos.z >= -0.3 || viewPos.z <= -4000.0) { return; }

      let R = quatToMat(splat.rot);
      let S = mat3x3f(
        splat.scale.x, 0.0, 0.0,
        0.0, splat.scale.y, 0.0,
        0.0, 0.0, splat.scale.z
      );
      let M = R * S;
      let cov3D = M * transpose(M);

      let zInv = 1.0 / viewPos.z;
      let zInv2 = zInv * zInv;

      let J = mat3x3f(
        uCam.focal.x * zInv, 0.0, -(uCam.focal.x * viewPos.x) * zInv2,
        0.0, uCam.focal.y * zInv, -(uCam.focal.y * viewPos.y) * zInv2,
        0.0, 0.0, 0.0
      );

      let W = mat3x3f(uCam.view[0].xyz, uCam.view[1].xyz, uCam.view[2].xyz);
      let T = J * W;
      var cov2D = T * cov3D * transpose(T);

      // 低通滤波抗锯齿
      cov2D[0][0] += 0.3;
      cov2D[1][1] += 0.3;

      let det = cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[0][1];
      if (det <= 0.000001) { return; }

      let invDet = 1.0 / det;
      let conic = vec3f(cov2D[1][1] * invDet, -cov2D[0][1] * invDet, cov2D[0][0] * invDet);

      let mid = 0.5 * (cov2D[0][0] + cov2D[1][1]);
      let lambda = mid + sqrt(max(0.01, mid * mid - det));
      
      // 【核心修复 2】：取消原本的 512 剔除限制，但设定硬件安全极值 4096 防止超出光栅化上限
      let radius = min(ceil(3.0 * sqrt(max(0.01, lambda))), 4096.0);
      if (radius <= 0.5) { return; }

      let clipPos = uCam.proj * viewPos4;
      let ndc = clipPos.xy / clipPos.w;
      let screenPos = (ndc * 0.5 + 0.5) * uCam.viewport;

      // 剔除屏幕外的高斯点
      if (screenPos.x + radius < 0.0 || screenPos.x - radius > uCam.viewport.x ||
          screenPos.y + radius < 0.0 || screenPos.y - radius > uCam.viewport.y) {
        return;
      }

      let visibleIdx = atomicAdd(&drawArgs.instanceCount, 1u);

      var out: ProjectedSplat;
      out.xy = ndc;
      out.conic = conic;
      out.radius = radius;
      out.color = unpack4x8unorm(splat.colorPacked);
      out.depth = -viewPos.z;
      out.clipZ = clipPos.z / clipPos.w;

      splatsOut[visibleIdx] = out;
      sortKeys[visibleIdx] = bitcast<u32>(out.depth); // 深度转为 bitcast 用于排序
      sortValues[visibleIdx] = visibleIdx;
    }
  `;

  // 1.2 GPU Clear Shader 
  // 【核心修复 3】：替换原来的 reset padding。彻底清空上一帧残留的“幽灵排序数据”！
  const clearKeysShaderCode = /* wgsl */ `
    struct SortUniform {
      numElements: u32,
      stage: u32,
      step: u32,
      _pad: u32,
    };

    @group(0) @binding(0) var<uniform> uSort: SortUniform;
    @group(0) @binding(1) var<storage, read_write> keys: array<u32>;
    @group(0) @binding(2) var<storage, read_write> values: array<u32>;

    @compute @workgroup_size(256, 1)
    fn cs_clear_keys(
      @builtin(workgroup_id) wid: vec3u,
      @builtin(local_invocation_id) lid: vec3u,
      @builtin(num_workgroups) nwid: vec3u
    ) {
      let idx = (wid.y * nwid.x + wid.x) * 256u + lid.x;
      if (idx >= uSort.numElements) { return; }
      
      // 清空所有的深度 Key。在降序排序中，0u 会自然沉底到数组最后，不会干扰画面。
      keys[idx] = 0u; 
      values[idx] = 0u;
    }
  `;

  // 1.3 GPU Bitonic Sort 
  const sortShaderCode = /* wgsl */ `
    struct SortUniform {
      numElements: u32,
      stage: u32,
      step: u32,
      _pad: u32,
    };

    @group(0) @binding(0) var<uniform> uSort: SortUniform;
    @group(0) @binding(1) var<storage, read_write> keys: array<u32>;
    @group(0) @binding(2) var<storage, read_write> values: array<u32>;

    @compute @workgroup_size(256, 1)
    fn cs_bitonic_step(
      @builtin(workgroup_id) wid: vec3u,
      @builtin(local_invocation_id) lid: vec3u,
      @builtin(num_workgroups) nwid: vec3u
    ) {
      let i = (wid.y * nwid.x + wid.x) * 256u + lid.x;
      if (i >= uSort.numElements) { return; }

      let step = uSort.step;
      let stage = uSort.stage;

      let pairDistance = 1u << (step - 1u);
      let blockWidth   = 2u * pairDistance;

      let leftId = (i / pairDistance) * blockWidth + (i % pairDistance);
      let rightId = leftId + pairDistance;

      if (rightId >= uSort.numElements) { return; }

      let sameDir = ((leftId / (1u << stage)) % 2u) == 0u;
      let keyL = keys[leftId];
      let keyR = keys[rightId];

      var swap = keyL < keyR;
      if (!sameDir) {
        swap = keyL > keyR;
      }

      if (swap) {
        keys[leftId] = keyR;
        keys[rightId] = keyL;
        let valL = values[leftId];
        values[leftId] = values[rightId];
        values[rightId] = valL;
      }
    }
  `;

  // 1.4 高清光栅化着色器
  const renderShaderCode = /* wgsl */ `
    struct CameraUniform {
      view: mat4x4f,
      proj: mat4x4f,
      camPos: vec3f,
      _pad0: f32,
      viewport: vec2f,
      focal: vec2f,
    };

    struct ProjectedSplat {
      xy: vec2f,
      conic: vec3f,
      radius: f32,
      color: vec4f,
      depth: f32,
      clipZ: f32,
    };

    @group(0) @binding(0) var<uniform> uCam: CameraUniform;
    @group(0) @binding(1) var<storage, read> splatsOut: array<ProjectedSplat>;
    @group(0) @binding(2) var<storage, read> sortedIndices: array<u32>;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) color: vec4f,
      @location(1) conic: vec3f,
      @location(2) uv: vec2f,
    };

    @vertex
    fn vs_render(@builtin(vertex_index) vIdx: u32, @builtin(instance_index) instIdx: u32) -> VertexOutput {
      let splatIndex = sortedIndices[instIdx];
      let g = splatsOut[splatIndex];

      let quadOffset = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
      );
      let offset = quadOffset[vIdx];
      let screenDelta = (offset * g.radius) / uCam.viewport * 2.0;

      return VertexOutput(
        // 【核心修复 4】：强制 Z 坐标为 0.5！
        // 彻底禁止 WebGPU 底层的固定裁剪管线，防止相机拉近时大面片越过近裁减面突然消失闪烁！
        vec4f(g.xy + screenDelta, 0.5, 1.0),
        g.color,
        g.conic,
        offset * g.radius
      );
    }

    @fragment
    fn fs_render(in: VertexOutput) -> @location(0) vec4f {
      let d = in.uv;
      let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) - in.conic.y * d.x * d.y;
      if (power > 0.0) { discard; }

      let alpha = in.color.a * exp(power);
      if (alpha < 0.005) { discard; }

      return vec4f(in.color.rgb * alpha, alpha);
    }
  `;

  // =========================================================================
  // 2. 2D 工作组调度辅助函数
  // =========================================================================
  const WORKGROUP_SIZE = 256;
  const MAX_WG_DIM_X = 1024;

  function dispatch2D(pass: GPUComputePassEncoder, totalInvocations: number) {
    const totalWorkgroups = Math.ceil(totalInvocations / WORKGROUP_SIZE);
    if (totalWorkgroups === 0) return;

    if (totalWorkgroups <= MAX_WG_DIM_X) {
      pass.dispatchWorkgroups(totalWorkgroups, 1, 1);
    } else {
      const dimX = MAX_WG_DIM_X;
      const dimY = Math.ceil(totalWorkgroups / dimX);
      pass.dispatchWorkgroups(dimX, dimY, 1);
    }
  }

  // =========================================================================
  // 3. 解析模型与管线绑定 (与前面相同)
  // =========================================================================
  function parsePLYRobust(buffer: ArrayBuffer) {
    const bytes = new Uint8Array(buffer);
    let headerEnd = -1;
    for (let i = 0; i < Math.min(bytes.length, 20000); i++) {
      if (bytes[i] === 101 && bytes[i + 1] === 110 && bytes[i + 2] === 100 &&
          bytes[i + 3] === 95 && bytes[i + 4] === 104 && bytes[i + 5] === 101 &&
          bytes[i + 6] === 97 && bytes[i + 7] === 100 && bytes[i + 8] === 101 &&
          bytes[i + 9] === 114) {
        let j = i + 10;
        while (bytes[j] === 0x0D || bytes[j] === 0x0A) j++;
        headerEnd = j;
        break;
      }
    }
    if (headerEnd === -1) throw new Error("Invalid PLY file");
    const headerText = new TextDecoder().decode(bytes.subarray(0, headerEnd));
    const lines = headerText.split(/\r?\n/);

    let count = 0;
    let currentOffset = 0;
    const propMap: Record<string, number> = {};
    const typeSizes: Record<string, number> = {
      char: 1, uchar: 1, int8: 1, uint8: 1, short: 2, ushort: 2, int16: 2, uint16: 2,
      int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8
    };

    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts[0] === "element" && parts[1] === "vertex") count = parseInt(parts[2]);
      else if (parts[0] === "property") { propMap[parts[2]] = currentOffset; currentOffset += typeSizes[parts[1]] || 4; }
    }

    const stride = currentOffset;
    const maxSupported = Math.floor(device.limits.maxStorageBufferBindingSize / 48);
    const splatCount = Math.min(count, maxSupported);
    const dataView = new DataView(buffer, headerEnd);
    const gpuData = new ArrayBuffer(splatCount * 48);
    const f32 = new Float32Array(gpuData);
    const u32 = new Uint32Array(gpuData);
    const SH_C0 = 0.28209479177387814;
    let sumX = 0, sumY = 0, sumZ = 0;

    for (let i = 0; i < splatCount; i++) {
      const row = i * stride; const base = i * 12;
      const x = dataView.getFloat32(row + propMap["x"], true);
      const y = -dataView.getFloat32(row + propMap["y"], true);
      const z = -dataView.getFloat32(row + propMap["z"], true);
      f32[base + 0] = x; f32[base + 1] = y; f32[base + 2] = z;
      sumX += x; sumY += y; sumZ += z;

      const r = Math.max(0, Math.min(255, Math.floor((0.5 + SH_C0 * dataView.getFloat32(row + propMap["f_dc_0"], true)) * 255)));
      const g = Math.max(0, Math.min(255, Math.floor((0.5 + SH_C0 * dataView.getFloat32(row + propMap["f_dc_1"], true)) * 255)));
      const b = Math.max(0, Math.min(255, Math.floor((0.5 + SH_C0 * dataView.getFloat32(row + propMap["f_dc_2"], true)) * 255)));
      const op = dataView.getFloat32(row + propMap["opacity"], true);
      const a = Math.max(0, Math.min(255, Math.floor((1.0 / (1.0 + Math.exp(-op))) * 255)));
      u32[base + 3] = (a << 24) | (b << 16) | (g << 8) | r;

      f32[base + 4] = Math.exp(dataView.getFloat32(row + propMap["scale_0"], true));
      f32[base + 5] = Math.exp(dataView.getFloat32(row + propMap["scale_1"], true));
      f32[base + 6] = Math.exp(dataView.getFloat32(row + propMap["scale_2"], true));
      f32[base + 7] = 0.0;

      const q0 = dataView.getFloat32(row + propMap["rot_0"], true);
      const q1 = dataView.getFloat32(row + propMap["rot_1"], true);
      const q2 = -dataView.getFloat32(row + propMap["rot_2"], true);
      const q3 = -dataView.getFloat32(row + propMap["rot_3"], true);
      const qlen = Math.hypot(q0, q1, q2, q3) || 1.0;
      f32[base + 8] = q0 / qlen; f32[base + 9] = q1 / qlen; f32[base + 10] = q2 / qlen; f32[base + 11] = q3 / qlen;
    }
    return { splatCount, gpuData, center: [sumX / splatCount, sumY / splatCount, sumZ / splatCount] as [number, number, number] };
  }

  function perspective(fovY: number, aspect: number, zNear: number, zFar: number) {
    const f = 1.0 / Math.tan(fovY / 2);
    return [
      f / aspect, 0, 0, 0, 0, f, 0, 0,
      0, 0, zFar / (zNear - zFar), -1, 0, 0, (zNear * zFar) / (zNear - zFar), 0
    ];
  }

  function lookAt(eye: number[], center: number[], up: number[]) {
    let zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    const rz = 1 / Math.hypot(zx, zy, zz); zx *= rz; zy *= rz; zz *= rz;
    let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
    const rx = 1 / Math.hypot(xx, xy, xz); xx *= rx; xy *= rx; xz *= rx;
    let yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    return [
      xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
      -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
      -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
      -(zx * eye[0] + zy * eye[1] + zz * eye[2]), 1
    ];
  }

  function nextPowerOfTwo(n: number) { return Math.pow(2, Math.ceil(Math.log2(Math.max(n, 2)))); }

  const moduleCulling = device.createShaderModule({ code: cullingShaderCode });
  const moduleClearKeys = device.createShaderModule({ code: clearKeysShaderCode }); // 更新名字
  const moduleSort = device.createShaderModule({ code: sortShaderCode });
  const moduleRender = device.createShaderModule({ code: renderShaderCode });

  const cameraBuffer = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const drawArgsBuffer = device.createBuffer({
    size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
  });

  const MAX_STAGES = 24; 
  const MAX_STEPS = (MAX_STAGES * (MAX_STAGES + 1)) / 2;
  const sortUniformBuffer = device.createBuffer({ size: MAX_STEPS * 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const bglCompute = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });

  const bglSort = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });

  const bglRender = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });

  const pipelineCompute = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bglCompute] }),
    compute: { module: moduleCulling, entryPoint: "cs_cull_and_project" },
  });

  const pipelineClearKeys = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bglSort] }),
    compute: { module: moduleClearKeys, entryPoint: "cs_clear_keys" },
  });

  const pipelineSort = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bglSort] }),
    compute: { module: moduleSort, entryPoint: "cs_bitonic_step" },
  });

  const pipelineRender = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bglRender] }),
    vertex: { module: moduleRender, entryPoint: "vs_render" },
    fragment: {
      module: moduleRender,
      entryPoint: "fs_render",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        },
      }],
    },
    primitive: { topology: "triangle-list" },
  });

  let splatCount = 0, sortCapacity = 0, sceneCenter = [0, 0, 0], camDist = 3.5, isReady = false;
  let bufferSplatsIn: GPUBuffer, bufferSplatsOut: GPUBuffer, bufferSortKeys: GPUBuffer, bufferSortValues: GPUBuffer;
  let bindGroupCompute: GPUBindGroup, bindGroupSort: GPUBindGroup, bindGroupRender: GPUBindGroup;
  
  let sortStepsOffsetMap: number[] = [];
  let numStages = 0;

  function loadPLYData(arrayBuffer: ArrayBuffer) {
    const res = parsePLYRobust(arrayBuffer);
    splatCount = res.splatCount;
    sceneCenter = res.center;
    sortCapacity = nextPowerOfTwo(splatCount);

    numStages = Math.log2(sortCapacity);
    const sortUniformData = new Uint32Array(MAX_STEPS * 64);
    sortStepsOffsetMap = [];
    let offsetIdx = 0;

    for (let stage = 1; stage <= numStages; stage++) {
      for (let step = stage; step >= 1; step--) {
        const base = offsetIdx * 64;
        sortUniformData[base + 0] = sortCapacity;
        sortUniformData[base + 1] = stage;
        sortUniformData[base + 2] = step;
        sortUniformData[base + 3] = 0; // 不再需要 validCount
        sortStepsOffsetMap.push(offsetIdx * 256);
        offsetIdx++;
      }
    }
    device.queue.writeBuffer(sortUniformBuffer, 0, sortUniformData);

    bufferSplatsIn = device.createBuffer({ size: res.gpuData.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Uint8Array(bufferSplatsIn.getMappedRange()).set(new Uint8Array(res.gpuData));
    bufferSplatsIn.unmap();

    bufferSplatsOut = device.createBuffer({ size: splatCount * 48, usage: GPUBufferUsage.STORAGE });
    bufferSortKeys = device.createBuffer({ size: sortCapacity * 4, usage: GPUBufferUsage.STORAGE });
    bufferSortValues = device.createBuffer({ size: sortCapacity * 4, usage: GPUBufferUsage.STORAGE });

    bindGroupCompute = device.createBindGroup({
      layout: bglCompute,
      entries: [
        { binding: 0, resource: { buffer: cameraBuffer } },
        { binding: 1, resource: { buffer: bufferSplatsIn } },
        { binding: 2, resource: { buffer: bufferSplatsOut } },
        { binding: 3, resource: { buffer: drawArgsBuffer } },
        { binding: 4, resource: { buffer: bufferSortKeys } },
        { binding: 5, resource: { buffer: bufferSortValues } },
      ],
    });

    bindGroupSort = device.createBindGroup({
      layout: bglSort,
      entries: [
        { binding: 0, resource: { buffer: sortUniformBuffer, size: 16 } },
        { binding: 1, resource: { buffer: bufferSortKeys } },
        { binding: 2, resource: { buffer: bufferSortValues } },
      ],
    });

    bindGroupRender = device.createBindGroup({
      layout: bglRender,
      entries: [
        { binding: 0, resource: { buffer: cameraBuffer } },
        { binding: 1, resource: { buffer: bufferSplatsOut } },
        { binding: 2, resource: { buffer: bufferSortValues } },
      ],
    });

    document.getElementById("gs-status")!.innerHTML = `<span style="color:#0f0;">Ready: ${splatCount.toLocaleString()} Splats</span>`;
    isReady = true;
  }

  // UI Setup
  const canvas = context.canvas as HTMLCanvasElement;
  let rotX = 0.2, rotY = 0.0;
  let isDragging = false, lastX = 0, lastY = 0;

  canvas.addEventListener("mousedown", (e) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener("mouseup", () => isDragging = false);
  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    rotY += (e.clientX - lastX) * 0.005;
    rotX += (e.clientY - lastY) * 0.005;
    rotX = Math.max(-Math.PI * 0.49, Math.min(Math.PI * 0.49, rotX));
    lastX = e.clientX; lastY = e.clientY;
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camDist *= e.deltaY > 0 ? 1.08 : 0.92;
  }, { passive: false });

  const ui = document.createElement("div");
  ui.style.cssText = "position:absolute;top:12px;left:12px;background:rgba(20,20,20,0.85);color:#fff;padding:12px 16px;border-radius:6px;font-family:sans-serif;font-size:12px;z-index:99;";
  ui.innerHTML = `<div style="font-weight:bold;margin-bottom:6px;color:#4CAF50;">WebGPU 3DGS (Zoom Anti-Flicker Fix)</div><input type="file" id="plyLoader" accept=".ply" style="display:block;margin-bottom:6px;" /><div id="gs-status" style="color:#ffeb3b;">Select an official .ply file</div>`;
  document.body.appendChild(ui);
  document.getElementById("plyLoader")!.addEventListener("change", (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    document.getElementById("gs-status")!.innerText = "Uploading to GPU...";
    const reader = new FileReader();
    reader.onload = () => loadPLYData(reader.result as ArrayBuffer);
    reader.readAsArrayBuffer(file);
  });

  // =========================================================================
  // 8. 帧循环
  // =========================================================================
  const cameraUniformData = new Float32Array(40);
  const initialDrawArgs = new Uint32Array([6, 0, 0, 0]);
  let animId = 0;

  function frame() {
    animId = requestAnimationFrame(frame);
    if (!isReady) return;

    const dpr = window.devicePixelRatio || 1;
    const displayWidth = Math.floor(canvas.clientWidth * dpr);
    const displayHeight = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth; canvas.height = displayHeight;
    }

    const aspect = canvas.width / canvas.height;
    const fovY = 45 * (Math.PI / 180);
    const focalY = canvas.height / (2.0 * Math.tan(fovY / 2.0));
    const eye = [
      sceneCenter[0] + Math.sin(rotY) * Math.cos(rotX) * camDist,
      sceneCenter[1] + Math.sin(rotX) * camDist,
      sceneCenter[2] + Math.cos(rotY) * Math.cos(rotX) * camDist
    ];
    const view = lookAt(eye, sceneCenter, [0, 1, 0]);
    const proj = perspective(fovY, aspect, 0.1, 2000.0);

    cameraUniformData.set(view, 0); cameraUniformData.set(proj, 16); cameraUniformData.set(eye, 32);
    cameraUniformData[35] = 0.0; cameraUniformData.set([canvas.width, canvas.height], 36); cameraUniformData.set([focalY, focalY], 38);
    device.queue.writeBuffer(cameraBuffer, 0, cameraUniformData);

    // 每帧充置 atomic counter 为 0
    device.queue.writeBuffer(drawArgsBuffer, 0, initialDrawArgs);

    const encoder = device.createCommandEncoder();

    // 1. 【核心修复 1】: 清零 Pass，彻底清除上一帧未被剔除的残留幽灵坐标，防止乱入排序
    const cpass = encoder.beginComputePass();
    cpass.setPipeline(pipelineClearKeys);
    cpass.setBindGroup(0, bindGroupSort, [sortStepsOffsetMap[0]]);
    dispatch2D(cpass, sortCapacity);

    // 2. Compute Pass: Culling + Project
    cpass.setPipeline(pipelineCompute);
    cpass.setBindGroup(0, bindGroupCompute);
    dispatch2D(cpass, splatCount);

    // 3. Compute Pass: GPU Bitonic Sort 
    cpass.setPipeline(pipelineSort);
    let stepIdx = 0;
    for (let stage = 1; stage <= numStages; stage++) {
      for (let step = stage; step >= 1; step--) {
        cpass.setBindGroup(0, bindGroupSort, [sortStepsOffsetMap[stepIdx]]);
        dispatch2D(cpass, sortCapacity / 2);
        stepIdx++;
      }
    }
    cpass.end();

    // 4. Render Pass: Indirect Draw
    const rpass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    rpass.setPipeline(pipelineRender);
    rpass.setBindGroup(0, bindGroupRender);
    rpass.drawIndirect(drawArgsBuffer, 0);
    rpass.end();

    device.queue.submit([encoder.finish()]);
  }

  frame();

  return () => {
    cancelAnimationFrame(animId);
    cameraBuffer.destroy(); sortUniformBuffer.destroy(); drawArgsBuffer.destroy();
    if (bufferSplatsIn) bufferSplatsIn.destroy();
    if (bufferSplatsOut) bufferSplatsOut.destroy();
    if (bufferSortKeys) bufferSortKeys.destroy();
    if (bufferSortValues) bufferSortValues.destroy();
    document.body.removeChild(ui);
  };
}