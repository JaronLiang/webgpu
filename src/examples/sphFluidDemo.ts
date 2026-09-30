// src/examples/sphFluidDemo.ts
import { Mat4 } from "../utils/math";
import type { SimpleGUI } from "../utils/gui";

// 相机 View 矩阵生成函数
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

export async function runSPHFluidDemo(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  const PARTICLE_COUNT = 8192;
  const WORKGROUP_SIZE = 64;

  // =====================================================================
  // 1. WGSL 着色器源码
  // =====================================================================
  const sphShaderCode = `
    struct SPHParams {
      h: f32,
      h2: f32,
      dt: f32,
      rho0: f32,
      stiffness: f32,
      viscosity: f32,
      particleMass: f32,
      particleCount: u32,
      gravity: vec3f,
      pad0: f32,
      boxMin: vec3f,
      pad1: f32,
      boxMax: vec3f,
      pad2: f32,
      gridDim: vec3u,
      gridCellSize: f32,
    };

    struct Particle {
      pos: vec3f,
      density: f32,
      vel: vec3f,
      pressure: f32,
      force: vec3f,
      pad: f32,
    };

    struct SortEntry {
      cellHash: u32,
      particleIndex: u32,
    };

    const PI: f32 = 3.141592653589793;

    @group(0) @binding(0) var<uniform> u: SPHParams;
    @group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
    @group(0) @binding(2) var<storage, read_write> sortTable: array<SortEntry>;
    @group(0) @binding(3) var<storage, read_write> cellOffsets: array<u32>;

    fn getCellCoord(pos: vec3f) -> vec3i {
      let rel = (pos - u.boxMin) / u.gridCellSize;
      return vec3i(floor(rel));
    }

    fn getCellHash(cell: vec3i) -> u32 {
      if (cell.x < 0 || cell.x >= i32(u.gridDim.x) ||
          cell.y < 0 || cell.y >= i32(u.gridDim.y) ||
          cell.z < 0 || cell.z >= i32(u.gridDim.z)) {
        return 0xFFFFFFFFu;
      }
      return u32(cell.x) + u32(cell.y) * u.gridDim.x + u32(cell.z) * u.gridDim.x * u.gridDim.y;
    }

    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn cs_calc_hashes(@builtin(global_invocation_id) id: vec3u) {
      let idx = id.x;
      if (idx >= u.particleCount) { return; }

      let cell = getCellCoord(particles[idx].pos);
      let hash = getCellHash(cell);
      sortTable[idx].cellHash = hash;
      sortTable[idx].particleIndex = idx;
    }

    struct BitonicParams {
      stage: u32,
      stepOfStage: u32,
    };
    @group(1) @binding(0) var<uniform> bts: BitonicParams;

    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn cs_bitonic_sort(@builtin(global_invocation_id) id: vec3u) {
      let i = id.x;
      if (i >= u.particleCount) { return; }

      let pairDistance = 1u << (bts.stage - bts.stepOfStage);
      let blockWidth   = 2u * pairDistance;

      let leftId = (i / pairDistance) * blockWidth + (i % pairDistance);
      let rightId = leftId + pairDistance;

      if (rightId >= u.particleCount) { return; }

      let sameDirectionBlock = ((i * 2u) / (1u << (bts.stage + 1u))) % 2u == 0u;

      let leftEntry = sortTable[leftId];
      let rightEntry = sortTable[rightId];

      var shouldSwap = leftEntry.cellHash > rightEntry.cellHash;
      if (!sameDirectionBlock) {
        shouldSwap = leftEntry.cellHash < rightEntry.cellHash;
      }

      if (shouldSwap) {
        sortTable[leftId] = rightEntry;
        sortTable[rightId] = leftEntry;
      }
    }

    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn cs_clear_offsets(@builtin(global_invocation_id) id: vec3u) {
      let totalCells = u.gridDim.x * u.gridDim.y * u.gridDim.z;
      if (id.x <= totalCells) {
        cellOffsets[id.x] = 0xFFFFFFFFu;
      }
    }

    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn cs_build_offsets(@builtin(global_invocation_id) id: vec3u) {
      let idx = id.x;
      if (idx >= u.particleCount) { return; }

      let hash = sortTable[idx].cellHash;
      if (hash == 0xFFFFFFFFu) { return; }

      if (idx == 0u) {
        cellOffsets[hash] = idx;
      } else {
        let prevHash = sortTable[idx - 1u].cellHash;
        if (hash != prevHash) {
          cellOffsets[hash] = idx;
        }
      }
    }

    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn cs_density_pressure(@builtin(global_invocation_id) id: vec3u) {
      let pIdx = id.x;
      if (pIdx >= u.particleCount) { return; }

      let pos_i = particles[pIdx].pos;
      let centerCell = getCellCoord(pos_i);
      let poly6 = 315.0 / (64.0 * PI * pow(u.h, 9.0));
      var density = 0.0;

      for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
          for (var x = -1; x <= 1; x++) {
            let neighborCell = centerCell + vec3i(x, y, z);
            let nHash = getCellHash(neighborCell);
            if (nHash == 0xFFFFFFFFu) { continue; }

            let startIndex = cellOffsets[nHash];
            if (startIndex == 0xFFFFFFFFu) { continue; }

            var cur = startIndex;
            while (cur < u.particleCount && sortTable[cur].cellHash == nHash) {
              let j = sortTable[cur].particleIndex;
              let diff = pos_i - particles[j].pos;
              let r2 = dot(diff, diff);

              if (r2 < u.h2) {
                density += u.particleMass * poly6 * pow(u.h2 - r2, 3.0);
              }
              cur++;
            }
          }
        }
      }

      density = max(density, u.rho0);
      particles[pIdx].density = density;
      particles[pIdx].pressure = u.stiffness * (density - u.rho0);
    }

    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn cs_force_integration(@builtin(global_invocation_id) id: vec3u) {
      let pIdx = id.x;
      if (pIdx >= u.particleCount) { return; }

      let pos_i = particles[pIdx].pos;
      let vel_i = particles[pIdx].vel;
      let rho_i = particles[pIdx].density;
      let press_i = particles[pIdx].pressure;

      let spikyGrad = -45.0 / (PI * pow(u.h, 6.0));
      let viscLap = 45.0 / (PI * pow(u.h, 6.0));

      var f_pressure = vec3f(0.0);
      var f_viscosity = vec3f(0.0);

      let centerCell = getCellCoord(pos_i);

      for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
          for (var x = -1; x <= 1; x++) {
            let neighborCell = centerCell + vec3i(x, y, z);
            let nHash = getCellHash(neighborCell);
            if (nHash == 0xFFFFFFFFu) { continue; }

            let startIndex = cellOffsets[nHash];
            if (startIndex == 0xFFFFFFFFu) { continue; }

            var cur = startIndex;
            while (cur < u.particleCount && sortTable[cur].cellHash == nHash) {
              let j = sortTable[cur].particleIndex;
              if (j != pIdx) {
                let diff = pos_i - particles[j].pos;
                let r = length(diff);

                if (r > 1e-4 && r < u.h) {
                  let rDir = diff / r;
                  let rho_j = particles[j].density;
                  let press_j = particles[j].pressure;

                  let pFactor = u.particleMass * (press_i / (rho_i * rho_i) + press_j / (rho_j * rho_j));
                  f_pressure += -rDir * pFactor * spikyGrad * pow(u.h - r, 2.0);

                  let vFactor = (particles[j].vel - vel_i) * (u.particleMass / rho_j);
                  f_viscosity += u.viscosity * vFactor * viscLap * (u.h - r);
                }
              }
              cur++;
            }
          }
        }
      }

      let f_gravity = u.gravity * rho_i;
      let totalForce = f_pressure + f_viscosity + f_gravity;

      let accel = totalForce / rho_i;
      var newVel = vel_i + accel * u.dt;
      var newPos = pos_i + newVel * u.dt;

      let damping = -0.45;
      let eps = 0.005;

      if (newPos.x < u.boxMin.x + eps) { newPos.x = u.boxMin.x + eps; newVel.x *= damping; }
      if (newPos.x > u.boxMax.x - eps) { newPos.x = u.boxMax.x - eps; newVel.x *= damping; }
      if (newPos.y < u.boxMin.y + eps) { newPos.y = u.boxMin.y + eps; newVel.y *= damping; }
      if (newPos.y > u.boxMax.y - eps) { newPos.y = u.boxMax.y - eps; newVel.y *= damping; }
      if (newPos.z < u.boxMin.z + eps) { newPos.z = u.boxMin.z + eps; newVel.z *= damping; }
      if (newPos.z > u.boxMax.z - eps) { newPos.z = u.boxMax.z - eps; newVel.z *= damping; }

      particles[pIdx].vel = newVel;
      particles[pIdx].pos = newPos;
      particles[pIdx].force = totalForce;
    }
  `;

  const renderShaderCode = `
    struct SceneParams {
      viewProj: mat4x4f,
      camPos: vec3f,
      particleRadius: f32,
    };
    @group(0) @binding(0) var<uniform> scene: SceneParams;
    @group(0) @binding(1) var<storage, read> particles: array<Particle>;

    struct Particle {
      pos: vec3f,
      density: f32,
      vel: vec3f,
      pressure: f32,
      force: vec3f,
      pad: f32,
    };

    struct VertexOutput {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
      @location(1) speed: f32,
    };

    @vertex
    fn vs_main(
      @builtin(vertex_index) vIdx: u32,
      @builtin(instance_index) iIdx: u32
    ) -> VertexOutput {
      var quadOffset = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
      );

      let p = particles[iIdx];
      let offset2D = quadOffset[vIdx] * scene.particleRadius;

      let toCam = normalize(scene.camPos - p.pos);
      let up = vec3f(0.0, 1.0, 0.0);
      let right = normalize(cross(up, toCam));
      let realUp = cross(toCam, right);

      let worldPos = p.pos + (right * offset2D.x + realUp * offset2D.y);

      var out: VertexOutput;
      out.pos = scene.viewProj * vec4f(worldPos, 1.0);
      out.uv = quadOffset[vIdx];
      out.speed = length(p.vel);
      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let dist2 = dot(in.uv, in.uv);
      if (dist2 > 1.0) {
        discard;
      }

      let normal = vec3f(in.uv, sqrt(1.0 - dist2));
      let lightDir = normalize(vec3f(0.4, 0.8, 0.5));
      let diff = max(dot(normal, lightDir), 0.0);
      let spec = pow(max(dot(normal, lightDir), 0.0), 32.0);

      let t = clamp(in.speed * 0.45, 0.0, 1.0);
      let waterColor = mix(vec3f(0.08, 0.35, 0.88), vec3f(0.3, 0.85, 0.98), t);
      let finalColor = waterColor * (diff * 0.65 + 0.35) + vec3f(spec * 0.5);

      return vec4f(finalColor, 0.9);
    }
  `;

  // =====================================================================
  // 2. 空间几何参数与 Buffer 初始化
  // =====================================================================
  const h = 0.045;
  const boxMin = [-0.6, 0.0, -0.6];
  const boxMax = [0.6, 1.2, 0.6];
  const gridDim = [
    Math.ceil((boxMax[0] - boxMin[0]) / h),
    Math.ceil((boxMax[1] - boxMin[1]) / h),
    Math.ceil((boxMax[2] - boxMin[2]) / h),
  ];
  const totalCells = gridDim[0] * gridDim[1] * gridDim[2];

  const particleData = new Float32Array(PARTICLE_COUNT * 12);
  let pIdx = 0;
  const side = Math.cbrt(PARTICLE_COUNT);
  const spacing = h * 0.45;

  for (let z = 0; z < side; z++) {
    for (let y = 0; y < side * 1.5; y++) {
      for (let x = 0; x < side; x++) {
        if (pIdx >= PARTICLE_COUNT) break;
        const offset = pIdx * 12;
        particleData[offset + 0] = boxMin[0] + 0.05 + x * spacing + (Math.random() - 0.5) * 0.005;
        particleData[offset + 1] = boxMin[1] + 0.1 + y * spacing;
        particleData[offset + 2] = boxMin[2] + 0.05 + z * spacing + (Math.random() - 0.5) * 0.005;
        particleData[offset + 3] = 1000.0;
        particleData[offset + 4] = 0;
        particleData[offset + 5] = 0;
        particleData[offset + 6] = 0;
        particleData[offset + 7] = 0;
        pIdx++;
      }
    }
  }

  const particleBuffer = device.createBuffer({
    size: particleData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(particleBuffer, 0, particleData);

  const sortTableBuffer = device.createBuffer({
    size: PARTICLE_COUNT * 8,
    usage: GPUBufferUsage.STORAGE,
  });

  const cellOffsetsBuffer = device.createBuffer({
    size: (totalCells + 1) * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  const sphParamsBuffer = device.createBuffer({
    size: 24 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sphConfig = {
    h: h,
    h2: h * h,
    dt: 0.003,
    rho0: 1000.0,
    stiffness: 45.0,
    viscosity: 0.06,
    particleMass: 0.02,
    gravity: [0.0, -9.8, 0.0],
  };

  function updateSPHUniforms() {
    const raw = new ArrayBuffer(24 * 4);
    const f32 = new Float32Array(raw);
    const u32 = new Uint32Array(raw);
    f32[0] = sphConfig.h;
    f32[1] = sphConfig.h2;
    f32[2] = sphConfig.dt;
    f32[3] = sphConfig.rho0;
    f32[4] = sphConfig.stiffness;
    f32[5] = sphConfig.viscosity;
    f32[6] = sphConfig.particleMass;
    u32[7] = PARTICLE_COUNT;
    f32[8] = sphConfig.gravity[0];
    f32[9] = sphConfig.gravity[1];
    f32[10] = sphConfig.gravity[2];
    f32[11] = 0.0;
    f32[12] = boxMin[0]; f32[13] = boxMin[1]; f32[14] = boxMin[2]; f32[15] = 0.0;
    f32[16] = boxMax[0]; f32[17] = boxMax[1]; f32[18] = boxMax[2]; f32[19] = 0.0;
    u32[20] = gridDim[0]; u32[21] = gridDim[1]; u32[22] = gridDim[2];
    f32[23] = h;
    device.queue.writeBuffer(sphParamsBuffer, 0, raw);
  }
  updateSPHUniforms();

  const renderUBO = device.createBuffer({ size: 24 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // =====================================================================
  // 3. 显式创建 BindGroupLayout 与 PipelineLayout (关键修复)
  // =====================================================================
  const sphShaderModule = device.createShaderModule({ code: sphShaderCode });

  // 主通用 SPH BindGroupLayout (包含全部 4 个 bindings)
  const sphMainBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });

  // 双调排序阶段参数 BindGroupLayout
  const bitonicBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });

  // 分别指定 PipelineLayout
  const sphCommonPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [sphMainBGL],
  });

  const bitonicPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [sphMainBGL, bitonicBGL],
  });

  const computePipeline_hashes = device.createComputePipeline({
    layout: sphCommonPipelineLayout,
    compute: { module: sphShaderModule, entryPoint: "cs_calc_hashes" },
  });

  const computePipeline_sort = device.createComputePipeline({
    layout: bitonicPipelineLayout,
    compute: { module: sphShaderModule, entryPoint: "cs_bitonic_sort" },
  });

  const computePipeline_clearOffsets = device.createComputePipeline({
    layout: sphCommonPipelineLayout,
    compute: { module: sphShaderModule, entryPoint: "cs_clear_offsets" },
  });

  const computePipeline_buildOffsets = device.createComputePipeline({
    layout: sphCommonPipelineLayout,
    compute: { module: sphShaderModule, entryPoint: "cs_build_offsets" },
  });

  const computePipeline_density = device.createComputePipeline({
    layout: sphCommonPipelineLayout,
    compute: { module: sphShaderModule, entryPoint: "cs_density_pressure" },
  });

  const computePipeline_force = device.createComputePipeline({
    layout: sphCommonPipelineLayout,
    compute: { module: sphShaderModule, entryPoint: "cs_force_integration" },
  });

  // 预生成双调排序所有阶段 BindGroup
  const numStages = Math.log2(PARTICLE_COUNT);
  const bitonicBindGroups: GPUBindGroup[] = [];
  const bitonicBuffers: GPUBuffer[] = [];

  for (let stage = 0; stage < numStages; stage++) {
    for (let stepOfStage = 0; stepOfStage <= stage; stepOfStage++) {
      const buf = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, new Uint32Array([stage, stepOfStage]));
      bitonicBuffers.push(buf);

      const bg = device.createBindGroup({
        layout: bitonicBGL,
        entries: [{ binding: 0, resource: { buffer: buf } }],
      });
      bitonicBindGroups.push(bg);
    }
  }

  // 渲染管线
  const renderShaderModule = device.createShaderModule({ code: renderShaderCode });
  const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: renderShaderModule, entryPoint: "vs_main" },
    fragment: {
      module: renderShaderModule,
      entryPoint: "fs_main",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        },
      }],
    },
    primitive: { topology: "triangle-list" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // 使用显式声明的 sphMainBGL 创建主 BindGroup
  const sphMainBG = device.createBindGroup({
    layout: sphMainBGL,
    entries: [
      { binding: 0, resource: { buffer: sphParamsBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
      { binding: 2, resource: { buffer: sortTableBuffer } },
      { binding: 3, resource: { buffer: cellOffsetsBuffer } },
    ],
  });

  const renderBG = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: renderUBO } },
      { binding: 1, resource: { buffer: particleBuffer } },
    ],
  });

  const depthTexture = device.createTexture({
    size: [canvas.width || 800, canvas.height || 600],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // =====================================================================
  // 4. GUI 控制面板
  // =====================================================================
  gui.add(sphConfig, "stiffness", 5.0, 100.0, 1.0).name("刚度 (Stiffness)").onChange(updateSPHUniforms);
  gui.add(sphConfig, "viscosity", 0.01, 0.5, 0.01).name("黏性 (Viscosity)").onChange(updateSPHUniforms);
  gui.addTextInfo("💧 <b>8192 粒子 SPH 流体动力学</b><br>全流程运行于 GPU Compute Shader，无 CPU 回读。");

  // =====================================================================
  // 5. 模拟与渲染主循环
  // =====================================================================
  let animId: number;
  let time = 0;

  function frame() {
    time += 0.007;
    const encoder = device.createCommandEncoder();

    const substeps = 2;
    for (let step = 0; step < substeps; step++) {
      const cpass = encoder.beginComputePass();

      // 1. 计算空间网格 Hash
      cpass.setPipeline(computePipeline_hashes);
      cpass.setBindGroup(0, sphMainBG);
      cpass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE));

      // 2. 双调并行排序
      cpass.setPipeline(computePipeline_sort);
      cpass.setBindGroup(0, sphMainBG);
      for (let i = 0; i < bitonicBindGroups.length; i++) {
        cpass.setBindGroup(1, bitonicBindGroups[i]);
        cpass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE));
      }

      // 3. 构建单胞起始偏移表
      cpass.setPipeline(computePipeline_clearOffsets);
      cpass.setBindGroup(0, sphMainBG);
      cpass.dispatchWorkgroups(Math.ceil((totalCells + 1) / WORKGROUP_SIZE));

      cpass.setPipeline(computePipeline_buildOffsets);
      cpass.setBindGroup(0, sphMainBG);
      cpass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE));

      // 4. 计算局部密度与压力
      cpass.setPipeline(computePipeline_density);
      cpass.setBindGroup(0, sphMainBG);
      cpass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE));

      // 5. 计算动量受力与位置积分
      cpass.setPipeline(computePipeline_force);
      cpass.setBindGroup(0, sphMainBG);
      cpass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE));

      cpass.end();
    }

    // 渲染 Pass
    const aspect = (canvas.width || 800) / (canvas.height || 600);
    const proj = Mat4.perspective((45 * Math.PI) / 180, aspect, 0.1, 100.0);
    const eye = [Math.sin(time) * 1.8, 1.2, Math.cos(time) * 1.8];
    const view = createLookAtMatrix(eye, [0, 0.4, 0], [0, 1, 0]);
    const viewProj = Mat4.multiply(proj, view);

    const renderData = new Float32Array(24);
    renderData.set(viewProj, 0);
    renderData[16] = eye[0];
    renderData[17] = eye[1];
    renderData[18] = eye[2];
    renderData[19] = h * 0.42;
    device.queue.writeBuffer(renderUBO, 0, renderData);

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.06, g: 0.07, b: 0.1, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBG);
    renderPass.draw(6, PARTICLE_COUNT, 0, 0);
    renderPass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    particleBuffer.destroy();
    sortTableBuffer.destroy();
    cellOffsetsBuffer.destroy();
    sphParamsBuffer.destroy();
    renderUBO.destroy();
    depthTexture.destroy();
    bitonicBuffers.forEach(b => b.destroy());
  };
}