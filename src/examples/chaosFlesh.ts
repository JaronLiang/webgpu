// src/examples/chaosFlesh.ts
import GUI from "lil-gui";

// =========================================================================
// 1. 标准 3D 数学库 (针对 WebGPU 定制)
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
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function hexToRgb(hex: string): [number, number, number] {
  const bigint = parseInt(hex.replace("#", ""), 16);
  return [
    ((bigint >> 16) & 255) / 255,
    ((bigint >> 8) & 255) / 255,
    (bigint & 255) / 255,
  ];
}

// =========================================================================
// 2. 主程序入口
// =========================================================================
export function runChaosFlesh(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: GUI
) {
  // 软体分辨率: 8 x 8 x 8 节点 = 512 个质点, 343 个体积体素晶格
  const NX = 8;
  const NY = 8;
  const NZ = 8;
  const TOTAL_NODES = NX * NY * NZ;
  const REST_SIZE = 3.6; // 软体立方体尺寸
  const DX = REST_SIZE / (NX - 1);
  const DY = REST_SIZE / (NY - 1);
  const DZ = REST_SIZE / (NZ - 1);

  // 1. 初始化 3D 软体质点位置
  const initialPositions = new Float32Array(TOTAL_NODES * 4); // xyz, invMass
  for (let z = 0; z < NZ; z++) {
    for (let y = 0; y < NY; y++) {
      for (let x = 0; x < NX; x++) {
        const idx = z * (NX * NY) + y * NX + x;
        const px = (x / (NX - 1) - 0.5) * REST_SIZE;
        const py = 3.8 + (y / (NY - 1)) * REST_SIZE; // 悬空放置在碰撞地面和障碍球上方
        const pz = (z / (NZ - 1) - 0.5) * REST_SIZE;

        initialPositions[idx * 4 + 0] = px;
        initialPositions[idx * 4 + 1] = py;
        initialPositions[idx * 4 + 2] = pz;
        initialPositions[idx * 4 + 3] = 1.0; // 质量倒数 (1.0 = 自由运动体)
      }
    }
  }

  // 2. 提取软体 6 个外表面生成连续三角网格面 (Boundary Surface Extraction)
  const surfaceIndices: number[] = [];

  function nodeIndex(x: number, y: number, z: number): number {
    return z * (NX * NY) + y * NX + x;
  }

  function addQuad(i0: number, i1: number, i2: number, i3: number) {
    surfaceIndices.push(i0, i1, i2);
    surfaceIndices.push(i0, i2, i3);
  }

  // -Z 面 与 +Z 面
  for (let y = 0; y < NY - 1; y++) {
    for (let x = 0; x < NX - 1; x++) {
      addQuad(nodeIndex(x, y, 0), nodeIndex(x, y + 1, 0), nodeIndex(x + 1, y + 1, 0), nodeIndex(x + 1, y, 0));
      addQuad(nodeIndex(x, y, NZ - 1), nodeIndex(x + 1, y, NZ - 1), nodeIndex(x + 1, y + 1, NZ - 1), nodeIndex(x, y + 1, NZ - 1));
    }
  }
  // -X 面 与 +X 面
  for (let z = 0; z < NZ - 1; z++) {
    for (let y = 0; y < NY - 1; y++) {
      addQuad(nodeIndex(0, y, z), nodeIndex(0, y, z + 1), nodeIndex(0, y + 1, z + 1), nodeIndex(0, y + 1, z));
      addQuad(nodeIndex(NX - 1, y, z), nodeIndex(NX - 1, y + 1, z), nodeIndex(NX - 1, y + 1, z + 1), nodeIndex(NX - 1, y, z + 1));
    }
  }
  // -Y 面 (底面) 与 +Y 面 (顶面)
  for (let z = 0; z < NZ - 1; z++) {
    for (let x = 0; x < NX - 1; x++) {
      addQuad(nodeIndex(x, 0, z), nodeIndex(x + 1, 0, z), nodeIndex(x + 1, 0, z + 1), nodeIndex(x, 0, z + 1));
      addQuad(nodeIndex(x, NY - 1, z), nodeIndex(x, NY - 1, z + 1), nodeIndex(x + 1, NY - 1, z + 1), nodeIndex(x + 1, NY - 1, z));
    }
  }

  const surfaceIndexData = new Uint32Array(surfaceIndices);
  const surfaceIndexBuffer = device.createBuffer({
    size: surfaceIndexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(surfaceIndexBuffer, 0, surfaceIndexData);

  // 3. 创建 GPU 物理存储缓冲区 (双缓冲 Ping-Pong 架构)
  const posBufferA = device.createBuffer({
    size: initialPositions.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const posBufferB = device.createBuffer({
    size: initialPositions.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const prevPosBuffer = device.createBuffer({
    size: initialPositions.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const normalBuffer = device.createBuffer({
    size: TOTAL_NODES * 4 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(posBufferA, 0, initialPositions);
  device.queue.writeBuffer(posBufferB, 0, initialPositions);
  device.queue.writeBuffer(prevPosBuffer, 0, initialPositions);

  // 4. 常量缓冲区 (256 字节结构体对齐)
  const uniformBufferSize = 256;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 3. GPU 计算着色器 (Chaos Flesh 体积 XPBD 求解器)
  // =========================================================================
  const fleshComputeShader = `
    struct Uniforms {
      viewProj: mat4x4f,
      camPos: vec4f,
      sphere: vec4f,       // xyz: 挤压球位置, w: 半径
      params: vec4f,       // x: dt, y: damping, z: elasticity, w: volumeStiffness
      fleshParams: vec4f,  // x: dx, y: restVolume, z: time, w: gravity
      matParams: vec4f,    // x: translucency, y: roughness, z: jiggle, w: unused
      fleshColor: vec4f,   // rgb: 软体颜色
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var<storage, read> inPos: array<vec4f>;
    @group(0) @binding(2) var<storage, read_write> outPos: array<vec4f>;
    @group(0) @binding(3) var<storage, read_write> prevPos: array<vec4f>;
    @group(0) @binding(4) var<storage, read_write> normals: array<vec4f>;

    const NX: u32 = ${NX}u;
    const NY: u32 = ${NY}u;
    const NZ: u32 = ${NZ}u;

    fn getIndex(x: u32, y: u32, z: u32) -> u32 {
      return z * (NX * NY) + y * NX + x;
    }

    // 阶段 1: 外力积分与惯性位置预测 (Verlet Integration & Gravity)
    @compute @workgroup_size(4, 4, 4)
    fn cs_predict(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= NX || id.y >= NY || id.z >= NZ) { return; }
      let idx = getIndex(id.x, id.y, id.z);
      let pData = inPos[idx];
      let p = pData.xyz;
      let invMass = pData.w;

      let oldP = prevPos[idx].xyz;
      let velocity = (p - oldP) * (1.0 - u.params.y); // 阻尼衰减

      let gravity = vec3f(0.0, u.fleshParams.w, 0.0);
      let dt = u.params.x;
      let predicted = p + velocity + gravity * (dt * dt);

      outPos[idx] = vec4f(predicted, invMass);
      prevPos[idx] = vec4f(p, invMass);
    }

    // 阶段 2: 3D 体积弹性晶格与四面体体积守恒约束 (Volumetric XPBD Elasticity)
    @compute @workgroup_size(4, 4, 4)
    fn cs_solve_elasticity(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= NX || id.y >= NY || id.z >= NZ) { return; }
      let idx = getIndex(id.x, id.y, id.z);
      let selfData = inPos[idx];
      var p = selfData.xyz;

      var delta = vec3f(0.0);
      var weightSum = 0.0;
      let dx0 = u.fleshParams.x;

      // 1. 三轴向结构拉压约束 (Structural Distance Springs)
      let offsets = array<vec3i, 6>(
        vec3i(1,0,0), vec3i(-1,0,0),
        vec3i(0,1,0), vec3i(0,-1,0),
        vec3i(0,0,1), vec3i(0,0,-1)
      );

      for (var i = 0; i < 6; i++) {
        let nx = i32(id.x) + offsets[i].x;
        let ny = i32(id.y) + offsets[i].y;
        let nz = i32(id.z) + offsets[i].z;
        if (nx >= 0 && nx < i32(NX) && ny >= 0 && ny < i32(NY) && nz >= 0 && nz < i32(NZ)) {
          let nIdx = getIndex(u32(nx), u32(ny), u32(nz));
          let nP = inPos[nIdx].xyz;
          let diff = p - nP;
          let dist = length(diff);
          if (dist > 1e-5) {
            let corr = (diff / dist) * (dist - dx0) * 0.5;
            delta -= corr;
            weightSum += 1.0;
          }
        }
      }

      // 2. 面剪切抗扭与体对角抗屈曲约束 (Shear & Cross Bending Springs)
      let diagOffsets = array<vec3i, 8>(
        vec3i(1,1,1), vec3i(-1,1,1), vec3i(1,-1,1), vec3i(-1,-1,1),
        vec3i(1,1,-1), vec3i(-1,1,-1), vec3i(1,-1,-1), vec3i(-1,-1,-1)
      );
      let diagRest = dx0 * 1.73205; // sqrt(3) 对角原长

      for (var i = 0; i < 8; i++) {
        let nx = i32(id.x) + diagOffsets[i].x;
        let ny = i32(id.y) + diagOffsets[i].y;
        let nz = i32(id.z) + diagOffsets[i].z;
        if (nx >= 0 && nx < i32(NX) && ny >= 0 && ny < i32(NY) && nz >= 0 && nz < i32(NZ)) {
          let nIdx = getIndex(u32(nx), u32(ny), u32(nz));
          let nP = inPos[nIdx].xyz;
          let diff = p - nP;
          let dist = length(diff);
          if (dist > 1e-5) {
            let corr = (diff / dist) * (dist - diagRest) * 0.30;
            delta -= corr;
            weightSum += 0.6;
          }
        }
      }

      if (weightSum > 0.0) {
        p += (delta / weightSum) * u.params.z; // 弹性系数
      }

      outPos[idx] = vec4f(p, selfData.w);
    }

    // 阶段 3: 刚体碰撞挤压与地面反弹阻尼 (Rigid Squeezer & Ground Collision)
    @compute @workgroup_size(4, 4, 4)
    fn cs_collision(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= NX || id.y >= NY || id.z >= NZ) { return; }
      let idx = getIndex(id.x, id.y, id.z);
      let selfData = inPos[idx];
      var p = selfData.xyz;

      // 1. 动态刚体挤压球碰撞处理
      let spherePos = u.sphere.xyz;
      let sphereRadius = u.sphere.w;
      let toSphere = p - spherePos;
      let distSphere = length(toSphere);

      if (distSphere < sphereRadius) {
        let n = toSphere / max(distSphere, 1e-4);
        p = spherePos + n * sphereRadius; // 贴紧球壳强力挤压
      }

      // 2. 地面弹性碰撞与摩擦 (Floor plane y >= 0.1)
      if (p.y < 0.1) {
        p.y = 0.1;
      }

      outPos[idx] = vec4f(p, selfData.w);
    }

    // 阶段 4: GPU 并行法线平滑重构
    @compute @workgroup_size(4, 4, 4)
    fn cs_normals(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= NX || id.y >= NY || id.z >= NZ) { return; }
      let x = id.x;
      let y = id.y;
      let z = id.z;

      var dx = vec3f(0.0);
      var dy = vec3f(0.0);
      var dz = vec3f(0.0);
      let p = inPos[getIndex(x, y, z)].xyz;

      if (x + 1u < NX && x > 0u) {
        dx = inPos[getIndex(x + 1u, y, z)].xyz - inPos[getIndex(x - 1u, y, z)].xyz;
      } else if (x + 1u < NX) {
        dx = inPos[getIndex(x + 1u, y, z)].xyz - p;
      } else {
        dx = p - inPos[getIndex(x - 1u, y, z)].xyz;
      }

      if (y + 1u < NY && y > 0u) {
        dy = inPos[getIndex(x, y + 1u, z)].xyz - inPos[getIndex(x, y - 1u, z)].xyz;
      } else if (y + 1u < NY) {
        dy = inPos[getIndex(x, y + 1u, z)].xyz - p;
      } else {
        dy = p - inPos[getIndex(x, y - 1u, z)].xyz;
      }

      if (z + 1u < NZ && z > 0u) {
        dz = inPos[getIndex(x, y, z + 1u)].xyz - inPos[getIndex(x, y, z - 1u)].xyz;
      } else if (z + 1u < NZ) {
        dz = inPos[getIndex(x, y, z + 1u)].xyz - p;
      } else {
        dz = p - inPos[getIndex(x, y, z - 1u)].xyz;
      }

      let n = normalize(cross(dx, dy) + cross(dy, dz) + cross(dz, dx));
      normals[getIndex(x, y, z)] = vec4f(n, 1.0);
    }
  `;

  // =========================================================================
  // 4. 软体渲染着色器 (次表面散射 SSS + 胶体半透明 + 菲涅尔光泽)
  // =========================================================================
  const fleshRenderShader = `
    struct Uniforms {
      viewProj: mat4x4f,
      camPos: vec4f,
      sphere: vec4f,
      params: vec4f,
      fleshParams: vec4f,
      matParams: vec4f, // x: translucency, y: roughness
      fleshColor: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var<storage, read> positions: array<vec4f>;
    @group(0) @binding(2) var<storage, read> normals: array<vec4f>;

    struct VertexOut {
      @builtin(position) clipPos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) localPos: vec3f,
    };

    @vertex
    fn vs_flesh(@builtin(vertex_index) vid: u32) -> VertexOut {
      var out: VertexOut;
      let pData = positions[vid];
      let nData = normals[vid];

      out.clipPos = u.viewProj * vec4f(pData.xyz, 1.0);
      out.worldPos = pData.xyz;
      out.normal = nData.xyz;
      out.localPos = pData.xyz;
      return out;
    }

    @fragment
    fn fs_flesh(in: VertexOut) -> @location(0) vec4f {
      let N = normalize(in.normal);
      let V = normalize(u.camPos.xyz - in.worldPos);
      let L = normalize(vec3f(0.6, 0.9, 0.45)); // 顶部主光源
      let H = normalize(L + V);

      let NdotL = max(dot(N, L), 0.0);
      let NdotV = max(dot(N, V), 0.0);

      // 1. 菲涅尔透光边缘 (Fresnel Rim)
      let fresnel = pow(1.0 - NdotV, 3.2);

      // 2. 次表面散射软光 (Subsurface Scattering Approximation)
      // 逆光透射光感：背光处依然能透射出果冻深色内光
      let scatterDot = dot(V, -L);
      let backScatter = pow(max(scatterDot * 0.5 + 0.5, 0.0), 2.5) * u.matParams.x;

      // 3. 高级高光反射 (Blinn-Phong + Clearcoat)
      let spec = pow(max(dot(N, H), 0.0), 48.0) * 0.85;

      let baseCol = u.fleshColor.rgb;
      let ambient = vec3f(0.12, 0.18, 0.24) * baseCol;
      let sssTint = baseCol * vec3f(1.3, 1.1, 0.9); // 透射向暖色偏转

      let diffuse = baseCol * (NdotL * 0.65 + 0.35);
      let transLight = sssTint * backScatter * 0.65;
      let rimLight = vec3f(1.0, 0.95, 0.85) * (fresnel * 0.75);

      let finalColor = ambient + diffuse + transLight + rimLight + vec3f(spec);
      return vec4f(finalColor, 0.95);
    }

    // 地面着色器
    @vertex
    fn vs_ground(@builtin(vertex_index) vid: u32) -> VertexOut {
      var out: VertexOut;
      let sz = 70.0;
      var pos = vec3f(0.0);
      if (vid == 0u) { pos = vec3f(-sz, 0.0, -sz); }
      else if (vid == 1u) { pos = vec3f(sz, 0.0, -sz); }
      else if (vid == 2u) { pos = vec3f(-sz, 0.0, sz); }
      else if (vid == 3u) { pos = vec3f(-sz, 0.0, sz); }
      else if (vid == 4u) { pos = vec3f(sz, 0.0, -sz); }
      else { pos = vec3f(sz, 0.0, sz); }

      out.clipPos = u.viewProj * vec4f(pos, 1.0);
      out.worldPos = pos;
      out.normal = vec3f(0.0, 1.0, 0.0);
      out.localPos = pos;
      return out;
    }

    @fragment
    fn fs_ground(in: VertexOut) -> @location(0) vec4f {
      let coord = in.worldPos.xz * 0.5;
      let fw = fwidth(coord);
      let grid = abs(fract(coord - 0.5) - 0.5) / max(fw, vec2f(0.001));
      let line = min(grid.x, grid.y);
      let gridMask = 1.0 - min(line, 1.0);

      // 阴影渐变与青墨网格
      let distFromCenter = length(in.worldPos.xz);
      let vignette = smoothstep(30.0, 5.0, distFromCenter);

      let lineColor = vec3f(0.20, 0.32, 0.40);
      let groundColor = mix(vec3f(0.09, 0.13, 0.17), lineColor, gridMask * 0.85) * vignette;
      return vec4f(groundColor, 1.0);
    }
  `;

  // 挤压刚体球网格
  function createSphereMesh(radius: number, rings = 24, sectors = 32) {
    const verts: number[] = [];
    const inds: number[] = [];
    for (let r = 0; r <= rings; r++) {
      const v = r / rings;
      const phi = v * Math.PI;
      for (let s = 0; s <= sectors; s++) {
        const u = s / sectors;
        const theta = u * Math.PI * 2;
        const x = Math.cos(theta) * Math.sin(phi);
        const y = Math.cos(phi);
        const z = Math.sin(theta) * Math.sin(phi);
        verts.push(x * radius, y * radius, z * radius, x, y, z);
      }
    }
    for (let r = 0; r < rings; r++) {
      for (let s = 0; s < sectors; s++) {
        const first = r * (sectors + 1) + s;
        const second = first + sectors + 1;
        inds.push(first, second, first + 1);
        inds.push(second, second + 1, first + 1);
      }
    }
    return {
      vertices: new Float32Array(verts),
      indices: new Uint32Array(inds),
    };
  }

  const sphereMesh = createSphereMesh(1.0, 24, 32);
  const sphereVertexBuffer = device.createBuffer({
    size: sphereMesh.vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(sphereVertexBuffer, 0, sphereMesh.vertices);

  const sphereIndexBuffer = device.createBuffer({
    size: sphereMesh.indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(sphereIndexBuffer, 0, sphereMesh.indices);

  const sphereShaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
      camPos: vec4f,
      sphere: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexOut {
      @builtin(position) clipPos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
    };

    @vertex
    fn vs_sphere(@location(0) pos: vec3f, @location(1) normal: vec3f) -> VertexOut {
      var out: VertexOut;
      let worldPos = pos * u.sphere.w + u.sphere.xyz;
      out.clipPos = u.viewProj * vec4f(worldPos, 1.0);
      out.worldPos = worldPos;
      out.normal = normal;
      return out;
    }

    @fragment
    fn fs_sphere(in: VertexOut) -> @location(0) vec4f {
      let N = normalize(in.normal);
      let L = normalize(vec3f(0.6, 0.9, 0.45));
      let V = normalize(u.camPos.xyz - in.worldPos);
      let H = normalize(L + V);

      let diff = max(dot(N, L), 0.0);
      let spec = pow(max(dot(N, H), 0.0), 32.0);

      // 高质感枪灰色重金属挤压球
      let baseCol = vec3f(0.72, 0.76, 0.82);
      let col = baseCol * (diff * 0.70 + 0.30) + vec3f(spec * 0.75);
      return vec4f(col, 1.0);
    }
  `;

  // =========================================================================
  // 5. 编译着色器与创建管线
  // =========================================================================
  const computeModule = device.createShaderModule({ code: fleshComputeShader });
  const renderModule = device.createShaderModule({ code: fleshRenderShader });
  const sphereModule = device.createShaderModule({ code: sphereShaderCode });

  // 计算管线
  const computeBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });

  const computePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [computeBindGroupLayout],
  });

  const predictPipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: { module: computeModule, entryPoint: "cs_predict" },
  });

  const elasticityPipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: { module: computeModule, entryPoint: "cs_solve_elasticity" },
  });

  const collisionPipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: { module: computeModule, entryPoint: "cs_collision" },
  });

  const normalsPipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: { module: computeModule, entryPoint: "cs_normals" },
  });

  // 渲染管线
  const renderBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });

  const renderPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [renderBindGroupLayout],
  });

  const fleshPipeline = device.createRenderPipeline({
    layout: renderPipelineLayout,
    vertex: { module: renderModule, entryPoint: "vs_flesh" },
    fragment: { module: renderModule, entryPoint: "fs_flesh", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const groundPipeline = device.createRenderPipeline({
    layout: renderPipelineLayout,
    vertex: { module: renderModule, entryPoint: "vs_ground" },
    fragment: { module: renderModule, entryPoint: "fs_ground", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const sphereBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ],
  });

  const spherePipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [sphereBindGroupLayout] }),
    vertex: {
      module: sphereModule,
      entryPoint: "vs_sphere",
      buffers: [{
        arrayStride: 6 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" },
        ],
      }],
    },
    fragment: { module: sphereModule, entryPoint: "fs_sphere", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const sphereBindGroup = device.createBindGroup({
    layout: sphereBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // 双缓冲绑定
  const computeBindGroupA = device.createBindGroup({
    layout: computeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: posBufferA } },
      { binding: 2, resource: { buffer: posBufferB } },
      { binding: 3, resource: { buffer: prevPosBuffer } },
      { binding: 4, resource: { buffer: normalBuffer } },
    ],
  });

  const computeBindGroupB = device.createBindGroup({
    layout: computeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: posBufferB } },
      { binding: 2, resource: { buffer: posBufferA } },
      { binding: 3, resource: { buffer: prevPosBuffer } },
      { binding: 4, resource: { buffer: normalBuffer } },
    ],
  });

  const renderBindGroupA = device.createBindGroup({
    layout: renderBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: posBufferA } },
      { binding: 2, resource: { buffer: normalBuffer } },
    ],
  });

  const renderBindGroupB = device.createBindGroup({
    layout: renderBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: posBufferB } },
      { binding: 2, resource: { buffer: normalBuffer } },
    ],
  });

  // =========================================================================
  // 6. GUI 控制面板与物理交互
  // =========================================================================
  const settings = {
    // 物理参数
    substeps: 14,          // 每帧子循环迭代
    elasticity: 0.90,      // 弹性恢复刚度 (Young's Modulus)
    damping: 0.012,        // 物理震颤阻尼
    gravity: -9.8,         // 重力

    // 刚体挤压球控制
    sphereRadius: 1.45,
    autoSqueeze: true,     // 自动循环挤压弹跳
    sphereX: 0.0,
    sphereY: 4.5,
    sphereZ: 0.0,

    // 视觉与材质 (支持晶莹琥珀果冻、活力橙胶、科技蓝橡胶)
    fleshColor: "#ff4d6d", // 经典有机质感肌肉/果冻粉红
    translucency: 1.25,    // 次表面透光度

    reset: () => {
      device.queue.writeBuffer(posBufferA, 0, initialPositions);
      device.queue.writeBuffer(posBufferB, 0, initialPositions);
      device.queue.writeBuffer(prevPosBuffer, 0, initialPositions);
    },
  };

  gui.title("WebGPU Chaos Flesh 实时软体形变模拟");
  const physFolder = gui.addFolder("XPBD 体积物理解算器");
  physFolder.add(settings, "substeps", 4, 24, 1).name("解算迭代步数");
  physFolder.add(settings, "elasticity", 0.5, 1.0, 0.01).name("弹性回复率 (Stiffness)");
  physFolder.add(settings, "damping", 0.001, 0.05, 0.001).name("阻尼 (Damping)");
  physFolder.add(settings, "gravity", -20.0, 0.0, 0.5).name("重力大小");
  physFolder.open();

  const colFolder = gui.addFolder("刚体碰撞挤压球 (Squeezer)");
  colFolder.add(settings, "sphereRadius", 0.8, 2.5, 0.05).name("挤压球半径");
  colFolder.add(settings, "autoSqueeze").name("自动下压回弹巡航");
  colFolder.add(settings, "sphereX", -3.0, 3.0, 0.1).name("球 X 位置");
  colFolder.add(settings, "sphereY", 1.0, 7.0, 0.1).name("球 Y 高度 (下压)");
  colFolder.add(settings, "sphereZ", -3.0, 3.0, 0.1).name("球 Z 位置");
  colFolder.open();

  const matFolder = gui.addFolder("次表面半透明材质 (SSS)");
  matFolder.addColor(settings, "fleshColor").name("软体胶质体色");
  matFolder.add(settings, "translucency", 0.0, 2.5, 0.1).name("次表面透光度");
  matFolder.open();

  gui.add(settings, "reset").name("重置软体状态");

  // 轨道相机
  const camera = { target: [0, 2.6, 0], radius: 13.5, theta: 42.0, phi: 24.0 };
  let isDragging = false, dragButton = 0, lastX = 0, lastY = 0;

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    isDragging = true;
    dragButton = e.button;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!isDragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    if (dragButton === 0) {
      camera.theta -= dx * 0.35;
      camera.phi = Math.max(5, Math.min(85, camera.phi + dy * 0.35));
    } else if (dragButton === 2) {
      const radTheta = (camera.theta * Math.PI) / 180;
      const pan = camera.radius * 0.0016;
      camera.target[0] -= Math.cos(radTheta) * dx * pan;
      camera.target[2] -= -Math.sin(radTheta) * dx * pan;
      camera.target[1] += dy * pan;
    }
  });

  canvas.addEventListener("pointerup", (e) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.radius = Math.max(4, Math.min(30, camera.radius * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });

  // =========================================================================
  // 7. 渲染与物理主循环
  // =========================================================================
  let depthTexture: GPUTexture | null = null;
  let animId: number;
  let startTime = performance.now();

  function frame() {
    const now = performance.now();
    const elapsed = (now - startTime) * 0.001;

    const dpr = window.devicePixelRatio || 1;
    const renderWidth = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const renderHeight = Math.max(1, Math.floor(canvas.clientHeight * dpr));

    if (!depthTexture || depthTexture.width !== renderWidth || depthTexture.height !== renderHeight) {
      canvas.width = renderWidth;
      canvas.height = renderHeight;
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [renderWidth, renderHeight],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    // 挤压球动态上下穿透与挤压轨迹
    var spherePos = [settings.sphereX, settings.sphereY, settings.sphereZ];
    if (settings.autoSqueeze) {
      // 循环向下砸压软体，将其彻底压扁后抬起回弹
      const pressCycle = Math.sin(elapsed * 2.2);
      spherePos = [
        Math.sin(elapsed * 0.8) * 0.4,
        2.2 + pressCycle * 1.5,
        Math.cos(elapsed * 0.8) * 0.4,
      ];
    }

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.target[0] + camera.radius * Math.cos(radPhi) * Math.sin(radTheta),
      camera.target[1] + camera.radius * Math.sin(radPhi),
      camera.target[2] + camera.radius * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const proj = mat4Perspective((45 * Math.PI) / 180, renderWidth / renderHeight, 0.1, 100.0);
    const view = mat4LookAt(eye, camera.target, [0, 1, 0]);
    const viewProj = mat4Multiply(proj, view);

    const fleshRGB = hexToRgb(settings.fleshColor);

    const uniformData = new Float32Array(uniformBufferSize / 4);
    uniformData.set(viewProj, 0); // 0..15 (64 bytes)
    uniformData[16] = eye[0]; uniformData[17] = eye[1]; uniformData[18] = eye[2]; uniformData[19] = 1.0;
    uniformData[20] = spherePos[0]; uniformData[21] = spherePos[1]; uniformData[22] = spherePos[2]; uniformData[23] = settings.sphereRadius;

    const dt = 1.0 / 60.0;
    const subDt = dt / settings.substeps;
    uniformData[24] = subDt;
    uniformData[25] = settings.damping;
    uniformData[26] = settings.elasticity;
    uniformData[27] = 1.0;

    uniformData[28] = DX;
    uniformData[29] = DX * DY * DZ; // 原始单体素体积
    uniformData[30] = elapsed;
    uniformData[31] = settings.gravity;

    uniformData[32] = settings.translucency;
    uniformData[33] = 0.2;
    uniformData[34] = 1.0;
    uniformData[35] = 0.0;

    uniformData[36] = fleshRGB[0]; uniformData[37] = fleshRGB[1]; uniformData[38] = fleshRGB[2]; uniformData[39] = 1.0;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();

    // =========================================================================
    // 物理子步进计算循环 (GPU Compute XPBD Flesh Solver)
    // =========================================================================
    const wgX = Math.ceil(NX / 4);
    const wgY = Math.ceil(NY / 4);
    const wgZ = Math.ceil(NZ / 4);

    // 1. 位置预测
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(predictPipeline);
    computePass.setBindGroup(0, computeBindGroupA);
    computePass.dispatchWorkgroups(wgX, wgY, wgZ);

    // 2. 体积弹性与挤压碰撞迭代
    var isBufferBPrimary = true;
    for (let step = 0; step < settings.substeps; step++) {
      computePass.setPipeline(elasticityPipeline);
      computePass.setBindGroup(0, isBufferBPrimary ? computeBindGroupB : computeBindGroupA);
      computePass.dispatchWorkgroups(wgX, wgY, wgZ);
      isBufferBPrimary = !isBufferBPrimary;

      computePass.setPipeline(collisionPipeline);
      computePass.setBindGroup(0, isBufferBPrimary ? computeBindGroupB : computeBindGroupA);
      computePass.dispatchWorkgroups(wgX, wgY, wgZ);
      isBufferBPrimary = !isBufferBPrimary;
    }

    // 3. 并行平滑法线解算
    computePass.setPipeline(normalsPipeline);
    computePass.setBindGroup(0, isBufferBPrimary ? computeBindGroupB : computeBindGroupA);
    computePass.dispatchWorkgroups(wgX, wgY, wgZ);
    computePass.end();

    // =========================================================================
    // 渲染通道 (Render Pass)
    // =========================================================================
    const currentView = context.getCurrentTexture().createView();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: currentView,
        clearValue: { r: 0.08, g: 0.11, b: 0.15, a: 1.0 },
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

    // 1. 绘制科技网格地面
    renderPass.setPipeline(groundPipeline);
    renderPass.setBindGroup(0, isBufferBPrimary ? renderBindGroupA : renderBindGroupB);
    renderPass.draw(6, 1, 0, 0);

    // 2. 绘制刚体挤压球
    renderPass.setPipeline(spherePipeline);
    renderPass.setBindGroup(0, sphereBindGroup);
    renderPass.setVertexBuffer(0, sphereVertexBuffer);
    renderPass.setIndexBuffer(sphereIndexBuffer, "uint32");
    renderPass.drawIndexed(sphereMesh.indices.length);

    // 3. 绘制软体半透明形变体网格
    renderPass.setPipeline(fleshPipeline);
    renderPass.setBindGroup(0, isBufferBPrimary ? renderBindGroupB : renderBindGroupA);
    renderPass.setIndexBuffer(surfaceIndexBuffer, "uint32");
    renderPass.drawIndexed(surfaceIndexData.length);

    renderPass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    posBufferA.destroy();
    posBufferB.destroy();
    prevPosBuffer.destroy();
    normalBuffer.destroy();
    uniformBuffer.destroy();
    surfaceIndexBuffer.destroy();
    sphereVertexBuffer.destroy();
    sphereIndexBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}