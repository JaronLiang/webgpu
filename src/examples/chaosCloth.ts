// src/examples/chaosCloth.ts
import GUI from "lil-gui";

// =========================================================================
// 1. 3D 矩阵数学库
// =========================================================================
function mat4Perspective(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const out = new Float32Array(16);
  const f = 1.0 / Math.tan(fovRad / 2);
  out[0] = f / aspect; out[5] = f;
  out[10] = far / (near - far); out[11] = -1; out[14] = (near * far) / (near - far);
  return out;
}
// 4x4 矩阵乘法 (WebGPU 列优先标准)
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
export function runChaosCloth(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: GUI
) {
  // 布料网格分辨率: 64 x 64 = 4096 质点, 7938 面片
  const GRID_W = 64;
  const GRID_H = 64;
  const NUM_PARTICLES = GRID_W * GRID_H;
  const CLOTH_SIZE = 7.5; // 布料尺寸
  const DX = CLOTH_SIZE / (GRID_W - 1);
  const DY = CLOTH_SIZE / (GRID_H - 1);

  // 1. 初始化布料初始顶点 (水平悬挂在上方)
  const initialPositions = new Float32Array(NUM_PARTICLES * 4); // xyz, invMass
  for (let y = 0; y < GRID_H; y++) {
    for (let x = 0; x < GRID_W; x++) {
      const idx = y * GRID_W + x;
      const px = (x / (GRID_W - 1) - 0.5) * CLOTH_SIZE;
      const py = 5.2; // 初始悬空高度
      const pz = (y / (GRID_H - 1) - 0.5) * CLOTH_SIZE;

      // 顶部固定两个角 (Pinning constraint)
      let invMass = 1.0;
      if (y === 0 && (x === 0 || x === GRID_W - 1 || x === Math.floor(GRID_W / 2))) {
        invMass = 0.0; // 质量倒数为 0 代表无限大质量（固定锚点）
      }

      initialPositions[idx * 4 + 0] = px;
      initialPositions[idx * 4 + 1] = py;
      initialPositions[idx * 4 + 2] = pz;
      initialPositions[idx * 4 + 3] = invMass;
    }
  }

  // 2. 生成布料三角形索引缓冲
  const indices: number[] = [];
  for (let y = 0; y < GRID_H - 1; y++) {
    for (let x = 0; x < GRID_W - 1; x++) {
      const i0 = y * GRID_W + x;
      const i1 = y * GRID_W + (x + 1);
      const i2 = (y + 1) * GRID_W + x;
      const i3 = (y + 1) * GRID_W + (x + 1);
      indices.push(i0, i2, i1);
      indices.push(i1, i2, i3);
    }
  }
  const indexData = new Uint32Array(indices);
  const indexBuffer = device.createBuffer({
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indexBuffer, 0, indexData);

  // 3. 创建 GPU 质点物理存储缓冲区
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
    size: NUM_PARTICLES * 4 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // 写入初始状态
  device.queue.writeBuffer(posBufferA, 0, initialPositions);
  device.queue.writeBuffer(posBufferB, 0, initialPositions);
  device.queue.writeBuffer(prevPosBuffer, 0, initialPositions);

  // 4. Uniform 物理参数缓冲
  const uniformBufferSize = 256;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 3. GPU 计算着色器 (Chaos XPBD / PBD 物理引擎核心)
  // =========================================================================
  const clothComputeShader = `
    struct Uniforms {
      viewProj: mat4x4f,
      camPos: vec4f,
      sphere: vec4f,       // xyz: 碰撞球中心, w: 半径
      wind: vec4f,         // xyz: 风力矢量, w: 湍流强度
      params: vec4f,       // x: dt, y: damping, z: stiffness, w: time
      clothParams: vec4f,  // x: restDistH, y: restDistV, z: restDistDiag, w: restDistBending
      matParams: vec4f,    // x: wireframe, y: doubleSided, z: sheen, w: unused
      clothColor: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var<storage, read> inPos: array<vec4f>;
    @group(0) @binding(2) var<storage, read_write> outPos: array<vec4f>;
    @group(0) @binding(3) var<storage, read_write> prevPos: array<vec4f>;
    @group(0) @binding(4) var<storage, read_write> normals: array<vec4f>;

    const GRID_W: u32 = ${GRID_W}u;
    const GRID_H: u32 = ${GRID_H}u;

    fn getIndex(x: u32, y: u32) -> u32 {
      return y * GRID_W + x;
    }

    // 阶段 1: 外力积分与位置预测 (Verlet Integration & Aerodynamic Wind)
    @compute @workgroup_size(8, 8)
    fn cs_predict(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= GRID_W || id.y >= GRID_H) { return; }
      let idx = getIndex(id.x, id.y);
      let pData = inPos[idx];
      let p = pData.xyz;
      let invMass = pData.w;

      // 固定锚点受外力不移动
      if (invMass <= 0.0) {
        outPos[idx] = vec4f(p, 0.0);
        return;
      }

      let oldP = prevPos[idx].xyz;
      var velocity = (p - oldP) * (1.0 - u.params.y); // 阻尼

      // 重力加速度
      let gravity = vec3f(0.0, -9.8, 0.0);

      // 空气动力学风力 (带高频正弦波动湍流)
      let t = u.params.w;
      let gust = sin(p.x * 1.5 + t * 4.0) * cos(p.z * 1.2 + t * 3.5) * u.wind.w;
      let dynamicWind = u.wind.xyz + vec3f(gust * 0.5, gust * 0.2, gust * 0.5);
      
      // 表面法线阻力
      let N = normals[idx].xyz;
      let relVel = dynamicWind - velocity;
      let normalForce = N * dot(N, relVel) * 1.2;
      let totalAccel = gravity + dynamicWind * 0.4 + normalForce;

      let dt = u.params.x;
      let predicted = p + velocity + totalAccel * (dt * dt);

      outPos[idx] = vec4f(predicted, invMass);
      prevPos[idx] = vec4f(p, invMass); // 保存用于速度反算
    }

    // 阶段 2: GPU 并行雅可比 PBD 弹性约束求解
    // 包含：结构拉伸约束(水平/竖直) + 剪切抗扭约束(对角) + 抗弯曲褶皱约束(双步跨越)
    @compute @workgroup_size(8, 8)
    fn cs_solve_constraints(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= GRID_W || id.y >= GRID_H) { return; }
      let idx = getIndex(id.x, id.y);
      let selfData = inPos[idx];
      var p = selfData.xyz;
      let invMass = selfData.w;

      if (invMass <= 0.0) {
        outPos[idx] = selfData;
        return;
      }

      var deltaSum = vec3f(0.0);
      var count = 0.0;

      // 1. 水平与竖直结构约束 (Structural Constraints)
      let offsets = array<vec2i, 4>(vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1));
      let restH = u.clothParams.x;
      let restV = u.clothParams.y;

      for (var i = 0; i < 4; i++) {
        let nx = i32(id.x) + offsets[i].x;
        let ny = i32(id.y) + offsets[i].y;
        if (nx >= 0 && nx < i32(GRID_W) && ny >= 0 && ny < i32(GRID_H)) {
          let nIdx = getIndex(u32(nx), u32(ny));
          let nP = inPos[nIdx].xyz;
          let diff = p - nP;
          let d = length(diff);
          if (d > 1e-5) {
            let targetD = select(restV, restH, offsets[i].y == 0);
            let corr = (diff / d) * (d - targetD) * 0.5;
            deltaSum -= corr;
            count += 1.0;
          }
        }
      }

      // 2. 对角剪切抗扭约束 (Shear Constraints)
      let diagOffsets = array<vec2i, 4>(vec2i(1, 1), vec2i(-1, 1), vec2i(1, -1), vec2i(-1, -1));
      let restDiag = u.clothParams.z;
      for (var i = 0; i < 4; i++) {
        let nx = i32(id.x) + diagOffsets[i].x;
        let ny = i32(id.y) + diagOffsets[i].y;
        if (nx >= 0 && nx < i32(GRID_W) && ny >= 0 && ny < i32(GRID_H)) {
          let nIdx = getIndex(u32(nx), u32(ny));
          let nP = inPos[nIdx].xyz;
          let diff = p - nP;
          let d = length(diff);
          if (d > 1e-5) {
            let corr = (diff / d) * (d - restDiag) * 0.35;
            deltaSum -= corr;
            count += 0.8;
          }
        }
      }

      // 3. 抗弯曲褶皱约束 (Bending Constraints, 跨越2个网格)
      let bendOffsets = array<vec2i, 4>(vec2i(2, 0), vec2i(-2, 0), vec2i(0, 2), vec2i(0, -2));
      let restBend = u.clothParams.w;
      for (var i = 0; i < 4; i++) {
        let nx = i32(id.x) + bendOffsets[i].x;
        let ny = i32(id.y) + bendOffsets[i].y;
        if (nx >= 0 && nx < i32(GRID_W) && ny >= 0 && ny < i32(GRID_H)) {
          let nIdx = getIndex(u32(nx), u32(ny));
          let nP = inPos[nIdx].xyz;
          let diff = p - nP;
          let d = length(diff);
          if (d > 1e-5) {
            let corr = (diff / d) * (d - restBend) * 0.25;
            deltaSum -= corr;
            count += 0.5;
          }
        }
      }

      if (count > 0.0) {
        p += (deltaSum / count) * u.params.z; // 刚度系数调节
      }

      outPos[idx] = vec4f(p, invMass);
    }

    // 阶段 3: 刚体碰撞与地面碰撞 (Rigid Sphere Collision & Floor)
    @compute @workgroup_size(8, 8)
    fn cs_collision(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= GRID_W || id.y >= GRID_H) { return; }
      let idx = getIndex(id.x, id.y);
      let selfData = inPos[idx];
      var p = selfData.xyz;
      let invMass = selfData.w;

      if (invMass <= 0.0) {
        outPos[idx] = selfData;
        return;
      }

      // 1. 动态碰撞球体求交 (推斥接触面)
      let sphereCenter = u.sphere.xyz;
      let sphereRadius = u.sphere.w;
      let toP = p - sphereCenter;
      let dist = length(toP);
      if (dist < sphereRadius) {
        let normal = toP / max(dist, 1e-4);
        p = sphereCenter + normal * sphereRadius; // 紧贴球壳推斥
      }

      // 2. 地面碰撞边界 (y >= 0.05)
      if (p.y < 0.05) {
        p.y = 0.05;
      }

      outPos[idx] = vec4f(p, invMass);
    }

    // 阶段 4: GPU 实时法线生成
    @compute @workgroup_size(8, 8)
    fn cs_normals(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= GRID_W || id.y >= GRID_H) { return; }
      let x = id.x;
      let y = id.y;

      let p = inPos[getIndex(x, y)].xyz;
      var dx = vec3f(0.0);
      var dy = vec3f(0.0);

      if (x + 1u < GRID_W && x > 0u) {
        dx = inPos[getIndex(x + 1u, y)].xyz - inPos[getIndex(x - 1u, y)].xyz;
      } else if (x + 1u < GRID_W) {
        dx = inPos[getIndex(x + 1u, y)].xyz - p;
      } else {
        dx = p - inPos[getIndex(x - 1u, y)].xyz;
      }

      if (y + 1u < GRID_H && y > 0u) {
        dy = inPos[getIndex(x, y + 1u)].xyz - inPos[getIndex(x, y - 1u)].xyz;
      } else if (y + 1u < GRID_H) {
        dy = inPos[getIndex(x, y + 1u)].xyz - p;
      } else {
        dy = p - inPos[getIndex(x, y - 1u)].xyz;
      }

      let n = normalize(cross(dy, dx));
      normals[getIndex(x, y)] = vec4f(n, 1.0);
    }
  `;

  // =========================================================================
  // 4. 布料与碰撞体渲染着色器 (双面 PBR 织物微表面光照 + 丝绸光泽 Sheen)
  // =========================================================================
  const renderShaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
      camPos: vec4f,
      sphere: vec4f,
      wind: vec4f,
      params: vec4f,
      clothParams: vec4f,
      matParams: vec4f, // x: wireframe, y: doubleSided, z: sheen
      clothColor: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var<storage, read> positions: array<vec4f>;
    @group(0) @binding(2) var<storage, read> normals: array<vec4f>;

    struct VertexOut {
      @builtin(position) clipPos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) uv: vec2f,
    };

    const GRID_W: u32 = ${GRID_W}u;
    const GRID_H: u32 = ${GRID_H}u;

    @vertex
    fn vs_cloth(@builtin(vertex_index) vid: u32) -> VertexOut {
      var out: VertexOut;
      let pData = positions[vid];
      let nData = normals[vid];

      let ux = f32(vid % GRID_W) / f32(GRID_W - 1u);
      let uy = f32(vid / GRID_W) / f32(GRID_H - 1u);

      out.clipPos = u.viewProj * vec4f(pData.xyz, 1.0);
      out.worldPos = pData.xyz;
      out.normal = nData.xyz;
      out.uv = vec2f(ux, uy);
      return out;
    }

    @fragment
    fn fs_cloth(in: VertexOut, @builtin(front_facing) isFront: bool) -> @location(0) vec4f {
      var N = normalize(in.normal);
      if (!isFront) { N = -N; } // 双面正反向法线

      let V = normalize(u.camPos.xyz - in.worldPos);
      let L = normalize(vec3f(0.5, 0.9, 0.4)); // 主日光方向
      let H = normalize(L + V);

      let NdotL = max(dot(N, L), 0.0);
      let NdotV = max(dot(N, V), 0.0);

      // 织物边缘高光 (Fabric Charlie / Sheen 丝绸柔光效果)
      let sheenFactor = pow(1.0 - NdotV, 3.5) * u.matParams.z;
      let sheenColor = vec3f(1.0, 0.95, 0.9);

      // 织物精细经纬线编织纹理 (Procedural Weave Pattern)
      let weaveUV = in.uv * 120.0;
      let weave = sin(weaveUV.x * 3.1415) * sin(weaveUV.y * 3.1415) * 0.08;

      let baseCol = u.clothColor.rgb + weave;
      let ambient = vec3f(0.12, 0.16, 0.22);
      let diffuse = baseCol * (NdotL * 0.75 + 0.25);
      
      let finalColor = ambient * baseCol + diffuse + sheenColor * sheenFactor;
      return vec4f(finalColor, 1.0);
    }

    // 碰撞球体绘制
    @vertex
    fn vs_sphere(@builtin(vertex_index) vid: u32) -> VertexOut {
      // 程序化生成低面数球体顶点
      var out: VertexOut;
      return out;
    }

    // 地面网格
    @vertex
    fn vs_ground(@builtin(vertex_index) vid: u32) -> VertexOut {
      var out: VertexOut;
      let sz = 60.0;
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
      out.uv = pos.xz;
      return out;
    }

    @fragment
    fn fs_ground(in: VertexOut) -> @location(0) vec4f {
      let coord = in.worldPos.xz * 0.5;
      let fw = fwidth(coord);
      let grid = abs(fract(coord - 0.5) - 0.5) / max(fw, vec2f(0.001));
      let line = min(grid.x, grid.y);
      let gridMask = 1.0 - min(line, 1.0);

      let groundCol = mix(vec3f(0.10, 0.14, 0.18), vec3f(0.18, 0.28, 0.35), gridMask * 0.7);
      return vec4f(groundCol, 1.0);
    }
  `;

  // 碰撞球多边形几何生成 (UV 球体)
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
      wind: vec4f,
      params: vec4f,
      clothParams: vec4f,
      matParams: vec4f,
      clothColor: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexOut {
      @builtin(position) clipPos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
    };

    @vertex
    fn vs_sphere_mesh(@location(0) pos: vec3f, @location(1) normal: vec3f) -> VertexOut {
      var out: VertexOut;
      let worldPos = pos * u.sphere.w + u.sphere.xyz;
      out.clipPos = u.viewProj * vec4f(worldPos, 1.0);
      out.worldPos = worldPos;
      out.normal = normal;
      return out;
    }

    @fragment
    fn fs_sphere_mesh(in: VertexOut) -> @location(0) vec4f {
      let N = normalize(in.normal);
      let L = normalize(vec3f(0.5, 0.9, 0.4));
      let V = normalize(u.camPos.xyz - in.worldPos);
      let H = normalize(L + V);

      let diff = max(dot(N, L), 0.0);
      let spec = pow(max(dot(N, H), 0.0), 32.0);

      // 高级铬合金/哑光金属球体
      let baseCol = vec3f(0.85, 0.88, 0.92);
      let col = baseCol * (diff * 0.75 + 0.25) + vec3f(spec * 0.6);
      return vec4f(col, 1.0);
    }
  `;

  // =========================================================================
  // 5. 编译着色器与构建 GPU 管线
  // =========================================================================
  const computeModule = device.createShaderModule({ code: clothComputeShader });
  const renderModule = device.createShaderModule({ code: renderShaderCode });
  const sphereModule = device.createShaderModule({ code: sphereShaderCode });

  // 计算管线 BindGroupLayout
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

  const solvePipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: { module: computeModule, entryPoint: "cs_solve_constraints" },
  });

  const collisionPipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: { module: computeModule, entryPoint: "cs_collision" },
  });

  const normalsPipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: { module: computeModule, entryPoint: "cs_normals" },
  });

  // 渲染管线 BindGroupLayout
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

  // 布料渲染管线 (双面渲染 CullMode: none)
  const clothPipeline = device.createRenderPipeline({
    layout: renderPipelineLayout,
    vertex: { module: renderModule, entryPoint: "vs_cloth" },
    fragment: { module: renderModule, entryPoint: "fs_cloth", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // 地面渲染管线
  const groundPipeline = device.createRenderPipeline({
    layout: renderPipelineLayout,
    vertex: { module: renderModule, entryPoint: "vs_ground" },
    fragment: { module: renderModule, entryPoint: "fs_ground", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // 碰撞球渲染管线
  const sphereBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ],
  });

  const spherePipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [sphereBindGroupLayout] }),
    vertex: {
      module: sphereModule,
      entryPoint: "vs_sphere_mesh",
      buffers: [{
        arrayStride: 6 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" },
        ],
      }],
    },
    fragment: { module: sphereModule, entryPoint: "fs_sphere_mesh", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const sphereBindGroup = device.createBindGroup({
    layout: sphereBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // 双缓冲绑定 (Ping-Pong Storage BindGroups)
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
  // 6. GUI 控制面板与交互
  // =========================================================================
  const settings = {
    // 物理参数
    substeps: 12,          // 每帧子循环迭代次数
    stiffness: 0.95,       // 布料刚度抗拉伸
    damping: 0.015,        // 阻尼
    windSpeed: 4.5,        // 风速
    windTurbulence: 1.8,   // 风力湍流度
    windDirection: 35,     // 风向角度

    // 碰撞球控制
    sphereRadius: 1.35,
    autoMoveSphere: true,  // 自动巡航让布料产生丰富褶皱
    sphereX: 0.0,
    sphereY: 2.6,
    sphereZ: 0.0,

    // 材质色彩
    clothColor: "#8e2b38", // 经典贵族酒红丝绸
    sheen: 0.85,           // 丝绸边缘反光度

    reset: () => {
      device.queue.writeBuffer(posBufferA, 0, initialPositions);
      device.queue.writeBuffer(posBufferB, 0, initialPositions);
      device.queue.writeBuffer(prevPosBuffer, 0, initialPositions);
    },
  };

  gui.title("WebGPU Chaos Cloth 物理布料模拟");
  const physFolder = gui.addFolder("物理解算器 (XPBD/PBD)");
  physFolder.add(settings, "substeps", 4, 24, 1).name("子迭代步数 (Substeps)");
  physFolder.add(settings, "stiffness", 0.5, 1.0, 0.01).name("抗拉伸刚度 (Stiffness)");
  physFolder.add(settings, "damping", 0.0, 0.05, 0.001).name("空气阻尼");
  physFolder.open();

  const windFolder = gui.addFolder("空气动力学风场");
  windFolder.add(settings, "windSpeed", 0.0, 15.0, 0.2).name("风速大小");
  windFolder.add(settings, "windTurbulence", 0.0, 5.0, 0.1).name("湍流晃动度");
  windFolder.add(settings, "windDirection", 0, 360, 5).name("风力方位角");
  windFolder.open();

  const colFolder = gui.addFolder("刚体碰撞体 (Collision Sphere)");
  colFolder.add(settings, "sphereRadius", 0.5, 2.5, 0.05).name("球体半径");
  colFolder.add(settings, "autoMoveSphere").name("球体自动穿梭巡航");
  colFolder.add(settings, "sphereX", -3.0, 3.0, 0.1).name("球体 X 位置");
  colFolder.add(settings, "sphereY", 0.5, 5.0, 0.1).name("球体 Y 位置");
  colFolder.add(settings, "sphereZ", -3.0, 3.0, 0.1).name("球体 Z 位置");
  colFolder.open();

  const matFolder = gui.addFolder("织物视觉表现");
  matFolder.addColor(settings, "clothColor").name("织物体色");
  matFolder.add(settings, "sheen", 0.0, 2.0, 0.05).name("丝绸 Sheen 强光");
  matFolder.open();

  gui.add(settings, "reset").name("重置布料位置");

  // 相机轨道控制
  const camera = { target: [0, 2.8, 0], radius: 12.0, theta: 45.0, phi: 25.0 };
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
      const pan = camera.radius * 0.0015;
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
    camera.radius = Math.max(3, Math.min(30, camera.radius * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });

  // =========================================================================
  // 7. 渲染与物理子步进主循环
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

    // 碰撞球自动轨迹运动
    var spherePos = [settings.sphereX, settings.sphereY, settings.sphereZ];
    if (settings.autoMoveSphere) {
      spherePos = [
        Math.sin(elapsed * 1.2) * 1.8,
        2.5 + Math.cos(elapsed * 1.5) * 0.9,
        Math.cos(elapsed * 1.0) * 1.6,
      ];
    }

    // 相机与投影矩阵
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

    // 风力向量计算
    const windRad = (settings.windDirection * Math.PI) / 180;
    const windVec = [
      Math.cos(windRad) * settings.windSpeed,
      Math.sin(elapsed * 2.0) * 0.5,
      Math.sin(windRad) * settings.windSpeed,
    ];

    const clothRGB = hexToRgb(settings.clothColor);

    // 组装常量缓冲区
    const uniformData = new Float32Array(uniformBufferSize / 4);
    uniformData.set(viewProj, 0); // 0..15 (64 bytes)
    uniformData[16] = eye[0]; uniformData[17] = eye[1]; uniformData[18] = eye[2]; uniformData[19] = 1.0;
    uniformData[20] = spherePos[0]; uniformData[21] = spherePos[1]; uniformData[22] = spherePos[2]; uniformData[23] = settings.sphereRadius;
    uniformData[24] = windVec[0]; uniformData[25] = windVec[1]; uniformData[26] = windVec[2]; uniformData[27] = settings.windTurbulence;

    const dt = 1.0 / 60.0;
    const subDt = dt / settings.substeps;
    uniformData[28] = subDt;
    uniformData[29] = settings.damping;
    uniformData[30] = settings.stiffness;
    uniformData[31] = elapsed;

    // 各向原长
    uniformData[32] = DX;
    uniformData[33] = DY;
    uniformData[34] = Math.hypot(DX, DY);
    uniformData[35] = DX * 2.0;

    uniformData[36] = 0.0;
    uniformData[37] = 1.0;
    uniformData[38] = settings.sheen;
    uniformData[39] = 0.0;

    uniformData[40] = clothRGB[0]; uniformData[41] = clothRGB[1]; uniformData[42] = clothRGB[2]; uniformData[43] = 1.0;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();

    // =========================================================================
    // 物理子步进计算循环 (GPU Compute XPBD Solver)
    // =========================================================================
    const workgroupsX = Math.ceil(GRID_W / 8);
    const workgroupsY = Math.ceil(GRID_H / 8);

    // 1. 位置预测 (外力积分)
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(predictPipeline);
    computePass.setBindGroup(0, computeBindGroupA);
    computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

    // 2. 约束求解与碰撞子迭代 (Jacobi Relaxation Ping-Pong)
    var isBufferBPrimary = true;
    for (let step = 0; step < settings.substeps; step++) {
      computePass.setPipeline(solvePipeline);
      computePass.setBindGroup(0, isBufferBPrimary ? computeBindGroupB : computeBindGroupA);
      computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
      isBufferBPrimary = !isBufferBPrimary;

      // 碰撞处理
      computePass.setPipeline(collisionPipeline);
      computePass.setBindGroup(0, isBufferBPrimary ? computeBindGroupB : computeBindGroupA);
      computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
      isBufferBPrimary = !isBufferBPrimary;
    }

    // 3. 计算顶点光照法线
    computePass.setPipeline(normalsPipeline);
    computePass.setBindGroup(0, isBufferBPrimary ? computeBindGroupB : computeBindGroupA);
    computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
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

    // 1. 绘制网格地面
    renderPass.setPipeline(groundPipeline);
    renderPass.setBindGroup(0, isBufferBPrimary ? renderBindGroupA : renderBindGroupB);
    renderPass.draw(6, 1, 0, 0);

    // 2. 绘制动态碰撞球体
    renderPass.setPipeline(spherePipeline);
    renderPass.setBindGroup(0, sphereBindGroup);
    renderPass.setVertexBuffer(0, sphereVertexBuffer);
    renderPass.setIndexBuffer(sphereIndexBuffer, "uint32");
    renderPass.drawIndexed(sphereMesh.indices.length);

    // 3. 绘制丝绸布料网格 (双面渲染)
    renderPass.setPipeline(clothPipeline);
    renderPass.setBindGroup(0, isBufferBPrimary ? renderBindGroupB : renderBindGroupA);
    renderPass.setIndexBuffer(indexBuffer, "uint32");
    renderPass.drawIndexed(indexData.length);

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
    indexBuffer.destroy();
    sphereVertexBuffer.destroy();
    sphereIndexBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}