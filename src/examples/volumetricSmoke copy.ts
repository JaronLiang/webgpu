// src/examples/volumetricSmoke.ts
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
export function runVolumetricSmoke(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: GUI
) {
  // 1. 实体方块几何数据（标准 2x2x2 立方体，带法线）
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
  const cubeVertexBuffer = device.createBuffer({
    size: cubeData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(cubeVertexBuffer, 0, cubeData);

  // 2. 场景障碍建筑物设置
  const buildingDefs = [
    { pos: [1.2, 4.0, -1.0], scale: [3.2, 4.0, 3.0], color: [0.12, 0.16, 0.20] },
    { pos: [-0.6, 2.0, 3.8], scale: [2.5, 2.0, 2.2], color: [0.11, 0.15, 0.19] },
    { pos: [3.0, 1.2, 3.2], scale: [1.4, 1.2, 1.4], color: [0.13, 0.17, 0.22] },
    { pos: [5.2, 1.8, 1.5], scale: [1.8, 1.8, 3.6], color: [0.10, 0.14, 0.18] },
    { pos: [-3.8, 1.6, -1.5], scale: [2.0, 1.6, 2.4], color: [0.12, 0.15, 0.19] },
  ];

  const buildingInstanceData = new Float32Array(buildingDefs.length * 16);
  buildingDefs.forEach((b, i) => {
    buildingInstanceData[i * 16 + 0] = b.pos[0];
    buildingInstanceData[i * 16 + 1] = b.pos[1];
    buildingInstanceData[i * 16 + 2] = b.pos[2];

    buildingInstanceData[i * 16 + 4] = b.scale[0];
    buildingInstanceData[i * 16 + 5] = b.scale[1];
    buildingInstanceData[i * 16 + 6] = b.scale[2];

    buildingInstanceData[i * 16 + 8] = b.color[0];
    buildingInstanceData[i * 16 + 9] = b.color[1];
    buildingInstanceData[i * 16 + 10] = b.color[2];
    buildingInstanceData[i * 16 + 11] = 1.0;
  });

  const buildingBuffer = device.createBuffer({
    size: buildingInstanceData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buildingBuffer, 0, buildingInstanceData);

  // 3. 常量缓冲区
  const uniformBufferSize = 256;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 4. 着色器实现
  // =========================================================================
  const sceneShaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
      invViewProj: mat4x4f,
      camPos: vec4f,
      lightDir: vec4f,
      shockwave: vec4f,
      params: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct Instance {
      pos: vec4f,
      scale: vec4f,
      color: vec4f,
      pad: vec4f,
    };
    @group(0) @binding(1) var<storage, read> instances: array<Instance>;

    struct VertexOut {
      @builtin(position) clipPos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) color: vec3f,
      @location(3) isGround: f32,
    };

    @vertex
    fn vs_building(@location(0) pos: vec3f, @location(1) normal: vec3f, @builtin(instance_index) instIdx: u32) -> VertexOut {
      var out: VertexOut;
      let inst = instances[instIdx];
      let worldPos = pos * inst.scale.xyz + inst.pos.xyz;
      out.clipPos = u.viewProj * vec4f(worldPos, 1.0);
      out.worldPos = worldPos;
      out.normal = normal;
      out.color = inst.color.rgb;
      out.isGround = 0.0;
      return out;
    }

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
      out.color = vec3f(0.10, 0.16, 0.20);
      out.isGround = 1.0;
      return out;
    }

    @fragment
    fn fs_scene(in: VertexOut) -> @location(0) vec4f {
      let N = normalize(in.normal);
      let L = normalize(u.lightDir.xyz);
      let diff = max(dot(N, L), 0.0);

      var baseColor = in.color;

      let coord = in.worldPos.xz * 0.75;
      let fw = fwidth(coord);

      if (in.isGround > 0.5) {
        let grid = abs(fract(coord - 0.5) - 0.5) / max(fw, vec2f(0.001));
        let line = min(grid.x, grid.y);
        let gridMask = 1.0 - min(line, 1.0);
        let lineColor = vec3f(0.18, 0.32, 0.38);
        baseColor = mix(baseColor, lineColor, gridMask * 0.7);
      }

      let topLight = max(N.y, 0.0) * 0.15;
      let finalColor = baseColor * (diff * 0.65 + 0.35 + topLight);
      return vec4f(finalColor, 1.0);
    }
  `;

  const smokeShaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
      invViewProj: mat4x4f,
      camPos: vec4f,
      lightDir: vec4f,
      shockwave: vec4f,
      params: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var depthTex: texture_depth_2d;

    struct FullscreenOut {
      @builtin(position) clipPos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_post(@builtin(vertex_index) vid: u32) -> FullscreenOut {
      var out: FullscreenOut;
      var uv = vec2f(0.0);
      if (vid == 0u) { uv = vec2f(-1.0, -1.0); }
      else if (vid == 1u) { uv = vec2f(3.0, -1.0); }
      else { uv = vec2f(-1.0, 3.0); }
      out.clipPos = vec4f(uv, 0.0, 1.0);
      out.uv = uv * 0.5 + 0.5;
      return out;
    }

    fn hash(p: vec3f) -> f32 {
      var p3 = fract(p * 0.1031);
      p3 += dot(p3, p3.zyx + 31.32);
      return fract((p3.x + p3.y) * p3.z);
    }

    fn noise(p: vec3f) -> f32 {
      let i = floor(p);
      let f = fract(p);
      let u = f * f * (3.0 - 2.0 * f);

      return mix(
        mix(mix(hash(i + vec3f(0.0, 0.0, 0.0)), hash(i + vec3f(1.0, 0.0, 0.0)), u.x),
            mix(hash(i + vec3f(0.0, 1.0, 0.0)), hash(i + vec3f(1.0, 1.0, 0.0)), u.x), u.y),
        mix(mix(hash(i + vec3f(0.0, 0.0, 1.0)), hash(i + vec3f(1.0, 0.0, 1.0)), u.x),
            mix(hash(i + vec3f(0.0, 1.0, 1.0)), hash(i + vec3f(1.0, 1.0, 1.0)), u.x), u.y),
        u.z
      );
    }

    fn fbm(p: vec3f) -> f32 {
      var v = 0.0;
      var a = 0.5;
      var shift = vec3f(100.0);
      var pos = p;
      for (var i = 0; i < 4; i++) {
        v += a * noise(pos);
        pos = pos * 2.02 + shift;
        a *= 0.5;
      }
      return v;
    }

    fn getSmokeDensity(p: vec3f) -> f32 {
      if (p.y < 0.1 || p.y > 10.0) { return 0.0; }

      let t = u.params.x * 0.65;
      let wind = vec3f(t * 1.5, -t * 0.2, -t * 0.9);

      let sourceOrigin = vec3f(-6.0, 0.2, 5.0);
      let toOrigin = p - sourceOrigin;
      let spreadDist = length(toOrigin.xz);

      let shapeRadius = 1.2 + spreadDist * 0.42;
      let heightAtten = smoothstep(10.0, 3.0, p.y) * smoothstep(0.0, 0.8, p.y);
      let distToCore = length(p.xz - mix(sourceOrigin.xz, vec2f(3.0, -4.0), clamp(spreadDist / 16.0, 0.0, 1.0)));
      var coreEnvelope = smoothstep(shapeRadius, 0.0, distToCore) * heightAtten;

      // 鼠标激波动态碰撞
      let shockAge = u.shockwave.z;
      if (shockAge >= 0.0 && shockAge < 3.5) {
        let shockCenter = vec3f(u.shockwave.x, 0.5, u.shockwave.y);
        let distToShock = length(p - shockCenter);
        let waveRadius = shockAge * 5.5;
        let waveWidth = 2.4;

        let ringDelta = distToShock - waveRadius;
        let wavePush = exp(-pow(ringDelta / waveWidth, 2.0)) * (1.0 / (shockAge * 0.7 + 0.8));
        let blastTurbulence = sin(distToShock * 2.5 - shockAge * 8.0) * wavePush;
        coreEnvelope = max(coreEnvelope, wavePush * 1.2 * smoothstep(8.0, 0.2, p.y));

        let pushDir = normalize(p - shockCenter + vec3f(0.001));
        let displacement = pushDir * wavePush * 2.0;
        let displacedP = p - displacement;

        let n = fbm(displacedP * 0.45 + wind);
        return max(0.0, (coreEnvelope * 1.5 + blastTurbulence * 0.4) * (n - 0.28) * 2.2);
      }

      let n1 = fbm(p * 0.48 - wind);
      let n2 = fbm(p * 1.1 + wind * 0.5);
      let density = (n1 * 0.75 + n2 * 0.25) - 0.32;

      return max(0.0, coreEnvelope * density * 2.4 * u.params.y);
    }

    @fragment
    fn fs_smoke(in: FullscreenOut) -> @location(0) vec4f {
      let rawDepth = textureLoad(depthTex, vec2u(in.clipPos.xy), 0);
      let ndc = vec4f(in.uv * 2.0 - 1.0, rawDepth, 1.0);
      let worldNear = u.invViewProj * vec4f(in.uv * 2.0 - 1.0, 0.0, 1.0);
      let worldFar  = u.invViewProj * vec4f(in.uv * 2.0 - 1.0, 1.0, 1.0);
      let pNear = worldNear.xyz / worldNear.w;
      let pFar  = worldFar.xyz / worldFar.w;

      let rayDir = normalize(pFar - pNear);
      let sceneWorld = u.invViewProj * ndc;
      let scenePos = sceneWorld.xyz / sceneWorld.w;
      let maxDist = length(scenePos - u.camPos.xyz);

      var tNear = 0.0;
      var tFar = maxDist;

      if (abs(rayDir.y) > 1e-4) {
        let t0 = (0.0 - u.camPos.y) / rayDir.y;
        let t1 = (10.0 - u.camPos.y) / rayDir.y;
        let tMinY = min(t0, t1);
        let tMaxY = max(t0, t1);
        tNear = max(tNear, max(0.0, tMinY));
        tFar = min(tFar, tMaxY);
      }

      if (tNear >= tFar) {
        return vec4f(0.0);
      }

      let steps = i32(u.params.w);
      let dt = (tFar - tNear) / f32(steps);
      let jitter = hash(vec3f(in.clipPos.xy, fract(u.params.x))) * dt;

      var curT = tNear + jitter;
      var transmittance = 1.0;
      var totalScatteredLight = vec3f(0.0);

      let L = normalize(u.lightDir.xyz);
      let V = -rayDir;
      let cosTheta = dot(V, -L);
      let g = 0.38;
      let phase = (1.0 / (4.0 * 3.14159)) * ((1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosTheta, 1.5));

      let lightColor = vec3f(1.15, 1.25, 1.35);
      let ambientColor = vec3f(0.18, 0.25, 0.32);

      for (var i = 0; i < steps; i++) {
        if (curT >= tFar || transmittance < 0.01) { break; }

        let curPos = u.camPos.xyz + rayDir * curT;
        let density = getSmokeDensity(curPos);

        if (density > 0.001) {
          let shadowStep = 0.45;
          let shadowPos = curPos + L * shadowStep;
          let shadowDensity = getSmokeDensity(shadowPos);
          let shadowAtten = exp(-shadowDensity * u.params.z * 1.8);

          let scatter = (ambientColor + lightColor * (shadowAtten * phase * 3.2)) * density;
          let sampleExtinction = exp(-density * u.params.z * dt);

          totalScatteredLight += scatter * transmittance * (1.0 - sampleExtinction);
          transmittance *= sampleExtinction;
        }

        curT += dt;
      }

      let alpha = 1.0 - transmittance;
      return vec4f(totalScatteredLight, alpha);
    }
  `;

  // =========================================================================
  // 5. 显式创建 BindGroupLayout 与 PipelineLayout
  // =========================================================================
  const sceneModule = device.createShaderModule({ code: sceneShaderCode });
  const smokeModule = device.createShaderModule({ code: smokeShaderCode });

  const sceneBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });

  const scenePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [sceneBindGroupLayout],
  });

  // 1. 建筑实体管线
  const scenePipeline = device.createRenderPipeline({
    layout: scenePipelineLayout,
    vertex: {
      module: sceneModule,
      entryPoint: "vs_building",
      buffers: [{
        arrayStride: 6 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" },
        ],
      }],
    },
    fragment: { module: sceneModule, entryPoint: "fs_scene", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // 2. 地面管线
  const groundPipeline = device.createRenderPipeline({
    layout: scenePipelineLayout,
    vertex: { module: sceneModule, entryPoint: "vs_ground" },
    fragment: { module: sceneModule, entryPoint: "fs_scene", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // 3. 烟雾后处理体积管线：移除 depthStencil 配置，避免与 Pass 冲突
  const smokePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: smokeModule, entryPoint: "vs_post" },
    fragment: {
      module: smokeModule,
      entryPoint: "fs_smoke",
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

  const sceneBindGroup = device.createBindGroup({
    layout: sceneBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: buildingBuffer } },
    ],
  });

  // =========================================================================
  // 6. GUI 控制与鼠标点击交互
  // =========================================================================
  const settings = {
    density: 1.4,
    absorption: 0.9,
    stepCount: 48,
    smokeSpeed: 1.0,
    shockStrength: 1.5,
  };

  gui.title("WebGPU 3D 体积流体烟雾");
  gui.add(settings, "density", 0.2, 4.0, 0.1).name("烟雾厚度 (Density)");
  gui.add(settings, "absorption", 0.2, 2.5, 0.05).name("光吸收系数 (Absorption)");
  gui.add(settings, "stepCount", 24, 80, 2).name("步进精度 (Step Count)");
  gui.add(settings, "smokeSpeed", 0.0, 3.0, 0.1).name("气流风速");
  gui.add(settings, "shockStrength", 0.5, 3.0, 0.1).name("点击冲击强度");

  const camera = { target: [0.5, 2.0, 1.0], radius: 21.0, theta: 28.0, phi: 24.0 };
  let isDragging = false, dragButton = 0, lastX = 0, lastY = 0;
  let clickDownPos = [0, 0];

  const shockwave = { x: 0.0, z: 0.0, age: -1.0, strength: 1.0 };
  const cachedInvViewProj = new Float32Array(16);

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    isDragging = true;
    dragButton = e.shiftKey ? 2 : e.button;
    lastX = e.clientX;
    lastY = e.clientY;
    clickDownPos = [e.clientX, e.clientY];
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
      const pan = camera.radius * 0.0018;
      camera.target[0] -= Math.cos(radTheta) * dx * pan;
      camera.target[2] -= -Math.sin(radTheta) * dx * pan;
      camera.target[1] += dy * pan;
    }
  });

  canvas.addEventListener("pointerup", (e) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}

    const moveDist = Math.hypot(e.clientX - clickDownPos[0], e.clientY - clickDownPos[1]);
    if (moveDist < 6 && e.button === 0) {
      triggerShockwaveAtMouse(e.clientX, e.clientY);
    }
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.radius = Math.max(5, Math.min(60, camera.radius * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });

  function triggerShockwaveAtMouse(clientX: number, clientY: number) {
    const rect = canvas.getBoundingClientRect();
    const ndcX = ((clientX - rect.left) / rect.width) * 2.0 - 1.0;
    const ndcY = -(((clientY - rect.top) / rect.height) * 2.0 - 1.0);

    const pNear = transformPoint([ndcX, ndcY, 0.0], cachedInvViewProj);
    const pFar  = transformPoint([ndcX, ndcY, 1.0], cachedInvViewProj);

    const dy = pFar[1] - pNear[1];
    if (Math.abs(dy) > 1e-4) {
      const t = (0.5 - pNear[1]) / dy;
      if (t > 0) {
        shockwave.x = pNear[0] + (pFar[0] - pNear[0]) * t;
        shockwave.z = pNear[2] + (pFar[2] - pNear[2]) * t;
        shockwave.age = 0.0;
        shockwave.strength = settings.shockStrength;
      }
    }
  }

  // =========================================================================
  // 7. 渲染帧循环
  // =========================================================================
  let depthTexture: GPUTexture | null = null;
  let smokeBindGroup: GPUBindGroup | null = null;
  let animId: number;
  let startTime = performance.now();
  let lastFrameTime = performance.now();

  function frame() {
    const now = performance.now();
    const dt = (now - lastFrameTime) * 0.001;
    lastFrameTime = now;
    const elapsed = (now - startTime) * 0.001 * settings.smokeSpeed;

    if (shockwave.age >= 0.0) {
      shockwave.age += dt;
      if (shockwave.age > 4.0) { shockwave.age = -1.0; }
    }

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
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });

      smokeBindGroup = device.createBindGroup({
        layout: smokePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: depthTexture.createView() },
        ],
      });
    }

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.target[0] + camera.radius * Math.cos(radPhi) * Math.sin(radTheta),
      camera.target[1] + camera.radius * Math.sin(radPhi),
      camera.target[2] + camera.radius * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const proj = mat4Perspective((42 * Math.PI) / 180, renderWidth / renderHeight, 0.5, 120.0);
    const view = mat4LookAt(eye, camera.target, [0, 1, 0]);
    const viewProj = mat4Multiply(proj, view);
    const invViewProj = mat4Inverse(viewProj);
    cachedInvViewProj.set(invViewProj);

    const uniformData = new Float32Array(uniformBufferSize / 4);
    uniformData.set(viewProj, 0);
    uniformData.set(invViewProj, 16);
    uniformData[32] = eye[0]; uniformData[33] = eye[1]; uniformData[34] = eye[2]; uniformData[35] = 1.0;
    uniformData[36] = 0.6; uniformData[37] = 0.75; uniformData[38] = 0.5; uniformData[39] = 0.0;
    uniformData[40] = shockwave.x;
    uniformData[41] = shockwave.z;
    uniformData[42] = shockwave.age;
    uniformData[43] = shockwave.strength;
    uniformData[44] = elapsed;
    uniformData[45] = settings.density;
    uniformData[46] = settings.absorption;
    uniformData[47] = settings.stepCount;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const currentTextureView = context.getCurrentTexture().createView();
    const depthView = depthTexture.createView();

    // 阶段 1: 渲染实体建筑与网格地面（写入颜色与深度缓冲）
    const scenePass = encoder.beginRenderPass({
      colorAttachments: [{
        view: currentTextureView,
        clearValue: { r: 0.12, g: 0.18, b: 0.22, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store", // 保留深度供阶段 2 读取
      },
    });

    scenePass.setPipeline(groundPipeline);
    scenePass.setBindGroup(0, sceneBindGroup);
    scenePass.draw(6, 1, 0, 0);

    scenePass.setPipeline(scenePipeline);
    scenePass.setBindGroup(0, sceneBindGroup);
    scenePass.setVertexBuffer(0, cubeVertexBuffer);
    scenePass.draw(36, buildingDefs.length, 0, 0);
    scenePass.end();

    // 阶段 2: 渲染 3D 体积光线步进烟雾
    // 【关键修复点】：不传入 depthStencilAttachment，彻底消除同一 Pass 中 Attachment 与 TextureBinding 的资源冲突
    if (smokeBindGroup) {
      const smokePass = encoder.beginRenderPass({
        colorAttachments: [{
          view: currentTextureView,
          loadOp: "load", // 叠加到阶段 1 的颜色缓冲之上
          storeOp: "store",
        }],
      });

      smokePass.setPipeline(smokePipeline);
      smokePass.setBindGroup(0, smokeBindGroup);
      smokePass.draw(3, 1, 0, 0);
      smokePass.end();
    }

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    cubeVertexBuffer.destroy();
    buildingBuffer.destroy();
    uniformBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}