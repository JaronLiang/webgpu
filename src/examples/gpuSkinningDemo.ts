// src/examples/gpuSkinningDemo.ts
import { Mat4 } from "../utils/math";
import type { SimpleGUI } from "../utils/gui";
import { GLTFLoader, mat4Multiply, type GLTFModelData, type GLTFNode } from "./GLTFLoader";

// ==========================================
// 1. 基础矩阵与四元数插值数学
// ==========================================
function slerpQuat(out: number[], a: number[], b: number[], t: number) {
  let ax = a[0], ay = a[1], az = a[2], aw = a[3];
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];
  let cosTheta = ax * bx + ay * by + az * bz + aw * bw;
  if (cosTheta < 0) {
    bx = -bx; by = -by; bz = -bz; bw = -bw;
    cosTheta = -cosTheta;
  }
  if (cosTheta > 0.9995) {
    out[0] = ax + t * (bx - ax);
    out[1] = ay + t * (by - ay);
    out[2] = az + t * (bz - az);
    out[3] = aw + t * (bw - aw);
  } else {
    const theta = Math.acos(Math.max(-1, Math.min(1, cosTheta)));
    const sinTheta = Math.sin(theta);
    const w1 = Math.sin((1 - t) * theta) / sinTheta;
    const w2 = Math.sin(t * theta) / sinTheta;
    out[0] = ax * w1 + bx * w2;
    out[1] = ay * w1 + by * w2;
    out[2] = az * w1 + bz * w2;
    out[3] = aw * w1 + bw * w2;
  }
}

function mat4FromRotationTranslationScale(out: Float32Array, q: number[], t: number[], s: number[]) {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const sx = s[0], sy = s[1], sz = s[2];

  out[0] = (1 - (yy + zz)) * sx;
  out[1] = (xy + wz) * sx;
  out[2] = (xz - wy) * sx;
  out[3] = 0;
  out[4] = (xy - wz) * sy;
  out[5] = (1 - (xx + zz)) * sy;
  out[6] = (yz + wx) * sy;
  out[7] = 0;
  out[8] = (xz + wy) * sz;
  out[9] = (yz - wx) * sz;
  out[10] = (1 - (xx + yy)) * sz;
  out[11] = 0;
  out[12] = t[0];
  out[13] = t[1];
  out[14] = t[2];
  out[15] = 1;
}

function mat4Invert(out: Float32Array, a: Float32Array): boolean {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return false;
  det = 1.0 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return true;
}

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

// ==========================================
// 2. 主执行函数
// ==========================================
export async function runGPUSkinningDemo(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  const hud = document.createElement("div");
  hud.style.cssText = `
    position: absolute; left: 16px; bottom: 16px; z-index: 100;
    background: rgba(15, 23, 42, 0.9); color: #38bdf8; font-family: monospace;
    font-size: 12px; line-height: 1.5; padding: 12px 16px; border-radius: 8px;
    border: 1px solid rgba(56, 189, 248, 0.3); pointer-events: none;
  `;
  hud.innerHTML = "⏳ 正在加载 glTF 骨骼蒙皮动画模型...";
  canvas.parentElement?.appendChild(hud);

  function createSolidTexture(r: number, g: number, b: number, a: number): GPUTexture {
    const tex = device.createTexture({
      size: [1, 1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: tex }, new Uint8Array([r, g, b, a]), { bytesPerRow: 4 }, [1, 1, 1]);
    return tex;
  }
  const defaultWhiteTexture = createSolidTexture(255, 255, 255, 255);
  const defaultNormalTexture = createSolidTexture(128, 128, 255, 255);
  const defaultSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  const sceneBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ],
  });

  const materialBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
  });

  const modelBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });

  // 着色器：带模型缩放与自适应蒙皮
  const pbrShaderCode = `
    struct SceneUniforms {
      viewProj: mat4x4f,
      cameraPos: vec4f,
      lightDir: vec4f,
      lightColor: vec4f,
    };
    @group(0) @binding(0) var<uniform> scene: SceneUniforms;

    struct PBRParams {
      baseColorFactor: vec4f,
      metallicFactor: f32,
      roughnessFactor: f32,
      hasBaseColorTex: f32,
      hasMetallicRoughnessTex: f32,
      emissiveFactor: vec3f,
      hasEmissiveTex: f32,
      occlusionStrength: f32,
      hasOcclusionTex: f32,
      hasNormalTex: f32,
      normalScale: f32,
    };
    @group(1) @binding(0) var<uniform> mat: PBRParams;
    @group(1) @binding(1) var uSampler: sampler;
    @group(1) @binding(2) var tBaseColor: texture_2d<f32>;

    struct ModelUniforms {
      modelMatrix: mat4x4f,
      isSkinned: f32,
      modelScale: f32,
      _p1: f32, _p2: f32,
    };
    @group(2) @binding(0) var<uniform> model: ModelUniforms;
    @group(2) @binding(1) var<storage, read> jointMatrices: array<mat4x4f>;

    struct VertexInput {
      @location(0) position: vec3f,
      @location(1) normal: vec3f,
      @location(2) uv: vec2f,
      @location(3) joints: vec4u,
      @location(4) weights: vec4f,
    };

    struct VertexOutput {
      @builtin(position) clipPos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) uv: vec2f,
    };

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
      var out: VertexOutput;

      var localPos = vec4f(in.position, 1.0);
      var localNorm = vec4f(in.normal, 0.0);

      let wSum = in.weights.x + in.weights.y + in.weights.z + in.weights.w;
      if (model.isSkinned > 0.5 && wSum > 0.001) {
        let w = in.weights / wSum;
        let skinMat = w.x * jointMatrices[in.joints.x] +
                      w.y * jointMatrices[in.joints.y] +
                      w.z * jointMatrices[in.joints.z] +
                      w.w * jointMatrices[in.joints.w];
        localPos = skinMat * localPos;
        localNorm = skinMat * localNorm;
      }

      // 缩放并转换至世界空间
      localPos = vec4f(localPos.xyz * model.modelScale, 1.0);
      let worldPosition = model.modelMatrix * localPos;

      out.worldPos = worldPosition.xyz;
      out.clipPos = scene.viewProj * worldPosition;
      out.normal = normalize((model.modelMatrix * localNorm).xyz);
      out.uv = in.uv;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      var albedo = mat.baseColorFactor.rgb;
      if (mat.hasBaseColorTex > 0.5) {
        albedo = albedo * textureSample(tBaseColor, uSampler, in.uv).rgb;
      }

      var N = normalize(in.normal);
      if (length(N) < 0.1) { N = vec3f(0.0, 1.0, 0.0); }

      let L = normalize(-scene.lightDir.xyz);
      let diff = max(dot(N, L), 0.0);
      let ambient = vec3f(0.25) * albedo;
      let col = ambient + albedo * scene.lightColor.rgb * diff;

      return vec4f(pow(col, vec3f(1.0 / 2.2)), 1.0);
    }
  `;

  const shaderModule = device.createShaderModule({ code: pbrShaderCode });

  // 加载模型
  const modelUrl = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Fox/glTF-Binary/Fox.glb";
  let modelData: GLTFModelData;
  try {
    modelData = await GLTFLoader.load(
      modelUrl, device, materialBindGroupLayout, defaultWhiteTexture, defaultNormalTexture, defaultSampler
    );
  } catch (err) {
    hud.innerHTML = `❌ 模型加载失败: ${err}`;
    return;
  }

  // ==========================================
  // 3. 关键：Fox 模型的标准缩放与高度居中归一化
  // ==========================================
  // Fox.glb 原始单位约 100 单位长，高度约 80，我们统一缩放 0.02 倍，使之长约 2.0，完美贴合任何相机
  const MODEL_AUTO_SCALE = 0.02;
  const targetLookAt = [0.0, 0.75, 0.0]; // 聚焦在狐狸背部黄金中心
  let distance = 3.5;                     // 适中的观察距离
  let theta = 45;
  let phi = 15;

  const nodeTRS = modelData.nodes.map(node => ({
    translation: (node as any).translation ? [...(node as any).translation] : [0, 0, 0],
    rotation: (node as any).rotation ? [...(node as any).rotation] : [0, 0, 0, 1],
    scale: (node as any).scale ? [...(node as any).scale] : [1, 1, 1],
  }));

  const animations = (modelData as any).animations || [];
  let currentAnimIndex = 0;
  let animTime = 0.0;
  let animSpeed = 1.0;
  let isPlaying = true;

  const MAX_JOINTS = 128;
  const jointMatricesArray = new Float32Array(MAX_JOINTS * 16);
  for (let j = 0; j < MAX_JOINTS; j++) {
    jointMatricesArray[j * 16 + 0] = 1;
    jointMatricesArray[j * 16 + 5] = 1;
    jointMatricesArray[j * 16 + 10] = 1;
    jointMatricesArray[j * 16 + 15] = 1;
  }
  const jointStorageBuffer = device.createBuffer({
    size: MAX_JOINTS * 64,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(jointStorageBuffer, 0, jointMatricesArray);

  const nodeModelBindGroups: (GPUBindGroup | null)[] = [];
  const nodeModelBuffers: (GPUBuffer | null)[] = [];

  for (const node of modelData.nodes) {
    if (node.meshIndex !== undefined) {
      const ubo = device.createBuffer({
        size: 80,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const bg = device.createBindGroup({
        layout: modelBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: ubo } },
          { binding: 1, resource: { buffer: jointStorageBuffer } },
        ],
      });
      nodeModelBuffers.push(ubo);
      nodeModelBindGroups.push(bg);
    } else {
      nodeModelBuffers.push(null);
      nodeModelBindGroups.push(null);
    }
  }

  function evaluateAnimation(anim: any, time: number) {
    if (!anim || !anim.channels || !anim.samplers) return;
    const duration = anim.duration || 1.0;
    const t = duration > 0 ? (time % duration) : 0;

    for (const ch of anim.channels) {
      const targetNode = ch.target?.node ?? ch.targetNode;
      const targetPath = ch.target?.path ?? ch.path;
      if (targetNode === undefined || !nodeTRS[targetNode]) continue;

      const sampler = typeof ch.sampler === "number" ? anim.samplers[ch.sampler] : ch.sampler;
      if (!sampler || !sampler.input || !sampler.output) continue;

      const input = sampler.input;
      const output = sampler.output;

      let k = 0;
      while (k < input.length - 1 && input[k + 1] <= t) k++;
      const kNext = Math.min(k + 1, input.length - 1);
      const dt = input[kNext] - input[k];
      const factor = dt > 0 ? Math.max(0, Math.min(1, (t - input[k]) / dt)) : 0;

      const trs = nodeTRS[targetNode];
      if (targetPath === "translation") {
        trs.translation[0] = output[k * 3] * (1 - factor) + output[kNext * 3] * factor;
        trs.translation[1] = output[k * 3 + 1] * (1 - factor) + output[kNext * 3 + 1] * factor;
        trs.translation[2] = output[k * 3 + 2] * (1 - factor) + output[kNext * 3 + 2] * factor;
      } else if (targetPath === "rotation") {
        const q0 = [output[k * 4], output[k * 4 + 1], output[k * 4 + 2], output[k * 4 + 3]];
        const q1 = [output[kNext * 4], output[kNext * 4 + 1], output[kNext * 4 + 2], output[kNext * 4 + 3]];
        slerpQuat(trs.rotation, q0, q1, factor);
      } else if (targetPath === "scale") {
        trs.scale[0] = output[k * 3] * (1 - factor) + output[kNext * 3] * factor;
        trs.scale[1] = output[k * 3 + 1] * (1 - factor) + output[kNext * 3 + 1] * factor;
        trs.scale[2] = output[k * 3 + 2] * (1 - factor) + output[kNext * 3 + 2] * factor;
      }
    }
  }

  function updateBoneHierarchy() {
    for (let i = 0; i < modelData.nodes.length; i++) {
      const node = modelData.nodes[i];
      const trs = nodeTRS[i];
      mat4FromRotationTranslationScale(node.localMatrix, trs.rotation, trs.translation, trs.scale);
    }
    function traverse(node: GLTFNode, parentMatrix: Float32Array | null) {
      if (parentMatrix) mat4Multiply(node.worldMatrix, parentMatrix, node.localMatrix);
      else node.worldMatrix.set(node.localMatrix);
      for (const childId of node.children) traverse(modelData.nodes[childId], node.worldMatrix);
    }
    for (const node of modelData.nodes) {
      if (node.parent === undefined) traverse(node, null);
    }
  }

  const invModelMatrix = new Float32Array(16);
  const tempMat = new Float32Array(16);
  const invBindTemp = new Float32Array(16);

  function updateSkins() {
    const skins = (modelData as any).skins;
    if (!skins || skins.length === 0) return;
    const skin = skins[0];
    const joints = skin.joints;
    const rawInvBind = skin.inverseBindMatrices;

    let invBindFloats: Float32Array | null = null;
    if (rawInvBind) {
      if (rawInvBind instanceof Float32Array) invBindFloats = rawInvBind;
      else if (Array.isArray(rawInvBind)) invBindFloats = new Float32Array(rawInvBind.flat ? rawInvBind.flat(Infinity) : rawInvBind);
    }

    mat4Invert(invModelMatrix, modelData.nodes[0].worldMatrix);

    for (let i = 0; i < joints.length && i < MAX_JOINTS; i++) {
      const jointNode = modelData.nodes[joints[i]];
      if (!jointNode) continue;
      if (invBindFloats && invBindFloats.length >= (i + 1) * 16) {
        invBindTemp.set(invBindFloats.subarray(i * 16, (i + 1) * 16));
        mat4Multiply(tempMat, jointNode.worldMatrix, invBindTemp);
        mat4Multiply(tempMat, invModelMatrix, tempMat);
      } else {
        mat4Multiply(tempMat, invModelMatrix, jointNode.worldMatrix);
      }
      jointMatricesArray.set(tempMat, i * 16);
    }
    device.queue.writeBuffer(jointStorageBuffer, 0, jointMatricesArray);
  }

  // 渲染管线配置
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [sceneBindGroupLayout, materialBindGroupLayout, modelBindGroupLayout],
    }),
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [
        { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }, // pos
        { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }] }, // normal
        { arrayStride: 8,  attributes: [{ shaderLocation: 2, offset: 0, format: "float32x2" }] }, // uv
        { arrayStride: 8,  attributes: [{ shaderLocation: 3, offset: 0, format: "uint16x4" }] },  // joints
        { arrayStride: 16, attributes: [{ shaderLocation: 4, offset: 0, format: "float32x4" }] }, // weights
      ],
    },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const sceneUniformBuffer = device.createBuffer({
    size: 128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const sceneBindGroup = device.createBindGroup({
    layout: sceneBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: sceneUniformBuffer } }],
  });

  // 全局备用缓冲
  const fallbackBuffer1MB = device.createBuffer({
    size: 1024 * 1024,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  function getVertexBuffer(geom: any, ...candidates: string[]): GPUBuffer | undefined {
    if (!geom) return undefined;
    if (typeof geom.getVertexBuffer === "function") {
      for (const name of candidates) {
        const buf = geom.getVertexBuffer(name);
        if (buf) return buf;
      }
    }
    const vbs = geom.vertexBuffers;
    if (vbs) {
      if (vbs instanceof Map) {
        for (const [k, v] of vbs.entries()) {
          const lk = String(k).toLowerCase();
          if (candidates.some(c => lk.includes(c.toLowerCase()))) return v?.buffer || v;
        }
      } else if (typeof vbs === "object") {
        for (const k of Object.keys(vbs)) {
          const lk = k.toLowerCase();
          if (candidates.some(c => lk.includes(c.toLowerCase()))) return vbs[k]?.buffer || vbs[k];
        }
      }
    }
    const attrs = geom.attributes;
    if (attrs) {
      if (attrs instanceof Map) {
        for (const [k, v] of attrs.entries()) {
          const lk = String(k).toLowerCase();
          if (candidates.some(c => lk.includes(c.toLowerCase())) && v?.buffer) return v.buffer;
        }
      } else if (typeof attrs === "object") {
        for (const k of Object.keys(attrs)) {
          const lk = k.toLowerCase();
          if (candidates.some(c => lk.includes(c.toLowerCase())) && attrs[k]?.buffer) return attrs[k].buffer;
        }
      }
    }
    return undefined;
  }

  // ==========================================
  // 4. 交互：缩放 (Zoom) / 平移 (Pan) / 旋转 (Orbit)
  // ==========================================
  let isRotating = false;
  let isPanning = false;
  let lastX = 0, lastY = 0;

  const onContextMenu = (e: MouseEvent) => e.preventDefault();

  const onPointerDown = (e: PointerEvent) => {
    lastX = e.clientX;
    lastY = e.clientY;
    if (e.button === 2 || e.shiftKey) {
      isPanning = true;
    } else if (e.button === 0) {
      isRotating = true;
    }
  };

  const onPointerMove = (e: PointerEvent) => {
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    if (isRotating) {
      theta -= dx * 0.4;
      phi = Math.max(-85, Math.min(85, phi + dy * 0.4));
    } else if (isPanning) {
      const radTheta = (theta * Math.PI) / 180;
      const panSpeed = distance * 0.0015;
      const right = [-Math.cos(radTheta), 0, Math.sin(radTheta)];
      targetLookAt[0] += right[0] * dx * panSpeed;
      targetLookAt[2] += right[2] * dx * panSpeed;
      targetLookAt[1] += dy * panSpeed;
    }
  };

  const onPointerUp = () => {
    isRotating = false;
    isPanning = false;
  };

  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    const zoomFactor = 1.0 + (e.deltaY > 0 ? 0.08 : -0.08);
    distance = Math.max(0.5, Math.min(30.0, distance * zoomFactor));
  };

  canvas.addEventListener("contextmenu", onContextMenu);
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  // GUI 面板
  const animNames = animations.length > 0
    ? animations.map((a: any, idx: number) => a.name || `Animation_${idx}`)
    : ["无动作"];

  const guiParams = {
    animation: animNames[0] || "None",
    playbackSpeed: 1.0,
    isPlaying: true,
    lightIntensity: 3.5,
    resetCamera: () => {
      targetLookAt[0] = 0.0;
      targetLookAt[1] = 0.75;
      targetLookAt[2] = 0.0;
      distance = 3.5;
      theta = 45;
      phi = 15;
    }
  };

  gui.clear();
  gui.addTextInfo("<b>🦊 glTF 骨骼蒙皮动画引擎</b>");
  if (animations.length > 0) {
    gui.add(guiParams, "animation", animNames).name("当前动作").onChange((name: any) => {
      currentAnimIndex = animNames.indexOf(String(name));
      animTime = 0.0;
    });
    gui.add(guiParams, "playbackSpeed", 0.1, 3.0, 0.1).name("播放倍速").onChange((v: any) => animSpeed = Number(v));
    gui.add(guiParams, "isPlaying").name("播放/暂停").onChange((v: any) => isPlaying = Boolean(v));
  }
  gui.add(guiParams, "lightIntensity", 0.5, 6.0, 0.1).name("光照强度");
  gui.addButton("🎯 重置/聚焦到狐狸中心", guiParams.resetCamera);
  gui.addTextInfo("• 左键: 环绕旋转<br>• 右键/Shift+左键: 平移<br>• 滚轮: 平滑缩放");

  let depthTexture = device.createTexture({
    size: [canvas.width || 800, canvas.height || 600],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // ==========================================
  // 5. 渲染循环
  // ==========================================
  let animId: number;
  let lastTimestamp = performance.now();
  let frameCount = 0;
  let fpsTimer = performance.now();
  let curFps = 60;

  function frame(timestamp: number) {
    const deltaSec = Math.min((timestamp - lastTimestamp) / 1000, 0.1);
    lastTimestamp = timestamp;

    frameCount++;
    if (timestamp - fpsTimer > 500) {
      curFps = Math.round((frameCount * 1000) / (timestamp - fpsTimer));
      frameCount = 0;
      fpsTimer = timestamp;
      hud.innerHTML = `
        🦊 <b>Fox 骨骼蒙皮动画运行中</b> | FPS: ${curFps}<br>
        • 动作: <b>${animNames[currentAnimIndex]}</b> | 播放倍速: ${animSpeed.toFixed(1)}x<br>
        • 相机视距: ${distance.toFixed(2)} | 目标中心: [${targetLookAt.map(v => v.toFixed(2)).join(", ")}]
      `;
    }

    if (isPlaying && animations.length > 0) {
      animTime += deltaSec * animSpeed;
      evaluateAnimation(animations[currentAnimIndex], animTime);
    }

    updateBoneHierarchy();
    updateSkins();

    if (depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    // 观察矩阵计算
    const radTheta = (theta * Math.PI) / 180;
    const radPhi = (phi * Math.PI) / 180;
    const eye = [
      targetLookAt[0] + distance * Math.cos(radPhi) * Math.sin(radTheta),
      targetLookAt[1] + distance * Math.sin(radPhi),
      targetLookAt[2] + distance * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const view = createLookAtMatrix(eye, targetLookAt, [0, 1, 0]);
    const aspect = (canvas.width || 800) / (canvas.height || 600);
    // 宽容稳定的视锥裁剪范围
    const proj = Mat4.perspective((45 * Math.PI) / 180, aspect, 0.1, 100.0);
    const viewProj = Mat4.multiply(proj, view);

    const sceneData = new Float32Array(32);
    sceneData.set(viewProj, 0);
    sceneData.set([eye[0], eye[1], eye[2], 1.0], 16);
    sceneData.set([-0.6, -0.8, -0.4, 0.0], 20); // 主光照方向
    const li = guiParams.lightIntensity;
    sceneData.set([1.0 * li, 0.98 * li, 0.92 * li, 1.0], 24);
    device.queue.writeBuffer(sceneUniformBuffer, 0, sceneData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.12, g: 0.15, b: 0.2, a: 1.0 },
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

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, sceneBindGroup);

    for (let i = 0; i < modelData.nodes.length; i++) {
      const node = modelData.nodes[i];
      if (node.meshIndex !== undefined) {
        const ubo = nodeModelBuffers[i];
        const bg = nodeModelBindGroups[i];

        if (ubo && bg) {
          const modelDataArray = new Float32Array(20);
          modelDataArray.set(node.worldMatrix, 0);
          modelDataArray[16] = (modelData as any).skins?.length > 0 ? 1.0 : 0.0;
          modelDataArray[17] = MODEL_AUTO_SCALE; // 传入自动归一化缩放系数
          device.queue.writeBuffer(ubo, 0, modelDataArray);
          pass.setBindGroup(2, bg);
        }

        const primitives = modelData.primitivesByMesh[node.meshIndex];
        for (const prim of primitives) {
          if (!prim.material.bindGroup) continue;
          pass.setBindGroup(1, prim.material.bindGroup);

          const geom = prim.geometry as any;
          const posBuffer = getVertexBuffer(geom, "position", "pos", "POSITION");
          const normBuffer = getVertexBuffer(geom, "normal", "norm", "NORMAL");
          const uvBuffer = getVertexBuffer(geom, "uv", "texcoord", "texcoord_0", "TEXCOORD_0") || fallbackBuffer1MB;
          const jointsBuffer = getVertexBuffer(geom, "joints", "joints_0", "JOINTS_0") || fallbackBuffer1MB;
          const weightsBuffer = getVertexBuffer(geom, "weights", "weights_0", "WEIGHTS_0") || fallbackBuffer1MB;

          if (posBuffer && normBuffer) {
            pass.setVertexBuffer(0, posBuffer);
            pass.setVertexBuffer(1, normBuffer);
            pass.setVertexBuffer(2, uvBuffer);
            pass.setVertexBuffer(3, jointsBuffer);
            pass.setVertexBuffer(4, weightsBuffer);

            if (prim.indexFormat && geom.indexBuffer) {
              pass.setIndexBuffer(geom.indexBuffer, prim.indexFormat);
              pass.drawIndexed(geom.indexCount);
            } else {
              pass.draw(geom.vertexCount);
            }
          }
        }
      }
    }

    pass.end();
    device.queue.submit([encoder.finish()]);

    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("contextmenu", onContextMenu);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);
    hud.remove();
    defaultWhiteTexture.destroy();
    defaultNormalTexture.destroy();
    fallbackBuffer1MB.destroy();
    sceneUniformBuffer.destroy();
    jointStorageBuffer.destroy();
    depthTexture.destroy();
    for (const buf of nodeModelBuffers) {
      if (buf) buf.destroy();
    }
  };
}