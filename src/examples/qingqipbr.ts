// src/examples/qingqipbr.ts
import { Mat4 } from "../utils/math";
import type { SimpleGUI } from "../utils/gui";
import { GLTFLoader, mat4Multiply, type GLTFModelData, type GLTFNode } from "./GLTFLoader";
import { PBRMaterial } from "./PBRMaterial";

// 3D 视轨观察相机 LookAt 矩阵
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

export async function runGLTFPBRqingqi(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // ==========================================
  // 1. 创建默认 1x1 占位贴图与采样器
  // ==========================================
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
    mipmapFilter: "linear",
    addressModeU: "repeat",
    addressModeV: "repeat",
  });

  // ==========================================
  // 2. BindGroupLayout 定义
  // ==========================================
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
    ],
  });

  // ==========================================
  // 3. 高反差车展级 Clearcoat 双层 PBR 着色器
  // ==========================================
  const pbrShaderCode = `
    struct SceneUniforms {
      viewProj: mat4x4f,
      cameraPos: vec4f,
      lightDir: vec4f,
      lightColor: vec4f,
      clearcoatParams: vec4f, // x: clearcoatFactor, y: clearcoatRoughness
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
    @group(1) @binding(3) var tMetallicRoughness: texture_2d<f32>;
    @group(1) @binding(4) var tNormal: texture_2d<f32>;
    @group(1) @binding(5) var tOcclusion: texture_2d<f32>;
    @group(1) @binding(6) var tEmissive: texture_2d<f32>;

    @group(2) @binding(0) var<uniform> modelMatrix: mat4x4f;

    struct VertexInput {
      @location(0) position: vec3f,
      @location(1) normal: vec3f,
      @location(2) uv: vec2f,
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
      let worldPosition = modelMatrix * vec4f(in.position, 1.0);
      out.worldPos = worldPosition.xyz;
      out.clipPos = scene.viewProj * worldPosition;
      out.normal = normalize((modelMatrix * vec4f(in.normal, 0.0)).xyz);
      out.uv = in.uv;
      return out;
    }

    const PI = 3.14159265359;

    fn cotangentFrame(N: vec3f, p: vec3f, uv: vec2f) -> mat3x3f {
      let dp1 = dpdx(p);
      let dp2 = dpdy(p);
      let duv1 = dpdx(uv);
      let duv2 = dpdy(uv);
      let dp2perp = cross(dp2, N);
      let dp1perp = cross(N, dp1);
      let T = dp2perp * duv1.x + dp1perp * duv2.x;
      let B = dp2perp * duv1.y + dp1perp * duv2.y;
      let invmax = inverseSqrt(max(dot(T, T), dot(B, B)));
      return mat3x3f(T * invmax, B * invmax, N);
    }

    fn distributionGGX(N: vec3f, H: vec3f, roughness: f32) -> f32 {
      let a = roughness * roughness;
      let a2 = a * a;
      let NdotH = max(dot(N, H), 0.0);
      let denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
      return a2 / (PI * denom * denom);
    }

    fn geometrySmith(N: vec3f, V: vec3f, L: vec3f, roughness: f32) -> f32 {
      let r = roughness + 1.0;
      let k = (r * r) / 8.0;
      let NdotV = max(dot(N, V), 0.0);
      let NdotL = max(dot(N, L), 0.0);
      return (NdotV / (NdotV * (1.0 - k) + k)) * (NdotL / (NdotL * (1.0 - k) + k));
    }

    fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
      return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    }

    // 车展级暗室长条形柔光箱环境倒影
    fn getStudioEnvironment(R: vec3f, roughness: f32) -> vec3f {
      let sky = vec3f(0.08, 0.10, 0.14);
      let ground = vec3f(0.02, 0.02, 0.03);
      var env = mix(ground, sky, clamp(R.y * 0.5 + 0.5, 0.0, 1.0));

      let gloss = 1.0 - roughness;

      // 顶部主长条柔光箱
      let light1 = pow(max(dot(R, normalize(vec3f(0.0, 1.0, 0.2))), 0.0), 32.0 / (roughness * roughness + 0.005));
      // 侧前方长条反光板
      let light2 = pow(max(dot(R, normalize(vec3f(-0.8, 0.35, -0.4))), 0.0), 48.0 / (roughness * roughness + 0.005));
      // 侧后方轮廓勾边光
      let light3 = pow(max(dot(R, normalize(vec3f(0.7, 0.3, 0.6))), 0.0), 64.0 / (roughness * roughness + 0.005));

      env += (vec3f(light1) * 3.5 + vec3f(light2) * 2.5 + vec3f(light3) * 1.5) * gloss;
      return env;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let sColor = textureSample(tBaseColor, uSampler, in.uv);
      let sMR = textureSample(tMetallicRoughness, uSampler, in.uv);
      let sNorm = textureSample(tNormal, uSampler, in.uv);
      let sOcc = textureSample(tOcclusion, uSampler, in.uv);
      let sEmiss = textureSample(tEmissive, uSampler, in.uv);

      // 1. 基底色彩提取
      var albedo = mat.baseColorFactor.rgb;
      if (mat.hasBaseColorTex > 0.5) {
        albedo *= sColor.rgb;
      }
      
      var metallic = clamp(mat.metallicFactor, 0.0, 1.0);
      var baseRoughness = clamp(mat.roughnessFactor, 0.04, 1.0);
      if (mat.hasMetallicRoughnessTex > 0.5) {
        metallic *= sMR.b;
        baseRoughness *= sMR.g;
      }
      baseRoughness = max(baseRoughness, 0.08);

      var N = normalize(in.normal);
      if (mat.hasNormalTex > 0.5) {
        let tbn = cotangentFrame(N, in.worldPos, in.uv);
        let rawNormal = sNorm.xyz * 2.0 - 1.0;
        let scaledNormal = vec3f(rawNormal.xy * mat.normalScale, rawNormal.z);
        N = normalize(tbn * scaledNormal);
      }

      let V = normalize(scene.cameraPos.xyz - in.worldPos);
      let NdotV = clamp(dot(N, V), 0.001, 1.0);
      let R = reflect(-V, N);

      // ==========================================
      // 2. 清漆层 (Clearcoat) — 晶透玻璃镜面
      // ==========================================
      let ccFactor = scene.clearcoatParams.x;
      let ccRoughness = clamp(scene.clearcoatParams.y, 0.005, 0.15);

      let ccF0 = vec3f(0.04);
      let ccFresnel = (ccF0 + (vec3f(1.0) - ccF0) * pow(1.0 - NdotV, 5.0)) * ccFactor;
      let transmission = vec3f(1.0) - ccFresnel;

      let ccEnvSpec = getStudioEnvironment(R, ccRoughness) * ccFresnel;

      // ==========================================
      // 3. 直接光照（双层微表面高光）
      // ==========================================
      var directLighting = vec3f(0.0);
      let L = normalize(-scene.lightDir.xyz);
      let H = normalize(V + L);
      let NdotL = max(dot(N, L), 0.0);

      if (NdotL > 0.0) {
        let VdotH = max(dot(V, H), 0.0);

        // 基底反射与漫反射
        var baseF0 = mix(vec3f(0.04), albedo, metallic);
        let baseF = fresnelSchlick(VdotH, baseF0);
        let baseD = distributionGGX(N, H, baseRoughness);
        let baseG = geometrySmith(N, V, L, baseRoughness);
        let specBase = (baseD * baseG * baseF) / max(4.0 * NdotV * NdotL, 0.001);
        let kD = (vec3f(1.0) - baseF) * (1.0 - metallic);
        let diffBase = kD * albedo / PI;

        // 清漆聚光针尖高光
        var specCC = vec3f(0.0);
        if (ccFactor > 0.01) {
          let D_cc = distributionGGX(N, H, ccRoughness);
          let G_cc = geometrySmith(N, V, L, 0.08);
          let F_cc = (ccF0 + (vec3f(1.0) - ccF0) * pow(1.0 - VdotH, 5.0)) * ccFactor;
          specCC = (D_cc * G_cc * F_cc) / max(4.0 * NdotV * NdotL, 0.001);
        }

        let radiance = scene.lightColor.rgb;
        directLighting += ((diffBase + specBase) * transmission + specCC) * radiance * NdotL;
      }

      // ==========================================
      // 4. 环境漫反射与遮挡（低底噪，暗部纯净通透）
      // ==========================================
      let hemiSky = vec3f(0.25, 0.28, 0.32);
      let hemiGround = vec3f(0.06, 0.06, 0.07);
      let hemi = mix(hemiGround, hemiSky, N.y * 0.5 + 0.5);
      let ambientDiffuse = hemi * albedo * (1.0 - metallic) * transmission;

      let baseEnv = getStudioEnvironment(R, baseRoughness) * mix(vec3f(0.04), albedo, metallic) * transmission;

      var ao = 1.0;
      if (mat.hasOcclusionTex > 0.5) {
        ao = 1.0 + mat.occlusionStrength * (sOcc.r - 1.0);
      }

      var emissive = mat.emissiveFactor;
      if (mat.hasEmissiveTex > 0.5) {
        emissive *= sEmiss.rgb;
      }

      var finalColor = (ambientDiffuse + baseEnv) * ao + directLighting + ccEnvSpec + emissive;

      // ==========================================
      // 5. 扩展 Reinhard 色调映射（告别灰白粉末层）
      // ==========================================
      let whitePoint = 4.0;
      finalColor = (finalColor * (vec3f(1.0) + finalColor / (whitePoint * whitePoint))) / (vec3f(1.0) + finalColor);

      finalColor = pow(finalColor, vec3f(1.0 / 2.2));
      return vec4f(finalColor, mat.baseColorFactor.a);
    }
  `;

  const shaderModule = device.createShaderModule({ code: pbrShaderCode });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [sceneBindGroupLayout, materialBindGroupLayout, modelBindGroupLayout],
    }),
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [
        { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] },
        { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }] },
        { arrayStride: 8,  attributes: [{ shaderLocation: 2, offset: 0, format: "float32x2" }] },
      ],
    },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const sceneUniformBuffer = device.createBuffer({
    size: 160,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sceneBindGroup = device.createBindGroup({
    layout: sceneBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: sceneUniformBuffer } }],
  });

  const fallbackUVBuffer = device.createBuffer({
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
        for (const name of candidates) { if (vbs.has(name)) return vbs.get(name); }
        for (const [k, v] of vbs.entries()) {
          if (candidates.some(c => String(k).toLowerCase().includes(c.toLowerCase()))) return v;
        }
      } else {
        for (const name of candidates) { if (vbs[name]) return vbs[name]; }
      }
    }
    const attrs = geom.attributes;
    if (attrs) {
      if (attrs instanceof Map) {
        for (const name of candidates) {
          const a = attrs.get(name);
          if (a?.buffer) return a.buffer;
        }
      } else {
        for (const name of candidates) {
          if (attrs[name]?.buffer) return attrs[name].buffer;
        }
      }
    }
    return undefined;
  }

  // ==========================================
  // 4. 相机与视图变量（定义在 loadModel 前）
  // ==========================================
  let distance = 3.5;
  let theta = 45;
  let phi = 20;

  // ==========================================
  // 5. 在线模型管理
  // ==========================================
  const onlineModels: Record<string, { url: string; dist: number }> = {
    "ClearCoat 材质球 (ClearCoatTest)": {
      url: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/ClearCoatTest/glTF-Binary/ClearCoatTest.glb",
      dist: 3.5,
    },
    "战损头盔 (DamagedHelmet)": {
      url: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/DamagedHelmet/glTF-Binary/DamagedHelmet.glb",
      dist: 2.8,
    },
    "碳纤维水壶 (WaterBottle)": {
      url: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/WaterBottle/glTF-Binary/WaterBottle.glb",
      dist: 3.0,
    },
  };

  let currentModelData: GLTFModelData | null = null;
  let nodeModelBindGroups: (GPUBindGroup | null)[] = [];
  let nodeModelBuffers: (GPUBuffer | null)[] = [];
  let activePBRMaterial: PBRMaterial | null = null;

  function updateWorldMatrices(nodes: GLTFNode[]) {
    function traverse(node: GLTFNode, parentMatrix: Float32Array | null) {
      if (parentMatrix) {
        mat4Multiply(node.worldMatrix, parentMatrix, node.localMatrix);
      } else {
        node.worldMatrix.set(node.localMatrix);
      }
      for (const childId of node.children) {
        traverse(nodes[childId], node.worldMatrix);
      }
    }
    for (const node of nodes) {
      if (node.parent === undefined) traverse(node, null);
    }
  }

  async function loadModel(modelKey: string) {
    for (const b of nodeModelBuffers) { if (b) b.destroy(); }
    nodeModelBuffers = [];
    nodeModelBindGroups = [];
    currentModelData = null;
    activePBRMaterial = null;

    const info = onlineModels[modelKey];
    if (info) {
      distance = info.dist;
    }

    try {
      const data = await GLTFLoader.load(
        info.url,
        device,
        materialBindGroupLayout,
        defaultWhiteTexture,
        defaultNormalTexture,
        defaultSampler
      );
      updateWorldMatrices(data.nodes);

      for (const node of data.nodes) {
        if (node.meshIndex !== undefined) {
          const ubo = device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });
          device.queue.writeBuffer(ubo, 0, node.worldMatrix as any);
          const bg = device.createBindGroup({
            layout: modelBindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: ubo } }],
          });
          nodeModelBuffers.push(ubo);
          nodeModelBindGroups.push(bg);
        } else {
          nodeModelBuffers.push(null);
          nodeModelBindGroups.push(null);
        }
      }

      if (data.primitivesByMesh.length > 0 && data.primitivesByMesh[0].length > 0) {
        activePBRMaterial = data.primitivesByMesh[0][0].material;
      }
      currentModelData = data;
    } catch (err) {
      console.error("加载模型失败:", err);
    }
  }

  // ==========================================
  // 6. GUI 控件
  // ==========================================
  const modelKeys = Object.keys(onlineModels);

  const renderParams = {
    modelIndex: 0,
    model: modelKeys[0],
    clearcoat: 1.0,           // 开启清漆
    clearcoatRoughness: 0.02, // 保持极细聚光点
    baseRoughness: 0.5,       // 保持底层漫射哑光
    metallic: 0.0,
    normalScale: 1.0,
    lightIntensity: 3.2,
  };

  gui.add(renderParams, "modelIndex", 0, modelKeys.length - 1, 1)
    .name("切换模型 (0:球 1:头盔 2:水壶)")
    .onChange((idx: number) => {
      const selectedIndex = Math.round(idx);
      renderParams.modelIndex = selectedIndex;
      const selectedKey = modelKeys[selectedIndex];
      if (selectedKey) {
        renderParams.model = selectedKey;
        loadModel(selectedKey);
      }
    });

  gui.add(renderParams, "clearcoat", 0.0, 1.0, 0.01).name("清漆强度 (Clearcoat)");
  gui.add(renderParams, "clearcoatRoughness", 0.001, 0.2, 0.005).name("清漆粗糙度 (CC Roughness)");

  gui.add(renderParams, "baseRoughness", 0.04, 1.0, 0.01).name("基底粗糙度").onChange(() => {
    if (activePBRMaterial && activePBRMaterial.uniformBuffer) {
      activePBRMaterial.roughnessFactor = renderParams.baseRoughness;
      const data = activePBRMaterial.getMaterialData();
      device.queue.writeBuffer(activePBRMaterial.uniformBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
    }
  });

  gui.add(renderParams, "lightIntensity", 0.5, 6.0, 0.1).name("光照强度");
  gui.addTextInfo("✨ <b>Clearcoat (清漆)</b>：外层针尖高光 + 内层柔和光晕，呈现真实车漆/碳纤维通透胶衣感。");

  // 初始加载模型
  await loadModel(modelKeys[renderParams.modelIndex]);

  // ==========================================
  // 7. 相机事件监听
  // ==========================================
  let isDragging = false;
  let lastX = 0, lastY = 0;
  const onPointerDown = (e: PointerEvent) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; };
  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    theta -= (e.clientX - lastX) * 0.5;
    phi = Math.max(-85, Math.min(85, phi + (e.clientY - lastY) * 0.5));
    lastX = e.clientX; lastY = e.clientY;
  };
  const onPointerUp = () => { isDragging = false; };
  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    distance = Math.max(0.5, Math.min(15.0, distance + e.deltaY * 0.003));
  };

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  let depthTexture = device.createTexture({
    size: [canvas.width || 800, canvas.height || 600],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  let animId: number;

  function frame() {
    if (depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    // 1. 视图变换
    const radTheta = (theta * Math.PI) / 180;
    const radPhi = (phi * Math.PI) / 180;
    const eye = [
      distance * Math.cos(radPhi) * Math.sin(radTheta),
      distance * Math.sin(radPhi),
      distance * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const view = createLookAtMatrix(eye, [0, 0, 0], [0, 1, 0]);
    const aspect = (canvas.width || 800) / (canvas.height || 600);
    const proj = Mat4.perspective((50 * Math.PI) / 180, aspect, 0.1, 100);
    const viewProj = Mat4.multiply(proj, view);

    // 2. 更新 Scene Uniform 缓冲
    const sceneData = new Float32Array(36);
    sceneData.set(viewProj, 0);                           // offset 0: mat4x4
    sceneData.set([eye[0], eye[1], eye[2], 1.0], 16);     // offset 16: cameraPos
    // 斜前方主光，让弯管正面形成饱满聚光点
    sceneData.set([0.3, -0.7, 0.65, 0.0], 20);            // offset 20: lightDir
    const li = renderParams.lightIntensity;
    sceneData.set([1.0 * li, 1.0 * li, 1.0 * li, 1.0], 24); // offset 24: 纯白光
    sceneData.set([renderParams.clearcoat, renderParams.clearcoatRoughness, 0.0, 0.0], 28);
    device.queue.writeBuffer(sceneUniformBuffer, 0, sceneData);

    // 3. 渲染通道
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        // 汽车展台经典的极深蓝灰深邃背景
        clearValue: { r: 0.04, g: 0.05, b: 0.07, a: 1.0 },
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

    if (currentModelData) {
      for (let i = 0; i < currentModelData.nodes.length; i++) {
        const node = currentModelData.nodes[i];
        if (node.meshIndex !== undefined) {
          const modelBG = nodeModelBindGroups[i];
          if (modelBG) pass.setBindGroup(2, modelBG);

          const primitives = currentModelData.primitivesByMesh[node.meshIndex];
          for (const prim of primitives) {
            if (!prim.material.bindGroup) continue;

            pass.setBindGroup(1, prim.material.bindGroup);
            const geom = prim.geometry as any;

            const posBuffer = getVertexBuffer(geom, "position", "pos", "POSITION");
            const normBuffer = getVertexBuffer(geom, "normal", "norm", "NORMAL");
            const uvBuffer = getVertexBuffer(geom, "uv", "texcoord", "texcoord_0", "TEXCOORD_0") || fallbackUVBuffer;

            if (posBuffer && normBuffer) {
              pass.setVertexBuffer(0, posBuffer);
              pass.setVertexBuffer(1, normBuffer);
              pass.setVertexBuffer(2, uvBuffer);

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
    }

    pass.end();
    device.queue.submit([encoder.finish()]);

    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  // 资源清理
  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);
    defaultWhiteTexture.destroy();
    defaultNormalTexture.destroy();
    fallbackUVBuffer.destroy();
    sceneUniformBuffer.destroy();
    depthTexture.destroy();
    for (const buf of nodeModelBuffers) { if (buf) buf.destroy(); }
  };
}