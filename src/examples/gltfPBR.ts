// src/examples/gltfPBR.ts
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

export async function runGLTFPBR(
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
  // 2. 定义材质与场景 BindGroupLayout
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
  // 3. 编写标准 Cook-Torrance PBR WGSL 着色器
  // ==========================================
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

    // 基于屏幕偏导数（dpdx/dpdy）在片元中动态重构切线空间
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
      let NdotH2 = NdotH * NdotH;
      let denom = (NdotH2 * (a2 - 1.0) + 1.0);
      return a2 / (PI * denom * denom);
    }

    fn geometrySmith(N: vec3f, V: vec3f, L: vec3f, roughness: f32) -> f32 {
      let r = roughness + 1.0;
      let k = (r * r) / 8.0;
      let NdotV = max(dot(N, V), 0.0);
      let NdotL = max(dot(N, L), 0.0);
      let ggx2 = NdotV / (NdotV * (1.0 - k) + k);
      let ggx1 = NdotL / (NdotL * (1.0 - k) + k);
      return ggx1 * ggx2;
    }

    fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
      return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let sColor = textureSample(tBaseColor, uSampler, in.uv);
      let sMR = textureSample(tMetallicRoughness, uSampler, in.uv);
      let sNorm = textureSample(tNormal, uSampler, in.uv);
      let sOcc = textureSample(tOcclusion, uSampler, in.uv);
      let sEmiss = textureSample(tEmissive, uSampler, in.uv);

      var albedo = mat.baseColorFactor.rgb;
      if (mat.hasBaseColorTex > 0.5) { albedo *= sColor.rgb; }
      
      var metallic = mat.metallicFactor;
      var roughness = mat.roughnessFactor;
      if (mat.hasMetallicRoughnessTex > 0.5) {
        metallic *= sMR.b;
        roughness *= sMR.g;
      }
      roughness = clamp(roughness, 0.04, 1.0);

      var N = normalize(in.normal);
      if (mat.hasNormalTex > 0.5) {
        let tbn = cotangentFrame(N, in.worldPos, in.uv);
        let rawNormal = sNorm.xyz * 2.0 - 1.0;
        let scaledNormal = vec3f(rawNormal.xy * mat.normalScale, rawNormal.z);
        N = normalize(tbn * scaledNormal);
      }

      let V = normalize(scene.cameraPos.xyz - in.worldPos);
      let L = normalize(-scene.lightDir.xyz);
      let H = normalize(V + L);

      let NdotV = max(dot(N, V), 0.001);
      let NdotL = max(dot(N, L), 0.001);

      var F0 = vec3f(0.04);
      F0 = mix(F0, albedo, metallic);

      let NDF = distributionGGX(N, H, roughness);
      let G = geometrySmith(N, V, L, roughness);
      let F = fresnelSchlick(max(dot(H, V), 0.0), F0);

      let numerator = NDF * G * F;
      let denominator = 4.0 * NdotV * NdotL;
      let specular = numerator / max(denominator, 0.001);

      let kS = F;
      let kD = (vec3f(1.0) - kS) * (1.0 - metallic);
      let diffuse = kD * albedo / PI;

      let radiance = scene.lightColor.rgb;
      let Lo = (diffuse + specular) * radiance * NdotL;

      var ambient = vec3f(0.04) * albedo;
      if (mat.hasOcclusionTex > 0.5) {
        ambient *= (1.0 + mat.occlusionStrength * (sOcc.r - 1.0));
      }

      var emissive = mat.emissiveFactor;
      if (mat.hasEmissiveTex > 0.5) {
        emissive *= sEmiss.rgb;
      }

      var color = ambient + Lo + emissive;

      color = color / (color + vec3f(1.0));
      color = pow(color, vec3f(1.0 / 2.2));

      return vec4f(color, mat.baseColorFactor.a);
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
        {
          arrayStride: 12, // Slot 0: POSITION
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
        },
        {
          arrayStride: 12, // Slot 1: NORMAL
          attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }],
        },
        {
          arrayStride: 8,  // Slot 2: UV
          attributes: [{ shaderLocation: 2, offset: 0, format: "float32x2" }],
        },
      ],
    },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // ==========================================
  // 4. Uniform 缓冲配置与场景参数
  // ==========================================
  const sceneUniformBuffer = device.createBuffer({
    size: 128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sceneBindGroup = device.createBindGroup({
    layout: sceneBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: sceneUniformBuffer } }],
  });

  // 全局备用默认 UV 缓冲（防止个别网格无 UV 导致 Slot 2 报错）
  const fallbackUVBuffer = device.createBuffer({
    size: 1024 * 1024, // 1MB 预分配
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  // ==========================================
  // 5. 几何体 Buffer 鲁棒提取函数 (核心修复)
  // ==========================================
  function getVertexBuffer(geom: any, ...candidates: string[]): GPUBuffer | undefined {
    if (!geom) return undefined;

    // 1. 如果对象提供了 getVertexBuffer 方法
    if (typeof geom.getVertexBuffer === "function") {
      for (const name of candidates) {
        const buf = geom.getVertexBuffer(name);
        if (buf) return buf;
      }
    }

    // 2. 检查 vertexBuffers 字典或 Map (兼容不同命名习惯，如 uv, texcoord, TEXCOORD_0 等)
    const vbs = geom.vertexBuffers;
    if (vbs) {
      if (vbs instanceof Map) {
        for (const name of candidates) {
          if (vbs.has(name)) return vbs.get(name);
        }
        for (const [k, v] of vbs.entries()) {
          const lk = String(k).toLowerCase();
          if (candidates.some(c => lk.includes(c.toLowerCase()))) return v;
        }
      } else if (typeof vbs === "object") {
        for (const name of candidates) {
          if (vbs[name]) return vbs[name];
        }
        for (const k of Object.keys(vbs)) {
          const lk = k.toLowerCase();
          if (candidates.some(c => lk.includes(c.toLowerCase()))) return vbs[k];
        }
      }
    }

    // 3. 检查 attributes 结构
    const attrs = geom.attributes;
    if (attrs) {
      if (attrs instanceof Map) {
        for (const name of candidates) {
          const a = attrs.get(name);
          if (a?.buffer) return a.buffer;
        }
      } else if (typeof attrs === "object") {
        for (const name of candidates) {
          if (attrs[name]?.buffer) return attrs[name].buffer;
        }
        for (const k of Object.keys(attrs)) {
          const lk = k.toLowerCase();
          if (candidates.some(c => lk.includes(c.toLowerCase())) && attrs[k]?.buffer) {
            return attrs[k].buffer;
          }
        }
      }
    }

    return undefined;
  }

  // ==========================================
  // 6. 加载模型
  // ==========================================
  const modelUrl = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/DamagedHelmet/glTF-Binary/DamagedHelmet.glb";
  
  let modelData: GLTFModelData;
  try {
    modelData = await GLTFLoader.load(
      modelUrl, device, materialBindGroupLayout, defaultWhiteTexture, defaultNormalTexture, defaultSampler
    );
  } catch (err) {
    console.error("glTF 加载失败:", err);
    return;
  }

  // 递归计算 Node 节点的世界空间矩阵
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
  updateWorldMatrices(modelData.nodes);

  // 为每个节点创建独立的 Model Matrix Uniform 缓冲及 BindGroup
  const nodeModelBindGroups: (GPUBindGroup | null)[] = [];
  const nodeModelBuffers: (GPUBuffer | null)[] = [];

  for (const node of modelData.nodes) {
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

  // ==========================================
  // 7. GUI 调节
  // ==========================================
  let activePBRMaterial: PBRMaterial | null = null;
  if (modelData.primitivesByMesh.length > 0 && modelData.primitivesByMesh[0].length > 0) {
    activePBRMaterial = modelData.primitivesByMesh[0][0].material;
  }

  const pbrParams = {
    metallic: activePBRMaterial ? activePBRMaterial.metallicFactor : 1.0,
    roughness: activePBRMaterial ? activePBRMaterial.roughnessFactor : 1.0,
    normalScale: activePBRMaterial ? activePBRMaterial.normalScale : 1.0,
    occlusion: activePBRMaterial ? activePBRMaterial.occlusionStrength : 1.0,
    lightIntensity: 2.5,
  };

  const applyMaterialParams = () => {
    if (!activePBRMaterial || !activePBRMaterial.uniformBuffer) return;
    activePBRMaterial.metallicFactor = pbrParams.metallic;
    activePBRMaterial.roughnessFactor = pbrParams.roughness;
    activePBRMaterial.normalScale = pbrParams.normalScale;
    activePBRMaterial.occlusionStrength = pbrParams.occlusion;

    const data = activePBRMaterial.getMaterialData();
    device.queue.writeBuffer(activePBRMaterial.uniformBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
  };

  gui.add(pbrParams, "metallic", 0.0, 1.0, 0.01).name("金属度 (Metallic)").onChange(applyMaterialParams);
  gui.add(pbrParams, "roughness", 0.04, 1.0, 0.01).name("粗糙度 (Roughness)").onChange(applyMaterialParams);
  gui.add(pbrParams, "normalScale", 0.0, 2.5, 0.1).name("法线强度 (Normal)").onChange(applyMaterialParams);
  gui.add(pbrParams, "occlusion", 0.0, 2.0, 0.1).name("环境遮挡 (AO)").onChange(applyMaterialParams);
  gui.add(pbrParams, "lightIntensity", 0.5, 6.0, 0.1).name("光照强度 (Light)");
  gui.addTextInfo("💡 基于微表面 Cook-Torrance BRDF<br>• 左键拖拽: 旋转观察<br>• 滚轮: 缩放远近");

  // ==========================================
  // 8. 相机交互与主循环
  // ==========================================
  let distance = 2.8;
  let theta = 45;
  let phi = 20;

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
    distance = Math.max(0.5, Math.min(10.0, distance + e.deltaY * 0.003));
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

    // 2. 写入场景级 Uniform
    const sceneData = new Float32Array(32);
    sceneData.set(viewProj, 0);
    sceneData.set([eye[0], eye[1], eye[2], 1.0], 16);
    sceneData.set([-0.5, -0.8, -0.4, 0.0], 20);
    const li = pbrParams.lightIntensity;
    sceneData.set([1.0 * li, 0.95 * li, 0.85 * li, 1.0], 24);
    device.queue.writeBuffer(sceneUniformBuffer, 0, sceneData);

    // 3. 执行渲染
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.08, g: 0.09, b: 0.12, a: 1.0 },
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

    // 遍历 glTF 节点树并渲染
    for (let i = 0; i < modelData.nodes.length; i++) {
      const node = modelData.nodes[i];
      if (node.meshIndex !== undefined) {
        const modelBG = nodeModelBindGroups[i];
        if (modelBG) {
          pass.setBindGroup(2, modelBG);
        }

        const primitives = modelData.primitivesByMesh[node.meshIndex];
        for (const prim of primitives) {
          if (!prim.material.bindGroup) continue;

          // 绑定材质
          pass.setBindGroup(1, prim.material.bindGroup);

          const geom = prim.geometry as any;

          // 动态解析缓冲，全面兼容各种枚举键名
          const posBuffer = getVertexBuffer(geom, "position", "pos", "POSITION");
          const normBuffer = getVertexBuffer(geom, "normal", "norm", "NORMAL");
          const uvBuffer = getVertexBuffer(geom, "uv", "texcoord", "texcoord_0", "texCoord", "TEXCOORD_0", "uv0") || fallbackUVBuffer;

          if (posBuffer && normBuffer) {
            pass.setVertexBuffer(0, posBuffer);
            pass.setVertexBuffer(1, normBuffer);
            // 确保 Slot 2 始终被绑定（找不到就使用备用 UV 缓冲兜底）
            pass.setVertexBuffer(2, uvBuffer);

            // 绘制索引或非索引几何体
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

  // 清理资源
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
    for (const buf of nodeModelBuffers) {
      if (buf) buf.destroy();
    }
  };
}