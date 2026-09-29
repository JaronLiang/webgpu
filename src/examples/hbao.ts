// src/examples/hbao.ts
import type { SimpleGUI } from "../utils/gui";

function createPerspectiveMatrix(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(fovRad / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect; out[5] = f;
  out[10] = far / (near - far); out[11] = -1;
  out[14] = (near * far) / (near - far);
  return out;
}

function createLookAtMatrix(eye: number[], center: number[], up: number[]): Float32Array {
  const z = [eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]];
  const lenZ = 1 / (Math.hypot(z[0], z[1], z[2]) || 1);
  z[0] *= lenZ; z[1] *= lenZ; z[2] *= lenZ;
  const x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]];
  const lenX = 1 / (Math.hypot(x[0], x[1], x[2]) || 1);
  x[0] *= lenX; x[1] *= lenX; x[2] *= lenX;
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const out = new Float32Array(16);
  out[0] = x[0]; out[1] = y[0]; out[2] = z[0]; out[3] = 0;
  out[4] = x[1]; out[5] = y[1]; out[6] = z[1]; out[7] = 0;
  out[8] = x[2]; out[9] = y[2]; out[10] = z[2]; out[11] = 0;
  out[12] = -(x[0]*eye[0] + x[1]*eye[1] + x[2]*eye[2]);
  out[13] = -(y[0]*eye[0] + y[1]*eye[1] + y[2]*eye[2]);
  out[14] = -(z[0]*eye[0] + z[1]*eye[1] + z[2]*eye[2]);
  out[15] = 1;
  return out;
}

function multiplyMat4(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      out[i * 4 + j] =
        a[j] * b[i * 4] + a[4 + j] * b[i * 4 + 1] + a[8 + j] * b[i * 4 + 2] + a[12 + j] * b[i * 4 + 3];
    }
  }
  return out;
}

export function runHBAO(
  device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat, canvas: HTMLCanvasElement, gui: SimpleGUI
) {
  // 1. 还原截图中的几何场景：地面 + 左侧高墙(带顶) + 右侧小方块
  const vertices = new Float32Array([
    // 地面 (亮灰)
    -10,0,-10, 0,1,0, 0.7,0.75,0.8,   10,0,-10, 0,1,0, 0.7,0.75,0.8,   10,0,10, 0,1,0, 0.7,0.75,0.8,
    -10,0,-10, 0,1,0, 0.7,0.75,0.8,   10,0,10, 0,1,0, 0.7,0.75,0.8,  -10,0,10, 0,1,0, 0.7,0.75,0.8,
    
    // 左侧 L 型高墙壁 (深灰)
    -3,0,-2, 1,0,0, 0.3,0.3,0.3,  -3,0,3, 1,0,0, 0.3,0.3,0.3,  -3,5,3, 1,0,0, 0.3,0.3,0.3,
    -3,0,-2, 1,0,0, 0.3,0.3,0.3,  -3,5,3, 1,0,0, 0.3,0.3,0.3,  -3,5,-2, 1,0,0, 0.3,0.3,0.3,
    // 高墙顶部延伸 (创造强烈的死角遮蔽)
    -3,5,-2, 0,-1,0, 0.3,0.3,0.3,  -3,5,3, 0,-1,0, 0.3,0.3,0.3,  2,5,3, 0,-1,0, 0.3,0.3,0.3,
    -3,5,-2, 0,-1,0, 0.3,0.3,0.3,  2,5,3, 0,-1,0, 0.3,0.3,0.3,   2,5,-2, 0,-1,0, 0.3,0.3,0.3,
    // 高墙顶部上方 (防穿模)
    -3,5.2,-2, 0,1,0, 0.8,0.8,0.7,  2,5.2,-2, 0,1,0, 0.8,0.8,0.7,  2,5.2,3, 0,1,0, 0.8,0.8,0.7,
    -3,5.2,-2, 0,1,0, 0.8,0.8,0.7,  2,5.2,3, 0,1,0, 0.8,0.8,0.7,  -3,5.2,3, 0,1,0, 0.8,0.8,0.7,
    
    // 右侧小方块 (深黑测试接触阴影)
    1,0,0, -1,0,0, 0.15,0.15,0.15,  1,0,2.5, -1,0,0, 0.15,0.15,0.15,  1,2.5,2.5, -1,0,0, 0.15,0.15,0.15,
    1,0,0, -1,0,0, 0.15,0.15,0.15,  1,2.5,2.5, -1,0,0, 0.15,0.15,0.15, 1,2.5,0, -1,0,0, 0.15,0.15,0.15,
    1,2.5,0, 0,1,0, 0.15,0.15,0.15,  3,2.5,0, 0,1,0, 0.15,0.15,0.15,   3,2.5,2.5, 0,1,0, 0.15,0.15,0.15,
    1,2.5,0, 0,1,0, 0.15,0.15,0.15,  3,2.5,2.5, 0,1,0, 0.15,0.15,0.15, 1,2.5,2.5, 0,1,0, 0.15,0.15,0.15,
    1,0,0, 0,0,-1, 0.15,0.15,0.15,  3,0,0, 0,0,-1, 0.15,0.15,0.15,   3,2.5,0, 0,0,-1, 0.15,0.15,0.15,
    1,0,0, 0,0,-1, 0.15,0.15,0.15,  3,2.5,0, 0,0,-1, 0.15,0.15,0.15, 1,2.5,0, 0,0,-1, 0.15,0.15,0.15,
  ]);
  const vBuffer = device.createBuffer({ size: vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vBuffer, 0, vertices);

  const quadData = new Float32Array([-1,-1, 1,-1, -1,1, 1,-1, 1,1, -1,1]);
  const quadBuffer = device.createBuffer({ size: quadData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(quadBuffer, 0, quadData);

  const uniformBuffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const pointSampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });
  const linearSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  // 2. Pass 1: G-Buffer 
  const gbufferShader = `
    struct Uniforms { viewProj: mat4x4f, view: mat4x4f, proj: mat4x4f, camParams: vec4f, hbaoParams: vec4f, miscParams: vec4f };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    struct VIn { @location(0) pos: vec3f, @location(1) norm: vec3f, @location(2) col: vec3f };
    struct VOut { @builtin(position) pos: vec4f, @location(0) normalVS: vec3f, @location(1) color: vec3f };
    
    @vertex fn vs(v: VIn) -> VOut {
      var o: VOut;
      o.pos = u.viewProj * vec4f(v.pos, 1.0);
      o.normalVS = (u.view * vec4f(v.norm, 0.0)).xyz;
      // 模拟一点天光
      let diff = max(dot(v.norm, normalize(vec3f(0.2, 1.0, 0.5))), 0.3) + 0.2;
      o.color = v.col * diff;
      return o;
    }
    struct GOut { @location(0) color: vec4f, @location(1) normalVS: vec4f };
    @fragment fn fs(in: VOut) -> GOut {
      var g: GOut;
      g.color = vec4f(in.color, 1.0);
      g.normalVS = vec4f(normalize(in.normalVS), 1.0);
      return g;
    }
  `;
  const gbufferPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: gbufferShader }), entryPoint: "vs", buffers: [{ arrayStride: 36, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x3" }, { shaderLocation: 2, offset: 24, format: "float32x3" }] }] },
    fragment: { module: device.createShaderModule({ code: gbufferShader }), entryPoint: "fs", targets: [{ format: "rgba16float" }, { format: "rgba16float" }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    primitive: { topology: "triangle-list" }
  });

  // 3. Pass 2: 修复噪点和自遮蔽的完美 HBAO 计算核心
  const hbaoShader = `
    struct Uniforms { viewProj: mat4x4f, view: mat4x4f, proj: mat4x4f, camParams: vec4f, hbaoParams: vec4f, miscParams: vec4f };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var samp: sampler;
    @group(0) @binding(2) var normalTex: texture_2d<f32>;
    @group(0) @binding(3) var depthTex: texture_depth_2d;

    @vertex fn vs(@location(0) pos: vec2f) -> @builtin(position) vec4f { return vec4f(pos, 0.0, 1.0); }

    fn getLinearDepth(rawDepth: f32) -> f32 {
      let near = u.camParams.x; let far = u.camParams.y;
      return (near * far) / (far - rawDepth * (far - near));
    }
    
    fn getViewPos(uv: vec2f, linearZ: f32) -> vec3f {
      let x = (uv.x * 2.0 - 1.0) * u.camParams.z * u.camParams.w * linearZ;
      let y = (1.0 - uv.y * 2.0) * u.camParams.z * linearZ;
      return vec3f(x, y, -linearZ);
    }

    // 业界神级：IGN交错梯度噪声 (完美分布，非常容易被模糊抹平)
    fn interleavedGradientNoise(fragCoord: vec2f) -> f32 {
      let magic = vec3f(0.06711056, 0.00583715, 52.9829189);
      return fract(magic.z * fract(dot(fragCoord, magic.xy)));
    }

    @fragment fn fs(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      let uv = fragCoord.xy / vec2f(u.miscParams.z, u.miscParams.w);
      let rawZ = textureSampleLevel(depthTex, samp, uv, 0);
      if (rawZ >= 0.99999) { return vec4f(1.0); }

      let linearZ = getLinearDepth(rawZ);
      let N = normalize(textureSampleLevel(normalTex, samp, uv, 0).xyz);
      
      // 【修复自遮蔽的关键 1】：加入 Normal Bias，把射线起点沿着法线推出去一点点
      let normalBias = u.hbaoParams.z;
      let P = getViewPos(uv, linearZ) + N * normalBias; 

      let radius = u.hbaoParams.x; 
      let intensity = u.hbaoParams.y;
      
      // 【修复自遮蔽的关键 2】：角度容差 (Angle Bias)，防止平面的微小起伏造成阴影
      let angleBias = 0.1; 
      
      let numDirections = i32(u.hbaoParams.w);
      let numSteps = 5;
      
      let stepSizeUV = (radius / linearZ) / f32(numSteps);
      
      // 使用 IGN 产生连贯且均匀的噪声
      let noise = interleavedGradientNoise(fragCoord.xy);
      let randAngle = noise * 6.2831853; 

      var totalOcclusion = 0.0;
      
      for (var d = 0; d < numDirections; d++) {
        let angle = randAngle + f32(d) * (6.2831853 / f32(numDirections));
        let dir = vec2f(cos(angle), sin(angle));
        
        var maxOcclusion = 0.0;
        
        for (var s = 1; s <= numSteps; s++) { // 注意从 1 开始步进
          // 步进长度加入 noise 偏移，打破同心圆分层 (Banding)
          let stepOffset = (f32(s) - 0.5 + noise) * stepSizeUV;
          let sampleUV = uv + dir * stepOffset;
          
          if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) { break; }

          let sDepth = getLinearDepth(textureSampleLevel(depthTex, samp, sampleUV, 0));
          let S = getViewPos(sampleUV, sDepth);
          
          let V = S - P;
          let distSq = dot(V, V);

          if (distSq < radius * radius) {
            let invDist = inverseSqrt(distSq);
            let cosH = dot(V, N) * invDist;
            
            // 基于距离的平滑衰减
            let falloff = 1.0 - (distSq / (radius * radius));
            
            let occlusion = max(0.0, cosH - angleBias) * falloff;
            maxOcclusion = max(maxOcclusion, occlusion);
          }
        }
        totalOcclusion += maxOcclusion;
      }
      
      let ao = clamp(1.0 - (totalOcclusion / f32(numDirections)) * intensity, 0.0, 1.0);
      return vec4f(ao, 0.0, 0.0, 1.0);
    }
  `;
  const hbaoPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: hbaoShader }), entryPoint: "vs", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: device.createShaderModule({ code: hbaoShader }), entryPoint: "fs", targets: [{ format: "r16float" }] },
    primitive: { topology: "triangle-list" }
  });

  // 4. Pass 3: 重写的高斯双边滤波器 (完美消除IGN噪点且保边)
  const blurShader = `
    struct Uniforms { viewProj: mat4x4f, view: mat4x4f, proj: mat4x4f, camParams: vec4f, hbaoParams: vec4f, miscParams: vec4f };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var sampLinear: sampler;
    @group(0) @binding(2) var sampPoint: sampler;
    @group(0) @binding(3) var colorTex: texture_2d<f32>;
    @group(0) @binding(4) var normalTex: texture_2d<f32>;
    @group(0) @binding(5) var depthTex: texture_depth_2d;
    @group(0) @binding(6) var aoTex: texture_2d<f32>;

    @vertex fn vs(@location(0) pos: vec2f) -> @builtin(position) vec4f { return vec4f(pos, 0.0, 1.0); }

    fn getLinearDepth(rawDepth: f32) -> f32 {
      let near = u.camParams.x; let far = u.camParams.y;
      return (near * far) / (far - rawDepth * (far - near));
    }

    @fragment fn fs(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
      let texSize = vec2f(u.miscParams.z, u.miscParams.w);
      let texelSize = 1.0 / texSize;
      let uv = fragCoord.xy * texelSize;

      let displayMode = i32(u.miscParams.x);
      let sceneColor = textureSampleLevel(colorTex, sampLinear, uv, 0).rgb;
      let centerDepthRaw = textureSampleLevel(depthTex, sampPoint, uv, 0);
      
      if (displayMode == 2 || centerDepthRaw >= 0.99999) { 
        return vec4f(sceneColor, 1.0); 
      }

      let centerDepth = getLinearDepth(centerDepthRaw);
      let centerNormal = normalize(textureSampleLevel(normalTex, sampPoint, uv, 0).xyz);
      let depthSharpness = u.miscParams.y; 

      var resultAO = 0.0;
      var weightSum = 0.0;
      
      // 5x5 高斯空间模糊核
      let blurRadius = 2;
      for (var x = -blurRadius; x <= blurRadius; x++) {
        for (var y = -blurRadius; y <= blurRadius; y++) {
          let offset = vec2f(f32(x), f32(y));
          let sampleUV = uv + offset * texelSize;
          
          let sampleAO = textureSampleLevel(aoTex, sampPoint, sampleUV, 0).r;
          let sampleDepth = getLinearDepth(textureSampleLevel(depthTex, sampPoint, sampleUV, 0));
          let sampleNormal = normalize(textureSampleLevel(normalTex, sampPoint, sampleUV, 0).xyz);
          
          // 空间高斯权重
          let spatialWeight = exp(-f32(x*x + y*y) / 8.0);
          
          // 深度保边权重 (深度差异大则权重断崖式下降)
          let depthDiff = abs(centerDepth - sampleDepth);
          let depthWeight = exp(-depthDiff * depthSharpness);
          
          // 法线保边权重
          let normalWeight = pow(max(0.0, dot(centerNormal, sampleNormal)), 8.0);
          
          let weight = spatialWeight * depthWeight * normalWeight;
          resultAO += sampleAO * weight;
          weightSum += weight;
        }
      }
      
      let finalAO = vec3f(pow(resultAO / weightSum, 1.2)); // 加上微弱的Gamma对比度

      if (displayMode == 1) { return vec4f(finalAO, 1.0); }
      return vec4f(sceneColor * finalAO, 1.0);
    }
  `;
  const blurPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: device.createShaderModule({ code: blurShader }), entryPoint: "vs", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: device.createShaderModule({ code: blurShader }), entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" }
  });

  // 5. 动态贴图管理
  let curWidth = 0, curHeight = 0;
  let gColorTex: GPUTexture, gNormalTex: GPUTexture, gDepthTex: GPUTexture, gAOTex: GPUTexture;
  let gbufferBindGroup: GPUBindGroup, hbaoBindGroup: GPUBindGroup, blurBindGroup: GPUBindGroup;

  function ensureTextures() {
    const w = canvas.width || 800; const h = canvas.height || 600;
    if (w === curWidth && h === curHeight && gNormalTex) return;
    curWidth = w; curHeight = h;

    if (gColorTex) { gColorTex.destroy(); gNormalTex.destroy(); gDepthTex.destroy(); gAOTex.destroy(); }

    gColorTex = device.createTexture({ size: [w, h], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    gNormalTex = device.createTexture({ size: [w, h], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    gDepthTex = device.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    gAOTex = device.createTexture({ size: [w, h], format: "r16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });

    gbufferBindGroup = device.createBindGroup({ layout: gbufferPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uniformBuffer } }] });
    hbaoBindGroup = device.createBindGroup({
      layout: hbaoPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } }, { binding: 1, resource: pointSampler },
        { binding: 2, resource: gNormalTex.createView() }, { binding: 3, resource: gDepthTex.createView() },
      ]
    });
    blurBindGroup = device.createBindGroup({
      layout: blurPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } }, { binding: 1, resource: linearSampler }, { binding: 2, resource: pointSampler },
        { binding: 3, resource: gColorTex.createView() }, { binding: 4, resource: gNormalTex.createView() },
        { binding: 5, resource: gDepthTex.createView() }, { binding: 6, resource: gAOTex.createView() },
      ]
    });
  }

  // 6. GUI (新增 Normal Bias 面板)
  const camera = { distance: 13.0, theta: 45, phi: 25, panY: 2.0 };
  const hbaoParams = { radius: 1.5, intensity: 2.0, normalBias: 0.05, directions: 4, displayMode: 0, blurSharpness: 30.0 };

  gui.addTextInfo("<b>工业级 HBAO (无噪点/无自遮蔽)</b><br>0: 混合叠加<br>1: 纯AO<br>2: 无AO");
  gui.add(hbaoParams, "displayMode", 0, 2, 1).name("视图模式(0-2)");
  gui.add(hbaoParams, "intensity", 0.0, 5.0, 0.1).name("AO 强度");
  gui.add(hbaoParams, "radius", 0.5, 4.0, 0.1).name("采样半径");
  gui.add(hbaoParams, "normalBias", 0.0, 0.2, 0.01).name("法线偏移(去脏污)");
  gui.add(hbaoParams, "directions", 2, 8, 1).name("射线方向数");
  gui.add(hbaoParams, "blurSharpness", 1.0, 100.0, 1.0).name("模糊保边锐度");
  gui.add(camera, "theta", -180, 180, 1).name("偏航角");
  gui.add(camera, "phi", 5, 85, 1).name("俯仰角");

  let isDragging = false, lastX = 0, lastY = 0;
  canvas.onpointerdown = (e) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); };
  canvas.onpointermove = (e) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.4;
    camera.phi = Math.max(5, Math.min(85, camera.phi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY; gui.updateDisplay();
  };
  canvas.onpointerup = (e) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} };

  const cpuData = new Float32Array(256 / 4);
  let animId: number;

  function frame() {
    ensureTextures();
    const aspect = curWidth / curHeight;
    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [camera.distance * Math.cos(radPhi) * Math.sin(radTheta), camera.panY + camera.distance * Math.sin(radPhi), camera.distance * Math.cos(radPhi) * Math.cos(radTheta)];

    const near = 0.1, far = 50.0, fov = (50 * Math.PI) / 180, tanHalfFov = Math.tan(fov / 2);
    const view = createLookAtMatrix(eye, [0, camera.panY, 0], [0, 1, 0]);
    const proj = createPerspectiveMatrix(fov, aspect, near, far);
    const viewProj = multiplyMat4(proj, view);

    cpuData.set(viewProj, 0); cpuData.set(view, 16); cpuData.set(proj, 32);
    cpuData.set([near, far, tanHalfFov, aspect], 48); 
    cpuData.set([hbaoParams.radius, hbaoParams.intensity, hbaoParams.normalBias, hbaoParams.directions], 52); 
    cpuData.set([hbaoParams.displayMode, hbaoParams.blurSharpness, curWidth, curHeight], 56); 
    device.queue.writeBuffer(uniformBuffer, 0, cpuData);

    const encoder = device.createCommandEncoder();

    // Pass 1: G-Buffer
    const pass1 = encoder.beginRenderPass({
      colorAttachments: [{ view: gColorTex.createView(), clearValue: { r: 0.1, g: 0.12, b: 0.15, a: 1.0 }, loadOp: "clear", storeOp: "store" }, { view: gNormalTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: gDepthTex.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" }
    });
    pass1.setPipeline(gbufferPipeline); pass1.setBindGroup(0, gbufferBindGroup); pass1.setVertexBuffer(0, vBuffer); pass1.draw(vertices.length / 9); pass1.end();

    // Pass 2: HBAO 计算
    const pass2 = encoder.beginRenderPass({
      colorAttachments: [{ view: gAOTex.createView(), clearValue: { r: 1.0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }]
    });
    pass2.setPipeline(hbaoPipeline); pass2.setBindGroup(0, hbaoBindGroup); pass2.setVertexBuffer(0, quadBuffer); pass2.draw(6); pass2.end();

    // Pass 3: 完美的高斯双边模糊 + 合成
    const pass3 = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" }]
    });
    pass3.setPipeline(blurPipeline); pass3.setBindGroup(0, blurBindGroup); pass3.setVertexBuffer(0, quadBuffer); pass3.draw(6); pass3.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }
  frame();

  return () => {
    cancelAnimationFrame(animId); vBuffer.destroy(); quadBuffer.destroy(); uniformBuffer.destroy();
    if (gColorTex) { gColorTex.destroy(); gNormalTex.destroy(); gDepthTex.destroy(); gAOTex.destroy(); }
  };
}