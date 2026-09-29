// src/examples/envmap.ts
import type { SimpleGUI } from "../utils/gui";

// 矩阵计算
function createPerspectiveMatrix(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(fovRad / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far / (near - far);
  out[11] = -1;
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

// 球体顶点数据生成
function createSphere(radius: number, segments: number) {
  const pos: number[] = [], norm: number[] = [], indices: number[] = [];
  for (let y = 0; y <= segments; y++) {
    const v = y / segments;
    const phi = v * Math.PI;
    for (let x = 0; x <= segments; x++) {
      const u = x / segments;
      const theta = u * Math.PI * 2;
      const px = radius * Math.sin(phi) * Math.cos(theta);
      const py = radius * Math.cos(phi);
      const pz = radius * Math.sin(phi) * Math.sin(theta);
      pos.push(px, py, pz);
      const l = Math.hypot(px, py, pz) || 1;
      norm.push(px / l, py / l, pz / l);
    }
  }
  for (let y = 0; y < segments; y++) {
    for (let x = 0; x < segments; x++) {
      const p1 = y * (segments + 1) + x;
      const p2 = p1 + segments + 1;
      indices.push(p1, p2, p1 + 1, p1 + 1, p2, p2 + 1);
    }
  }
  const vertices = new Float32Array((pos.length / 3) * 6);
  for (let i = 0; i < pos.length / 3; i++) {
    vertices[i * 6 + 0] = pos[i * 3 + 0];
    vertices[i * 6 + 1] = pos[i * 3 + 1];
    vertices[i * 6 + 2] = pos[i * 3 + 2];
    vertices[i * 6 + 3] = norm[i * 3 + 0];
    vertices[i * 6 + 4] = norm[i * 3 + 1];
    vertices[i * 6 + 5] = norm[i * 3 + 2];
  }
  return { vertices, indices: new Uint16Array(indices) };
}

export function runEnvMap(
  device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat, canvas: HTMLCanvasElement, gui: SimpleGUI
) {
  const sphere = createSphere(2.0, 64);
  const sphereVBuffer = device.createBuffer({ size: sphere.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(sphereVBuffer, 0, sphere.vertices);
  const sphereIBuffer = device.createBuffer({ size: sphere.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(sphereIBuffer, 0, sphere.indices);

  // 天空盒立方体顶点 (包围相机)
  const skyboxVerts = new Float32Array([
    -1,  1, -1,  -1, -1, -1,   1, -1, -1,   1, -1, -1,   1,  1, -1,  -1,  1, -1,
    -1, -1,  1,  -1, -1, -1,  -1,  1, -1,  -1,  1, -1,  -1,  1,  1,  -1, -1,  1,
     1, -1, -1,   1, -1,  1,   1,  1,  1,   1,  1,  1,   1,  1, -1,   1, -1, -1,
    -1, -1,  1,   1, -1,  1,   1,  1,  1,   1,  1,  1,  -1,  1,  1,  -1, -1,  1,
    -1,  1, -1,   1,  1, -1,   1,  1,  1,   1,  1,  1,  -1,  1,  1,  -1,  1, -1,
    -1, -1, -1,  -1, -1,  1,   1, -1, -1,   1, -1, -1,  -1, -1,  1,   1, -1,  1
  ]);
  const skyboxVBuffer = device.createBuffer({ size: skyboxVerts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(skyboxVBuffer, 0, skyboxVerts);

  // 【修复核心】：严格遵循 WebGPU / DirectX / Metal 的 Cubemap 轴向定义
  const cubeSize = 256;
  const cubeTex = device.createTexture({
    size: [cubeSize, cubeSize, 6],
    dimension: "2d",
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  for (let face = 0; face < 6; face++) {
    const data = new Uint8Array(cubeSize * cubeSize * 4);
    for (let y = 0; y < cubeSize; y++) {
      for (let x = 0; x < cubeSize; x++) {
        // [-1, 1] 映射
        const u = (x + 0.5) / cubeSize * 2.0 - 1.0;
        const v = 1.0 - (y + 0.5) / cubeSize * 2.0;

        let dir = [0, 0, 0];
        switch (face) {
          case 0: dir = [ 1.0,   v,  -u]; break; // +X
          case 1: dir = [-1.0,   v,   u]; break; // -X
          case 2: dir = [   u, 1.0,  -v]; break; // +Y
          case 3: dir = [   u,-1.0,   v]; break; // -Y
          case 4: dir = [   u,   v, 1.0]; break; // +Z
          case 5: dir = [  -u,   v,-1.0]; break; // -Z
        }

        const len = Math.hypot(dir[0], dir[1], dir[2]);
        const nx = dir[0] / len;
        const ny = dir[1] / len;
        const nz = dir[2] / len;

        // 生成真实自然色彩：天顶深蓝、地平线暖霞、太阳光芒与绿色地表
        let r = 0, g = 0, b = 0;
        if (ny >= 0.0) {
          // 天空与云
          const skyT = Math.pow(ny, 0.4);
          r = 0.25 * skyT + 0.65 * (1.0 - skyT);
          g = 0.55 * skyT + 0.80 * (1.0 - skyT);
          b = 0.95 * skyT + 0.95 * (1.0 - skyT);

          // 拟真太阳光晕 (朝向特定方向)
          const sunDir = [0.4, 0.6, -0.6];
          const sunDot = Math.max(0, nx * sunDir[0] + ny * sunDir[1] + nz * sunDir[2]);
          const sun = Math.pow(sunDot, 64) * 1.5 + Math.pow(sunDot, 8) * 0.4;
          r += sun * 1.0; g += sun * 0.9; b += sun * 0.7;
        } else {
          // 地面与倒影水面
          const groundT = Math.min(1.0, -ny * 4.0);
          r = 0.25 * (1.0 - groundT) + 0.15 * groundT;
          g = 0.28 * (1.0 - groundT) + 0.22 * groundT;
          b = 0.32 * (1.0 - groundT) + 0.18 * groundT;
        }

        const idx = (y * cubeSize + x) * 4;
        data[idx]     = Math.min(255, Math.max(0, Math.floor(r * 255)));
        data[idx + 1] = Math.min(255, Math.max(0, Math.floor(g * 255)));
        data[idx + 2] = Math.min(255, Math.max(0, Math.floor(b * 255)));
        data[idx + 3] = 255;
      }
    }
    device.queue.writeTexture(
      { texture: cubeTex, origin: [0, 0, face] },
      data,
      { bytesPerRow: cubeSize * 4, rowsPerImage: cubeSize },
      [cubeSize, cubeSize, 1]
    );
  }

  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  const uniformBuffer = device.createBuffer({ size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const shaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
      skyViewProj: mat4x4f,
      eyePos: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var samp: sampler;
    @group(0) @binding(2) var envMap: texture_cube<f32>;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) worldNormal: vec3f,
    };

    @vertex fn vs_sphere(@location(0) pos: vec3f, @location(1) normal: vec3f) -> VertexOut {
      var out: VertexOut;
      out.pos = u.viewProj * vec4f(pos, 1.0);
      out.worldPos = pos;
      out.worldNormal = normal;
      return out;
    }

    @fragment fn fs_sphere(in: VertexOut) -> @location(0) vec4f {
      let V = normalize(in.worldPos - u.eyePos.xyz);
      let N = normalize(in.worldNormal);
      let R = reflect(V, N);

      // 从立方体贴图采样反射光
      let reflCol = textureSample(envMap, samp, R).rgb;

      // 菲涅尔镀铬反射
      let fresnel = 0.2 + 0.8 * pow(1.0 - max(dot(-V, N), 0.0), 4.0);
      return vec4f(reflCol * (0.85 + 0.15 * fresnel), 1.0);
    }

    // 天空盒渲染
    struct SkyOut {
      @builtin(position) pos: vec4f,
      @location(0) uvDir: vec3f,
    };

    @vertex fn vs_sky(@location(0) pos: vec3f) -> SkyOut {
      var out: SkyOut;
      out.uvDir = pos;
      let clip = u.skyViewProj * vec4f(pos, 1.0);
      out.pos = clip.xyww; // 强制位于远裁剪面 Z=1.0
      return out;
    }

    @fragment fn fs_sky(in: SkyOut) -> @location(0) vec4f {
      return textureSample(envMap, samp, normalize(in.uvDir));
    }
  `;

  const module = device.createShaderModule({ code: shaderCode });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "cube" } },
    ]
  });

  const spherePipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module, entryPoint: "vs_sphere",
      buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "float32x3" }] }],
    },
    fragment: { module, entryPoint: "fs_sphere", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const skyPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module, entryPoint: "vs_sky",
      buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }],
    },
    fragment: { module, entryPoint: "fs_sky", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: cubeTex.createView({ dimension: "cube" }) },
    ],
  });

  let depthTexture = device.createTexture({
    size: [canvas.width || 800, canvas.height || 600],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const camera = { distance: 6.0, theta: 45, phi: 20 };
  gui.addTextInfo("<b>环境反射 (Environment Map)</b><br>修正 Cubemap 轴向对齐的标准镜面球");
  gui.add(camera, "theta", -180, 180, 1).name("偏航角");
  gui.add(camera, "phi", -85, 85, 1).name("俯仰角");

  let isDragging = false, lastX = 0, lastY = 0;
  canvas.onpointerdown = (e) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); };
  canvas.onpointermove = (e) => {
    if (!isDragging) return;
    camera.theta -= (e.clientX - lastX) * 0.4;
    camera.phi = Math.max(-85, Math.min(85, camera.phi + (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY;
    gui.updateDisplay();
  };
  canvas.onpointerup = (e) => { isDragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} };

  let animId: number;
  function frame() {
    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.distance * Math.cos(radPhi) * Math.sin(radTheta),
      camera.distance * Math.sin(radPhi),
      camera.distance * Math.cos(radPhi) * Math.cos(radTheta),
    ];

    const aspect = (canvas.width || 800) / (canvas.height || 600);
    const view = createLookAtMatrix(eye, [0, 0, 0], [0, 1, 0]);
    const proj = createPerspectiveMatrix((50 * Math.PI) / 180, aspect, 0.1, 100);
    const viewProj = multiplyMat4(proj, view);

    // 天空盒视角矩阵剔除位移（保持在原地不动）
    const skyView = new Float32Array(view);
    skyView[12] = 0; skyView[13] = 0; skyView[14] = 0;
    const skyViewProj = multiplyMat4(proj, skyView);

    device.queue.writeBuffer(uniformBuffer, 0, viewProj.buffer as ArrayBuffer);
    device.queue.writeBuffer(uniformBuffer, 64, skyViewProj.buffer as ArrayBuffer);
    device.queue.writeBuffer(uniformBuffer, 128, new Float32Array([eye[0], eye[1], eye[2], 1.0]));

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: depthTexture.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" }
    });

    pass.setBindGroup(0, bindGroup);

    // 1. 渲染球体
    pass.setPipeline(spherePipeline);
    pass.setVertexBuffer(0, sphereVBuffer);
    pass.setIndexBuffer(sphereIBuffer, "uint16");
    pass.drawIndexed(sphere.indices.length);

    // 2. 渲染天空盒
    pass.setPipeline(skyPipeline);
    pass.setVertexBuffer(0, skyboxVBuffer);
    pass.draw(36);

    pass.end();
    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }
  frame();

  return () => {
    cancelAnimationFrame(animId);
    sphereVBuffer.destroy(); sphereIBuffer.destroy(); skyboxVBuffer.destroy();
    uniformBuffer.destroy(); cubeTex.destroy(); depthTexture.destroy();
  };
}