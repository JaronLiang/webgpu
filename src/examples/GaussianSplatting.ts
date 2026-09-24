// src/examples/gaussianSplatting.ts

export function runGaussianSplatting(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // =========================================================================
  // 步骤 1: WGSL 着色器 (保持高精度和正确的 Alpha 预乘)
  // =========================================================================
  const shaderCode = `
    struct CameraUniform {
      view: mat4x4f,
      proj: mat4x4f,
      camPos: vec3f,
      viewport: vec2f,
      focal: vec2f,
    };

    struct Gaussian {
      pos: vec3f,             // 12 bytes
      pad1: f32,              // 4 bytes -> 16
      scale: vec3f,           // 12 bytes
      pad2: f32,              // 4 bytes -> 16
      rot: vec4f,             // 16 bytes
      color: vec4f,           // 16 bytes -> 总计 64 bytes
    };

    @group(0) @binding(0) var<uniform> uCam: CameraUniform;
    @group(0) @binding(1) var<storage, read> splats: array<Gaussian>;
    @group(0) @binding(2) var<storage, read> indices: array<u32>;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) color: vec4f,
      @location(1) conic: vec3f,
      @location(2) uv: vec2f,
    };

    fn computeRotMatrix(q: vec4f) -> mat3x3f {
      let r = q.x; let x = q.y; let y = q.z; let z = q.w;
      return mat3x3f(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y - r*z),       2.0*(x*z + r*y),
        2.0*(x*y + r*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - r*x),
        2.0*(x*z - r*y),       2.0*(y*z + r*x),       1.0 - 2.0*(x*x + y*y)
      );
    }

    @vertex
    fn vs_main(@builtin(vertex_index) vIdx: u32, @builtin(instance_index) instIdx: u32) -> VertexOutput {
      let idx = indices[instIdx];
      let g = splats[idx];

      let R = computeRotMatrix(g.rot);
      let S = mat3x3f(g.scale.x, 0.0, 0.0, 0.0, g.scale.y, 0.0, 0.0, 0.0, g.scale.z);
      let M = R * S;
      let cov3D = M * transpose(M);

      let viewPos4 = uCam.view * vec4f(g.pos, 1.0);
      let viewPos = viewPos4.xyz;
      
      // 剔除相机背后的点
      if (viewPos.z >= -0.2) {
        return VertexOutput(vec4f(0.0), vec4f(0.0), vec3f(0.0), vec2f(0.0));
      }

      let rz = 1.0 / viewPos.z;
      let rz2 = rz * rz;
      let J = mat3x3f(
        uCam.focal.x * rz, 0.0, -(uCam.focal.x * viewPos.x) * rz2,
        0.0, uCam.focal.y * rz, -(uCam.focal.y * viewPos.y) * rz2,
        0.0, 0.0, 0.0
      );

      let W = mat3x3f(uCam.view[0].xyz, uCam.view[1].xyz, uCam.view[2].xyz);
      let T = W * J;
      var cov2D = transpose(T) * transpose(cov3D) * T;
      
      cov2D[0][0] += 0.3;
      cov2D[1][1] += 0.3;

      let a = cov2D[0][0]; let b = cov2D[0][1]; let c = cov2D[1][1];
      let det = a * c - b * b;
      let invDet = 1.0 / max(det, 0.0000001);
      let conic = vec3f(c * invDet, -b * invDet, a * invDet);

      let mid = 0.5 * (a + c);
      let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
      let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
      let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

      let quadOffset = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
      );
      let offset = quadOffset[vIdx];
      let screenOffset = offset * radius / uCam.viewport;

      let clipPos = uCam.proj * viewPos4;
      let finalPos = vec4f(
        clipPos.x / clipPos.w + screenOffset.x,
        clipPos.y / clipPos.w + screenOffset.y,
        clipPos.z / clipPos.w,
        1.0
      );

      return VertexOutput(finalPos, g.color, conic, offset * radius);
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let d = in.uv;
      let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) - in.conic.y * d.x * d.y;
      if (power > 0.0) { discard; }
      let alpha = min(0.99, in.color.a * exp(power));
      if (alpha < 1.0 / 255.0) { discard; }
      return vec4f(in.color.rgb * alpha, alpha); // Alpha 预乘
    }
  `;

  // =========================================================================
  // 步骤 2: 构建极速基数排序 (Radix Sort) 与矩阵库
  // =========================================================================
  function sortSplats(depths: Float32Array, indices: Uint32Array, count: number) {
    // 修复 TypeScript ArrayBufferLike 强校验报错
    const depthInt = new Uint32Array(depths.buffer as ArrayBuffer);
    
    let maxDepth = 0; 
    let minDepth = 0xFFFFFFFF;
    
    // 将浮点数深度映射到正整数区间以便位运算
    for (let i = 0; i < count; i++) {
      let d = depthInt[i];
      // Float 转正向 Uint 保序技巧
      d = (d >> 31) === 0 ? d : d ^ 0xFFFFFFFF;
      depthInt[i] = d;
      if (d > maxDepth) maxDepth = d;
      if (d < minDepth) minDepth = d;
    }

    const range = maxDepth - minDepth;
    if (range === 0) return;

    // 8-bit Radix Sort (4 Pass)
    const outIndices = new Uint32Array(count);
    const histograms = new Uint32Array(256 * 4);
    
    for (let i = 0; i < count; i++) {
      const val = depthInt[i];
      histograms[0xFF & val]                 ++;
      histograms[256 + (0xFF & (val >> 8))]  ++;
      histograms[512 + (0xFF & (val >> 16))] ++;
      histograms[768 + (0xFF & (val >> 24))] ++;
    }

    const offsets = new Uint32Array(256 * 4);
    for (let j = 0; j < 4; j++) {
      let sum = 0;
      for (let i = 0; i < 256; i++) {
        offsets[j * 256 + i] = sum;
        sum += histograms[j * 256 + i];
      }
    }

    let src = indices, dst = outIndices;
    for (let pass = 0; pass < 4; pass++) {
      const shift = pass * 8;
      const passOffset = pass * 256;
      for (let i = 0; i < count; i++) {
        const idx = src[i];
        const val = depthInt[idx];
        const bucket = 0xFF & (val >> shift);
        const pos = offsets[passOffset + bucket]++;
        dst[pos] = idx;
      }
      let temp = src; src = dst; dst = temp;
    }
    // 逆序以实现从后向前渲染
    for(let i = 0; i < count; i++) indices[i] = src[count - 1 - i];
  }

  function perspective(fov: number, aspect: number, near: number, far: number) {
    const f = 1.0 / Math.tan(fov / 2);
    return [
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, far / (near - far), -1,
      0, 0, (near * far) / (near - far), 0
    ];
  }

  function lookAt(eye: number[], center: number[], up: number[]) {
    let z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
    let len = 1 / Math.hypot(z0, z1, z2); z0 *= len; z1 *= len; z2 *= len;
    let x0 = up[1] * z2 - up[2] * z1, x1 = up[2] * z0 - up[0] * z2, x2 = up[0] * z1 - up[1] * z0;
    len = 1 / Math.hypot(x0, x1, x2); x0 *= len; x1 *= len; x2 *= len;
    let y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;
    return [
      x0, y0, z0, 0, x1, y1, z1, 0, x2, y2, z2, 0,
      -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]), -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]), -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]), 1
    ];
  }

  // =========================================================================
  // 步骤 3: 核心状态与 GUI (本地文件加载器)
  // =========================================================================
  let splatCount = 0;
  let splatBuffer: GPUBuffer | null = null;
  let indexBuffer: GPUBuffer | null = null;
  let bindGroup: GPUBindGroup | null = null;
  let splatDataRaw: Float32Array; // 用于 CPU 深度提取
  let isModelLoaded = false;

  const cameraUniformData = new Float32Array(40);
  const cameraBuffer = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module: device.createShaderModule({ code: shaderCode }), entryPoint: "vs_main" },
    fragment: { 
      module: device.createShaderModule({ code: shaderCode }), 
      entryPoint: "fs_main",
      targets: [{ 
        format, 
        blend: { 
          color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" }
        }
      }]
    },
    primitive: { topology: "triangle-list" },
  });

  // UI 构建
  const gui = document.createElement("div");
  gui.style.cssText = "position:absolute; top:10px; left:10px; background:rgba(20,20,20,0.9); color:white; padding:15px; border-radius:8px; font-family:sans-serif; font-size:13px; z-index:100; box-shadow: 0 4px 10px rgba(0,0,0,0.5);";
  gui.innerHTML = `
    <h3 style="margin:0 0 10px 0; color:#4CAF50;">3DGS .ply Renderer</h3>
    <p style="margin:5px 0; font-size:11px; color:#aaa;">Choose an official .ply model to render.</p>
    <input type="file" id="plyInput" accept=".ply" style="margin-bottom:10px;" />
    <div id="status" style="color:yellow;">Waiting for file...</div>
  `;
  document.body.appendChild(gui);

  // =========================================================================
  // 步骤 4: 官方 .ply 文件解析器 (解析训练后的公式变量)
  // =========================================================================
  document.getElementById("plyInput")!.addEventListener("change", (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    
    document.getElementById("status")!.innerText = "Reading file...";
    const reader = new FileReader();
    reader.onload = () => {
      document.getElementById("status")!.innerText = "Parsing PLY... (May take a moment)";
      setTimeout(() => parsePLY(reader.result as ArrayBuffer), 50); // 给UI渲染留一点时间
    };
    reader.readAsArrayBuffer(file);
  });

  function parsePLY(buffer: ArrayBuffer) {
    const textDecoder = new TextDecoder();
    const headerStr = textDecoder.decode(new Uint8Array(buffer, 0, Math.min(2000, buffer.byteLength)));
    const headerEnd = headerStr.indexOf("end_header\n") + "end_header\n".length;
    
    // 获取顶点数量
    const vertexMatch = headerStr.match(/element vertex (\d+)/);
    if (!vertexMatch) return alert("Invalid PLY file");
    splatCount = parseInt(vertexMatch[1]);
    
    // 解析属性偏移量
    const properties = headerStr.split('\n').filter(l => l.startsWith('property '));
    let offset = 0;
    const propOffsets: Record<string, number> = {};
    properties.forEach(p => {
      const parts = p.split(' ');
      const name = parts[2];
      propOffsets[name] = offset;
      offset += 4; // 官方高斯模型均为 float32 (4 bytes)
    });
    
    const vertexByteSize = offset;
    const dataView = new DataView(buffer, headerEnd);
    
    // 初始化显存需要的数据结构 (每个 splat 64 字节，对应 16 个 f32)
    const splatData = new Float32Array(splatCount * 16);
    splatDataRaw = new Float32Array(splatCount * 3); // 额外存一份纯 XYZ 供 CPU 快速提取排序
    
    const SH_C0 = 0.28209479177387814; // 球面调和函数 0阶常数

    for (let i = 0; i < splatCount; i++) {
      const byteOffset = i * vertexByteSize;
      const sIdx = i * 16;

      // 1. Position
      const x = dataView.getFloat32(byteOffset + propOffsets["x"], true);
      const y = dataView.getFloat32(byteOffset + propOffsets["y"], true);
      const z = dataView.getFloat32(byteOffset + propOffsets["z"], true);
      splatData[sIdx + 0] = x; splatData[sIdx + 1] = y; splatData[sIdx + 2] = z;
      splatDataRaw[i * 3 + 0] = x; splatDataRaw[i * 3 + 1] = y; splatDataRaw[i * 3 + 2] = z;

      // 2. Scale (Log to Exp)
      splatData[sIdx + 4] = Math.exp(dataView.getFloat32(byteOffset + propOffsets["scale_0"], true));
      splatData[sIdx + 5] = Math.exp(dataView.getFloat32(byteOffset + propOffsets["scale_1"], true));
      splatData[sIdx + 6] = Math.exp(dataView.getFloat32(byteOffset + propOffsets["scale_2"], true));

      // 3. Rotation (Normalize Quaternion)
      const rot0 = dataView.getFloat32(byteOffset + propOffsets["rot_0"], true);
      const rot1 = dataView.getFloat32(byteOffset + propOffsets["rot_1"], true);
      const rot2 = dataView.getFloat32(byteOffset + propOffsets["rot_2"], true);
      const rot3 = dataView.getFloat32(byteOffset + propOffsets["rot_3"], true);
      const len = Math.sqrt(rot0*rot0 + rot1*rot1 + rot2*rot2 + rot3*rot3);
      splatData[sIdx + 8] = rot0 / len;
      splatData[sIdx + 9] = rot1 / len;
      splatData[sIdx + 10] = rot2 / len;
      splatData[sIdx + 11] = rot3 / len;

      // 4. Color (SH degree 0 to RGB) & Opacity (Inverse Sigmoid to Alpha)
      const f_dc_0 = dataView.getFloat32(byteOffset + propOffsets["f_dc_0"], true);
      const f_dc_1 = dataView.getFloat32(byteOffset + propOffsets["f_dc_1"], true);
      const f_dc_2 = dataView.getFloat32(byteOffset + propOffsets["f_dc_2"], true);
      const opacity = dataView.getFloat32(byteOffset + propOffsets["opacity"], true);

      splatData[sIdx + 12] = Math.max(0, Math.min(1, 0.5 + SH_C0 * f_dc_0)); // R
      splatData[sIdx + 13] = Math.max(0, Math.min(1, 0.5 + SH_C0 * f_dc_1)); // G
      splatData[sIdx + 14] = Math.max(0, Math.min(1, 0.5 + SH_C0 * f_dc_2)); // B
      splatData[sIdx + 15] = 1.0 / (1.0 + Math.exp(-opacity)); // A (Sigmoid)
    }

    // 重建 GPU Buffers
    if (splatBuffer) splatBuffer.destroy();
    if (indexBuffer) indexBuffer.destroy();

    splatBuffer = device.createBuffer({
      size: splatData.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true,
    });
    new Float32Array(splatBuffer.getMappedRange()).set(splatData);
    splatBuffer.unmap();

    const initialIndices = new Uint32Array(splatCount);
    for (let i = 0; i < splatCount; i++) initialIndices[i] = i;
    
    indexBuffer = device.createBuffer({
      size: initialIndices.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indexBuffer, 0, initialIndices);

    bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: cameraBuffer } },
        { binding: 1, resource: { buffer: splatBuffer } },
        { binding: 2, resource: { buffer: indexBuffer } },
      ],
    });

    document.getElementById("status")!.innerHTML = `<span style="color:#0f0;">Loaded ${splatCount.toLocaleString()} Splats!</span>`;
    isModelLoaded = true;
  }

  // =========================================================================
  // 步骤 5: 交互与渲染循环
  // =========================================================================
  let rotX = 0, rotY = 0, camDist = 5.0;
  let isDragging = false, lastX = 0, lastY = 0;
  const canvasEl = context.canvas as HTMLCanvasElement;

  canvasEl.addEventListener("mousedown", (e) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener("mouseup", () => isDragging = false);
  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    rotY += (e.clientX - lastX) * 0.01;
    rotX += (e.clientY - lastY) * 0.01;
    rotX = Math.max(-Math.PI/2, Math.min(Math.PI/2, rotX));
    lastX = e.clientX; lastY = e.clientY;
  });
  canvasEl.addEventListener("wheel", (e) => {
    e.preventDefault(); camDist *= e.deltaY > 0 ? 1.1 : 0.9;
  }, { passive: false });

  // 预分配内存以避免每帧垃圾回收
  let depthsArray = new Float32Array(0);
  let indicesArray = new Uint32Array(0);

  let animId: number;
  function render() {
    animId = requestAnimationFrame(render);
    if (!isModelLoaded || !bindGroup) return;

    // 1. 相机矩阵计算
    const fov = 45 * Math.PI / 180;
    const aspect = canvasEl.width / canvasEl.height;
    
    const camPos = [
      Math.sin(rotY) * Math.cos(rotX) * camDist,
      Math.sin(rotX) * camDist,
      Math.cos(rotY) * Math.cos(rotX) * camDist
    ];
    const viewMat = lookAt(camPos, [0, 0, 0], [0, 1, 0]);
    const projMat = perspective(fov, aspect, 0.1, 100.0);

    const focalY = canvasEl.height / (2.0 * Math.tan(fov / 2.0));

    cameraUniformData.set(viewMat, 0);
    cameraUniformData.set(projMat, 16);
    cameraUniformData.set(camPos, 32);
    cameraUniformData.set([canvasEl.width, canvasEl.height], 36);
    cameraUniformData.set([focalY, focalY], 38);
    device.queue.writeBuffer(cameraBuffer, 0, cameraUniformData);

    // 2. 高速 CPU 深度基数排序
    if (depthsArray.length !== splatCount) {
      depthsArray = new Float32Array(splatCount);
      indicesArray = new Uint32Array(splatCount);
      for(let i=0; i<splatCount; i++) indicesArray[i] = i;
    }

    const vx = viewMat[2], vy = viewMat[6], vz = viewMat[10]; 
    for(let i = 0; i < splatCount; i++) {
        depthsArray[i] = splatDataRaw[i*3]*vx + splatDataRaw[i*3+1]*vy + splatDataRaw[i*3+2]*vz;
    }
    
    // 执行极速 Radix Sort
    sortSplats(depthsArray, indicesArray, splatCount);
    device.queue.writeBuffer(indexBuffer!, 0, indicesArray);

    // 3. WebGPU 绘制
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, splatCount, 0, 0); 
    pass.end();

    device.queue.submit([encoder.finish()]);
  }
  
  animId = requestAnimationFrame(render);

  return () => {
    cancelAnimationFrame(animId);
    if (splatBuffer) splatBuffer.destroy();
    if (indexBuffer) indexBuffer.destroy();
    cameraBuffer.destroy();
    document.body.removeChild(gui);
  };
}