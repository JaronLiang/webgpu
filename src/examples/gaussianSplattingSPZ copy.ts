// src/examples/gaussianSplattingSPZ.ts
import GUI from "lil-gui";

// ==================== 矩阵与投影数学 ====================
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

export interface GaussianCloudData {
  count: number;
  positions: Float32Array; // [x, y, z] * count
  scales: Float32Array;    // [sx, sy, sz] * count
  rotations: Float32Array; // [qx, qy, qz, qw] * count
  colors: Float32Array;    // [r, g, b, a] * count
  center: [number, number, number];
  radius: number;
}

// ==================== 官方标准 SPZ 流式解包器 ====================
async function parseSPZ(buffer: ArrayBuffer): Promise<GaussianCloudData> {
  // 1. 原生 Gzip 解压缩
  const ds = new DecompressionStream("gzip");
  const writer = ds.writable.getWriter();
  writer.write(new Uint8Array(buffer));
  writer.close();
  const decompressed = await new Response(ds.readable).arrayBuffer();

  const view = new DataView(decompressed);
  const magic = view.getUint32(0, true);
  const version = view.getUint32(4, true);
  const count = view.getUint32(8, true);
  const shDegree = view.getUint8(12);
  const fractionalBits = view.getUint8(13);

  console.log(`[SPZ] 魔数: 0x${magic.toString(16)}, 版本: ${version}, 粒子数: ${count}, 定点精度: ${fractionalBits}`);

  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const rotations = new Float32Array(count * 4);
  const colors = new Float32Array(count * 4);

  const uint8 = new Uint8Array(decompressed);
  let offset = 16;

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  // 2. 坐标解析 (支持 SPZ v2 的 24-bit 紧凑编码与浮点编码)
  if (fractionalBits > 0) {
    const scaleFactor = 1.0 / (1 << fractionalBits);
    for (let i = 0; i < count * 3; i++) {
      const b0 = uint8[offset++];
      const b1 = uint8[offset++];
      const b2 = uint8[offset++];
      let val = b0 | (b1 << 8) | (b2 << 16);
      if (val & 0x800000) val |= 0xff000000;
      const coord = val * scaleFactor;
      positions[i] = coord;

      if (i % 3 === 0) { minX = Math.min(minX, coord); maxX = Math.max(maxX, coord); }
      if (i % 3 === 1) { minY = Math.min(minY, coord); maxY = Math.max(maxY, coord); }
      if (i % 3 === 2) { minZ = Math.min(minZ, coord); maxZ = Math.max(maxZ, coord); }
    }
  } else {
    for (let i = 0; i < count * 3; i++) {
      const coord = view.getFloat32(offset, true);
      offset += 4;
      positions[i] = coord;
      if (i % 3 === 0) { minX = Math.min(minX, coord); maxX = Math.max(maxX, coord); }
      if (i % 3 === 1) { minY = Math.min(minY, coord); maxY = Math.max(maxY, coord); }
      if (i % 3 === 2) { minZ = Math.min(minZ, coord); maxZ = Math.max(maxZ, coord); }
    }
  }

  // 3. 不透明度 Alpha (Sigmoid 逆变换还原)
  const alphas = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const rawA = uint8[offset++];
    alphas[i] = 1.0 / (1.0 + Math.exp(-((rawA - 128.0) / 16.0)));
  }

  // 4. 球谐 DC 颜色基底 (SH0 -> RGB)
  for (let i = 0; i < count; i++) {
    colors[i * 4 + 0] = uint8[offset + 0] / 255.0;
    colors[i * 4 + 1] = uint8[offset + 1] / 255.0;
    colors[i * 4 + 2] = uint8[offset + 2] / 255.0;
    colors[i * 4 + 3] = alphas[i];
    offset += 3;
  }

  // 跳过高阶球谐系数
  const shExtraDim = (shDegree > 0) ? ((shDegree + 1) * (shDegree + 1) - 1) * 3 : 0;
  offset += count * shExtraDim;

  // 5. 尺度 Scales (对数空间还原)
  for (let i = 0; i < count * 3; i++) {
    const raw = uint8[offset++];
    scales[i] = Math.exp((raw / 16.0) - 10.0);
  }

  // 6. 旋转四元数 (Smallest-three 归一化四元数)
  for (let i = 0; i < count; i++) {
    const r0 = (uint8[offset + 0] - 128.0) / 127.0;
    const r1 = (uint8[offset + 1] - 128.0) / 127.0;
    const r2 = (uint8[offset + 2] - 128.0) / 127.0;
    const sumSq = r0 * r0 + r1 * r1 + r2 * r2;
    const r3 = Math.sqrt(Math.max(0.0, 1.0 - sumSq));
    rotations[i * 4 + 0] = r0;
    rotations[i * 4 + 1] = r1;
    rotations[i * 4 + 2] = r2;
    rotations[i * 4 + 3] = r3;
    offset += 3;
  }

  const cx = (minX + maxX) * 0.5 || 0;
  const cy = (minY + maxY) * 0.5 || 0;
  const cz = (minZ + maxZ) * 0.5 || 0;
  const boundRadius = Math.max(maxX - minX, maxY - minY, maxZ - minZ) * 0.5 || 2.0;

  return {
    count,
    positions,
    scales,
    rotations,
    colors,
    center: [cx, cy, cz],
    radius: boundRadius,
  };
}

// 生成离线星云场景
function createSyntheticScene(count = 45000): GaussianCloudData {
  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const rotations = new Float32Array(count * 4);
  const colors = new Float32Array(count * 4);

  for (let i = 0; i < count; i++) {
    const t = (i / count) * Math.PI * 30.0;
    const r = Math.pow(Math.random(), 0.5) * 1.8;
    const arm = (i % 5) * ((Math.PI * 2) / 5);

    const x = Math.cos(t * 0.2 + arm) * r + (Math.random() - 0.5) * 0.15;
    const z = Math.sin(t * 0.2 + arm) * r + (Math.random() - 0.5) * 0.15;
    const y = ((Math.random() - 0.5) * 0.3) * (2.0 - r);

    positions[i * 3 + 0] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    scales[i * 3 + 0] = 0.012 + Math.random() * 0.015;
    scales[i * 3 + 1] = 0.004 + Math.random() * 0.005;
    scales[i * 3 + 2] = 0.012 + Math.random() * 0.015;

    rotations[i * 4 + 0] = 0;
    rotations[i * 4 + 1] = Math.sin(t * 0.1);
    rotations[i * 4 + 2] = 0;
    rotations[i * 4 + 3] = Math.cos(t * 0.1);

    colors[i * 4 + 0] = 0.2 + 0.8 * Math.sin(r * 2.0);
    colors[i * 4 + 1] = 0.4 + 0.6 * Math.cos(r * 3.0);
    colors[i * 4 + 2] = 0.95;
    colors[i * 4 + 3] = 0.4 + Math.random() * 0.4;
  }

  return {
    count,
    positions,
    scales,
    rotations,
    colors,
    center: [0, 0, 0],
    radius: 2.0
  };
}

// ==================== 主渲染流程 ====================
export function runGaussianSplattingspz(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: any
) {
  // 1. 先初始化交互相机的初始参数，避免 TDZ 报错
  const camera = {
    target: [0, 0, 0],
    radius: 4.5,
    theta: 35.0,
    phi: 22.0,
  };

  const quadVertices = new Float32Array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
    -1.0,  1.0,
     1.0, -1.0,
     1.0,  1.0,
  ]);
  const quadBuffer = device.createBuffer({
    size: quadVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(quadBuffer, 0, quadVertices);

  const uniformBuffer = device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const gsShaderWGSL = `
    struct Uniforms {
      view: mat4x4f,
      proj: mat4x4f,
      camPos: vec4f,
      viewport: vec2f,
      focal: vec2f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct Gaussian {
      pos: vec4f,
      scale: vec4f,
      rot: vec4f,
      color: vec4f,
    };
    @group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
    @group(0) @binding(2) var<storage, read> sortedIndices: array<u32>;

    struct VertexOutput {
      @builtin(position) pos: vec4f,
      @location(0) color: vec4f,
      @location(1) conic: vec3f,
      @location(2) coord: vec2f,
    };

    fn quatToMat3(q: vec4f) -> mat3x3f {
      let x = q.x; let y = q.y; let z = q.z; let w = q.w;
      return mat3x3f(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z), 2.0*(x*z - w*y),
        2.0*(x*y - w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x),
        2.0*(x*z + w*y), 2.0*(y*z - w*x), 1.0 - 2.0*(x*x + y*y)
      );
    }

    @vertex
    fn vs_main(
      @builtin(instance_index) instIdx: u32,
      @location(0) quadPos: vec2f
    ) -> VertexOutput {
      var out: VertexOutput;

      let gIdx = sortedIndices[instIdx];
      let g = gaussians[gIdx];

      let pView = (u.view * vec4f(g.pos.xyz, 1.0)).xyz;
      if (pView.z >= -0.15) {
        out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
        return out;
      }

      let R = quatToMat3(g.rot);
      let S = mat3x3f(
        g.scale.x, 0.0, 0.0,
        0.0, g.scale.y, 0.0,
        0.0, 0.0, g.scale.z
      );
      let M = R * S;
      let Sigma = M * transpose(M);

      let fx = u.focal.x;
      let fy = u.focal.y;
      let rz = 1.0 / pView.z;
      let rz2 = rz * rz;

      let J = mat3x3f(
        fx * rz, 0.0, -fx * pView.x * rz2,
        0.0, fy * rz, -fy * pView.y * rz2,
        0.0, 0.0, 0.0
      );

      let W = mat3x3f(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
      let T = J * W;
      let cov2D = T * Sigma * transpose(T);

      let a = cov2D[0][0] + 0.25;
      let b = cov2D[0][1];
      let c = cov2D[1][1] + 0.25;

      let det = a * c - b * b;
      if (det <= 0.0001) {
        out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
        return out;
      }

      let conic = vec3f(c / det, -b / det, a / det);

      let mid = 0.5 * (a + c);
      let term = sqrt(max(0.1, mid * mid - det));
      let lambda1 = mid + term;
      let lambda2 = max(0.1, mid - term);
      let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

      let pProj = u.proj * vec4f(pView, 1.0);
      let centerNDC = pProj.xy / pProj.w;

      let offsetPixels = quadPos * radius;
      let offsetNDC = offsetPixels / (u.viewport * 0.5);

      out.pos = vec4f(centerNDC + offsetNDC, pProj.z / pProj.w, 1.0);
      out.color = g.color;
      out.conic = conic;
      out.coord = offsetPixels;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let d = in.coord;
      let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) - in.conic.y * d.x * d.y;

      if (power > 0.0) { discard; }

      let alpha = in.color.a * exp(power);
      if (alpha < 0.015) { discard; }

      let rgb = in.color.rgb;
      return vec4f(rgb * alpha, alpha);
    }
  `;

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: gsShaderWGSL }),
      entryPoint: "vs_main",
      buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }]
    },
    fragment: {
      module: device.createShaderModule({ code: gsShaderWGSL }),
      entryPoint: "fs_main",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" }
        }
      }]
    },
    primitive: { topology: "triangle-list" }
  });

  let gaussianBuffer: GPUBuffer;
  let sortedIndexBuffer: GPUBuffer;
  let bindGroup: GPUBindGroup;

  let currentData = createSyntheticScene(45000);
  let sortedIndices = new Uint32Array(currentData.count);

  function uploadDataToGPU(data: GaussianCloudData) {
    currentData = data;
    sortedIndices = new Uint32Array(data.count);
    for (let i = 0; i < data.count; i++) sortedIndices[i] = i;

    const packed = new Float32Array(data.count * 16);
    for (let i = 0; i < data.count; i++) {
      const o = i * 16;
      packed[o + 0] = data.positions[i * 3 + 0];
      packed[o + 1] = data.positions[i * 3 + 1];
      packed[o + 2] = data.positions[i * 3 + 2];
      packed[o + 3] = 1.0;

      packed[o + 4] = data.scales[i * 3 + 0];
      packed[o + 5] = data.scales[i * 3 + 1];
      packed[o + 6] = data.scales[i * 3 + 2];
      packed[o + 7] = 0.0;

      packed[o + 8] = data.rotations[i * 4 + 0];
      packed[o + 9] = data.rotations[i * 4 + 1];
      packed[o + 10] = data.rotations[i * 4 + 2];
      packed[o + 11] = data.rotations[i * 4 + 3];

      packed[o + 12] = data.colors[i * 4 + 0];
      packed[o + 13] = data.colors[i * 4 + 1];
      packed[o + 14] = data.colors[i * 4 + 2];
      packed[o + 15] = data.colors[i * 4 + 3];
    }

    if (gaussianBuffer) gaussianBuffer.destroy();
    gaussianBuffer = device.createBuffer({
      size: packed.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(gaussianBuffer, 0, packed);

    if (sortedIndexBuffer) sortedIndexBuffer.destroy();
    sortedIndexBuffer = device.createBuffer({
      size: sortedIndices.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(sortedIndexBuffer, 0, sortedIndices);

    bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: gaussianBuffer } },
        { binding: 2, resource: { buffer: sortedIndexBuffer } }
      ]
    });

    // 此时 camera 已经初始化，安全访问
    camera.target = [...data.center];
    camera.radius = data.radius * 2.2;
  }

  // 初次上传模型
  uploadDataToGPU(currentData);

  const onlinePresets: Record<string, string> = {
    "测试星云花 (默认离线)": "builtin",
    "NIANTIC 官方雕塑 (Statue)": "https://nianticlabs.github.io/spz/sample.spz",
  };

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".spz";
  fileInput.style.display = "none";
  document.body.appendChild(fileInput);

  const settings = {
    status: "准备就绪 (内置星云)",
    pointCount: currentData.count,
    selectedPreset: "测试星云花 (默认离线)",
    customUrl: "",
    autoRotate: true,
    selectLocalFile: () => {
      fileInput.value = "";
      fileInput.click();
    },
    loadModel: async () => {
      let targetUrl = settings.customUrl.trim();
      if (!targetUrl) {
        targetUrl = onlinePresets[settings.selectedPreset];
      }

      if (targetUrl === "builtin") {
        uploadDataToGPU(createSyntheticScene(45000));
        settings.pointCount = 45000;
        settings.status = "已切换为内置离线星云";
        return;
      }

      try {
        settings.status = "正在下载 SPZ 数据...";
        const resp = await fetch(targetUrl, { mode: "cors" });
        if (!resp.ok) throw new Error(`HTTP 异常 ${resp.status}`);
        const buf = await resp.arrayBuffer();
        settings.status = "正在流式解压与构建高斯网格...";
        const gsData = await parseSPZ(buf);
        uploadDataToGPU(gsData);
        settings.pointCount = gsData.count;
        settings.status = `渲染就绪 (${gsData.count} 点)`;
      } catch (err: any) {
        settings.status = `加载失败: ${err.message}`;
        console.error("加载 SPZ 失败:", err);
      }
    }
  };

  fileInput.onchange = async () => {
    if (fileInput.files && fileInput.files.length > 0) {
      const file = fileInput.files[0];
      settings.status = `正在解析本地: ${file.name}...`;
      try {
        const buf = await file.arrayBuffer();
        const gsData = await parseSPZ(buf);
        uploadDataToGPU(gsData);
        settings.pointCount = gsData.count;
        settings.status = `已载入本地模型: ${file.name} (${gsData.count} 点)`;
      } catch (err: any) {
        settings.status = `本地解析失败: ${err.message}`;
      }
    }
  };

  // GUI 控制栏
  gui.title("WebGPU 3D高斯Splatting (优化版)");
  gui.add(settings, "status").name("运行状态").listen().disable();
  gui.add(settings, "pointCount").name("高斯粒子数").listen().disable();
  gui.add(settings, "selectedPreset", Object.keys(onlinePresets)).name("预设模型");
  gui.add(settings, "customUrl").name("或输入 SPZ 链接");
  gui.add(settings, "loadModel").name("🚀 加载选中/网络模型");
  gui.add(settings, "selectLocalFile").name("📂 从本地选择 .spz 文件");
  gui.add(settings, "autoRotate").name("视角自动环绕");

  // 拖拽文件到画布支持
  canvas.addEventListener("dragover", (e) => e.preventDefault());
  canvas.addEventListener("drop", async (e) => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      settings.status = `正在载入: ${file.name}...`;
      const buf = await file.arrayBuffer();
      const gsData = await parseSPZ(buf);
      uploadDataToGPU(gsData);
      settings.pointCount = gsData.count;
      settings.status = `完成: ${file.name}`;
    }
  });

  // 轨道相机交互手势
  let isDragging = false, dragButton = 0, lastX = 0, lastY = 0;
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    isDragging = true;
    dragButton = e.button;
    if (e.shiftKey) dragButton = 2;
    lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!isDragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;

    if (dragButton === 0) {
      camera.theta -= dx * 0.35;
      camera.phi = Math.max(-88, Math.min(88, camera.phi + dy * 0.35));
    } else if (dragButton === 2) {
      const radTheta = (camera.theta * Math.PI) / 180;
      const panSpeed = camera.radius * 0.0015;
      camera.target[0] -= Math.cos(radTheta) * dx * panSpeed;
      camera.target[2] -= -Math.sin(radTheta) * dx * panSpeed;
      camera.target[1] += dy * panSpeed;
    }
  });
  canvas.addEventListener("pointerup", (e) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.radius = Math.max(0.1, camera.radius * Math.exp(e.deltaY * 0.001));
  }, { passive: false });

  // 深度排序
  let lastSortTime = 0;
  function sortGaussians(camEye: number[], viewDir: number[]) {
    const pos = currentData.positions;
    const count = currentData.count;
    const depths = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const dx = pos[i * 3 + 0] - camEye[0];
      const dy = pos[i * 3 + 1] - camEye[1];
      const dz = pos[i * 3 + 2] - camEye[2];
      depths[i] = dx * viewDir[0] + dy * viewDir[1] + dz * viewDir[2];
    }
    sortedIndices.sort((a, b) => depths[b] - depths[a]);
    device.queue.writeBuffer(sortedIndexBuffer, 0, sortedIndices);
  }

  let animId: number;
  const uniformCPU = new Float32Array(48);

  function frame(timestamp: number) {
    if (settings.autoRotate && !isDragging) {
      camera.theta += 0.25;
    }

    const width = canvas.width || 800;
    const height = canvas.height || 600;
    const aspect = width / height;

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;

    const eye = [
      camera.target[0] + camera.radius * Math.cos(radPhi) * Math.sin(radTheta),
      camera.target[1] + camera.radius * Math.sin(radPhi),
      camera.target[2] + camera.radius * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const view = createLookAtMatrix(eye, camera.target, [0, 1, 0]);
    const fov = (48 * Math.PI) / 180;
    const proj = createPerspectiveMatrix(fov, aspect, 0.05, 5000.0);

    const focalY = height / (2.0 * Math.tan(fov / 2.0));
    const focalX = focalY;

    if (timestamp - lastSortTime > 70) {
      const viewDir = [
        camera.target[0] - eye[0],
        camera.target[1] - eye[1],
        camera.target[2] - eye[2]
      ];
      sortGaussians(eye, viewDir);
      lastSortTime = timestamp;
    }

    uniformCPU.set(view, 0);
    uniformCPU.set(proj, 16);
    uniformCPU.set([eye[0], eye[1], eye[2], 1.0], 32);
    uniformCPU.set([width, height, focalX, focalY], 36);
    device.queue.writeBuffer(uniformBuffer, 0, uniformCPU);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0.03, g: 0.04, b: 0.06, a: 1.0 },
        storeOp: "store"
      }]
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, quadBuffer);
    pass.draw(6, currentData.count, 0, 0);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    if (fileInput.parentNode) fileInput.parentNode.removeChild(fileInput);
    quadBuffer.destroy();
    uniformBuffer.destroy();
    if (gaussianBuffer) gaussianBuffer.destroy();
    if (sortedIndexBuffer) sortedIndexBuffer.destroy();
  };
}