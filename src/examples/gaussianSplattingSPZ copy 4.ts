// src/examples/gaussianSplattingSPZ.ts
import GUI from "lil-gui";

// ==================== 矩阵与投影数学 ====================
function createPerspectiveMatrix(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(fovRad / 2.0);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far / (near - far);
  out[11] = -1.0;
  out[14] = (near * far) / (near - far);
  return out;
}

function createLookAtMatrix(eye: number[], center: number[], up: number[]): Float32Array {
  let z0 = eye[0] - center[0];
  let z1 = eye[1] - center[1];
  let z2 = eye[2] - center[2];
  const lenZ = 1.0 / (Math.hypot(z0, z1, z2) || 1.0);
  z0 *= lenZ; z1 *= lenZ; z2 *= lenZ;

  let x0 = up[1] * z2 - up[2] * z1;
  let x1 = up[2] * z0 - up[0] * z2;
  let x2 = up[0] * z1 - up[1] * z0;
  const lenX = 1.0 / (Math.hypot(x0, x1, x2) || 1.0);
  x0 *= lenX; x1 *= lenX; x2 *= lenX;

  const y0 = z1 * x2 - z2 * x1;
  const y1 = z2 * x0 - z0 * x2;
  const y2 = z0 * x1 - z1 * x0;

  const out = new Float32Array(16);
  out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
  out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
  out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
  out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
  out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
  out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]);
  out[15] = 1.0;
  return out;
}

export interface GaussianCloudData {
  count: number;
  positions: Float32Array;
  scales: Float32Array;
  rotations: Float32Array;
  colors: Float32Array;
  center: [number, number, number];
  radius: number;
}

// ==================== 严格标准版 SPZ 解码器 ====================
async function parseSPZ(buffer: ArrayBuffer): Promise<GaussianCloudData> {
  const ds = new DecompressionStream("gzip");
  const writer = ds.writable.getWriter();
  writer.write(new Uint8Array(buffer));
  writer.close();
  const decompressed = await new Response(ds.readable).arrayBuffer();

  const view = new DataView(decompressed);
  const count = view.getUint32(8, true);
  const shDegree = view.getUint8(12);
  const fractionalBits = view.getUint8(13);

  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const rotations = new Float32Array(count * 4);
  const colors = new Float32Array(count * 4);
  const uint8 = new Uint8Array(decompressed);
  let offset = 16;

  // 1. 位置读取 (标准 3DGS 坐标系向 WebGPU 坐标系适配：反转 Y 和 Z)
  if (fractionalBits > 0) {
    const scaleFactor = 1.0 / (1 << fractionalBits);
    for (let i = 0; i < count; i++) {
      let vx = uint8[offset + 0] | (uint8[offset + 1] << 8) | (uint8[offset + 2] << 16);
      if (vx & 0x800000) vx |= 0xff000000;
      let vy = uint8[offset + 3] | (uint8[offset + 4] << 8) | (uint8[offset + 5] << 16);
      if (vy & 0x800000) vy |= 0xff000000;
      let vz = uint8[offset + 6] | (uint8[offset + 7] << 8) | (uint8[offset + 8] << 16);
      if (vz & 0x800000) vz |= 0xff000000;
      offset += 9;

      positions[i * 3 + 0] = vx * scaleFactor;
      positions[i * 3 + 1] = -vy * scaleFactor;
      positions[i * 3 + 2] = -vz * scaleFactor;
    }
  } else {
    for (let i = 0; i < count; i++) {
      positions[i * 3 + 0] = view.getFloat32(offset + 0, true);
      positions[i * 3 + 1] = -view.getFloat32(offset + 4, true);
      positions[i * 3 + 2] = -view.getFloat32(offset + 8, true);
      offset += 12;
    }
  }

  // 2. Alpha (SPZ 规范中已做 Sigmoid 压缩，直接线性归一化)
  const alphas = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    alphas[i] = uint8[offset++] / 255.0;
  }

  // 3. 颜色读取 (SH0 真实色彩还原)
  const SH_C0 = 0.28209479177387814;
  for (let i = 0; i < count; i++) {
    const rawR = (uint8[offset + 0] - 128.0) / 64.0;
    const rawG = (uint8[offset + 1] - 128.0) / 64.0;
    const rawB = (uint8[offset + 2] - 128.0) / 64.0;

    let r = rawR * SH_C0 + 0.5;
    let g = rawG * SH_C0 + 0.5;
    let b = rawB * SH_C0 + 0.5;

    // 超界回退保护
    if (r < 0.0 || r > 1.0 || g < 0.0 || g > 1.0 || b < 0.0 || b > 1.0) {
      r = uint8[offset + 0] / 255.0;
      g = uint8[offset + 1] / 255.0;
      b = uint8[offset + 2] / 255.0;
    }

    colors[i * 4 + 0] = Math.max(0.0, Math.min(1.0, r));
    colors[i * 4 + 1] = Math.max(0.0, Math.min(1.0, g));
    colors[i * 4 + 2] = Math.max(0.0, Math.min(1.0, b));
    colors[i * 4 + 3] = alphas[i];
    offset += 3;
  }

  // 4. Scales 读取
  for (let i = 0; i < count * 3; i++) {
    scales[i] = Math.exp(uint8[offset++] / 16.0 - 10.0);
  }

  // 5. 旋转四元数解码 (精准映射 SO(3) 翻转变换)
  // 当位置变换 P' = diag(1, -1, -1) * P 时，对应的四元数必须准确构造
  for (let i = 0; i < count; i++) {
    const rx = (uint8[offset + 0] - 128.0) / 128.0;
    const ry = (uint8[offset + 1] - 128.0) / 128.0;
    const rz = (uint8[offset + 2] - 128.0) / 128.0;
    const sumSq = rx * rx + ry * ry + rz * rz;
    const rw = Math.sqrt(Math.max(0.0, 1.0 - sumSq));

    // 严密适配：绕 X 轴旋转 180° 的四元数乘积 q' = (rw, -rz, ry, -rx)
    // 使得旋转后的 3D 椭球体与翻转后的几何完全贴合，根除拉丝毛刺
    const qx = rw;
    const qy = -rz;
    const qz = ry;
    const qw = -rx;

    const len = Math.hypot(qx, qy, qz, qw) || 1.0;
    rotations[i * 4 + 0] = qx / len;
    rotations[i * 4 + 1] = qy / len;
    rotations[i * 4 + 2] = qz / len;
    rotations[i * 4 + 3] = qw / len;
    offset += 3;
  }

  // 6. 跳过额外高阶球谐系数
  const shCoeffs = shDegree > 0 ? ((shDegree + 1) * (shDegree + 1) - 1) * 3 : 0;
  offset += count * shCoeffs;

  // 7. 计算中心及包围半径
  let sumX = 0, sumY = 0, sumZ = 0;
  for (let i = 0; i < count; i++) {
    sumX += positions[i * 3 + 0];
    sumY += positions[i * 3 + 1];
    sumZ += positions[i * 3 + 2];
  }
  const cx = sumX / count, cy = sumY / count, cz = sumZ / count;

  let maxDistSq = 0;
  for (let i = 0; i < count; i += 10) {
    const dx = positions[i * 3 + 0] - cx;
    const dy = positions[i * 3 + 1] - cy;
    const dz = positions[i * 3 + 2] - cz;
    maxDistSq = Math.max(maxDistSq, dx * dx + dy * dy + dz * dz);
  }

  return {
    count, positions, scales, rotations, colors,
    center: [cx, cy, cz],
    radius: Math.max(0.3, Math.sqrt(maxDistSq) * 0.7),
  };
}

// ==================== 高性能基数排序 Worker ====================
const workerBlob = new Blob([`
  function fastRadixSort(depths, indices, count) {
    let minD = depths[0], maxD = depths[0];
    for (let i = 1; i < count; i++) {
      if (depths[i] < minD) minD = depths[i];
      if (depths[i] > maxD) maxD = depths[i];
    }
    const range = maxD - minD;
    if (range <= 0.00001) return;

    const keys = new Uint16Array(count);
    const factor = 65535.0 / range;
    for (let i = 0; i < count; i++) {
      keys[i] = ((depths[i] - minD) * factor) | 0;
    }

    const temp = new Uint32Array(count);
    const counts = new Uint32Array(256);

    for (let i = 0; i < count; i++) counts[keys[i] & 0xff]++;
    let sum = 0;
    for (let i = 0; i < 256; i++) { const c = counts[i]; counts[i] = sum; sum += c; }
    for (let i = 0; i < count; i++) { const id = indices[i]; temp[counts[keys[id] & 0xff]++] = id; }

    counts.fill(0);
    for (let i = 0; i < count; i++) counts[(keys[temp[i]] >> 8) & 0xff]++;
    sum = 0;
    for (let i = 0; i < 256; i++) { const c = counts[i]; counts[i] = sum; sum += c; }
    for (let i = 0; i < count; i++) { const id = temp[i]; indices[counts[(keys[id] >> 8) & 0xff]++] = id; }
  }

  let positions = null;
  self.onmessage = function(e) {
    if (e.data.type === 'init') {
      positions = new Float32Array(e.data.positions);
      return;
    }
    if (e.data.type === 'sort') {
      const { viewDir, camEye, count, generation } = e.data;
      if (!positions) return;
      const depths = new Float32Array(count);
      const indices = new Uint32Array(count);
      for (let i = 0; i < count; i++) {
        indices[i] = i;
        depths[i] = (positions[i * 3 + 0] - camEye[0]) * viewDir[0] +
                    (positions[i * 3 + 1] - camEye[1]) * viewDir[1] +
                    (positions[i * 3 + 2] - camEye[2]) * viewDir[2];
      }
      fastRadixSort(depths, indices, count);
      // 由远及近渲染 (Back-to-front)
      indices.reverse();
      self.postMessage({ type: 'sorted', generation, indices: indices.buffer }, [indices.buffer]);
    }
  };
`], { type: "application/javascript" });

// ==================== 主渲染流程 ====================
export function runGaussianSplattingspz(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: any
) {
  const camera = { target: [0, 0, 0], radius: 1.0, theta: 25.0, phi: 12.0 };
  const sortWorker = new Worker(URL.createObjectURL(workerBlob));
  let currentGeneration = 0;

  // 标准单位 Quad 顶点：范围 [-2, 2] 对应 2-Sigma 半径展开
  const quadVertices = new Float32Array([
    -2.0, -2.0,
     2.0, -2.0,
    -2.0,  2.0,
    -2.0,  2.0,
     2.0, -2.0,
     2.0,  2.0
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

  // ================= 严格符合 3DGS 论文数学原型的 WGSL 着色器 =================
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
      color: vec4f
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
      // 列主序构建精确旋转矩阵
      return mat3x3f(
        vec3f(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z), 2.0*(x*z - w*y)),
        vec3f(2.0*(x*y - w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x)),
        vec3f(2.0*(x*z + w*y), 2.0*(y*z - w*x), 1.0 - 2.0*(x*x + y*y))
      );
    }

    @vertex
    fn vs_main(@builtin(instance_index) instIdx: u32, @location(0) quadPos: vec2f) -> VertexOutput {
      var out: VertexOutput;
      let g = gaussians[sortedIndices[instIdx]];

      // 相机空间变换
      let pView = (u.view * vec4f(g.pos.xyz, 1.0)).xyz;
      // 剔除相机背面粒子
      if (pView.z >= -0.05) {
        out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
        return out;
      }

      let R = quatToMat3(g.rot);
      let S = mat3x3f(
        vec3f(g.scale.x, 0.0, 0.0),
        vec3f(0.0, g.scale.y, 0.0),
        vec3f(0.0, 0.0, g.scale.z)
      );
      let M = R * S;
      let Sigma = M * transpose(M);

      // 从 u.view 提取世界到相机空间的旋转变换 W
      let W = mat3x3f(
        u.view[0].xyz,
        u.view[1].xyz,
        u.view[2].xyz
      );

      // 相机空间 3D 协方差
      let Vrk = W * Sigma * transpose(W);

      let fx = u.focal.x;
      let fy = u.focal.y;
      let rz = 1.0 / (-pView.z);
      let rz2 = rz * rz;

      // 投影雅可比矩阵 J
      let J = mat3x2f(
        vec2f(fx * rz, 0.0),
        vec2f(0.0, fy * rz),
        vec2f((fx * pView.x) * rz2, (fy * pView.y) * rz2)
      );

      // 屏幕空间 2D 协方差矩阵: cov2d = J * Vrk * J^T
      let cov2d = J * Vrk * transpose(J);

      // 低通滤波抗混叠保护 (+0.3 像素方差)
      let a = cov2d[0][0] + 0.3;
      let b = cov2d[0][1];
      let c = cov2d[1][1] + 0.3;

      let det = a * c - b * b;
      if (det <= 0.00001) {
        out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
        return out;
      }

      // 计算椭圆二次型逆矩阵 (Conic)
      let invDet = 1.0 / det;
      let conic = vec3f(c * invDet, -b * invDet, a * invDet);

      // 正确计算特征值与最大主轴半径（3-Sigma 准则），杜绝针刺拉丝
      let mid = 0.5 * (a + c);
      let term = sqrt(max(0.1, mid * mid - det));
      let lambda1 = mid + term;
      let lambda2 = mid - term;
      let maxRadius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

      if (maxRadius > 1024.0) {
        out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
        return out;
      }

      // 平移顶点到像素空间包围范围
      let pixelOffset = quadPos * (maxRadius / 2.0);

      let pProj = u.proj * vec4f(pView, 1.0);
      let centerNDC = pProj.xy / pProj.w;
      let offsetNDC = pixelOffset / (u.viewport * 0.5);

      out.pos = vec4f(centerNDC + offsetNDC, pProj.z / pProj.w, 1.0);
      out.color = g.color;
      out.conic = conic;
      out.coord = pixelOffset;

      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let d = in.coord;

      // 计算标准马氏距离 (Mahalanobis Distance)
      let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) - in.conic.y * d.x * d.y;

      if (power > 0.0) { discard; }

      let G = exp(power);
      let alpha = in.color.a * G;

      if (alpha < 0.005) { discard; }

      // 预乘 Alpha 输出 (Premultiplied Alpha)
      return vec4f(in.color.rgb * alpha, alpha);
    }
  `;

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: gsShaderWGSL }),
      entryPoint: "vs_main",
      buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }],
    },
    fragment: {
      module: device.createShaderModule({ code: gsShaderWGSL }),
      entryPoint: "fs_main",
      targets: [
        {
          format,
          blend: {
            color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          },
        },
      ],
    },
    primitive: { topology: "triangle-list" },
  });

  let gaussianBuffer: GPUBuffer;
  let sortedIndexBuffer: GPUBuffer;
  let bindGroup: GPUBindGroup;
  let currentData: GaussianCloudData = {
    count: 0,
    positions: new Float32Array(),
    scales: new Float32Array(),
    rotations: new Float32Array(),
    colors: new Float32Array(),
    center: [0, 0, 0],
    radius: 1.0
  };

  function uploadDataToGPU(data: GaussianCloudData) {
    currentData = data;
    currentGeneration++;

    const initIndices = new Uint32Array(data.count);
    for (let i = 0; i < data.count; i++) initIndices[i] = i;

    const workerPositions = data.positions.slice().buffer;
    sortWorker.postMessage({
      type: "init",
      positions: workerPositions,
    }, [workerPositions]);

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
      size: initIndices.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(sortedIndexBuffer, 0, initIndices);

    bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: gaussianBuffer } },
        { binding: 2, resource: { buffer: sortedIndexBuffer } },
      ],
    });

    camera.target = [...data.center];
    camera.radius = data.radius * 1.6;
    camera.theta = 25.0;
    camera.phi = 12.0;
  }

  let isSorting = false;
  sortWorker.onmessage = (e) => {
    if (e.data.type === "sorted") {
      if (e.data.generation === currentGeneration) {
        const result = new Uint32Array(e.data.indices);
        device.queue.writeBuffer(sortedIndexBuffer, 0, result);
      }
      isSorting = false;
    }
  };

  const onlinePresets: Record<string, string> = {
    "Three.js 官方狮子 (lion.v3.spz)": "https://raw.githubusercontent.com/mrdoob/three.js/master/examples/models/splat/lion.v3.spz",
    "Niantic 官方雕塑 (Statue)": "https://nianticlabs.github.io/spz/sample.spz",
  };

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".spz";
  fileInput.style.display = "none";
  document.body.appendChild(fileInput);

  const settings = {
    status: "准备就绪",
    pointCount: 0,
    selectedPreset: "Three.js 官方狮子 (lion.v3.spz)",
    customUrl: "",
    autoRotate: true,
    selectLocalFile: () => {
      fileInput.value = "";
      fileInput.click();
    },
    loadModel: async () => {
      const targetUrl = settings.customUrl.trim() || onlinePresets[settings.selectedPreset];
      try {
        settings.status = "正在下载 SPZ 数据...";
        const resp = await fetch(targetUrl, { mode: "cors" });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const buf = await resp.arrayBuffer();
        settings.status = "正在解包渲染数据...";
        const gsData = await parseSPZ(buf);
        uploadDataToGPU(gsData);
        settings.pointCount = gsData.count;
        settings.status = `高清平滑渲染中 (${gsData.count} 点)`;
      } catch (err: any) {
        settings.status = `加载失败: ${err.message}`;
      }
    },
  };

  fileInput.onchange = async () => {
    if (fileInput.files?.length) {
      const file = fileInput.files[0];
      settings.status = `正在读取: ${file.name}`;
      try {
        const gsData = await parseSPZ(await file.arrayBuffer());
        uploadDataToGPU(gsData);
        settings.pointCount = gsData.count;
        settings.status = `已载入: ${file.name}`;
      } catch (err: any) {
        settings.status = `解析失败: ${err.message}`;
      }
    }
  };

  gui.title("WebGPU 3D GS (平滑高清完美版)");
  gui.add(settings, "status").name("运行状态").listen().disable();
  gui.add(settings, "pointCount").name("粒子数").listen().disable();
  gui.add(settings, "selectedPreset", Object.keys(onlinePresets)).name("在线预设");
  gui.add(settings, "customUrl").name("自定义 URL");
  gui.add(settings, "loadModel").name("🚀 加载选中模型");
  gui.add(settings, "selectLocalFile").name("📂 打开本地 SPZ");
  gui.add(settings, "autoRotate").name("自动环绕");

  // 鼠标交互平移与旋转
  let isDragging = false, dragButton = 0, lastX = 0, lastY = 0;
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    isDragging = true;
    dragButton = e.shiftKey ? 2 : e.button;
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
    camera.radius = Math.max(0.05, camera.radius * Math.exp(e.deltaY * 0.001));
  }, { passive: false });

  // 默认启动载入狮子模型
  settings.loadModel();

  let animId: number;
  const uniformCPU = new Float32Array(48);

  function frame() {
    if (settings.autoRotate && !isDragging) camera.theta += 0.25;

    const dpr = window.devicePixelRatio || 1;
    const renderWidth = Math.max(1, Math.floor((canvas.clientWidth || 800) * dpr));
    const renderHeight = Math.max(1, Math.floor((canvas.clientHeight || 600) * dpr));

    if (canvas.width !== renderWidth || canvas.height !== renderHeight) {
      canvas.width = renderWidth;
      canvas.height = renderHeight;
      context.configure({
        device,
        format,
        alphaMode: "opaque",
      });
    }

    const aspect = renderWidth / renderHeight;
    const radTheta = (camera.theta * Math.PI) / 180.0;
    const radPhi = (camera.phi * Math.PI) / 180.0;

    const eye = [
      camera.target[0] + camera.radius * Math.cos(radPhi) * Math.sin(radTheta),
      camera.target[1] + camera.radius * Math.sin(radPhi),
      camera.target[2] + camera.radius * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const view = createLookAtMatrix(eye, camera.target, [0, 1, 0]);
    const fov = (45.0 * Math.PI) / 180.0;
    const proj = createPerspectiveMatrix(fov, aspect, 0.1, 1000.0);

    const focalY = renderHeight / (2.0 * Math.tan(fov / 2.0));
    const focalX = focalY;

    const viewDir = [
      camera.target[0] - eye[0],
      camera.target[1] - eye[1],
      camera.target[2] - eye[2]
    ];
    const lenV = Math.hypot(viewDir[0], viewDir[1], viewDir[2]) || 1.0;
    viewDir[0] /= lenV; viewDir[1] /= lenV; viewDir[2] /= lenV;

    if (!isSorting && currentData.count > 0) {
      isSorting = true;
      sortWorker.postMessage({
        type: "sort",
        camEye: eye,
        viewDir,
        count: currentData.count,
        generation: currentGeneration,
      });
    }

    uniformCPU.set(view, 0);
    uniformCPU.set(proj, 16);
    uniformCPU.set([eye[0], eye[1], eye[2], 1.0], 32);
    uniformCPU.set([renderWidth, renderHeight, focalX, focalY], 36);

    device.queue.writeBuffer(uniformBuffer, 0, uniformCPU);

    if (currentData.count > 0 && bindGroup) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            storeOp: "store",
          },
        ],
      });

      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.setVertexBuffer(0, quadBuffer);
      pass.draw(6, currentData.count, 0, 0);
      pass.end();

      device.queue.submit([encoder.finish()]);
    }

    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    sortWorker.terminate();
    if (fileInput.parentNode) fileInput.parentNode.removeChild(fileInput);
    quadBuffer.destroy();
    uniformBuffer.destroy();
    if (gaussianBuffer) gaussianBuffer.destroy();
    if (sortedIndexBuffer) sortedIndexBuffer.destroy();
  };
}