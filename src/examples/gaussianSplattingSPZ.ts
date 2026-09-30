// src/examples/gaussianSplattingSPZ.ts
import GUI from "lil-gui";

// ==================== 矩阵与投影数学 ====================
function createPerspectiveMatrix(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(fovRad / 2.0);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  // WebGPU 规范 NDC Z 范围 [0, 1]
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

// ==================== 官方纯正 SPZ 解码器 ====================
async function parseSPZ(buffer: ArrayBuffer): Promise<GaussianCloudData> {
  const ds = new DecompressionStream("gzip");
  const writer = ds.writable.getWriter();
  writer.write(new Uint8Array(buffer));
  writer.close();
  const decompressed = await new Response(ds.readable).arrayBuffer();

  const view = new DataView(decompressed);
  const count = view.getUint32(8, true);
  const fractionalBits = view.getUint8(13);

  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const rotations = new Float32Array(count * 4);
  const colors = new Float32Array(count * 4);
  const uint8 = new Uint8Array(decompressed);
  let offset = 16;

  // 1. 位置读取 (转至 WebGPU 正立坐标系)
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
      positions[i * 3 + 1] = -vy * scaleFactor; // 狮头向上正立
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

  // 2. Alpha 读取
  const alphas = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    alphas[i] = uint8[offset++] / 255.0;
  }

  // 3. 颜色读取 (SPZ 真实原生高动态颜色)
  for (let i = 0; i < count; i++) {
    colors[i * 4 + 0] = uint8[offset + 0] / 255.0;
    colors[i * 4 + 1] = uint8[offset + 1] / 255.0;
    colors[i * 4 + 2] = uint8[offset + 2] / 255.0;
    colors[i * 4 + 3] = alphas[i];
    offset += 3;
  }

  // 4. Scales 读取
  for (let i = 0; i < count * 3; i++) {
    scales[i] = Math.exp(uint8[offset++] / 16.0 - 10.0);
  }

  // 5. 旋转四元数解码：精确匹配 y, z 轴翻转，高斯椭球严格顺着皮肤曲面生长
  for (let i = 0; i < count; i++) {
    const qx = (uint8[offset + 0] - 128.0) / 128.0;
    const qy = (uint8[offset + 1] - 128.0) / 128.0;
    const qz = (uint8[offset + 2] - 128.0) / 128.0;
    const sumSq = qx * qx + qy * qy + qz * qz;
    const qw = Math.sqrt(Math.max(0.0, 1.0 - sumSq));

    const fx = qx;
    const fy = -qy;
    const fz = -qz;
    const fw = qw;

    const len = Math.hypot(fx, fy, fz, fw) || 1.0;
    rotations[i * 4 + 0] = fx / len;
    rotations[i * 4 + 1] = fy / len;
    rotations[i * 4 + 2] = fz / len;
    rotations[i * 4 + 3] = fw / len;
    offset += 3;
  }

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

// ==================== 高性能排序 Worker ====================
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
      const { viewRowZ, count, generation } = e.data;
      if (!positions) return;
      const depths = new Float32Array(count);
      const indices = new Uint32Array(count);
      
      for (let i = 0; i < count; i++) {
        indices[i] = i;
        depths[i] = positions[i * 3 + 0] * viewRowZ[0] +
                    positions[i * 3 + 1] * viewRowZ[1] +
                    positions[i * 3 + 2] * viewRowZ[2] + viewRowZ[3];
      }
      fastRadixSort(depths, indices, count);
      // 自后向前绘制 (Back-to-Front)
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
  // 严格匹配参考图的霸气特写正视视角
  const camera = { target: [0, 0, 0], radius: 1.0, theta: 0.0, phi: 0.0 };
  const sortWorker = new Worker(URL.createObjectURL(workerBlob));
  let currentGeneration = 0;

  // Quad 几何体 [-3, 3] 覆盖 3-sigma 衰减尾部
  const quadVertices = new Float32Array([
    -3.0, -3.0,
     3.0, -3.0,
    -3.0,  3.0,
    -3.0,  3.0,
     3.0, -3.0,
     3.0,  3.0
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

  // ================= 彻底消灭白刺与噪斑的 WGSL 着色器 =================
  const gsShaderWGSL = `
    struct Uniforms {
      view: mat4x4f,
      proj: mat4x4f,
      camPos: vec4f,
      viewport: vec2f,
      focal: vec2f,
      params: vec4f, // x: kernelSize
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
      @location(1) uv: vec2f,
    };

    fn quatToMat3(q: vec4f) -> mat3x3f {
      let x = q.x; let y = q.y; let z = q.z; let w = q.w;
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

      let viewCenter4 = u.view * vec4f(g.pos.xyz, 1.0);
      let viewCenter = viewCenter4.xyz;
      let centerClip = u.proj * viewCenter4;

      if (viewCenter.z >= -0.05) {
        out.pos = vec4f(2.0, 2.0, 2.0, 1.0);
        return out;
      }

      // 1. 3D 协方差 Sigma = R * S * S^T * R^T
      let R = quatToMat3(g.rot);
      let S = mat3x3f(
        vec3f(g.scale.x, 0.0, 0.0),
        vec3f(0.0, g.scale.y, 0.0),
        vec3f(0.0, 0.0, g.scale.z)
      );
      let M = R * S;
      let Sigma = M * transpose(M);

      // 2. 变换至观察空间: V = W * Sigma * W^T
      let r0 = vec3f(u.view[0].x, u.view[1].x, u.view[2].x);
      let r1 = vec3f(u.view[0].y, u.view[1].y, u.view[2].y);

      let cov0 = Sigma[0]; let cov1 = Sigma[1]; let cov2 = Sigma[2];
      let vc0 = vec3f(dot(r0, cov0), dot(r0, cov1), dot(r0, cov2));
      let vc1 = vec3f(dot(r1, cov0), dot(r1, cov1), dot(r1, cov2));

      let c00 = dot(vc0, r0);
      let c01 = dot(vc0, r1);
      let c11 = dot(vc1, r1);

      // 3. 关键核心突破：使用仿射 EWA 雅可比投影！
      // 彻底消除导致满天白刺的透视二次非线性交叉项 (J02, J12)
      let z_depth = min(viewCenter.z, -0.05);
      let invZ = 1.0 / (-z_depth);
      let sX = u.focal.x * invZ;
      let sY = u.focal.y * invZ;

      let aBase = sX * sX * c00;
      let b     = sX * sY * c01;
      let cBase = sY * sY * c11;

      // 4. 精细低通抗锯齿膨胀核
      let a = aBase + u.params.x;
      let c = cBase + u.params.x;

      // 5. 特征值与椭圆主轴解析求解
      let halfTrace = 0.5 * (a + c);
      let radius = sqrt(max(0.25 * (a - c) * (a - c) + b * b, 0.0000001));
      let lambda1 = max(halfTrace + radius, 0.0000001);
      let lambda2 = max(halfTrace - radius, 0.0000001);

      var axis1 = vec2f(1.0, 0.0);
      if (radius > 0.00001) {
        let angle = 0.5 * atan2(2.0 * b, a - c);
        axis1 = vec2f(cos(angle), sin(angle));
      }
      let axis2 = vec2f(-axis1.y, axis1.x);

      let scale1 = min(sqrt(lambda1), 512.0);
      let scale2 = min(sqrt(lambda2), 512.0);

      let offsetPixels = axis1 * (quadPos.x * scale1) + axis2 * (quadPos.y * scale2);
      let offsetNdc = (offsetPixels * 2.0) / u.viewport;

      out.pos = centerClip + vec4f(offsetNdc * centerClip.w, 0.0, 0.0);
      // 原汁原味的高保真原生颜色，彻底告别彩虹噪斑与灰白雾感
      out.color = vec4f(g.color.rgb, g.color.a);
      out.uv = quadPos;

      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let r2 = dot(in.uv, in.uv);
      
      // 3-sigma 边界平滑趋近于零，消除生硬毛边
      if (r2 > 9.0) { discard; }

      let G = exp(-0.5 * r2);
      let alpha = in.color.a * G;

      if (alpha < (1.0 / 255.0)) { discard; }

      // 预乘 Alpha 输出
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

    // 每个高斯 16 个 float
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

    // 完美复刻参考图的特写镜头机位 (正视狮面、充满屏幕、立体深邃)
    camera.target = [data.center[0], data.center[1] + data.radius * 0.05, data.center[2]];
    camera.radius = data.radius * 1.05;
    camera.theta = 0.0;
    camera.phi = 2.0;
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
    kernel2DSize: 0.15, // 锐化低通滤波核，从模糊毛球转变为根根分明的发丝
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
        settings.status = `高清毛发特写渲染中 (${gsData.count} 点)`;
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

  gui.title("WebGPU 3D GS (顶级清晰质感还原版)");
  gui.add(settings, "status").name("运行状态").listen().disable();
  gui.add(settings, "pointCount").name("粒子数").listen().disable();
  gui.add(settings, "selectedPreset", Object.keys(onlinePresets)).name("在线预设");
  gui.add(settings, "customUrl").name("自定义 URL");
  gui.add(settings, "loadModel").name("🚀 加载选中模型");
  gui.add(settings, "selectLocalFile").name("📂 打开本地 SPZ");
  gui.add(settings, "autoRotate").name("自动环绕");
  gui.add(settings, "kernel2DSize", 0.05, 0.4, 0.01).name("画面锐度/滤波核");

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

    // View 矩阵第三行严格用于透视空间深度排序
    const viewRowZ = [view[2], view[6], view[10], view[14]];

    if (!isSorting && currentData.count > 0) {
      isSorting = true;
      sortWorker.postMessage({
        type: "sort",
        viewRowZ,
        count: currentData.count,
        generation: currentGeneration,
      });
    }

    uniformCPU.set(view, 0);
    uniformCPU.set(proj, 16);
    uniformCPU.set([eye[0], eye[1], eye[2], 1.0], 32);
    uniformCPU.set([renderWidth, renderHeight, focalX, focalY], 36);
    uniformCPU.set([settings.kernel2DSize, 0.0, 0.0, 0.0], 40);

    device.queue.writeBuffer(uniformBuffer, 0, uniformCPU);

    if (currentData.count > 0 && bindGroup) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0.015, g: 0.015, b: 0.015, a: 1.0 },
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