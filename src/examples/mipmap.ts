// src/examples/runSampler.ts
import { Mat4 } from "../utils/math";
import type { SimpleGUI } from "../utils/gui";

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

export function runMipSampler(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // 1. 创建向前大幅度延伸的 3D 纵深跑道平面网格 (Pos3 + UV2)
  // 近端 Z=2，远端 Z=-45，让 MipMap 在纵深方向形成完美的 LOD 阶梯展示
  // prettier-ignore
  const vertexData = new Float32Array([
    // pos(x,y,z)        uv(u,v)
    -3.5, 0.0,  3.0,    -1.5,   0.0,
     3.5, 0.0,  3.0,     1.5,   0.0,
     3.5, 0.0, -50.0,    1.5,  28.0,
    -3.5, 0.0, -50.0,   -1.5,  28.0,
  ]);
  const indexData = new Uint16Array([0, 1, 2, 0, 2, 3]);

  const vBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vBuffer, 0, vertexData);

  const iBuffer = device.createBuffer({
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(iBuffer, 0, indexData);

  // 2. 创建支持 8 级 MipMap 的纹理 (128x128 -> 1x1)
  // 每一级 Mip 填充不同的独立色彩，肉眼直接识别当前像素正在采样的 Mip 层级
  const mipLevelCount = 8;
  const baseSize = 128;

  const texture = device.createTexture({
    size: [baseSize, baseSize, 1],
    mipLevelCount,
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  // Mip 对应颜色：红 -> 橙 -> 黄 -> 绿 -> 青 -> 蓝 -> 紫 -> 白
  const mipColors = [
    [230, 50, 50],   // Level 0 (128x128): 红
    [245, 130, 30],  // Level 1 (64x64):   橙
    [245, 215, 30],  // Level 2 (32x32):   黄
    [50, 205, 80],   // Level 3 (16x16):   绿
    [30, 200, 220],  // Level 4 (8x8):     青
    [60, 110, 240],  // Level 5 (4x4):     蓝
    [170, 70, 230],  // Level 6 (2x2):     紫
    [255, 255, 255], // Level 7 (1x1):     白
  ];

  for (let level = 0; level < mipLevelCount; level++) {
    const size = baseSize >> level;
    const pixels = new Uint8Array(size * size * 4);
    const color = mipColors[level];

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const idx = (y * size + x) * 4;
        let r = color[0], g = color[1], b = color[2];

        // 在 Level 0 绘制网格棋盘，极近距离时观察 Magnification (放大过滤) 的锯齿 vs 柔和插值
        if (level === 0 && ((x >> 3) + (y >> 3)) % 2 === 0) {
          r = Math.floor(r * 0.65);
          g = Math.floor(g * 0.65);
          b = Math.floor(b * 0.65);
        }
        // 外围留 1 像素亮边，便于识别平铺与边界
        if (x === 0 || y === 0 || x === size - 1 || y === size - 1) {
          r = 255; g = 255; b = 255;
        }

        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
        pixels[idx + 3] = 255;
      }
    }

    device.queue.writeTexture(
      { texture, mipLevel: level },
      pixels,
      { bytesPerRow: size * 4 },
      [size, size, 1]
    );
  }

  const textureView = texture.createView();

  // Uniform Buffer：mvp (64 bytes) + lodBias (4 bytes) + padding (12 bytes) = 80 bytes
  const uniformBuffer = device.createBuffer({
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  let depthTexture = device.createTexture({
    size: [canvas.width || 800, canvas.height || 600],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // 3. WGSL 着色器：使用 textureSampleBias 支持动态 LOD 偏移
  const shaderCode = `
    struct Uniforms {
      mvp: mat4x4f,
      lodBias: f32,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var mySampler: sampler;
    @group(0) @binding(2) var myTexture: texture_2d<f32>;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@location(0) pos: vec3f, @location(1) uv: vec2f) -> VertexOut {
      var out: VertexOut;
      out.pos = u.mvp * vec4f(pos, 1.0);
      out.uv = uv;
      return out;
    }

    @fragment
    fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
      // 通过 bias 实时调整硬件计算出的 Mip 级别偏置
      return textureSampleBias(myTexture, mySampler, uv, u.lodBias);
    }
  `;

  const module = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 20,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x2" },
          ],
        },
      ],
    },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // 4. 模式定义与采样器配置
  const addressModes: GPUAddressMode[] = ["clamp-to-edge", "repeat", "mirror-repeat"];
  const filterModes: GPUFilterMode[] = ["nearest", "linear"];
  const mipmapModes: GPUMipmapFilterMode[] = ["nearest", "linear"];

  const config = {
    magFilter: 0,    // 0: nearest, 1: linear (放大过滤)
    minFilter: 1,    // 0: nearest, 1: linear (缩小过滤)
    mipmapFilter: 1, // 0: nearest, 1: linear (MipMap层间过滤/三线性)
    lodMinClamp: 0,  // 最低 Mip 级别限制（0 为最清晰 Level 0）
    lodMaxClamp: 7,  // 最高 Mip 级别限制（7 为最模糊 1x1）
    lodBias: 0.0,    // LOD 整体偏移 [-4.0, 4.0]
    addressModeU: 1, // 0: clamp, 1: repeat, 2: mirror
    addressModeV: 1,
  };

  let currentSampler: GPUSampler | null = null;
  let bindGroup: GPUBindGroup | null = null;
  let lastSamplerKey = "";

  function rebuildSamplerAndBindGroup() {
    const minClamp = Math.min(config.lodMinClamp, config.lodMaxClamp);
    const maxClamp = Math.max(config.lodMinClamp, config.lodMaxClamp);

    const mag = filterModes[Math.round(config.magFilter)] || "nearest";
    const min = filterModes[Math.round(config.minFilter)] || "linear";
    const mip = mipmapModes[Math.round(config.mipmapFilter)] || "linear";
    const uMode = addressModes[Math.round(config.addressModeU)] || "repeat";
    const vMode = addressModes[Math.round(config.addressModeV)] || "repeat";

    const key = `${mag}_${min}_${mip}_${minClamp}_${maxClamp}_${uMode}_${vMode}`;
    if (key === lastSamplerKey && bindGroup) return;
    lastSamplerKey = key;

    currentSampler = device.createSampler({
      magFilter: mag,
      minFilter: min,
      mipmapFilter: mip,
      lodMinClamp: minClamp,
      lodMaxClamp: maxClamp,
      addressModeU: uMode,
      addressModeV: vMode,
    });

    bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: currentSampler },
        { binding: 2, resource: textureView },
      ],
    });
  }

  rebuildSamplerAndBindGroup();

  // 5. GUI 控件挂载
  gui.add(config, "mipmapFilter", 0, 1, 1).name("MipMap层间过滤 (0:近邻 1:线性插值)");
  gui.add(config, "minFilter", 0, 1, 1).name("缩小过滤 MinFilter (0:近邻 1:线性)");
  gui.add(config, "magFilter", 0, 1, 1).name("放大过滤 MagFilter (0:近邻 1:线性)");
  gui.add(config, "lodBias", -4.0, 4.0, 0.1).name("LOD 偏移偏置 (lodBias)");
  gui.add(config, "lodMinClamp", 0, 7, 1).name("LOD 最小限制 (lodMinClamp)");
  gui.add(config, "lodMaxClamp", 0, 7, 1).name("LOD 最大限制 (lodMaxClamp)");
  gui.add(config, "addressModeU", 0, 2, 1).name("U 寻址 (0:拉伸 1:重复 2:镜像)");
  gui.add(config, "addressModeV", 0, 2, 1).name("V 寻址 (0:拉伸 1:重复 2:镜像)");

  // 快捷切换与重置
  gui.addButton("切换 MipMap插值 (0:阶梯分明 ⇋ 1:三线性平滑)", () => {
    config.mipmapFilter = config.mipmapFilter === 0 ? 1 : 0;
    gui.updateDisplay();
  });
  gui.addButton("循环寻址模式 (clamp ➔ repeat ➔ mirror)", () => {
    const next = (config.addressModeU + 1) % 3;
    config.addressModeU = next;
    config.addressModeV = next;
    gui.updateDisplay();
  });

  // 相机设置（默认从近处朝向跑道纵深观察）
  const defaultCamera = { distance: 6.0, theta: 0, phi: 18, panX: 0.0, panY: 0.8, fov: 60 };
  const camera = { ...defaultCamera };

  gui.add(camera, "distance", 1.5, 20.0, 0.1).name("相机距离 (靠近看放大/远离看缩小)");
  gui.add(camera, "phi", 2, 85, 1).name("俯仰角 (Pitch)");
  gui.addButton("重置所有参数", () => {
    Object.assign(camera, defaultCamera);
    config.magFilter = 0;
    config.minFilter = 1;
    config.mipmapFilter = 1;
    config.lodMinClamp = 0;
    config.lodMaxClamp = 7;
    config.lodBias = 0.0;
    config.addressModeU = 1;
    config.addressModeV = 1;
    gui.updateDisplay();
  });

  gui.addTextInfo(
    "<b>各 Mip 级别色谱：</b><br>" +
    "• Level 0: <span style='color:#e63232'>■ 红(棋盘)</span> (128x128，近处放大)<br>" +
    "• Level 1: <span style='color:#f5821e'>■ 橙</span> | Level 2: <span style='color:#e6c81e'>■ 黄</span><br>" +
    "• Level 3: <span style='color:#32cd50'>■ 绿</span> | Level 4: <span style='color:#1ec8dc'>■ 青</span><br>" +
    "• Level 5: <span style='color:#3c6ef0'>■ 蓝</span> | Level 6: <span style='color:#aa46e6'>■ 紫</span><br>" +
    "• Level 7: <span style='color:#ffffff'>■ 白</span> (最远处 1x1)<br><br>" +
    "<b>教学观察技巧：</b><br>" +
    "1. 将 <b>mipmapFilter</b> 设为 0，地面呈现硬边界阶梯色带；设为 1 则色带间柔和过渡（三线性过滤）。<br>" +
    "2. 拖动 <b>lodBias</b>，可观察色阶整体向近处推移或退后。<br>" +
    "3. 滚轮拉近到红色地面，切换 <b>magFilter</b> 观察棋盘格马赛克与线性平滑区别。"
  );

  // 6. 相机交互
  let isDragging = false;
  let dragButton = 0;
  let lastX = 0, lastY = 0;

  const onPointerDown = (e: PointerEvent) => {
    isDragging = true;
    dragButton = e.shiftKey ? 2 : e.button;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: PointerEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    if (dragButton === 0) {
      camera.theta -= dx * 0.4;
      camera.phi = Math.max(2, Math.min(85, camera.phi + dy * 0.4));
    } else if (dragButton === 2) {
      const factor = camera.distance * 0.0018;
      camera.panX -= dx * factor;
      camera.panY += dy * factor;
    }
    gui.updateDisplay();
  };

  const onPointerUp = (e: PointerEvent) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  };

  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    camera.distance = Math.max(1.5, Math.min(250.0, camera.distance + e.deltaY * 0.005));
    gui.updateDisplay();
  };

  const onContextMenu = (e: MouseEvent) => e.preventDefault();

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("contextmenu", onContextMenu);

  // 7. 渲染主循环
  let animId: number;
  const uniformData = new Float32Array(20);

  function frame() {
    rebuildSamplerAndBindGroup();

    if (depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;

    const eyeX = camera.panX + camera.distance * Math.cos(radPhi) * Math.sin(radTheta);
    const eyeY = camera.panY + camera.distance * Math.sin(radPhi);
    const eyeZ = 2.0 + camera.distance * Math.cos(radPhi) * Math.cos(radTheta);

    const viewMatrix = createLookAtMatrix([eyeX, eyeY, eyeZ], [camera.panX, camera.panY, -8.0], [0, 1, 0]);
    const aspect = (canvas.width || 800) / (canvas.height || 600);
    const projMatrix = Mat4.perspective((camera.fov * Math.PI) / 180, aspect, 0.1, 100);
    const mvpMatrix = Mat4.multiply(projMatrix, viewMatrix);

    uniformData.set(mvpMatrix, 0);
    uniformData[16] = config.lodBias; // 传入 lodBias
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

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
    pass.setBindGroup(0, bindGroup!);
    pass.setVertexBuffer(0, vBuffer);
    pass.setIndexBuffer(iBuffer, "uint16");
    pass.drawIndexed(6);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  frame();

  // 8. 资源清理
  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);
    canvas.removeEventListener("contextmenu", onContextMenu);
    vBuffer.destroy();
    iBuffer.destroy();
    uniformBuffer.destroy();
    depthTexture.destroy();
    texture.destroy();
  };
}