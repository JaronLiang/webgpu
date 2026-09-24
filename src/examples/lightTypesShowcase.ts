import type { SimpleGUI } from "../utils/gui";

// ==========================================
// 1. 基础 3D 矩阵数学 (免第三方依赖)
// ==========================================
namespace Math3D {
  export function lookAt(eye: number[], center: number[], up: number[]): Float32Array {
    const z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
    const lenZ = 1 / (Math.hypot(z0, z1, z2) || 1);
    const zx = z0 * lenZ, zy = z1 * lenZ, zz = z2 * lenZ;

    const x0 = up[1] * zz - up[2] * zy, x1 = up[2] * zx - up[0] * zz, x2 = up[0] * zy - up[1] * zx;
    const lenX = 1 / (Math.hypot(x0, x1, x2) || 1);
    const xx = x0 * lenX, xy = x1 * lenX, xz = x2 * lenX;

    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;

    const out = new Float32Array(16);
    out[0] = xx;  out[1] = yx;  out[2] = zx;  out[3] = 0;
    out[4] = xy;  out[5] = yy;  out[6] = zy;  out[7] = 0;
    out[8] = xz;  out[9] = yz;  out[10] = zz; out[11] = 0;
    out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    out[15] = 1;
    return out;
  }

  export function perspective(fovRad: number, aspect: number, near: number, far: number): Float32Array {
    const f = 1.0 / Math.tan(fovRad / 2.0);
    const nf = 1 / (near - far);
    const out = new Float32Array(16);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = far * nf;
    out[11] = -1.0;
    out[14] = near * far * nf;
    return out;
  }

  export function ortho(left: number, right: number, bottom: number, top: number, near: number, far: number): Float32Array {
    const lr = 1 / (left - right);
    const bt = 1 / (bottom - top);
    const nf = 1 / (near - far);
    const out = new Float32Array(16);
    out[0] = -2 * lr;
    out[5] = -2 * bt;
    out[10] = nf;
    out[12] = (left + right) * lr;
    out[13] = (top + bottom) * bt;
    out[14] = near * nf;
    out[15] = 1;
    return out;
  }

  export function multiply(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      const ai0 = a[i], ai1 = a[i + 4], ai2 = a[i + 8], ai3 = a[i + 12];
      out[i]      = ai0 * b[0]  + ai1 * b[1]  + ai2 * b[2]  + ai3 * b[3];
      out[i + 4]  = ai0 * b[4]  + ai1 * b[5]  + ai2 * b[6]  + ai3 * b[7];
      out[i + 8]  = ai0 * b[8]  + ai1 * b[9]  + ai2 * b[10] + ai3 * b[11];
      out[i + 12] = ai0 * b[12] + ai1 * b[13] + ai2 * b[14] + ai3 * b[15];
    }
    return out;
  }
}

// ==========================================
// 2. 主场景入口函数
// ==========================================
export function runLightTypesShowcase(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: SimpleGUI
) {
  // ------------------------------------------
  // 场景几何体: 地板 + 立方体 + 两根投射柱 (Pos:3, Normal:3, Color:3)
  // ------------------------------------------
  const vertices: number[] = [];

  function pushQuad(
    p1: number[], p2: number[], p3: number[], p4: number[],
    n: number[], col: number[]
  ) {
    vertices.push(
      ...p1, ...n, ...col,
      ...p2, ...n, ...col,
      ...p3, ...n, ...col,
      ...p1, ...n, ...col,
      ...p3, ...n, ...col,
      ...p4, ...n, ...col
    );
  }

  // A. 大面积地板 (接收软/硬投影)
  const floorHalf = 12.0;
  const floorCol = [0.85, 0.85, 0.88];
  pushQuad(
    [-floorHalf, 0,  floorHalf],
    [ floorHalf, 0,  floorHalf],
    [ floorHalf, 0, -floorHalf],
    [-floorHalf, 0, -floorHalf],
    [0, 1, 0],
    floorCol
  );

  // B. 生成立方体工具
  function addBox(cx: number, cy: number, cz: number, sx: number, sy: number, sz: number, col: number[]) {
    const x0 = cx - sx / 2, x1 = cx + sx / 2;
    const y0 = cy - sy / 2, y1 = cy + sy / 2;
    const z0 = cz - sz / 2, z1 = cz + sz / 2;

    pushQuad([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], [ 0,  1,  0], col); // 上
    pushQuad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], [ 0, -1,  0], col); // 下
    pushQuad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [ 0,  0,  1], col); // 前
    pushQuad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [ 0,  0, -1], col); // 后
    pushQuad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [-1,  0,  0], col); // 左
    pushQuad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [ 1,  0,  0], col); // 右
  }

  // 摆放几何体
  addBox(0.0, 0.75, 0.0, 1.5, 1.5, 1.5, [0.95, 0.35, 0.2]); // 红色主立方体
  addBox(-2.5, 1.25, -1.0, 0.4, 2.5, 0.4, [0.2, 0.7, 0.9]); // 蓝色细立柱
  addBox(2.2, 0.8, 1.2, 0.6, 1.6, 0.6, [0.3, 0.85, 0.4]);  // 绿色小方柱

  const vBuffer = device.createBuffer({
    size: vertices.length * 4,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vBuffer, 0, new Float32Array(vertices));
  const vertexCount = vertices.length / 9;

  // ------------------------------------------
  // 3. 阴影贴图配置
  // ------------------------------------------
  const shadowMapSize = 2048;
  const shadowDepthTexture = device.createTexture({
    size: [shadowMapSize, shadowMapSize],
    format: "depth32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  const shadowSampler = device.createSampler({
    compare: "less",
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  // ------------------------------------------
  // 4. Uniform Buffers 与相机/光源状态
  // ------------------------------------------
  const sceneState = {
    lightType: 2, // 默认面光源
    intensity: 1.5,
    posX: 3.5,
    posY: 6.0,
    posZ: 3.0,
    lightSize: 1.5,
    autoRotate: 1, // 1: 开启, 0: 关闭
  };

  const camera = { distance: 11.0, phi: 30, theta: 45, panX: 0, panY: 0, fov: 50 };

  const uniformBuffer = device.createBuffer({
    size: (16 + 16 + 4 + 4 + 4) * 4, // 176 字节
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ------------------------------------------
  // 5. WGSL 着色器实现
  // ------------------------------------------
  const shaderWGSL = `
    struct SceneData {
      lightViewProj: mat4x4f,
      camViewProj: mat4x4f,
      lightPos_Type: vec4f,   // xyz: 光源坐标, w: 类型 (0:平行, 1:点, 2:面, 3:体)
      camPos_Size: vec4f,     // xyz: 相机坐标, w: 半影尺寸
      params: vec4f,          // x: 强度
    };

    @group(0) @binding(0) var<uniform> u: SceneData;
    @group(0) @binding(1) var shadowMap: texture_depth_2d;
    @group(0) @binding(2) var shadowSampler: sampler_comparison;

    // --- Shadow Pass ---
    @vertex
    fn vs_shadow(@location(0) pos: vec3f) -> @builtin(position) vec4f {
      return u.lightViewProj * vec4f(pos, 1.0);
    }

    // --- Main Pass ---
    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) color: vec3f,
      @location(3) shadowPos: vec4f,
    };

    @vertex
    fn vs_main(
      @location(0) pos: vec3f,
      @location(1) normal: vec3f,
      @location(2) color: vec3f
    ) -> VertexOut {
      var out: VertexOut;
      out.worldPos = pos;
      out.normal = normal;
      out.color = color;
      out.shadowPos = u.lightViewProj * vec4f(pos, 1.0);
      out.pos = u.camViewProj * vec4f(pos, 1.0);
      return out;
    }

    // 16 采样点泊松圆盘 (用于多采样柔和滤波)
    const POISSON_DISK: array<vec2f, 16> = array<vec2f, 16>(
      vec2f(-0.94201624, -0.39906216), vec2f(0.94558609, -0.76890725),
      vec2f(-0.094184101, -0.92938870), vec2f(0.34495938, 0.29387760),
      vec2f(-0.91588581, 0.45771432), vec2f(-0.81544232, -0.87912464),
      vec2f(-0.38277543, 0.27676845), vec2f(0.97484398, 0.75648379),
      vec2f(0.44323325, -0.97511554), vec2f(0.53742981, -0.47373420),
      vec2f(-0.26496911, -0.41893023), vec2f(0.79197514, 0.19090188),
      vec2f(-0.24188840, 0.99706507), vec2f(-0.81409955, 0.91437590),
      vec2f(0.19984126, 0.78641367), vec2f(0.14383161, -0.14100790)
    );

    // 解决非统一流错误：始终无分支调用采样器，使用 select 处理边界
    fn sampleShadow(shadowPos: vec4f, shadowRadius: f32) -> f32 {
      let proj = shadowPos.xyz / shadowPos.w;
      let uv = proj.xy * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
      let currentDepth = proj.z;

      let inBounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 && currentDepth <= 1.0;
      let bias = 0.0035;
      let texelSize = vec2f(1.0 / 2048.0);

      var shadow = 0.0;
      // 保持统一控制流：无条件执行循环与采样
      for (var i = 0; i < 16; i++) {
        let offset = POISSON_DISK[i] * texelSize * shadowRadius;
        // 使用 textureSampleCompareLevel 显式在 0 级采样，避免 mip 导数问题
        shadow += textureSampleCompareLevel(shadowMap, shadowSampler, uv + offset, currentDepth - bias);
      }
      let rawVis = shadow / 16.0;
      return select(1.0, rawVis, inBounds);
    }

    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4f {
      let lType = i32(round(u.lightPos_Type.w));
      let lPos = u.lightPos_Type.xyz;
      let lSize = u.camPos_Size.w;
      let intensity = u.params.x;

      let N = normalize(in.normal);
      let V = normalize(u.camPos_Size.xyz - in.worldPos);

      var L: vec3f;
      var attenuation = 1.0;
      var shadowFilterRadius = 1.0;
      var roughness = 0.3;

      let distToLight = length(lPos - in.worldPos);

      // 遵循严格的 WGSL switch 语法规则
      switch (lType) {
        case 0: { // 平行光 (Directional)
          L = normalize(lPos);
          attenuation = 1.0;
          shadowFilterRadius = 1.0; // 锐利边缘
          roughness = 0.25;
        }
        case 1: { // 点光源 (Point)
          L = normalize(lPos - in.worldPos);
          attenuation = 1.0 / (1.0 + 0.1 * distToLight + 0.05 * distToLight * distToLight);
          shadowFilterRadius = 1.5;
          roughness = 0.3;
        }
        case 2: { // 面光源 (Area)
          L = normalize(lPos - in.worldPos);
          attenuation = 1.0 / (1.0 + 0.08 * distToLight + 0.03 * distToLight * distToLight);
          shadowFilterRadius = max(lSize * 3.5, 1.0); // 宽半影软阴影
          roughness = 0.55;
        }
        case 3: { // 体光源 (Volumetric / Sphere)
          let lightVec = lPos - in.worldPos;
          let centerDist = length(lightVec);
          L = lightVec / centerDist;
          attenuation = 1.0 / (1.0 + 0.05 * centerDist + 0.02 * (centerDist * centerDist));
          shadowFilterRadius = max(lSize * 4.8, 1.5); // 超大扩散软半影
          roughness = 0.45;
        }
        default: {
          L = normalize(lPos - in.worldPos);
          attenuation = 1.0;
          shadowFilterRadius = 1.0;
        }
      }

      // 漫反射 (Diffuse)
      let nDotL = max(dot(N, L), 0.0);
      let diffuse = nDotL * in.color * intensity * attenuation;

      // 镜面高光 (Specular - Blinn Phong)
      let H = normalize(L + V);
      let specPower = (1.0 - roughness) * 128.0;
      let specular = pow(max(dot(N, H), 0.0), specPower) * 0.4 * intensity * attenuation;

      // 阴影遮罩
      let shadowVis = sampleShadow(in.shadowPos, shadowFilterRadius);

      // 环境光
      let ambient = vec3f(0.12, 0.13, 0.16) * in.color;

      // 最终合成与 Gamma 校正
      let finalColor = ambient + (diffuse + specular) * shadowVis;
      return vec4f(pow(finalColor, vec3f(1.0 / 2.2)), 1.0);
    }
  `;

  const shaderModule = device.createShaderModule({ code: shaderWGSL });

  // ------------------------------------------
  // 6. 管线构建
  // ------------------------------------------
  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 9 * 4,
    attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" },  // pos
      { shaderLocation: 1, offset: 12, format: "float32x3" }, // normal
      { shaderLocation: 2, offset: 24, format: "float32x3" }, // color
    ],
  };

  const shadowPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shaderModule,
      entryPoint: "vs_shadow",
      buffers: [vertexBufferLayout],
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth32float",
    },
    primitive: { topology: "triangle-list", cullMode: "front" },
  });

  const mainPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [vertexBufferLayout],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth24plus",
    },
    primitive: { topology: "triangle-list", cullMode: "back" },
  });

  const mainBindGroup = device.createBindGroup({
    layout: mainPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: shadowDepthTexture.createView() },
      { binding: 2, resource: shadowSampler },
    ],
  });

const shadowBindGroup = device.createBindGroup({
    layout: shadowPipeline.getBindGroupLayout(0),
    entries: [
      // 阴影 Pass (vs_shadow) 中只用到了 Uniform，所以只绑定 0 即可
      { binding: 0, resource: { buffer: uniformBuffer } },
    ],
  });

  // ------------------------------------------
  // 7. GUI 控制（严格兼容数字滑块与按钮）
  // ------------------------------------------
  gui.add(sceneState, "lightType", 0, 3, 1).name("光源类型 (0-3)");
  gui.add(sceneState, "intensity", 0.1, 4.0, 0.1).name("光照强度");
  gui.add(sceneState, "lightSize", 0.2, 3.5, 0.1).name("光源尺寸/半影");
  gui.add(sceneState, "posY", 2.0, 12.0, 0.2).name("光源高度 (Y)");
  gui.add(sceneState, "autoRotate", 0, 1, 1).name("自动旋转 (1/0)");

  // 便捷切换按钮
  gui.addButton("切为: 平行光", () => { sceneState.lightType = 0; gui.updateDisplay?.(); });
  gui.addButton("切为: 点光源", () => { sceneState.lightType = 1; gui.updateDisplay?.(); });
  gui.addButton("切为: 面光源", () => { sceneState.lightType = 2; gui.updateDisplay?.(); });
  gui.addButton("切为: 体光源", () => { sceneState.lightType = 3; gui.updateDisplay?.(); });

  gui.addTextInfo(
    "<b>光源类型模式说明:</b><br>" +
    "• <b>0: 平行光</b> - 恒定强度、平行射入、硬边阴影<br>" +
    "• <b>1: 点光源</b> - 距离平方衰减、锐利阴影<br>" +
    "• <b>2: 面光源</b> - 宽半影、柔和软阴影<br>" +
    "• <b>3: 体光源</b> - 球体发光、渐进超柔软阴影"
  );

  // ------------------------------------------
  // 8. 鼠标与触控相机控制
  // ------------------------------------------
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
      camera.phi = Math.max(-85, Math.min(85, camera.phi + dy * 0.4));
    } else if (dragButton === 2) {
      const f = camera.distance * 0.0015;
      camera.panX -= dx * f;
      camera.panY += dy * f;
    }
  };

  const onPointerUp = (e: PointerEvent) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  };

  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    camera.distance = Math.max(3.0, Math.min(25.0, camera.distance + e.deltaY * 0.008));
  };

  const onContextMenu = (e: MouseEvent) => e.preventDefault();

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("contextmenu", onContextMenu);

  // ------------------------------------------
  // 9. 渲染主循环
  // ------------------------------------------
  let animId: number;
  let depthTexture: GPUTexture | null = null;
  const uniformData = new Float32Array(44);

  function frame(timestamp: number) {
    if (!depthTexture || depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width || 800, canvas.height || 600],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    // 光源旋转运动
    if (sceneState.autoRotate === 1) {
      const angle = timestamp * 0.0008;
      const radius = 5.0;
      sceneState.posX = Math.cos(angle) * radius;
      sceneState.posZ = Math.sin(angle) * radius;
    }

    // 1. 光源视图与投影矩阵
    const lPos = [sceneState.posX, sceneState.posY, sceneState.posZ];
    const lightView = Math3D.lookAt(lPos, [0, 0, 0], [0, 1, 0]);
    let lightProj: Float32Array;

    if (sceneState.lightType === 0) {
      lightProj = Math3D.ortho(-8, 8, -8, 8, 0.5, 25.0);
    } else {
      lightProj = Math3D.perspective((80 * Math.PI) / 180, 1.0, 0.5, 25.0);
    }
    const lightViewProj = Math3D.multiply(lightProj, lightView);

    // 2. 主相机视图与投影矩阵
    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.panX + camera.distance * Math.cos(radPhi) * Math.sin(radTheta),
      camera.panY + camera.distance * Math.sin(radPhi),
      camera.distance * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const camView = Math3D.lookAt(eye, [camera.panX, camera.panY, 0], [0, 1, 0]);
    const aspect = (canvas.width || 800) / (canvas.height || 600);
    const camProj = Math3D.perspective((camera.fov * Math.PI) / 180, aspect, 0.1, 100.0);
    const camViewProj = Math3D.multiply(camProj, camView);

    // 3. 上传 Uniform 数据
    uniformData.set(lightViewProj, 0);
    uniformData.set(camViewProj, 16);

    uniformData[32] = lPos[0];
    uniformData[33] = lPos[1];
    uniformData[34] = lPos[2];
    uniformData[35] = sceneState.lightType;

    uniformData[36] = eye[0];
    uniformData[37] = eye[1];
    uniformData[38] = eye[2];
    uniformData[39] = sceneState.lightSize;

    uniformData[40] = sceneState.intensity;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();

    // Pass 1: 渲染阴影深度
    const shadowPass = encoder.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: shadowDepthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    shadowPass.setPipeline(shadowPipeline);
    shadowPass.setBindGroup(0, shadowBindGroup);
    shadowPass.setVertexBuffer(0, vBuffer);
    shadowPass.draw(vertexCount);
    shadowPass.end();

    // Pass 2: 渲染主相机画面
    const mainPass = encoder.beginRenderPass({
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
    mainPass.setPipeline(mainPipeline);
    mainPass.setBindGroup(0, mainBindGroup);
    mainPass.setVertexBuffer(0, vBuffer);
    mainPass.draw(vertexCount);
    mainPass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  // ------------------------------------------
  // 10. 清理销毁
  // ------------------------------------------
  return () => {
    cancelAnimationFrame(animId);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);
    canvas.removeEventListener("contextmenu", onContextMenu);

    vBuffer.destroy();
    uniformBuffer.destroy();
    shadowDepthTexture.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}