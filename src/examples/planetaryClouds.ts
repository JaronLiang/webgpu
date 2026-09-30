// src/examples/planetaryClouds.ts
import GUI from "lil-gui";

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

function mat4Invert(m: Float32Array): Float32Array {
  const out = new Float32Array(16);
  const b00 = m[0] * m[5] - m[1] * m[4], b01 = m[0] * m[6] - m[2] * m[4];
  const b02 = m[0] * m[7] - m[3] * m[4], b03 = m[1] * m[6] - m[2] * m[5];
  const b04 = m[1] * m[7] - m[3] * m[5], b05 = m[2] * m[7] - m[3] * m[6];
  const b06 = m[8] * m[13] - m[9] * m[12], b07 = m[8] * m[14] - m[10] * m[12];
  const b08 = m[8] * m[15] - m[11] * m[12], b09 = m[9] * m[14] - m[10] * m[13];
  const b10 = m[9] * m[15] - m[11] * m[13], b11 = m[10] * m[15] - m[11] * m[14];
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return out;
  const inv = 1.0 / det;
  out[0] = (m[5]*b11 - m[6]*b10 + m[7]*b09)*inv;
  out[1] = (-m[1]*b11 + m[2]*b10 - m[3]*b09)*inv;
  out[2] = (m[13]*b05 - m[14]*b04 + m[15]*b03)*inv;
  out[3] = (-m[9]*b05 + m[10]*b04 - m[11]*b03)*inv;
  out[4] = (-m[4]*b11 + m[6]*b08 - m[7]*b07)*inv;
  out[5] = (m[0]*b11 - m[2]*b08 + m[3]*b07)*inv;
  out[6] = (-m[12]*b05 + m[14]*b02 - m[15]*b01)*inv;
  out[7] = (m[8]*b05 - m[10]*b02 + m[11]*b01)*inv;
  out[8] = (m[4]*b10 - m[5]*b08 + m[7]*b06)*inv;
  out[9] = (-m[0]*b10 + m[1]*b08 - m[3]*b06)*inv;
  out[10] = (m[12]*b04 - m[13]*b02 + m[15]*b00)*inv;
  out[11] = (-m[8]*b04 + m[9]*b02 - m[11]*b00)*inv;
  out[12] = (-m[4]*b09 + m[5]*b07 - m[6]*b06)*inv;
  out[13] = (m[0]*b09 - m[1]*b07 + m[2]*b06)*inv;
  out[14] = (-m[12]*b03 + m[13]*b01 - m[14]*b00)*inv;
  out[15] = (m[8]*b03 - m[9]*b01 + m[10]*b00)*inv;
  return out;
}

export function runPlanetaryClouds(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: any
) {
  // 全屏 Quad
  const quadData = new Float32Array([-1,-1, 1,-1, -1,1, 1,-1, 1,1, -1,1]);
  const quadBuffer = device.createBuffer({
    size: quadData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(quadBuffer, 0, quadData);

  const uniformBuffer = device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const cpuUniformData = new Float32Array(64);

  // 强化后的体积云 WGSL Shader
  const cloudsShaderWGSL = `
    struct Uniforms {
      invViewProj: mat4x4f,
      camPos: vec4f,        // xyz: camPos, w: time
      sunDir: vec4f,        // xyz: sunDir, w: coverage
      cloudParams: vec4f,   // x: cloudBottom, y: cloudTop, z: density, w: absorption
      renderParams: vec4f,  // x: maxDistance, y: stepCount, z: detailScale, w: eccentricity
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexOutput {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs_main(@location(0) pos: vec2f) -> VertexOutput {
      var out: VertexOutput;
      out.pos = vec4f(pos, 0.0, 1.0);
      out.uv = pos * 0.5 + 0.5;
      return out;
    }

    // 屏幕空间随机抖动 (Interleaved Gradient Noise) 彻底消除步进同心圆条纹
    fn interleavedGradientNoise(pixel: vec2f) -> f32 {
      return fract(52.9829189 * fract(dot(pixel, vec2f(0.06711056, 0.00583715))));
    }

    // 快速高质量 3D 值噪声
    fn hash33(p: vec3f) -> vec3f {
      var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
      p3 += dot(p3, p3.yxz + 33.33);
      return fract((p3.xxy + p3.yxx) * p3.zyx);
    }

    fn noise(p: vec3f) -> f32 {
      let i = floor(p);
      let f = fract(p);
      let u = f * f * (3.0 - 2.0 * f);

      let n000 = dot(hash33(i + vec3f(0.0, 0.0, 0.0)) - 0.5, f - vec3f(0.0, 0.0, 0.0));
      let n100 = dot(hash33(i + vec3f(1.0, 0.0, 0.0)) - 0.5, f - vec3f(1.0, 0.0, 0.0));
      let n010 = dot(hash33(i + vec3f(0.0, 1.0, 0.0)) - 0.5, f - vec3f(0.0, 1.0, 0.0));
      let n110 = dot(hash33(i + vec3f(1.0, 1.0, 0.0)) - 0.5, f - vec3f(1.0, 1.0, 0.0));
      let n001 = dot(hash33(i + vec3f(0.0, 0.0, 1.0)) - 0.5, f - vec3f(0.0, 0.0, 1.0));
      let n101 = dot(hash33(i + vec3f(1.0, 0.0, 1.0)) - 0.5, f - vec3f(1.0, 0.0, 1.0));
      let n011 = dot(hash33(i + vec3f(0.0, 1.0, 1.0)) - 0.5, f - vec3f(0.0, 1.0, 1.0));
      let n111 = dot(hash33(i + vec3f(1.0, 1.0, 1.0)) - 0.5, f - vec3f(1.0, 1.0, 1.0));

      return 0.5 + mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z
      );
    }

    fn fbm(p: vec3f) -> f32 {
      var v = 0.0;
      var a = 0.5;
      var pos = p;
      for (var i = 0; i < 4; i++) {
        v += a * noise(pos);
        pos = pos * 2.12 + vec3f(1.3, 3.7, 5.1);
        a *= 0.5;
      }
      return v;
    }

    // 密度采样器：包含高度剖面、风力漂移、边缘侵蚀
    fn sampleCloudDensity(p: vec3f) -> f32 {
      let b = u.cloudParams.x;
      let t = u.cloudParams.y;
      if (p.y < b || p.y > t) {
        return 0.0;
      }

      let hFraction = (p.y - b) / (t - b);
      // 云底部平展，云层内部饱满，顶部羽化
      let heightProfile = smoothstep(0.0, 0.2, hFraction) * smoothstep(1.0, 0.7, hFraction);

      let time = u.camPos.w;
      let wind = vec3f(time * 0.12, 0.0, time * 0.06);
      let coord = (p * 0.0003) + wind;

      let baseNoise = fbm(coord);
      let coverage = u.sunDir.w;
      
      // 阈值过渡更柔和
      var d = smoothstep(1.0 - coverage, 1.0 - coverage + 0.35, baseNoise) * heightProfile;

      if (d <= 0.001) { return 0.0; }

      // 高频卷曲边缘侵蚀
      let detail = fbm(coord * 3.2 - wind * 0.5);
      d = clamp(d - detail * 0.2, 0.0, 1.0);

      return d * u.cloudParams.z * 0.8;
    }

    // 双向 Henyey-Greenstein 相位函数（前向强银边 + 后向微散射）
    fn dualHenyeyGreenstein(cosTheta: f32, g1: f32, g2: f32, blend: f32) -> f32 {
      let p1 = (1.0 - g1 * g1) / (4.0 * 3.14159 * pow(1.0 + g1 * g1 - 2.0 * g1 * cosTheta, 1.5));
      let p2 = (1.0 - g2 * g2) / (4.0 * 3.14159 * pow(1.0 + g2 * g2 - 2.0 * g2 * cosTheta, 1.5));
      return mix(p1, p2, blend);
    }

    // 太阳光透射采样（加入 Powder Sugar 糖粉效应）
    fn sampleLight(pos: vec3f, sunDir: vec3f) -> vec2f {
      var p = pos;
      let stepSize = 140.0;
      var opticalDensity = 0.0;
      for (var i = 0; i < 4; i++) {
        p += sunDir * stepSize;
        opticalDensity += sampleCloudDensity(p) * stepSize;
      }
      let beers = exp(-opticalDensity * u.cloudParams.w);
      // 糖粉效应：防止云暗部死黑，呈现蓬松的微多重散射
      let powder = 1.0 - exp(-opticalDensity * u.cloudParams.w * 2.0);
      return vec2f(beers, powder);
    }

    // 电影级 ACES Tone Mapping 解决刺眼白斑
    fn tonemapACES(x: vec3f) -> vec3f {
      let a = 2.51;
      let b = 0.03;
      let c = 2.43;
      let d = 0.59;
      let e = 0.14;
      return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let ndc = vec4f(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0, 1.0, 1.0);
      let unproj = u.invViewProj * ndc;
      let rd = normalize(unproj.xyz / unproj.w - u.camPos.xyz);
      let ro = u.camPos.xyz;
      let sunDir = normalize(u.sunDir.xyz);

      // 大气物理天穹颜色（随太阳高度角动态变化）
      let horizon = clamp(rd.y, 0.0, 1.0);
      let sunDot = max(dot(rd, sunDir), 0.0);
      let sunElevation = sunDir.y;

      let skyDeepBlue = vec3f(0.12, 0.28, 0.58);
      let skyHorizonBlue = vec3f(0.48, 0.65, 0.88);
      let sunsetOrange = vec3f(0.95, 0.45, 0.15);

      var horizonColor = mix(sunsetOrange, skyHorizonBlue, clamp(sunElevation * 3.0, 0.0, 1.0));
      var skyColor = mix(horizonColor, skyDeepBlue, pow(horizon, 0.4));
      // 真实太阳圆盘光晕
      skyColor += vec3f(1.0, 0.9, 0.7) * pow(sunDot, 180.0) * 1.5;

      let cloudBottom = u.cloudParams.x;
      let cloudTop = u.cloudParams.y;

      // 平面层光线求交（同时支持从上方俯瞰、从下方仰视、从云层内部穿行）
      var tMin = 0.0;
      var tMax = u.renderParams.x;

      if (abs(rd.y) > 1e-5) {
        let t1 = (cloudBottom - ro.y) / rd.y;
        let t2 = (cloudTop - ro.y) / rd.y;
        let nearP = min(t1, t2);
        let farP = max(t1, t2);
        tMin = max(0.0, nearP);
        tMax = min(tMax, farP);
      } else {
        if (ro.y < cloudBottom || ro.y > cloudTop) {
          return vec4f(tonemapACES(skyColor), 1.0);
        }
      }

      if (tMin >= tMax || tMax <= 0.0) {
        return vec4f(tonemapACES(skyColor), 1.0);
      }

      // 步进初始化与像素抖动去噪
      let stepCount = i32(u.renderParams.y);
      let dt = (tMax - tMin) / f32(stepCount);
      let dither = interleavedGradientNoise(in.pos.xy);
      var t = tMin + dt * dither; // 随机化起始步长消除分层

      var accumulatedColor = vec3f(0.0);
      var transmittance = 1.0;

      let cosTheta = dot(rd, sunDir);
      // 前向 0.75 + 后向 -0.25 双峰散射
      let phase = dualHenyeyGreenstein(cosTheta, 0.72, -0.22, 0.25);
      let sunLightColor = vec3f(1.8, 1.6, 1.4) * max(sunElevation * 1.5, 0.2);

      for (var i = 0; i < 64; i++) {
        if (i >= stepCount || transmittance < 0.015) { break; }

        let samplePos = ro + rd * (t + dt * 0.5);
        let density = sampleCloudDensity(samplePos);

        if (density > 0.001) {
          let lightSample = sampleLight(samplePos, sunDir);
          let beers = lightSample.x;
          let powder = lightSample.y;

          // 结合高度梯度的立体漫反射环境光
          let hFrac = clamp((samplePos.y - cloudBottom) / (cloudTop - cloudBottom), 0.0, 1.0);
          let skyAmbient = mix(vec3f(0.2, 0.25, 0.35), vec3f(0.85, 0.92, 1.05), hFrac);

          // 糖粉效应增强明暗反差但避免死黑
          let directLight = sunLightColor * beers * max(powder * 2.0, 0.15) * phase;
          let totalLight = directLight + skyAmbient * 0.5;

          let sampleAtten = exp(-density * u.cloudParams.w * dt);

          // 能量守恒累加
          accumulatedColor += transmittance * (1.0 - sampleAtten) * totalLight;
          transmittance *= sampleAtten;
        }

        t += dt;
      }

      let finalLinear = accumulatedColor + skyColor * transmittance;
      
      // ACES 色调映射 + 伽马矫正 (sRGB)
      let tonemapped = tonemapACES(finalLinear);
      let finalSrgb = pow(tonemapped, vec3f(1.0 / 2.2));

      return vec4f(finalSrgb, 1.0);
    }
  `;

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: cloudsShaderWGSL }),
      entryPoint: "vs_main",
      buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }]
    },
    fragment: {
      module: device.createShaderModule({ code: cloudsShaderWGSL }),
      entryPoint: "fs_main",
      targets: [{ format }]
    },
    primitive: { topology: "triangle-list" }
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
  });

  // ========== 增强型轨道漫游相机状态 ==========
  const camera = {
    target: [0, 2000, 0], // 旋转观察中心 (设在云层中部附近)
    radius: 7500.0,       // 距离中心距离（方便纵观全局）
    theta: 45.0,          // 水平偏航角
    phi: 28.0,            // 俯仰角 (正值可俯瞰云海)
  };

  const cloudSettings = {
    coverage: 0.52,
    density: 1.6,
    cloudBottom: 1200.0,
    cloudThickness: 2400.0,
    windSpeed: 1.2,
    sunElevation: 28.0,
    sunAzimuth: 130.0,
    stepCount: 42,
    autoRotate: false,
    presetView: "俯瞰云海",
  };

  // GUI 配置
  gui.title("全球行星级体积云系统 (优化版)");

  const fView = gui.addFolder("视角预设");
  fView.add(cloudSettings, "presetView", ["俯瞰云海", "地面仰望", "云层穿行"]).name("预设视角").onChange((v: string) => {
    if (v === "俯瞰云海") {
      camera.target = [0, 2200, 0];
      camera.radius = 8000;
      camera.phi = 35;
    } else if (v === "地面仰望") {
      camera.target = [0, 1600, 0];
      camera.radius = 2500;
      camera.phi = -20;
    } else if (v === "云层穿行") {
      camera.target = [0, 2400, 0];
      camera.radius = 600;
      camera.phi = 5;
    }
  });
  fView.add(cloudSettings, "autoRotate").name("视角自动环绕");

  const fClouds = gui.addFolder("云层形貌");
  fClouds.add(cloudSettings, "coverage", 0.1, 0.9, 0.01).name("云层覆盖率");
  fClouds.add(cloudSettings, "density", 0.5, 4.0, 0.1).name("云雾浓度");
  fClouds.add(cloudSettings, "cloudBottom", 500.0, 4000.0, 100.0).name("云底高度(m)");
  fClouds.add(cloudSettings, "cloudThickness", 800.0, 5000.0, 100.0).name("云层厚度(m)");
  fClouds.add(cloudSettings, "windSpeed", 0.0, 5.0, 0.1).name("风场流速");

  const fLight = gui.addFolder("光照与步进");
  fLight.add(cloudSettings, "sunElevation", -5.0, 85.0, 1.0).name("太阳高度角");
  fLight.add(cloudSettings, "sunAzimuth", 0.0, 360.0, 1.0).name("太阳方位角");
  fLight.add(cloudSettings, "stepCount", 24, 64, 1).name("步进采样数");

  // ========== 交互事件监听：旋转、缩放、平移 ==========
  let isDragging = false;
  let dragButton = 0; // 0: 左键(旋转), 2: 右键(平移)
  let lastX = 0, lastY = 0;

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  canvas.addEventListener("pointerdown", (e) => {
    isDragging = true;
    dragButton = e.button;
    if (e.shiftKey) dragButton = 2; // Shift + 左键也作为平移
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
      // 左键：旋转视角
      camera.theta -= dx * 0.3;
      camera.phi = Math.max(-88, Math.min(88, camera.phi + dy * 0.3));
    } else if (dragButton === 2) {
      // 右键 / Shift+左键：相机平移
      const radTheta = (camera.theta * Math.PI) / 180;
      // 沿相机水平右侧与前侧移动观察中心
      const rightX = Math.cos(radTheta);
      const rightZ = -Math.sin(radTheta);
      const panSpeed = camera.radius * 0.0012;

      camera.target[0] -= rightX * dx * panSpeed;
      camera.target[2] -= rightZ * dx * panSpeed;
      camera.target[1] += dy * panSpeed * 1.5;
    }
  });

  canvas.addEventListener("pointerup", (e) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  });

  // 滚轮缩放：根据距离自适应缩放速率
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomFactor = Math.exp(e.deltaY * 0.001);
    camera.radius = Math.max(100.0, Math.min(30000.0, camera.radius * zoomFactor));
  }, { passive: false });

  let animId: number;
  let timeSeconds = 0;

  function frame() {
    timeSeconds += 0.016 * cloudSettings.windSpeed;

    if (cloudSettings.autoRotate && !isDragging) {
      camera.theta += 0.12;
    }

    const aspect = (canvas.width || 800) / (canvas.height || 600);
    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;

    // 计算相机世界位置 Eye
    const eyeX = camera.target[0] + camera.radius * Math.cos(radPhi) * Math.sin(radTheta);
    const eyeY = camera.target[1] + camera.radius * Math.sin(radPhi);
    const eyeZ = camera.target[2] + camera.radius * Math.cos(radPhi) * Math.cos(radTheta);

    const view = createLookAtMatrix([eyeX, eyeY, eyeZ], camera.target, [0, 1, 0]);
    const proj = createPerspectiveMatrix((55 * Math.PI) / 180, aspect, 50.0, 50000.0);

    // 计算反投影矩阵
    const viewProj = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        viewProj[i * 4 + j] =
          proj[j] * view[i * 4] + proj[4 + j] * view[i * 4 + 1] + proj[8 + j] * view[i * 4 + 2] + proj[12 + j] * view[i * 4 + 3];
      }
    }
    const invViewProj = mat4Invert(viewProj);

    // 计算太阳方向
    const sunElRad = (cloudSettings.sunElevation * Math.PI) / 180;
    const sunAzRad = (cloudSettings.sunAzimuth * Math.PI) / 180;
    const sunDir = [
      Math.cos(sunElRad) * Math.sin(sunAzRad),
      Math.sin(sunElRad),
      Math.cos(sunElRad) * Math.cos(sunAzRad),
    ];

    const cloudTop = cloudSettings.cloudBottom + cloudSettings.cloudThickness;

    // 传输 Uniform 缓冲
    cpuUniformData.set(invViewProj, 0);
    cpuUniformData.set([eyeX, eyeY, eyeZ, timeSeconds], 16);
    cpuUniformData.set([sunDir[0], sunDir[1], sunDir[2], cloudSettings.coverage], 20);
    cpuUniformData.set([cloudSettings.cloudBottom, cloudTop, cloudSettings.density, 0.0035], 24);
    cpuUniformData.set([35000.0, cloudSettings.stepCount, 1.0, 0.72], 28);

    device.queue.writeBuffer(uniformBuffer, 0, cpuUniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0.1, g: 0.15, b: 0.25, a: 1.0 },
        storeOp: "store"
      }]
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, quadBuffer);
    pass.draw(6);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(frame);
  }

  frame();

  return () => {
    cancelAnimationFrame(animId);
    quadBuffer.destroy();
    uniformBuffer.destroy();
  };
}