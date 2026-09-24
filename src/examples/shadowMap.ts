// src/examples/dynamicShadowMap.ts
export function runDynamicShadowMap(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // ==========================================
  // 1. 构建 3D 场景几何体 (地面 + 盒子)
  // 包含: Position(3), Color(3), Normal(3) = 9个 Float32
  // ==========================================
  const verts: number[] = [];
  
  // 地面 (灰色, Y=0)
  const size = 3.0;
  const groundColor = [0.6, 0.6, 0.6];
  const groundNormal = [0, 1, 0];
  verts.push(
    -size, 0, -size, ...groundColor, ...groundNormal,
     size, 0, -size, ...groundColor, ...groundNormal,
     size, 0,  size, ...groundColor, ...groundNormal,
    -size, 0, -size, ...groundColor, ...groundNormal,
     size, 0,  size, ...groundColor, ...groundNormal,
    -size, 0,  size, ...groundColor, ...groundNormal,
  );

  // 盒子 (中心位于 0, 0.5, 0，边长为 1.0)
  const p = [
    [-0.5, 1.0, -0.5], [0.5, 1.0, -0.5], [0.5, 1.0, 0.5], [-0.5, 1.0, 0.5], // 顶部4点
    [-0.5, 0.0, -0.5], [0.5, 0.0, -0.5], [0.5, 0.0, 0.5], [-0.5, 0.0, 0.5]  // 底部4点
  ];
  const addFace = (p1: number[], p2: number[], p3: number[], p4: number[], normal: number[]) => {
    const color = [0.9, 0.4, 0.2]; // 橙红色盒子
    verts.push(
      ...p1, ...color, ...normal,  ...p2, ...color, ...normal,  ...p3, ...color, ...normal,
      ...p1, ...color, ...normal,  ...p3, ...color, ...normal,  ...p4, ...color, ...normal
    );
  };
  addFace(p[3], p[2], p[1], p[0], [ 0,  1,  0]); // 上
  addFace(p[4], p[5], p[6], p[7], [ 0, -1,  0]); // 下
  addFace(p[7], p[6], p[2], p[3], [ 0,  0,  1]); // 前
  addFace(p[5], p[4], p[0], p[1], [ 0,  0, -1]); // 后
  addFace(p[4], p[7], p[3], p[0], [-1,  0,  0]); // 左
  addFace(p[6], p[5], p[1], p[2], [ 1,  0,  0]); // 右

  const vertices = new Float32Array(verts);
  const vertexCount = vertices.length / 9;

  const vertexBuffer = device.createBuffer({
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertices);

  // ==========================================
  // 2. 创建纹理与 Uniform 数据
  // ==========================================
  const shadowMapSize = 1024;
  const shadowTexture = device.createTexture({
    size: [shadowMapSize, shadowMapSize],
    format: "depth32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  
  // 动态数据 Buffer: lightPos(4) + cameraPos(4) + aspect(4, 对齐)
  const uniformData = new Float32Array(12);
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const shadowSampler = device.createSampler({
    compare: "less", 
    magFilter: "linear", minFilter: "linear",
    addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
  });

  // ==========================================
  // 3. 编写强悍的 WGSL 着色器 (内嵌 3D 矩阵数学)
  // ==========================================
  const shaderCode = `
    struct Uniforms {
      lightPos: vec4f,
      cameraPos: vec4f,
      aspect: f32, // 用于相机透视投影
    };
    @group(0) @binding(0) var<uniform> uniforms: Uniforms;

    // --- 在 WGSL 中手写 3D 矩阵算法，免除 JS 端矩阵库依赖！ ---
    fn lookAt(eye: vec3f, center: vec3f, up: vec3f) -> mat4x4f {
      let f = normalize(center - eye); // 前向
      let s = normalize(cross(f, up)); // 右向
      let u = cross(s, f);             // 上向
      return mat4x4f(
        vec4f(s.x, u.x, -f.x, 0.0),
        vec4f(s.y, u.y, -f.y, 0.0),
        vec4f(s.z, u.z, -f.z, 0.0),
        vec4f(-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0)
      );
    }

    fn ortho(l: f32, r: f32, b: f32, t: f32, n: f32, f_far: f32) -> mat4x4f {
      return mat4x4f(
        vec4f(2.0 / (r - l), 0.0, 0.0, 0.0),
        vec4f(0.0, 2.0 / (t - b), 0.0, 0.0),
        vec4f(0.0, 0.0, -1.0 / (f_far - n), 0.0),
        vec4f(-(r + l)/(r - l), -(t + b)/(t - b), -n / (f_far - n), 1.0)
      );
    }

    fn perspective(fovY: f32, aspect: f32, near: f32, far: f32) -> mat4x4f {
      let f = 1.0 / tan(fovY / 2.0);
      return mat4x4f(
        vec4f(f / aspect, 0.0, 0.0, 0.0),
        vec4f(0.0, f, 0.0, 0.0),
        vec4f(0.0, 0.0, far / (near - far), -1.0),
        vec4f(0.0, 0.0, (near * far) / (near - far), 0.0)
      );
    }

    // ========== Pass 1: Shadow Map ==========
    @vertex
    fn shadow_vs(@location(0) pos: vec3f) -> @builtin(position) vec4f {
      // 从光源视角的正交投影
      let lightView = lookAt(uniforms.lightPos.xyz, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0));
      let lightProj = ortho(-4.0, 4.0, -4.0, 4.0, 0.1, 20.0);
      return (lightProj * lightView) * vec4f(pos, 1.0);
    }

    // ========== Pass 2: Main Camera ==========
    @group(0) @binding(1) var shadowMap: texture_depth_2d;
    @group(0) @binding(2) var shadowSampler: sampler_comparison;

    struct VertexOut {
      @builtin(position) pos: vec4f,
      @location(0) shadowPos: vec4f,
      @location(1) color: vec3f,
      @location(2) normal: vec3f,
      @location(3) worldPos: vec3f,
    };

    @vertex
    fn main_vs(
      @location(0) pos: vec3f, 
      @location(1) color: vec3f, 
      @location(2) normal: vec3f
    ) -> VertexOut {
      var out: VertexOut;
      
      // 相机矩阵
      let camView = lookAt(uniforms.cameraPos.xyz, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0));
      let camProj = perspective(radians(45.0), uniforms.aspect, 0.1, 100.0);
      out.pos = (camProj * camView) * vec4f(pos, 1.0);
      
      // 光源矩阵 (用于计算阴影坐标)
      let lightView = lookAt(uniforms.lightPos.xyz, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0));
      let lightProj = ortho(-4.0, 4.0, -4.0, 4.0, 0.1, 20.0);
      out.shadowPos = (lightProj * lightView) * vec4f(pos, 1.0);

      out.color = color;
      out.normal = normal;
      out.worldPos = pos;
      return out;
    }

    @fragment
    fn main_fs(in: VertexOut) -> @location(0) vec4f {
      // 1. 计算阴影可见度
      var shadowUV = in.shadowPos.xy * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
      let depthBias = 0.005; // 关键：解决 3D 阴影痤疮
      let rawVisibility = textureSampleCompare(shadowMap, shadowSampler, shadowUV, in.shadowPos.z - depthBias);
      let inBounds = shadowUV.x >= 0.0 && shadowUV.x <= 1.0 && shadowUV.y >= 0.0 && shadowUV.y <= 1.0;
      let visibility = select(1.0, rawVisibility, inBounds);

      // 2. 基础光照 (漫反射)
      let lightDir = normalize(uniforms.lightPos.xyz - in.worldPos);
      let diffuse = max(dot(normalize(in.normal), lightDir), 0.0);

      // 3. 最终混合
      let ambient = 0.2;
      let lighting = ambient + diffuse * 0.8 * visibility;
      
      return vec4f(in.color * lighting, 1.0);
    }
  `;
  const module = device.createShaderModule({ code: shaderCode });

  // ==========================================
  // 4. 定义管线和绑定组
  // ==========================================
  // Shadow Pass Layout (只用 Uniform)
  const shadowLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
  });

  // Main Pass Layout (Uniform + Texture + Sampler)
  const mainLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
    ],
  });

  const vertexBuffersConfig: GPUVertexBufferLayout[] = [{
    arrayStride: 36, // 9 个 Float32 * 4 字节
    attributes: [
      { shaderLocation: 0, offset: 0,  format: "float32x3" }, // Pos
      { shaderLocation: 1, offset: 12, format: "float32x3" }, // Color
      { shaderLocation: 2, offset: 24, format: "float32x3" }, // Normal
    ],
  }];

  const shadowPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [shadowLayout] }),
    vertex: { module, entryPoint: "shadow_vs", buffers: vertexBuffersConfig },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth32float" },
    primitive: { topology: "triangle-list", cullMode: "back" }, // 剔除背面防悬浮
  });

  const mainPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [mainLayout] }),
    vertex: { module, entryPoint: "main_vs", buffers: vertexBuffersConfig },
    fragment: { module, entryPoint: "main_fs", targets: [{ format }] },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
    primitive: { topology: "triangle-list", cullMode: "back" },
  });

  const shadowBindGroup = device.createBindGroup({
    layout: shadowLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const mainBindGroup = device.createBindGroup({
    layout: mainLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: shadowTexture.createView() },
      { binding: 2, resource: shadowSampler },
    ],
  });

  // ==========================================
  // 5. 动态渲染循环 (Animation Loop)
  // ==========================================
  let animationFrameId: number;
  let cameraDepthTexture: GPUTexture | null = null;

  function render(time: number) {
    // 确保 Canvas 尺寸变化时更新 Depth Texture 和 Aspect
    const canvas = context.canvas as HTMLCanvasElement;
    if (!cameraDepthTexture || cameraDepthTexture.width !== canvas.width || cameraDepthTexture.height !== canvas.height) {
      if (cameraDepthTexture) cameraDepthTexture.destroy();
      cameraDepthTexture = device.createTexture({
        size: [canvas.width, canvas.height], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    // 计算动态光源轨迹 (绕中心旋转)
    const t = time * 0.001; 
    const radius = 4.0;
    uniformData[0] = Math.cos(t) * radius;  // Light X
    uniformData[1] = 5.0;                   // Light Y (高度)
    uniformData[2] = Math.sin(t) * radius;  // Light Z
    
    // 相机固定位置，从斜上方俯视
    uniformData[4] = 4.0; // Camera X
    uniformData[5] = 4.0; // Camera Y
    uniformData[6] = 5.0; // Camera Z
    
    uniformData[8] = canvas.width / canvas.height; // Aspect

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();

    // 绘制 Pass 1: 阴影图生成
    const shadowPass = encoder.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: shadowTexture.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });
    shadowPass.setPipeline(shadowPipeline);
    shadowPass.setBindGroup(0, shadowBindGroup);
    shadowPass.setVertexBuffer(0, vertexBuffer);
    shadowPass.draw(vertexCount);
    shadowPass.end();

    // 绘制 Pass 2: 主相机场景
    const mainPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 }, // 深蓝色天空背景
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: cameraDepthTexture.createView(),
        depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });
    mainPass.setPipeline(mainPipeline);
    mainPass.setBindGroup(0, mainBindGroup);
    mainPass.setVertexBuffer(0, vertexBuffer);
    mainPass.draw(vertexCount);
    mainPass.end();

    device.queue.submit([encoder.finish()]);

    animationFrameId = requestAnimationFrame(render);
  }

  // 启动循环
  animationFrameId = requestAnimationFrame(render);

  // 清理函数
  return () => {
    cancelAnimationFrame(animationFrameId);
    vertexBuffer.destroy();
    uniformBuffer.destroy();
    shadowTexture.destroy();
    if (cameraDepthTexture) cameraDepthTexture.destroy();
  };
}