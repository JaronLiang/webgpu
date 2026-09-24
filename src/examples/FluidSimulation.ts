// src/examples/fluidSimulation.ts

export function runFluidSimulation(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  const NUM_PARTICLES = 2500; // 稍微增加粒子数，效果更好

  // =========================================================================
  // 步骤 1: 编写 WGSL 着色器
  // =========================================================================
  
  // 公共结构体 (精确 48 字节，完美对齐)
  const commonWGSL = `
    struct UniformConfig {
      dt: f32,             // 0
      gravity: f32,        // 1
      stiffness: f32,      // 2
      restingDensity: f32, // 3
      viscosity: f32,      // 4
      radius: f32,         // 5 (平滑核半径)
      boxSizeX: f32,       // 6 (物理盒子一半的宽)
      boxSizeY: f32,       // 7 (物理盒子一半的高)
      camOffsetX: f32,     // 8 (相机X)
      camOffsetY: f32,     // 9 (相机Y)
      camZoom: f32,        // 10 (相机缩放)
      aspectRatio: f32,    // 11 (屏幕宽高比) -> 12 * 4 = 48 bytes
    };

    struct Particle {
      pos: vec2f,
      vel: vec2f,
      force: vec2f,
      density: f32,
      pressure: f32,
    };
  `;

  // Compute Shader: 物理计算
  const computeShaderCode = commonWGSL + `
    @group(0) @binding(0) var<uniform> uConfig: UniformConfig;
    @group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

    @compute @workgroup_size(64)
    fn cp_density_pressure(@builtin(global_invocation_id) id: vec3u) {
      let i = id.x;
      if (i >= ${NUM_PARTICLES}u) { return; }

      let p_i = particles[i];
      var density = 0.0;
      let h = uConfig.radius;
      let h2 = h * h;

      for (var j = 0u; j < ${NUM_PARTICLES}u; j++) {
        let p_j = particles[j];
        let diff = p_i.pos - p_j.pos;
        let r2 = dot(diff, diff);
        if (r2 < h2) {
          let q = 1.0 - (r2 / h2);
          density += q * q * q; 
        }
      }
      density = max(density, 0.001); 
      let pressure = uConfig.stiffness * (density - uConfig.restingDensity);

      particles[i].density = density;
      particles[i].pressure = pressure;
    }

    @compute @workgroup_size(64)
    fn cp_forces_integrate(@builtin(global_invocation_id) id: vec3u) {
      let i = id.x;
      if (i >= ${NUM_PARTICLES}u) { return; }

      var p_i = particles[i];
      var force = vec2f(0.0, -uConfig.gravity * p_i.density);
      
      let h = uConfig.radius;
      let h2 = h * h;

      for (var j = 0u; j < ${NUM_PARTICLES}u; j++) {
        if (i == j) { continue; }
        let p_j = particles[j];
        let diff = p_i.pos - p_j.pos;
        let r2 = dot(diff, diff);

        if (r2 < h2 && r2 > 0.0001) {
          let r = sqrt(r2);
          let dir = diff / r;
          let q = 1.0 - (r / h);

          let pressureForce = -dir * (p_i.pressure + p_j.pressure) / (2.0 * p_j.density) * (q * q);
          let velDiff = p_j.vel - p_i.vel;
          let visForce = uConfig.viscosity * velDiff / p_j.density * q;

          force += pressureForce + visForce;
        }
      }

      let acc = force / p_i.density;
      p_i.vel += acc * uConfig.dt;
      
      // 速度限制防止穿模爆炸
      let maxVel = 8.0;
      if (length(p_i.vel) > maxVel) {
         p_i.vel = normalize(p_i.vel) * maxVel;
      }
      p_i.pos += p_i.vel * uConfig.dt;

      // 物理边界碰撞 (方盒子)
      let damp = 0.5;
      if (p_i.pos.x < -uConfig.boxSizeX) { p_i.pos.x = -uConfig.boxSizeX; p_i.vel.x *= -damp; }
      if (p_i.pos.x >  uConfig.boxSizeX) { p_i.pos.x =  uConfig.boxSizeX; p_i.vel.x *= -damp; }
      if (p_i.pos.y < -uConfig.boxSizeY) { p_i.pos.y = -uConfig.boxSizeY; p_i.vel.y *= -damp; }
      if (p_i.pos.y >  uConfig.boxSizeY) { p_i.pos.y =  uConfig.boxSizeY; p_i.vel.y *= -damp; }

      particles[i] = p_i;
    }
  `;

  // Render Shader: 粒子渲染 (带有相机矩阵变换)
  const renderShaderCode = commonWGSL + `
    @group(0) @binding(0) var<uniform> uConfig: UniformConfig;
    @group(0) @binding(1) var<storage, read> particles: array<Particle>;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) uv: vec2f,
      @location(1) density: f32,
    };

    @vertex
    fn vs_main(
      @builtin(vertex_index) vIdx: u32,
      @builtin(instance_index) instIdx: u32
    ) -> VertexOutput {
      let quad = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
      );
      
      let p = particles[instIdx];
      let offset = quad[vIdx];
      
      // 相机变换: World -> View -> Clip
      let viewPos = (p.pos + vec2f(uConfig.camOffsetX, uConfig.camOffsetY)) * uConfig.camZoom;
      let clipPos = vec2f(viewPos.x / uConfig.aspectRatio, viewPos.y);

      // 粒子显示大小随缩放变化，并修正圆形的拉伸
      let particleScreenSize = uConfig.radius * uConfig.camZoom * 0.8;
      let quadOffset = vec2f(offset.x / uConfig.aspectRatio, offset.y);

      var out: VertexOutput;
      out.position = vec4f(clipPos + quadOffset * particleScreenSize, 0.0, 1.0);
      out.uv = offset;
      out.density = p.density;
      return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let dist = length(in.uv);
      if (dist > 1.0) { discard; }
      
      let alpha = (1.0 - dist) * 0.9;
      let baseColor = vec3f(0.0, 0.5, 1.0);
      let highDensityColor = vec3f(0.7, 0.9, 1.0);
      let color = mix(baseColor, highDensityColor, clamp((in.density - uConfig.restingDensity) * 0.05, 0.0, 1.0));

      return vec4f(color, alpha);
    }
  `;

  // Render Shader: 画边界框
  const boxShaderCode = commonWGSL + `
    @group(0) @binding(0) var<uniform> uConfig: UniformConfig;
    @vertex
    fn vs_box(@location(0) pos: vec2f) -> @builtin(position) vec4f {
      let viewPos = (pos + vec2f(uConfig.camOffsetX, uConfig.camOffsetY)) * uConfig.camZoom;
      let clipPos = vec2f(viewPos.x / uConfig.aspectRatio, viewPos.y);
      return vec4f(clipPos, 0.0, 1.0);
    }
    @fragment
    fn fs_box() -> @location(0) vec4f {
      return vec4f(1.0, 1.0, 1.0, 0.5); // 半透明白线
    }
  `;

  // =========================================================================
  // 步骤 2: 显存配置与初始化 (大坝决堤场景)
  // =========================================================================
  const fluidParams = {
    dt: 0.012, gravity: 9.8, stiffness: 60.0,
    restingDensity: 15.0, viscosity: 0.2, radius: 0.06,
    boxX: 1.5, boxY: 1.5 // 物理盒子大小: 宽3米，高3米
  };

  const uniformData = new Float32Array(12); // 48 bytes
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const particleData = new Float32Array(NUM_PARTICLES * 8);
  
  // 初始化函数: 将粒子打包在盒子的左侧上方
  function resetParticles() {
    const cols = Math.floor(Math.sqrt(NUM_PARTICLES));
    const spacing = fluidParams.radius * 0.8;
    const startX = -fluidParams.boxX + 0.1;
    const startY = fluidParams.boxY - 0.1 - (cols * spacing);

    for (let i = 0; i < NUM_PARTICLES; i++) {
      particleData[i * 8 + 0] = startX + (i % cols) * spacing;
      particleData[i * 8 + 1] = startY + Math.floor(i / cols) * spacing;
      particleData[i * 8 + 2] = 0; // vel X
      particleData[i * 8 + 3] = 0; // vel Y
    }
    device.queue.writeBuffer(particleBuffer, 0, particleData);
  }

  const particleBuffer = device.createBuffer({
    size: particleData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  resetParticles();

  // 盒子顶点数据 (-1.5 到 1.5 的方框)
  const boxVertices = new Float32Array([
    -fluidParams.boxX, -fluidParams.boxY,
     fluidParams.boxX, -fluidParams.boxY,
     fluidParams.boxX,  fluidParams.boxY,
    -fluidParams.boxX,  fluidParams.boxY,
    -fluidParams.boxX, -fluidParams.boxY, // 闭合回起点
  ]);
  const boxVertexBuffer = device.createBuffer({
    size: boxVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(boxVertexBuffer, 0, boxVertices);

  // =========================================================================
  // 步骤 3: Pipeline Layout & Bindings
  // =========================================================================
  const computeBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const computeBindGroup = device.createBindGroup({
    layout: computeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
    ],
  });

  const renderBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });
  const renderBindGroup = device.createBindGroup({
    layout: renderBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
    ],
  });

  // 绘制盒子的 BindGroup (只需要 Uniform)
  const boxBindGroupLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
  });
  const boxBindGroup = device.createBindGroup({
    layout: boxBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // =========================================================================
  // 步骤 4: 创建 管线
  // =========================================================================
  const computeModule = device.createShaderModule({ code: computeShaderCode });
  const renderModule = device.createShaderModule({ code: renderShaderCode });
  const boxModule = device.createShaderModule({ code: boxShaderCode });

  const computeDensityPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
    compute: { module: computeModule, entryPoint: "cp_density_pressure" },
  });
  const computeForcesPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
    compute: { module: computeModule, entryPoint: "cp_forces_integrate" },
  });

  const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
    vertex: { module: renderModule, entryPoint: "vs_main" },
    fragment: { 
      module: renderModule, 
      entryPoint: "fs_main", 
      targets: [{ 
        format, 
        blend: { 
          color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one", operation: "add" }
        }
      }] 
    },
    primitive: { topology: "triangle-list" },
  });

  const boxPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [boxBindGroupLayout] }),
    vertex: {
      module: boxModule,
      entryPoint: "vs_box",
      buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }]
    },
    fragment: { module: boxModule, entryPoint: "fs_box", targets: [{ format }] },
    primitive: { topology: "line-strip" }, // 绘制线框
  });

  // =========================================================================
  // 步骤 5: 相机交互系统 (鼠标拖拽与滚轮)
  // =========================================================================
  let camState = { offsetX: 0, offsetY: 0, zoom: 0.6 }; // 默认稍微拉远一点
  let isDragging = false;
  let lastMouse = { x: 0, y: 0 };
  
  const canvasEl = context.canvas as HTMLCanvasElement;

  const onMouseDown = (e: MouseEvent) => { isDragging = true; lastMouse = { x: e.clientX, y: e.clientY }; };
  const onMouseUp = () => { isDragging = false; };
  const onMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    // 将屏幕像素移动转换为世界坐标系移动
    const aspect = canvasEl.width / canvasEl.height;
    camState.offsetX += (dx / canvasEl.width) * 2.0 * aspect / camState.zoom;
    camState.offsetY -= (dy / canvasEl.height) * 2.0 / camState.zoom; // WebGPU Y 轴向上
    lastMouse = { x: e.clientX, y: e.clientY };
  };
  const onWheel = (e: WheelEvent) => {
    e.preventDefault(); // 阻止页面滚动
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    camState.zoom = Math.max(0.1, Math.min(camState.zoom * zoomFactor, 10.0));
  };

  canvasEl.addEventListener('mousedown', onMouseDown);
  window.addEventListener('mouseup', onMouseUp);
  window.addEventListener('mousemove', onMouseMove);
  canvasEl.addEventListener('wheel', onWheel, { passive: false });

  // =========================================================================
  // 步骤 6: GUI 面板
  // =========================================================================
  const guiContainer = document.createElement("div");
  guiContainer.style.cssText = "position:absolute; top:10px; right:10px; background:rgba(20,20,20,0.85); color:white; padding:15px; border-radius:8px; font-family:sans-serif; font-size:12px; z-index:999; box-shadow: 0 4px 6px rgba(0,0,0,0.3); width: 220px; user-select:none;";
  document.body.appendChild(guiContainer);

  guiContainer.innerHTML += "<h3 style='margin-top:0; border-bottom:1px solid #444; padding-bottom:5px;'>Fluid Params</h3>";
  guiContainer.innerHTML += "<p style='color:#aaa; font-size:10px; margin-top:0;'>Drag to Pan, Scroll to Zoom</p>";

  const addSlider = (label: string, min: number, max: number, step: number, key: keyof typeof fluidParams) => {
    const wrapper = document.createElement("div"); wrapper.style.marginBottom = "8px";
    const labelEl = document.createElement("div"); labelEl.innerText = `${label}: ${fluidParams[key]}`; labelEl.style.marginBottom = "4px";
    const input = document.createElement("input");
    input.type = "range"; input.min = min.toString(); input.max = max.toString(); input.step = step.toString(); input.value = fluidParams[key].toString(); input.style.width = "100%";
    input.oninput = (e) => {
      const val = parseFloat((e.target as HTMLInputElement).value);
      (fluidParams as any)[key] = val;
      labelEl.innerText = `${label}: ${val}`;
    };
    wrapper.appendChild(labelEl); wrapper.appendChild(input); guiContainer.appendChild(wrapper);
  };

  addSlider("Gravity", 0.0, 25.0, 0.1, "gravity");
  addSlider("Stiffness", 1.0, 200.0, 1.0, "stiffness");
  addSlider("Viscosity", 0.0, 2.0, 0.01, "viscosity");
  
  const resetBtn = document.createElement("button");
  resetBtn.innerText = "Restart Dam Break";
  resetBtn.style.cssText = "width:100%; padding:8px; margin-top:10px; cursor:pointer; background:#0078D7; color:white; border:none; border-radius:4px; font-weight:bold;";
  resetBtn.onclick = resetParticles;
  guiContainer.appendChild(resetBtn);

  // =========================================================================
  // 步骤 7: 渲染循环
  // =========================================================================
  let animId: number;
  const workgroupCount = Math.ceil(NUM_PARTICLES / 64);

  function render() {
    const aspect = canvasEl.width / canvasEl.height;

    // 组装 48 bytes 的 Uniform (严格按WGSL顺序)
    uniformData[0] = fluidParams.dt;
    uniformData[1] = fluidParams.gravity;
    uniformData[2] = fluidParams.stiffness;
    uniformData[3] = fluidParams.restingDensity;
    uniformData[4] = fluidParams.viscosity;
    uniformData[5] = fluidParams.radius;
    uniformData[6] = fluidParams.boxX;
    uniformData[7] = fluidParams.boxY;
    uniformData[8] = camState.offsetX;
    uniformData[9] = camState.offsetY;
    uniformData[10] = camState.zoom;
    uniformData[11] = aspect;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();

    // 1. 发起 Compute Pass
    const computePass = encoder.beginComputePass();
    computePass.setBindGroup(0, computeBindGroup);
    computePass.setPipeline(computeDensityPipeline);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.setPipeline(computeForcesPipeline);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();

    // 2. 发起 Render Pass
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.02, g: 0.05, b: 0.1, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    
    // 渲染流体
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(6, NUM_PARTICLES, 0, 0); 

    // 渲染边界线框
    renderPass.setPipeline(boxPipeline);
    renderPass.setBindGroup(0, boxBindGroup);
    renderPass.setVertexBuffer(0, boxVertexBuffer);
    renderPass.draw(5, 1, 0, 0);

    renderPass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(render);
  }
  
  render();

  return () => {
    cancelAnimationFrame(animId);
    uniformBuffer.destroy();
    particleBuffer.destroy();
    boxVertexBuffer.destroy();
    document.body.removeChild(guiContainer);
    
    // 移除事件监听防内存泄漏
    canvasEl.removeEventListener('mousedown', onMouseDown);
    window.removeEventListener('mouseup', onMouseUp);
    window.removeEventListener('mousemove', onMouseMove);
    canvasEl.removeEventListener('wheel', onWheel);
  };
}