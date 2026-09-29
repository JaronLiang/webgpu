// src/examples/aa_msaa.ts
export function runMSAA(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat
) {
  // 1. 生成细分球体网格 (80x80 纬线/经线)
  const latBands = 80, lonBands = 80;
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  for (let lat = 0; lat <= latBands; lat++) {
    const theta = (lat * Math.PI) / latBands;
    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);

    for (let lon = 0; lon <= lonBands; lon++) {
      const phi = (lon * 2 * Math.PI) / lonBands;
      const x = Math.cos(phi) * sinTheta;
      const y = cosTheta;
      const z = Math.sin(phi) * sinTheta;

      positions.push(x * 0.95, y * 0.95, z * 0.95);
      normals.push(x, y, z);
    }
  }

  for (let lat = 0; lat < latBands; lat++) {
    for (let lon = 0; lon < lonBands; lon++) {
      const first = lat * (lonBands + 1) + lon;
      const second = first + lonBands + 1;
      indices.push(first, second, first + 1);
      indices.push(second, second + 1, first + 1);
    }
  }

  // 顶点与索引缓冲区
  const vertBuffer = device.createBuffer({
    size: positions.length * 4 * 2, // pos(3) + norm(3)
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  const vertData = new Float32Array(positions.length * 2);
  for (let i = 0; i < positions.length / 3; i++) {
    vertData[i * 6 + 0] = positions[i * 3 + 0];
    vertData[i * 6 + 1] = positions[i * 3 + 1];
    vertData[i * 6 + 2] = positions[i * 3 + 2];
    vertData[i * 6 + 3] = normals[i * 3 + 0];
    vertData[i * 6 + 4] = normals[i * 3 + 1];
    vertData[i * 6 + 5] = normals[i * 3 + 2];
  }
  device.queue.writeBuffer(vertBuffer, 0, vertData);

  const idxBuffer = device.createBuffer({
    size: indices.length * 2,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(idxBuffer, 0, new Uint16Array(indices));

  // 2. 着色器代码
  const shaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
      camPos: vec4f,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VertexInput {
      @location(0) pos: vec3f,
      @location(1) norm: vec3f,
    };

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) worldNorm: vec3f,
    };

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
      var out: VertexOutput;
      out.position = u.viewProj * vec4f(in.pos, 1.0);
      out.worldPos = in.pos;
      out.worldNorm = in.norm;
      return out;
    }

    fn aces(color: vec3f) -> vec3f {
      let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
      return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3f(0.0), vec3f(1.0));
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
      let N = normalize(in.worldNorm);
      let V = normalize(u.camPos.xyz - in.worldPos);
      let R = reflect(-V, N);

      // 黄金材质与影棚光照
      let albedo = vec3f(1.00, 0.76, 0.28);
      let NdotV = max(dot(N, V), 0.01);
      let F0 = albedo;
      let F = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

      // 主影棚光箱
      let lDir = normalize(vec3f(0.65, 0.65, 0.4));
      let spec = pow(max(dot(R, lDir), 0.0), 32.0) * 8.0;
      // 辅冷光
      let lDir2 = normalize(vec3f(-0.85, 0.2, -0.48));
      let spec2 = pow(max(dot(R, lDir2), 0.0), 16.0) * 3.0;

      let color = (albedo * 0.15 + vec3f(spec) * vec3f(1.0, 0.9, 0.7) + vec3f(spec2) * vec3f(0.3, 0.6, 1.0)) * F;
      return vec4f(pow(aces(color), vec3f(1.0 / 2.2)), 1.0);
    }
  `;

  // 3. 4x MSAA 渲染管线
  const sampleCount = 4;
  const uniformBuffer = device.createBuffer({
    size: 16 * 4 + 4 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const module = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module,
      entryPoint: "vs_main",
      buffers: [{
        arrayStride: 6 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" },
        ],
      }],
    },
    fragment: {
      module,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list", cullMode: "back" },
    multisample: { count: sampleCount }, // 开启 4 重采样
  });

  // 4. 管理 MSAA 颜色纹理与深度纹理
  let msaaTexture: GPUTexture | null = null;
  function updateMSAATexture(width: number, height: number) {
    if (msaaTexture) msaaTexture.destroy();
    msaaTexture = device.createTexture({
      size: [width, height],
      sampleCount,
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  const canvas = context.canvas as HTMLCanvasElement;
  updateMSAATexture(canvas.width, canvas.height);

  let animId: number;
  function render() {
    if (canvas.width !== msaaTexture?.width || canvas.height !== msaaTexture?.height) {
      updateMSAATexture(canvas.width, canvas.height);
    }

    const t = performance.now() * 0.0008;
    const camPos = [Math.sin(t) * 2.8, 0.5, Math.cos(t) * 2.8, 1.0];
    
    // 透视与视图变换矩阵
    const aspect = canvas.width / canvas.height;
    const fov = Math.PI / 4;
    const f = 1.0 / Math.tan(fov / 2);
    const near = 0.1, far = 100.0;
    
    // LookAt(cam, [0,0,0], [0,1,0])
    const zAxis = normalize([camPos[0], camPos[1], camPos[2]]);
    const xAxis = normalize(cross([0, 1, 0], zAxis));
    const yAxis = cross(zAxis, xAxis);

    const viewProj = new Float32Array([
      (f / aspect) * xAxis[0], f * yAxis[0], zAxis[0] * (far / (near - far)), -zAxis[0],
      (f / aspect) * xAxis[1], f * yAxis[1], zAxis[1] * (far / (near - far)), -zAxis[1],
      (f / aspect) * xAxis[2], f * yAxis[2], zAxis[2] * (far / (near - far)), -zAxis[2],
      0, 0, (near * far) / (near - far), 0
    ]);

    device.queue.writeBuffer(uniformBuffer, 0, viewProj);
    device.queue.writeBuffer(uniformBuffer, 64, new Float32Array(camPos));

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: msaaTexture!.createView(), // MSAA 纹理渲染目标
        resolveTarget: context.getCurrentTexture().createView(), // 自动解析到最终画布
        clearValue: { r: 0.03, g: 0.04, b: 0.05, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertBuffer);
    pass.setIndexBuffer(idxBuffer, "uint16");
    pass.drawIndexed(indices.length);
    pass.end();

    device.queue.submit([encoder.finish()]);
    animId = requestAnimationFrame(render);
  }
  render();

  function cross(a: number[], b: number[]): number[] {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }
  function normalize(v: number[]): number[] {
    const l = Math.hypot(...v);
    return [v[0] / l, v[1] / l, v[2] / l];
  }

  return () => {
    cancelAnimationFrame(animId);
    vertBuffer.destroy();
    idxBuffer.destroy();
    uniformBuffer.destroy();
    msaaTexture?.destroy();
  };
}