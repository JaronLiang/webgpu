export class PBRMaterial {
    public baseColorFactor: [number, number, number, number] = [1.0, 1.0, 1.0, 1.0];
    public metallicFactor: number = 1.0;
    public roughnessFactor: number = 1.0;
    public emissiveFactor: [number, number, number] = [0.0, 0.0, 0.0];
    public occlusionStrength: number = 1.0;
    public normalScale: number = 1.0;

    // 5 张核心 PBR 贴图
    public baseColorTexture?: GPUTexture;
    public metallicRoughnessTexture?: GPUTexture;
    public normalTexture?: GPUTexture;
    public occlusionTexture?: GPUTexture;
    public emissiveTexture?: GPUTexture;

    public uniformBuffer?: GPUBuffer;
    public bindGroup?: GPUBindGroup;

    constructor(params?: Partial<PBRMaterial>) {
        if (params) {
            Object.assign(this, params);
        }
    }

    /**
     * 生成与 GPU 对齐的 PBR 数据（严格 64 字节）
     */
    public getMaterialData(): Float32Array {
        const data = new Float32Array(16);
        data[0] = this.baseColorFactor[0];
        data[1] = this.baseColorFactor[1];
        data[2] = this.baseColorFactor[2];
        data[3] = this.baseColorFactor[3];

        data[4] = this.metallicFactor;
        data[5] = this.roughnessFactor;
        data[6] = this.baseColorTexture ? 1.0 : 0.0;
        data[7] = this.metallicRoughnessTexture ? 1.0 : 0.0;

        data[8] = this.emissiveFactor[0];
        data[9] = this.emissiveFactor[1];
        data[10] = this.emissiveFactor[2];
        data[11] = this.emissiveTexture ? 1.0 : 0.0;

        data[12] = this.occlusionStrength;
        data[13] = this.occlusionTexture ? 1.0 : 0.0;
        data[14] = this.normalTexture ? 1.0 : 0.0;
        data[15] = this.normalScale;

        return data;
    }

    public buildGPUResources(
        device: GPUDevice,
        layout: GPUBindGroupLayout,
        defaultTexture: GPUTexture,
        defaultNormalTexture: GPUTexture,
        defaultSampler: GPUSampler
    ) {
        const data = this.getMaterialData();
        this.uniformBuffer = device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.uniformBuffer, 0, data.buffer, data.byteOffset, data.byteLength);

        this.bindGroup = device.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: defaultSampler },
                { binding: 2, resource: (this.baseColorTexture || defaultTexture).createView() },
                { binding: 3, resource: (this.metallicRoughnessTexture || defaultTexture).createView() },
                { binding: 4, resource: (this.normalTexture || defaultNormalTexture).createView() },
                { binding: 5, resource: (this.occlusionTexture || defaultTexture).createView() },
                { binding: 6, resource: (this.emissiveTexture || defaultTexture).createView() },
            ],
        });
    }
}