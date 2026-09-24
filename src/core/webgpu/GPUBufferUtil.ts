export class GPUBufferUtil {
    /**
     * 创建并填充 WebGPU Buffer
     */
    static createBuffer(
        device: GPUDevice,
        data: Float32Array | Uint16Array | Uint32Array,
        usage: GPUBufferUsageFlags
    ): GPUBuffer {
        const buffer = device.createBuffer({
            size: (data.byteLength + 3) & ~3, // 4 字节对齐
            usage: usage | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });

        // 写入数据（添加 as any 解决 TS5+ ArrayBufferLike 类型检查限制）
        if (data instanceof Float32Array) {
            new Float32Array(buffer.getMappedRange()).set(data as any);
        } else if (data instanceof Uint16Array) {
            new Uint16Array(buffer.getMappedRange()).set(data as any);
        } else if (data instanceof Uint32Array) {
            new Uint32Array(buffer.getMappedRange()).set(data as any);
        }

        buffer.unmap();
        return buffer;
    }
}