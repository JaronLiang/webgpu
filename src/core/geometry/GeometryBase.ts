import { VertexAttribute, VertexAttributeName } from "./VertexAttribute";
import { GPUBufferUtil } from "../webgpu/GPUBufferUtil";

export class GeometryBase {
    public attributes: Map<string, VertexAttribute> = new Map();
    public indexBuffer?: GPUBuffer;
    public indexCount: number = 0;
    public vertexCount: number = 0;

    /**
     * 设置顶点属性
     */
    public setAttribute(name: VertexAttributeName, data: Float32Array, itemSize: number) {
        this.attributes.set(name, {
            name,
            data,
            itemSize
        });
        if (name === VertexAttributeName.position) {
            this.vertexCount = data.length / itemSize;
        }
    }

    /**
     * 设置索引
     */
    public setIndices(indices: Uint16Array | Uint32Array) {
        this.attributes.set(VertexAttributeName.indices, {
            name: VertexAttributeName.indices,
            data: indices,
            itemSize: 1
        });
        this.indexCount = indices.length;
    }

    public getAttribute(name: VertexAttributeName): VertexAttribute | undefined {
        return this.attributes.get(name);
    }

    /**
     * 上传数据至 GPU
     */
    public upload(device: GPUDevice) {
        for (const [name, attr] of this.attributes) {
            if (name === VertexAttributeName.indices) {
                this.indexBuffer = GPUBufferUtil.createBuffer(
                    device,
                    attr.data,
                    GPUBufferUsage.INDEX
                );
            } else {
                attr.buffer = GPUBufferUtil.createBuffer(
                    device,
                    attr.data,
                    GPUBufferUsage.VERTEX
                );
            }
        }
    }
}