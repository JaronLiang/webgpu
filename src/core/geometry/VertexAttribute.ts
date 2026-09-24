export enum VertexAttributeName {
    position = "position",
    normal = "normal",
    uv = "uv",
    tangent = "tangent",
    color = "color",
    indices = "indices"
}

export interface VertexAttribute {
    name: VertexAttributeName;
    data: Float32Array | Uint16Array | Uint32Array;
    itemSize: number;
    buffer?: GPUBuffer;
}