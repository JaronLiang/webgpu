import { GeometryBase } from "../core/geometry/GeometryBase";
import { VertexAttributeName } from "../core/geometry/VertexAttribute";
import { PBRMaterial } from "../core/material/PBRMaterial";

export interface AnimationSampler {
    input: Float32Array;   // 关键帧时间戳 (秒)
    output: Float32Array;  // 关键帧采样值
    interpolation: "LINEAR" | "STEP";
}

export interface AnimationChannel {
    sampler: AnimationSampler;
    targetNodeIndex: number;
    targetPath: "translation" | "rotation" | "scale";
}

export interface AnimationClip {
    name: string;
    channels: AnimationChannel[];
    duration: number;
}

export interface GLTFSkin {
    inverseBindMatrices: Float32Array[];
    joints: number[]; // 对应 Node 索引数组
}

export interface GLTFNode {
    id: number;
    name: string;
    children: number[];
    parent?: number;
    
    // 初始局部 TRS
    translation: [number, number, number];
    rotation: [number, number, number, number];
    scale: [number, number, number];
    
    localMatrix: Float32Array;
    worldMatrix: Float32Array;

    meshIndex?: number;
    skinIndex?: number;
}

export interface GLTFPrimitive {
    geometry: GeometryBase;
    material: PBRMaterial;
    indexFormat?: GPUIndexFormat;
}

export interface GLTFModelData {
    nodes: GLTFNode[];
    skins: GLTFSkin[];
    animations: AnimationClip[];
    primitivesByMesh: GLTFPrimitive[][];
}

function slerp(out: Float32Array | number[], a: ArrayLike<number>, b: ArrayLike<number>, t: number) {
    let ax = a[0], ay = a[1], az = a[2], aw = a[3];
    let bx = b[0], by = b[1], bz = b[2], bw = b[3];
    let cosom = ax * bx + ay * by + az * bz + aw * bw;
    if (cosom < 0.0) {
        cosom = -cosom;
        bx = -bx; by = -by; bz = -bz; bw = -bw;
    }
    let omega, sinom, scale0, scale1;
    if ((1.0 - cosom) > 0.000001) {
        omega = Math.acos(cosom);
        sinom = Math.sin(omega);
        scale0 = Math.sin((1.0 - t) * omega) / sinom;
        scale1 = Math.sin(t * omega) / sinom;
    } else {
        scale0 = 1.0 - t;
        scale1 = t;
    }
    out[0] = scale0 * ax + scale1 * bx;
    out[1] = scale0 * ay + scale1 * by;
    out[2] = scale0 * az + scale1 * bz;
    out[3] = scale0 * aw + scale1 * bw;
}

export function mat4Multiply(out: Float32Array, a: Float32Array, b: Float32Array): Float32Array {
    const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

    let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    out[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

    b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
    out[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

    b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
    out[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

    b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
    out[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    return out;
}

export function mat4FromRotationTranslationScale(q: ArrayLike<number>, t: ArrayLike<number>, s: ArrayLike<number>): Float32Array {
    const x = q[0], y = q[1], z = q[2], w = q[3];
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;

    const sx = s[0], sy = s[1], sz = s[2];
    const out = new Float32Array(16);
    out[0] = (1 - (yy + zz)) * sx;
    out[1] = (xy + wz) * sx;
    out[2] = (xz - wy) * sx;
    out[3] = 0;

    out[4] = (xy - wz) * sy;
    out[5] = (1 - (xx + zz)) * sy;
    out[6] = (yz + wx) * sy;
    out[7] = 0;

    out[8] = (xz + wy) * sz;
    out[9] = (yz - wx) * sz;
    out[10] = (1 - (xx + yy)) * sz;
    out[11] = 0;

    out[12] = t[0];
    out[13] = t[1];
    out[14] = t[2];
    out[15] = 1;
    return out;
}

export class GLTFLoader {
    static async load(
        url: string,
        device: GPUDevice,
        materialLayout: GPUBindGroupLayout,
        defaultTexture: GPUTexture,
        defaultNormalTexture: GPUTexture,
        defaultSampler: GPUSampler
    ): Promise<GLTFModelData> {
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();

        let json: any;
        let binaryChunk: ArrayBuffer | null = null;

        const magic = new Uint32Array(arrayBuffer, 0, 1)[0];
        if (magic === 0x46546c67) {
            const dataView = new DataView(arrayBuffer);
            let byteOffset = 12;
            while (byteOffset < arrayBuffer.byteLength) {
                const chunkLength = dataView.getUint32(byteOffset, true);
                const chunkType = dataView.getUint32(byteOffset + 4, true);
                byteOffset += 8;
                if (chunkType === 0x4e4f534a) {
                    json = JSON.parse(new TextDecoder().decode(new Uint8Array(arrayBuffer, byteOffset, chunkLength)));
                } else if (chunkType === 0x004e4942) {
                    binaryChunk = arrayBuffer.slice(byteOffset, byteOffset + chunkLength);
                }
                byteOffset += chunkLength;
            }
        } else {
            json = JSON.parse(new TextDecoder().decode(arrayBuffer));
        }

        const buffers: ArrayBuffer[] = [];
        for (let i = 0; i < (json.buffers || []).length; i++) {
            if (i === 0 && binaryChunk) {
                buffers.push(binaryChunk);
            } else {
                const uri = new URL(json.buffers[i].uri, url).href;
                const res = await fetch(uri);
                buffers.push(await res.arrayBuffer());
            }
        }

        // 解析纹理
        const imageBitmaps: ImageBitmap[] = [];
        if (json.images) {
            for (const img of json.images) {
                let blob: Blob;
                if (img.bufferView !== undefined) {
                    const bv = json.bufferViews[img.bufferView];
                    const buf = buffers[bv.buffer];
                    const slice = buf.slice(bv.byteOffset || 0, (bv.byteOffset || 0) + bv.byteLength);
                    blob = new Blob([slice], { type: img.mimeType || "image/png" });
                } else {
                    const imgUrl = new URL(img.uri, url).href;
                    const res = await fetch(imgUrl);
                    blob = await res.blob();
                }
                const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });
                imageBitmaps.push(bitmap);
            }
        }

        function createGPUTexture(imageIndex: number, isSRGB: boolean): GPUTexture {
            const bitmap = imageBitmaps[imageIndex];
            const format: GPUTextureFormat = isSRGB ? "rgba8unorm-srgb" : "rgba8unorm";
            const texture = device.createTexture({
                size: [bitmap.width, bitmap.height, 1],
                format,
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
            });
            device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [bitmap.width, bitmap.height]);
            return texture;
        }

        function getAccessorData(accessorIndex: number) {
            const accessor = json.accessors[accessorIndex];
            const bufferView = json.bufferViews[accessor.bufferView];
            const buffer = buffers[bufferView.buffer];
            const byteOffset = (bufferView.byteOffset || 0) + (accessor.byteOffset || 0);

            let count = 1;
            if (accessor.type === "VEC2") count = 2;
            if (accessor.type === "VEC3") count = 3;
            if (accessor.type === "VEC4") count = 4;
            if (accessor.type === "MAT4") count = 16;

            const total = accessor.count * count;
            switch (accessor.componentType) {
                case 5126: return { data: new Float32Array(buffer, byteOffset, total), componentType: 5126 };
                case 5123: return { data: new Uint16Array(buffer, byteOffset, total), componentType: 5123 };
                case 5121: return { data: new Uint8Array(buffer, byteOffset, total), componentType: 5121 };
                case 5125: return { data: new Uint32Array(buffer, byteOffset, total), componentType: 5125 };
                default: throw new Error(`Unsupported ComponentType: ${accessor.componentType}`);
            }
        }

        // 解析 Mesh Primitives
        const primitivesByMesh: GLTFPrimitive[][] = [];
        for (const mesh of json.meshes || []) {
            const primitives: GLTFPrimitive[] = [];
            for (const prim of mesh.primitives) {
                const geom = new GeometryBase();
                let indexFormat: GPUIndexFormat | undefined = undefined;

                if (prim.attributes.POSITION !== undefined) {
                    geom.setAttribute(VertexAttributeName.position, getAccessorData(prim.attributes.POSITION).data as Float32Array, 3);
                }
                if (prim.attributes.NORMAL !== undefined) {
                    geom.setAttribute(VertexAttributeName.normal, getAccessorData(prim.attributes.NORMAL).data as Float32Array, 3);
                }
                if (prim.attributes.TEXCOORD_0 !== undefined) {
                    geom.setAttribute(VertexAttributeName.uv, getAccessorData(prim.attributes.TEXCOORD_0).data as Float32Array, 2);
                } else {
                    geom.setAttribute(VertexAttributeName.uv, new Float32Array(geom.vertexCount * 2), 2);
                }
                if (prim.attributes.TANGENT !== undefined) {
                    geom.setAttribute(VertexAttributeName.tangent, getAccessorData(prim.attributes.TANGENT).data as Float32Array, 4);
                } else {
                    const defaultTangents = new Float32Array(geom.vertexCount * 4);
                    for (let i = 0; i < geom.vertexCount; i++) {
                        defaultTangents[i * 4] = 1.0;
                        defaultTangents[i * 4 + 3] = 1.0;
                    }
                    geom.setAttribute(VertexAttributeName.tangent, defaultTangents, 4);
                }

                // 重点：骨骼关节 (Joints) 与权重 (Weights)
                if (prim.attributes.JOINTS_0 !== undefined) {
                    const rawJoints = getAccessorData(prim.attributes.JOINTS_0).data;
                    const jointsF32 = new Float32Array(rawJoints.length);
                    for (let i = 0; i < rawJoints.length; i++) jointsF32[i] = rawJoints[i];
                    // 作为自定义属性或预留属性槽上传
                    geom.setAttribute("joints" as any, jointsF32, 4);
                } else {
                    geom.setAttribute("joints" as any, new Float32Array(geom.vertexCount * 4), 4);
                }

                if (prim.attributes.WEIGHTS_0 !== undefined) {
                    geom.setAttribute("weights" as any, getAccessorData(prim.attributes.WEIGHTS_0).data as Float32Array, 4);
                } else {
                    const defaultWeights = new Float32Array(geom.vertexCount * 4);
                    for (let i = 0; i < geom.vertexCount; i++) defaultWeights[i * 4] = 1.0;
                    geom.setAttribute("weights" as any, defaultWeights, 4);
                }

                if (prim.indices !== undefined) {
                    const idxInfo = getAccessorData(prim.indices);
                    geom.setIndices(idxInfo.data as Uint16Array | Uint32Array);
                    indexFormat = idxInfo.componentType === 5125 ? "uint32" : "uint16";
                }
                geom.upload(device);

                const pbrMat = new PBRMaterial();
                if (prim.material !== undefined && json.materials) {
                    const matData = json.materials[prim.material];
                    if (matData.pbrMetallicRoughness) {
                        const pbr = matData.pbrMetallicRoughness;
                        if (pbr.baseColorFactor) pbrMat.baseColorFactor = pbr.baseColorFactor;
                        if (pbr.metallicFactor !== undefined) pbrMat.metallicFactor = pbr.metallicFactor;
                        if (pbr.roughnessFactor !== undefined) pbrMat.roughnessFactor = pbr.roughnessFactor;
                        if (pbr.baseColorTexture !== undefined) {
                            pbrMat.baseColorTexture = createGPUTexture(json.textures[pbr.baseColorTexture.index].source, true);
                        }
                        if (pbr.metallicRoughnessTexture !== undefined) {
                            pbrMat.metallicRoughnessTexture = createGPUTexture(json.textures[pbr.metallicRoughnessTexture.index].source, false);
                        }
                    }
                    if (matData.normalTexture !== undefined) {
                        pbrMat.normalTexture = createGPUTexture(json.textures[matData.normalTexture.index].source, false);
                    }
                    if (matData.occlusionTexture !== undefined) {
                        pbrMat.occlusionTexture = createGPUTexture(json.textures[matData.occlusionTexture.index].source, false);
                    }
                    if (matData.emissiveTexture !== undefined) {
                        pbrMat.emissiveTexture = createGPUTexture(json.textures[matData.emissiveTexture.index].source, true);
                    }
                }
                pbrMat.buildGPUResources(device, materialLayout, defaultTexture, defaultNormalTexture, defaultSampler);
                primitives.push({ geometry: geom, material: pbrMat, indexFormat });
            }
            primitivesByMesh.push(primitives);
        }

        // 解析 Nodes 层级结构
        const nodes: GLTFNode[] = (json.nodes || []).map((n: any, idx: number) => {
            const translation: [number, number, number] = n.translation ? [n.translation[0], n.translation[1], n.translation[2]] : [0, 0, 0];
            const rotation: [number, number, number, number] = n.rotation ? [n.rotation[0], n.rotation[1], n.rotation[2], n.rotation[3]] : [0, 0, 0, 1];
            const scale: [number, number, number] = n.scale ? [n.scale[0], n.scale[1], n.scale[2]] : [1, 1, 1];

            return {
                id: idx,
                name: n.name || `Node_${idx}`,
                children: n.children || [],
                translation,
                rotation,
                scale,
                localMatrix: mat4FromRotationTranslationScale(rotation, translation, scale),
                worldMatrix: new Float32Array(16),
                meshIndex: n.mesh,
                skinIndex: n.skin,
            };
        });

        // 绑定父节点
        for (const node of nodes) {
            for (const childId of node.children) {
                nodes[childId].parent = node.id;
            }
        }

        // 解析 Skins (骨骼蒙皮)
       // 解析 Skins (骨骼蒙皮)
        const skins: GLTFSkin[] = [];
        if (json.skins) {
            for (const s of json.skins) {
                const invBindMatrices: Float32Array[] = [];
                
                // 容错处理：glTF 中 inverseBindMatrices 是可选的，若未定义则使用单位矩阵
                if (s.inverseBindMatrices !== undefined) {
                    const invBindData = getAccessorData(s.inverseBindMatrices).data as Float32Array;
                    for (let i = 0; i < s.joints.length; i++) {
                        invBindMatrices.push(invBindData.slice(i * 16, (i + 1) * 16));
                    }
                } else {
                    for (let i = 0; i < s.joints.length; i++) {
                        invBindMatrices.push(new Float32Array([
                            1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1
                        ]));
                    }
                }
                
                // 这里显式赋值，消除变量名不一致与类型错误
                skins.push({ 
                    inverseBindMatrices: invBindMatrices, 
                    joints: s.joints 
                });
            }
        }

        // 解析 Animations (帧动画/骨骼动画)
        const animations: AnimationClip[] = [];
        if (json.animations) {
            for (const anim of json.animations) {
                let maxDuration = 0;
                const samplers: AnimationSampler[] = anim.samplers.map((s: any) => {
                    const input = getAccessorData(s.input).data as Float32Array;
                    const output = getAccessorData(s.output).data as Float32Array;
                    const lastTime = input[input.length - 1];
                    if (lastTime > maxDuration) maxDuration = lastTime;
                    return {
                        input,
                        output,
                        interpolation: s.interpolation || "LINEAR",
                    };
                });

                const channels: AnimationChannel[] = anim.channels.map((ch: any) => ({
                    sampler: samplers[ch.sampler],
                    targetNodeIndex: ch.target.node,
                    targetPath: ch.target.path,
                }));

                animations.push({
                    name: anim.name || "DefaultAnimation",
                    channels,
                    duration: maxDuration,
                });
            }
        }

        return { nodes, skins, animations, primitivesByMesh };
    }
}