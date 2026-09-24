// src/webgpu/TextureUtil.ts
export class TextureUtil {
    static async createTextureFromUrl(device: GPUDevice, url: string): Promise<GPUTexture> {
        const response = await fetch(url);
        const blob = await response.blob();
        const imageBitmap = await createImageBitmap(blob);
        return this.createTextureFromBitmap(device, imageBitmap);
    }

    static createTextureFromBitmap(device: GPUDevice, imageBitmap: ImageBitmap): GPUTexture {
        const texture = device.createTexture({
            size: [imageBitmap.width, imageBitmap.height, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        device.queue.copyExternalImageToTexture(
            { source: imageBitmap },
            { texture: texture },
            [imageBitmap.width, imageBitmap.height]
        );
        return texture;
    }

    static createSolidTexture(device: GPUDevice, color: [number, number, number, number]): GPUTexture {
        const texture = device.createTexture({
            size: [1, 1, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture(
            { texture },
            new Uint8Array(color),
            { bytesPerRow: 4, rowsPerImage: 1 },
            [1, 1, 1]
        );
        return texture;
    }
}