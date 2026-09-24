// src/utils/math.ts
export class Mat4 {
  // 创建透视投影矩阵
  static perspective(fovy: number, aspect: number, near: number, far: number): Float32Array {
    const out = new Float32Array(16);
    const f = 1.0 / Math.tan(fovy / 2);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = far / (near - far);
    out[11] = -1;
    out[14] = (near * far) / (near - far);
    return out;
  }

  // 矩阵相乘 (A * B)
  static multiply(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; ++i) {
      for (let j = 0; j < 4; ++j) {
        let sum = 0;
        for (let k = 0; k < 4; ++k) {
          sum += a[k * 4 + i] * b[j * 4 + k];
        }
        out[j * 4 + i] = sum;
      }
    }
    return out;
  }

  // 生成围绕 X, Y 轴旋转的模型视图矩阵
  static createModelView(rx: number, ry: number, tz: number): Float32Array {
    const out = new Float32Array(16);
    const radX = (rx * Math.PI) / 180;
    const radY = (ry * Math.PI) / 180;
    const cx = Math.cos(radX), sx = Math.sin(radX);
    const cy = Math.cos(radY), sy = Math.sin(radY);

    out[0] = cy;
    out[1] = sx * sy;
    out[2] = -cx * sy;
    out[4] = 0;
    out[5] = cx;
    out[6] = sx;
    out[8] = sy;
    out[9] = -sx * cy;
    out[10] = cx * cy;
    out[14] = tz;
    out[15] = 1;
    return out;
  }

  // 【新增】平移矩阵
  static translation(x: number, y: number, z: number): Float32Array {
    const out = new Float32Array(16);
    out[0] = 1;
    out[5] = 1;
    out[10] = 1;
    out[15] = 1;
    out[12] = x;
    out[13] = y;
    out[14] = z;
    return out;
  }

  // 【新增】缩放矩阵
  static scaling(x: number, y: number, z: number): Float32Array {
    const out = new Float32Array(16);
    out[0] = x;
    out[5] = y;
    out[10] = z;
    out[15] = 1;
    return out;
  }
}