// src/examples/controlRig.ts
import GUI from "lil-gui";

// =========================================================================
// 1. 标准 3D 矩阵与四元数数学库 (针对 WebGPU 定制)
// =========================================================================
function mat4Perspective(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const out = new Float32Array(16);
  const f = 1.0 / Math.tan(fovRad / 2);
  out[0] = f / aspect; out[5] = f;
  out[10] = far / (near - far); out[11] = -1; out[14] = (near * far) / (near - far);
  return out;
}

function mat4LookAt(eye: number[], center: number[], up: number[]): Float32Array {
  const out = new Float32Array(16);
  let z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
  let len = 1 / Math.hypot(z0, z1, z2); z0 *= len; z1 *= len; z2 *= len;
  let x0 = up[1] * z2 - up[2] * z1, x1 = up[2] * z0 - up[0] * z2, x2 = up[0] * z1 - up[1] * z0;
  len = 1 / Math.hypot(x0, x1, x2); x0 *= len; x1 *= len; x2 *= len;
  let y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;
  out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
  out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
  out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
  out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
  out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
  out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]); out[15] = 1;
  return out;
}

function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function mat4Inverse(m: Float32Array): Float32Array {
  const out = new Float32Array(16);
  const m00 = m[0], m01 = m[1], m02 = m[2], m03 = m[3];
  const m10 = m[4], m11 = m[5], m12 = m[6], m13 = m[7];
  const m20 = m[8], m21 = m[9], m22 = m[10], m23 = m[11];
  const m30 = m[12], m31 = m[13], m32 = m[14], m33 = m[15];

  const b00 = m00 * m11 - m01 * m10, b01 = m00 * m12 - m02 * m10, b02 = m00 * m13 - m03 * m10;
  const b03 = m01 * m12 - m02 * m11, b04 = m01 * m13 - m03 * m11, b05 = m02 * m13 - m03 * m12;
  const b06 = m20 * m31 - m21 * m30, b07 = m20 * m32 - m22 * m30, b08 = m20 * m33 - m23 * m30;
  const b09 = m21 * m32 - m22 * m31, b10 = m21 * m33 - m23 * m31, b11 = m22 * m33 - m23 * m32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return out;
  det = 1.0 / det;

  out[0] = (m11 * b11 - m12 * b10 + m13 * b09) * det;
  out[1] = (m02 * b10 - m01 * b11 - m03 * b09) * det;
  out[2] = (m31 * b05 - m32 * b04 + m33 * b03) * det;
  out[3] = (m22 * b04 - m21 * b05 - m23 * b03) * det;
  out[4] = (m12 * b08 - m10 * b11 - m13 * b07) * det;
  out[5] = (m00 * b11 - m02 * b08 + m03 * b07) * det;
  out[6] = (m32 * b02 - m30 * b05 - m33 * b01) * det;
  out[7] = (m20 * b05 - m22 * b02 + m23 * b01) * det;
  out[8] = (m10 * b10 - m11 * b08 + m13 * b06) * det;
  out[9] = (m01 * b08 - m00 * b10 - m03 * b06) * det;
  out[10] = (m30 * b04 - m31 * b02 + m33 * b00) * det;
  out[11] = (m21 * b02 - m20 * b04 - m23 * b00) * det;
  out[12] = (m11 * b07 - m10 * b09 - m12 * b06) * det;
  out[13] = (m00 * b09 - m01 * b07 + m02 * b06) * det;
  out[14] = (m31 * b01 - m30 * b03 - m32 * b00) * det;
  out[15] = (m20 * b03 - m21 * b01 + m22 * b00) * det;
  return out;
}

function mat4FromRotationTranslation(q: number[], v: number[]): Float32Array {
  const out = new Float32Array(16);
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;

  out[0] = 1 - (yy + zz); out[1] = xy + wz; out[2] = xz - wy; out[3] = 0;
  out[4] = xy - wz; out[5] = 1 - (xx + zz); out[6] = yz + wx; out[7] = 0;
  out[8] = xz + wy; out[9] = yz - wx; out[10] = 1 - (xx + yy); out[11] = 0;
  out[12] = v[0]; out[13] = v[1]; out[14] = v[2]; out[15] = 1;
  return out;
}

function quatMultiply(a: number[], b: number[]): number[] {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

function quatFromAxisAngle(axis: number[], rad: number): number[] {
  const half = rad * 0.5;
  const s = Math.sin(half);
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(half)];
}

function quatSlerp(a: number[], b: number[], t: number): number[] {
  let cosHalfTheta = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let bFinal = b;
  if (cosHalfTheta < 0) {
    bFinal = [-b[0], -b[1], -b[2], -b[3]];
    cosHalfTheta = -cosHalfTheta;
  }
  if (Math.abs(cosHalfTheta) >= 1.0) {
    return [a[0], a[1], a[2], a[3]];
  }
  const halfTheta = Math.acos(cosHalfTheta);
  const sinHalfTheta = Math.sqrt(1.0 - cosHalfTheta * cosHalfTheta);
  if (Math.abs(sinHalfTheta) < 0.001) {
    return [
      a[0] * 0.5 + bFinal[0] * 0.5,
      a[1] * 0.5 + bFinal[1] * 0.5,
      a[2] * 0.5 + bFinal[2] * 0.5,
      a[3] * 0.5 + bFinal[3] * 0.5,
    ];
  }
  const ratioA = Math.sin((1 - t) * halfTheta) / sinHalfTheta;
  const ratioB = Math.sin(t * halfTheta) / sinHalfTheta;
  return [
    a[0] * ratioA + bFinal[0] * ratioB,
    a[1] * ratioA + bFinal[1] * ratioB,
    a[2] * ratioA + bFinal[2] * ratioB,
    a[3] * ratioA + bFinal[3] * ratioB,
  ];
}

// =========================================================================
// 2. 17 骨骼人体骨架定义与拓扑映射 (Humanoid Bone Definition)
// =========================================================================
const BONES = {
  Hips: 0,
  Spine: 1,
  Chest: 2,
  Neck: 3,
  Head: 4,
  LeftArm: 5,
  LeftForeArm: 6,
  LeftHand: 7,
  RightArm: 8,
  RightForeArm: 9,
  RightHand: 10,
  LeftUpLeg: 11,
  LeftLeg: 12,
  LeftFoot: 13,
  RightUpLeg: 14,
  RightLeg: 15,
  RightFoot: 16,
};
const BONE_COUNT = 17;

const BONE_PARENTS: number[] = [
  -1,               // 0: Hips (Root)
  BONES.Hips,       // 1: Spine
  BONES.Spine,      // 2: Chest
  BONES.Chest,      // 3: Neck
  BONES.Neck,       // 4: Head
  BONES.Chest,      // 5: LeftArm
  BONES.LeftArm,    // 6: LeftForeArm
  BONES.LeftForeArm,// 7: LeftHand
  BONES.Chest,      // 8: RightArm
  BONES.RightArm,   // 9: RightForeArm
  BONES.RightForeArm,// 10: RightHand
  BONES.Hips,       // 11: LeftUpLeg
  BONES.LeftUpLeg,  // 12: LeftLeg
  BONES.LeftLeg,    // 13: LeftFoot
  BONES.Hips,       // 14: RightUpLeg
  BONES.RightUpLeg, // 15: RightLeg
  BONES.RightLeg,   // 16: RightFoot
];

interface BoneNode {
  name: string;
  parent: number;
  restWorldPos: number[]; // 静止 T-Pose 绝对坐标
  localPos: number[];     // 相对父级的局部坐标
}

function buildSkeleton(scaleX = 1.0, scaleY = 1.0, scaleZ = 1.0): { bones: BoneNode[]; invBinds: Float32Array[] } {
  const rawPositions: { [key: number]: number[] } = {
    [BONES.Hips]:        [0.0, 2.05 * scaleY, 0.0],
    [BONES.Spine]:       [0.0, 2.45 * scaleY, 0.0],
    [BONES.Chest]:       [0.0, 2.90 * scaleY, 0.0],
    [BONES.Neck]:        [0.0, 3.35 * scaleY, 0.0],
    [BONES.Head]:        [0.0, 3.75 * scaleY, 0.0],

    [BONES.LeftArm]:     [-0.55 * scaleX, 3.05 * scaleY, 0.0],
    [BONES.LeftForeArm]: [-1.15 * scaleX, 3.05 * scaleY, 0.0],
    [BONES.LeftHand]:    [-1.75 * scaleX, 3.05 * scaleY, 0.0],

    [BONES.RightArm]:    [ 0.55 * scaleX, 3.05 * scaleY, 0.0],
    [BONES.RightForeArm]:[ 1.15 * scaleX, 3.05 * scaleY, 0.0],
    [BONES.RightHand]:   [ 1.75 * scaleX, 3.05 * scaleY, 0.0],

    [BONES.LeftUpLeg]:   [-0.34 * scaleX, 2.00 * scaleY, 0.0],
    [BONES.LeftLeg]:     [-0.34 * scaleX, 1.05 * scaleY, 0.0],
    [BONES.LeftFoot]:    [-0.34 * scaleX, 0.15 * scaleY, 0.0],

    [BONES.RightUpLeg]:  [ 0.34 * scaleX, 2.00 * scaleY, 0.0],
    [BONES.RightLeg]:    [ 0.34 * scaleX, 1.05 * scaleY, 0.0],
    [BONES.RightFoot]:   [ 0.34 * scaleX, 0.15 * scaleY, 0.0],
  };

  const bones: BoneNode[] = [];
  const invBinds: Float32Array[] = [];

  for (let i = 0; i < BONE_COUNT; i++) {
    const p = rawPositions[i];
    const parentIdx = BONE_PARENTS[i];
    let localPos = [p[0], p[1], p[2]];
    if (parentIdx >= 0) {
      const parentP = rawPositions[parentIdx];
      localPos = [p[0] - parentP[0], p[1] - parentP[1], p[2] - parentP[2]];
    }

    bones.push({
      name: Object.keys(BONES)[i],
      parent: parentIdx,
      restWorldPos: p,
      localPos,
    });

    const bindWorldMat = mat4FromRotationTranslation([0, 0, 0, 1], p);
    invBinds.push(mat4Inverse(bindWorldMat));
  }

  return { bones, invBinds };
}

// =========================================================================
// 3. 科技机甲人型模型网格生成 (高保真肢体与关节球)
// =========================================================================
function createModelSpaceHumanoidMesh(bones: BoneNode[]): Float32Array {
  const verts: number[] = [];

  // 生成圆柱/圆台肢体 (如大小臂、大腿小腿、躯干)
  function addLimb(pA: number[], pB: number[], bone0: number, bone1: number, radiusA: number, radiusB: number) {
    const segs = 14;
    const dx = pB[0] - pA[0], dy = pB[1] - pA[1], dz = pB[2] - pA[2];
    const len = Math.hypot(dx, dy, dz);
    if (len < 1e-4) return;

    const dir = [dx / len, dy / len, dz / len];
    let up = [0, 1, 0];
    if (Math.abs(dir[1]) > 0.9) { up = [1, 0, 0]; }
    const side = [
      up[1] * dir[2] - up[2] * dir[1],
      up[2] * dir[0] - up[0] * dir[2],
      up[0] * dir[1] - up[1] * dir[0],
    ];
    const sLen = 1 / Math.hypot(side[0], side[1], side[2]);
    side[0] *= sLen; side[1] *= sLen; side[2] *= sLen;

    const ortho = [
      dir[1] * side[2] - dir[2] * side[1],
      dir[2] * side[0] - dir[0] * side[2],
      dir[0] * side[1] - dir[1] * side[0],
    ];

    for (let i = 0; i < segs; i++) {
      const a0 = (i / segs) * Math.PI * 2;
      const a1 = ((i + 1) / segs) * Math.PI * 2;
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);

      const r0 = [side[0] * c0 + ortho[0] * s0, side[1] * c0 + ortho[1] * s0, side[2] * c0 + ortho[2] * s0];
      const r1 = [side[0] * c1 + ortho[0] * s1, side[1] * c1 + ortho[1] * s1, side[2] * c1 + ortho[2] * s1];

      const vA0 = [pA[0] + r0[0] * radiusA, pA[1] + r0[1] * radiusA, pA[2] + r0[2] * radiusA];
      const vA1 = [pA[0] + r1[0] * radiusA, pA[1] + r1[1] * radiusA, pA[2] + r1[2] * radiusA];

      const vB0 = [pB[0] + r0[0] * radiusB, pB[1] + r0[1] * radiusB, pB[2] + r0[2] * radiusB];
      const vB1 = [pB[0] + r1[0] * radiusB, pB[1] + r1[1] * radiusB, pB[2] + r1[2] * radiusB];

      // Tri 1
      verts.push(vA0[0], vA0[1], vA0[2], r0[0], r0[1], r0[2], bone0, bone1, 0.85, 0.15);
      verts.push(vB0[0], vB0[1], vB0[2], r0[0], r0[1], r0[2], bone0, bone1, 0.15, 0.85);
      verts.push(vA1[0], vA1[1], vA1[2], r1[0], r1[1], r1[2], bone0, bone1, 0.85, 0.15);

      // Tri 2
      verts.push(vA1[0], vA1[1], vA1[2], r1[0], r1[1], r1[2], bone0, bone1, 0.85, 0.15);
      verts.push(vB0[0], vB0[1], vB0[2], r0[0], r0[1], r0[2], bone0, bone1, 0.15, 0.85);
      verts.push(vB1[0], vB1[1], vB1[2], r1[0], r1[1], r1[2], bone0, bone1, 0.15, 0.85);
    }
  }

  // 关节球 (如肩、肘、膝、髋关节)
  function addJointSphere(center: number[], bone: number, radius: number) {
    const lat = 8, lon = 12;
    for (let j = 0; j < lat; j++) {
      const theta0 = (j / lat) * Math.PI;
      const theta1 = ((j + 1) / lat) * Math.PI;
      for (let i = 0; i < lon; i++) {
        const phi0 = (i / lon) * Math.PI * 2;
        const phi1 = ((i + 1) / lon) * Math.PI * 2;

        const p = (theta: number, phi: number) => {
          const x = Math.sin(theta) * Math.cos(phi);
          const y = Math.cos(theta);
          const z = Math.sin(theta) * Math.sin(phi);
          return [center[0] + x * radius, center[1] + y * radius, center[2] + z * radius, x, y, z];
        };

        const p00 = p(theta0, phi0), p10 = p(theta1, phi0), p01 = p(theta0, phi1), p11 = p(theta1, phi1);

        verts.push(p00[0], p00[1], p00[2], p00[3], p00[4], p00[5], bone, bone, 1.0, 0.0);
        verts.push(p10[0], p10[1], p10[2], p10[3], p10[4], p10[5], bone, bone, 1.0, 0.0);
        verts.push(p01[0], p01[1], p01[2], p01[3], p01[4], p01[5], bone, bone, 1.0, 0.0);

        verts.push(p01[0], p01[1], p01[2], p01[3], p01[4], p01[5], bone, bone, 1.0, 0.0);
        verts.push(p10[0], p10[1], p10[2], p10[3], p10[4], p10[5], bone, bone, 1.0, 0.0);
        verts.push(p11[0], p11[1], p11[2], p11[3], p11[4], p11[5], bone, bone, 1.0, 0.0);
      }
    }
  }

  // 1. 头部造型：球体头部 + 科技护目镜
  const headPos = bones[BONES.Head].restWorldPos;
  addJointSphere([headPos[0], headPos[1] + 0.12, headPos[2]], BONES.Head, 0.22);
  addLimb(bones[BONES.Neck].restWorldPos, [headPos[0], headPos[1] + 0.05, headPos[2]], BONES.Neck, BONES.Head, 0.12, 0.16);

  // 2. 躯干主胸甲与骨盆
  addLimb(bones[BONES.Hips].restWorldPos, bones[BONES.Spine].restWorldPos, BONES.Hips, BONES.Spine, 0.32, 0.30);
  addLimb(bones[BONES.Spine].restWorldPos, bones[BONES.Chest].restWorldPos, BONES.Spine, BONES.Chest, 0.30, 0.35);
  addLimb(bones[BONES.Chest].restWorldPos, bones[BONES.Neck].restWorldPos, BONES.Chest, BONES.Neck, 0.35, 0.16);

  // 3. 上肢：肩球、大臂、肘球、小臂、手腕手掌
  addJointSphere(bones[BONES.LeftArm].restWorldPos, BONES.LeftArm, 0.15);
  addLimb(bones[BONES.Chest].restWorldPos, bones[BONES.LeftArm].restWorldPos, BONES.Chest, BONES.LeftArm, 0.28, 0.15);
  addLimb(bones[BONES.LeftArm].restWorldPos, bones[BONES.LeftForeArm].restWorldPos, BONES.LeftArm, BONES.LeftForeArm, 0.14, 0.12);
  addJointSphere(bones[BONES.LeftForeArm].restWorldPos, BONES.LeftForeArm, 0.13);
  addLimb(bones[BONES.LeftForeArm].restWorldPos, bones[BONES.LeftHand].restWorldPos, BONES.LeftForeArm, BONES.LeftHand, 0.12, 0.10);
  addJointSphere(bones[BONES.LeftHand].restWorldPos, BONES.LeftHand, 0.11);

  addJointSphere(bones[BONES.RightArm].restWorldPos, BONES.RightArm, 0.15);
  addLimb(bones[BONES.Chest].restWorldPos, bones[BONES.RightArm].restWorldPos, BONES.Chest, BONES.RightArm, 0.28, 0.15);
  addLimb(bones[BONES.RightArm].restWorldPos, bones[BONES.RightForeArm].restWorldPos, BONES.RightArm, BONES.RightForeArm, 0.14, 0.12);
  addJointSphere(bones[BONES.RightForeArm].restWorldPos, BONES.RightForeArm, 0.13);
  addLimb(bones[BONES.RightForeArm].restWorldPos, bones[BONES.RightHand].restWorldPos, BONES.RightForeArm, BONES.RightHand, 0.12, 0.10);
  addJointSphere(bones[BONES.RightHand].restWorldPos, BONES.RightHand, 0.11);

  // 4. 下肢：髋球、大腿、膝球、小腿、脚掌
  addJointSphere(bones[BONES.LeftUpLeg].restWorldPos, BONES.LeftUpLeg, 0.18);
  addLimb(bones[BONES.Hips].restWorldPos, bones[BONES.LeftUpLeg].restWorldPos, BONES.Hips, BONES.LeftUpLeg, 0.28, 0.19);
  addLimb(bones[BONES.LeftUpLeg].restWorldPos, bones[BONES.LeftLeg].restWorldPos, BONES.LeftUpLeg, BONES.LeftLeg, 0.19, 0.15);
  addJointSphere(bones[BONES.LeftLeg].restWorldPos, BONES.LeftLeg, 0.16);
  addLimb(bones[BONES.LeftLeg].restWorldPos, bones[BONES.LeftFoot].restWorldPos, BONES.LeftLeg, BONES.LeftFoot, 0.15, 0.13);
  addLimb(bones[BONES.LeftFoot].restWorldPos, [bones[BONES.LeftFoot].restWorldPos[0], bones[BONES.LeftFoot].restWorldPos[1], bones[BONES.LeftFoot].restWorldPos[2] + 0.24], BONES.LeftFoot, BONES.LeftFoot, 0.13, 0.10);

  addJointSphere(bones[BONES.RightUpLeg].restWorldPos, BONES.RightUpLeg, 0.18);
  addLimb(bones[BONES.Hips].restWorldPos, bones[BONES.RightUpLeg].restWorldPos, BONES.Hips, BONES.RightUpLeg, 0.28, 0.19);
  addLimb(bones[BONES.RightUpLeg].restWorldPos, bones[BONES.RightLeg].restWorldPos, BONES.RightUpLeg, BONES.RightLeg, 0.19, 0.15);
  addJointSphere(bones[BONES.RightLeg].restWorldPos, BONES.RightLeg, 0.16);
  addLimb(bones[BONES.RightLeg].restWorldPos, bones[BONES.RightFoot].restWorldPos, BONES.RightLeg, BONES.RightFoot, 0.15, 0.13);
  addLimb(bones[BONES.RightFoot].restWorldPos, [bones[BONES.RightFoot].restWorldPos[0], bones[BONES.RightFoot].restWorldPos[1], bones[BONES.RightFoot].restWorldPos[2] + 0.24], BONES.RightFoot, BONES.RightFoot, 0.13, 0.10);

  return new Float32Array(verts);
}

// =========================================================================
// 4. 主程序入口
// =========================================================================
export function runControlRig(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  gui: GUI
) {
  // 1. 初始化源角色 (标准修长机甲) 与 目标角色 (矮壮重装机甲，腿长减少 35%，肩膀增宽 30%)
  const source = buildSkeleton(1.0, 1.0, 1.0);
  const target = buildSkeleton(1.3, 0.70, 1.15);

  const sourceMesh = createModelSpaceHumanoidMesh(source.bones);
  const targetMesh = createModelSpaceHumanoidMesh(target.bones);

  const sourceVertexBuffer = device.createBuffer({
    size: sourceMesh.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(sourceVertexBuffer, 0, sourceMesh as any);

  const targetVertexBuffer = device.createBuffer({
    size: targetMesh.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(targetVertexBuffer, 0, targetMesh as any);

  // 2. 骨骼矩阵调色板
  const bonePaletteData = new Float32Array(BONE_COUNT * 16 * 2);
  const bonePaletteBuffer = device.createBuffer({
    size: bonePaletteData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // 3. 【核心修复】：为源角色与目标角色创建两个独立的 UniformBuffer，彻底杜绝数据冲撞覆盖！
  const uniformBufferSize = 256;
  const uniformBufferSource = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const uniformBufferTarget = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =========================================================================
  // 5. WGSL 着色器实现
  // =========================================================================
  const rigShaderCode = `
    struct Uniforms {
      viewProj: mat4x4f,
      camPos: vec4f,
      lightDir: vec4f,
      params: vec4f, // x: xOffset, y: charType (0: Source Blue, 1: Target Orange)
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var<storage, read> boneMatrices: array<mat4x4f>;

    struct VertexInput {
      @location(0) pos: vec3f,
      @location(1) normal: vec3f,
      @location(2) boneIdx0: f32,
      @location(3) boneIdx1: f32,
      @location(4) weight0: f32,
      @location(5) weight1: f32,
    };

    struct VertexOut {
      @builtin(position) clipPos: vec4f,
      @location(0) worldPos: vec3f,
      @location(1) normal: vec3f,
      @location(2) charType: f32,
    };

    @vertex
    fn vs_skinned(in: VertexInput) -> VertexOut {
      var out: VertexOut;
      let charIdx = u32(u.params.y);
      let boneOffset = charIdx * ${BONE_COUNT}u;

      let b0 = u32(in.boneIdx0) + boneOffset;
      let b1 = u32(in.boneIdx1) + boneOffset;

      let m0 = boneMatrices[b0];
      let m1 = boneMatrices[b1];

      // 经典双骨骼线性加权蒙皮
      let skinMat = m0 * in.weight0 + m1 * in.weight1;
      var wPos = skinMat * vec4f(in.pos, 1.0);
      wPos.x += u.params.x; // 左右平移分立两侧

      out.clipPos = u.viewProj * wPos;
      out.worldPos = wPos.xyz;
      out.normal = (skinMat * vec4f(in.normal, 0.0)).xyz;
      out.charType = u.params.y;
      return out;
    }

    @fragment
    fn fs_skinned(in: VertexOut) -> @location(0) vec4f {
      let N = normalize(in.normal);
      let L = normalize(u.lightDir.xyz);
      let V = normalize(u.camPos.xyz - in.worldPos);
      let H = normalize(L + V);

      let diff = max(dot(N, L), 0.0);
      let spec = pow(max(dot(N, H), 0.0), 32.0);
      let fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);

      // 左：源角色 (科技蓝机甲) | 右：目标重定向角色 (活力先锋橙)
      let baseColor = select(vec3f(0.20, 0.65, 0.88), vec3f(0.96, 0.46, 0.18), in.charType > 0.5);

      let ambient = vec3f(0.12, 0.16, 0.22) * baseColor;
      let diffuse = baseColor * (diff * 0.72);
      let rim = vec3f(1.0) * fresnel * 0.40;
      let specular = vec3f(0.85) * (spec * 0.40);

      return vec4f(ambient + diffuse + rim + specular, 1.0);
    }

    // 地面着色器
    @vertex
    fn vs_ground(@builtin(vertex_index) vid: u32) -> VertexOut {
      var out: VertexOut;
      let sz = 70.0;
      var pos = vec3f(0.0);
      if (vid == 0u) { pos = vec3f(-sz, 0.0, -sz); }
      else if (vid == 1u) { pos = vec3f(sz, 0.0, -sz); }
      else if (vid == 2u) { pos = vec3f(-sz, 0.0, sz); }
      else if (vid == 3u) { pos = vec3f(-sz, 0.0, sz); }
      else if (vid == 4u) { pos = vec3f(sz, 0.0, -sz); }
      else { pos = vec3f(sz, 0.0, sz); }

      out.clipPos = u.viewProj * vec4f(pos, 1.0);
      out.worldPos = pos;
      out.normal = vec3f(0.0, 1.0, 0.0);
      out.charType = 2.0;
      return out;
    }

    @fragment
    fn fs_ground(in: VertexOut) -> @location(0) vec4f {
      let coord = in.worldPos.xz * 0.5;
      let fw = fwidth(coord);
      let grid = abs(fract(coord - 0.5) - 0.5) / max(fw, vec2f(0.001));
      let line = min(grid.x, grid.y);
      let gridMask = 1.0 - min(line, 1.0);

      let gridCol = vec3f(0.18, 0.28, 0.35);
      let groundCol = mix(vec3f(0.08, 0.12, 0.16), gridCol, gridMask * 0.8);
      return vec4f(groundCol, 1.0);
    }
  `;

  // =========================================================================
  // 6. 创建管线与双 BindGroup
  // =========================================================================
  const shaderModule = device.createShaderModule({ code: rigShaderCode });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const skinnedPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_skinned",
      buffers: [{
        arrayStride: 10 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" },
          { shaderLocation: 2, offset: 24, format: "float32" },
          { shaderLocation: 3, offset: 28, format: "float32" },
          { shaderLocation: 4, offset: 32, format: "float32" },
          { shaderLocation: 5, offset: 36, format: "float32" },
        ],
      }],
    },
    fragment: { module: shaderModule, entryPoint: "fs_skinned", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  const groundPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: shaderModule, entryPoint: "vs_ground" },
    fragment: { module: shaderModule, entryPoint: "fs_ground", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
  });

  // 源角色与目标角色各自独立的 BindGroup
  const bindGroupSource = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBufferSource } },
      { binding: 1, resource: { buffer: bonePaletteBuffer } },
    ],
  });

  const bindGroupTarget = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBufferTarget } },
      { binding: 1, resource: { buffer: bonePaletteBuffer } },
    ],
  });

  // =========================================================================
  // 7. GUI 控制与动作片段解算
  // =========================================================================
  const settings = {
    animationClip: "Running",
    scaleRootMotion: true,
    lookAtTargetX: 0.0,
    lookAtTargetY: 3.2,
  };

  gui.title("WebGPU Control Rig 动画重定向");
  gui.add(settings, "animationClip", ["Running", "Walking", "MartialKick", "LookAtIK"]).name("动画片段选择");

  const retargetFolder = gui.addFolder("IK Retargeter 重定向配置");
  retargetFolder.add(settings, "scaleRootMotion").name("步长与骨盆高度自适应");
  retargetFolder.open();

  const ikFolder = gui.addFolder("Control Rig 实时注视 (Look-At)");
  ikFolder.add(settings, "lookAtTargetX", -3.0, 3.0, 0.1).name("注视 X");
  ikFolder.add(settings, "lookAtTargetY", 1.0, 6.0, 0.1).name("注视 Y");
  ikFolder.open();

  const camera = { target: [0, 2.1, 0], radius: 9.0, theta: 45.0, phi: 16.0 };
  let isDragging = false, dragButton = 0, lastX = 0, lastY = 0;

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    isDragging = true;
    dragButton = e.button;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!isDragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    if (dragButton === 0) {
      camera.theta -= dx * 0.35;
      camera.phi = Math.max(5, Math.min(85, camera.phi + dy * 0.35));
    } else if (dragButton === 2) {
      const radTheta = (camera.theta * Math.PI) / 180;
      const pan = camera.radius * 0.0016;
      camera.target[0] -= Math.cos(radTheta) * dx * pan;
      camera.target[2] -= -Math.sin(radTheta) * dx * pan;
      camera.target[1] += dy * pan;
    }
  });

  canvas.addEventListener("pointerup", (e) => {
    isDragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.radius = Math.max(3, Math.min(30, camera.radius * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });

  // 动画片段求值
  function evaluateClip(clip: string, time: number): { [key: number]: number[] } {
    const rots: { [key: number]: number[] } = {};

    if (clip === "Running") {
      const speed = time * 7.5;
      const legL = Math.sin(speed) * 0.72;
      const legR = -legL;

      rots[BONES.Hips] = quatFromAxisAngle([0, 1, 0], Math.sin(speed) * 0.15);
      rots[BONES.LeftUpLeg] = quatFromAxisAngle([1, 0, 0], legL);
      rots[BONES.LeftLeg] = quatFromAxisAngle([1, 0, 0], Math.max(0, -legL * 1.35));
      rots[BONES.RightUpLeg] = quatFromAxisAngle([1, 0, 0], legR);
      rots[BONES.RightLeg] = quatFromAxisAngle([1, 0, 0], Math.max(0, -legR * 1.35));

      // 双臂反向自然摆动
      rots[BONES.LeftArm] = quatFromAxisAngle([1, 0, 0], -legL * 0.85);
      rots[BONES.LeftForeArm] = quatFromAxisAngle([1, 0, 0], 0.65);
      rots[BONES.RightArm] = quatFromAxisAngle([1, 0, 0], -legR * 0.85);
      rots[BONES.RightForeArm] = quatFromAxisAngle([1, 0, 0], 0.65);
    } else if (clip === "Walking") {
      const speed = time * 4.2;
      const legL = Math.sin(speed) * 0.45;
      rots[BONES.LeftUpLeg] = quatFromAxisAngle([1, 0, 0], legL);
      rots[BONES.LeftLeg] = quatFromAxisAngle([1, 0, 0], Math.max(0, -legL * 0.85));
      rots[BONES.RightUpLeg] = quatFromAxisAngle([1, 0, 0], -legL);
      rots[BONES.RightLeg] = quatFromAxisAngle([1, 0, 0], Math.max(0, legL * 0.85));

      rots[BONES.LeftArm] = quatFromAxisAngle([1, 0, 0], -legL * 0.6);
      rots[BONES.RightArm] = quatFromAxisAngle([1, 0, 0], legL * 0.6);
    } else if (clip === "MartialKick") {
      const t = (Math.sin(time * 3.2) + 1.0) * 0.5;
      rots[BONES.Spine] = quatFromAxisAngle([0, 0, 1], t * -0.35);
      rots[BONES.RightUpLeg] = quatFromAxisAngle([0, 0, 1], t * 1.75); // 高鞭腿侧踢
      rots[BONES.RightLeg] = quatFromAxisAngle([0, 0, 1], t * 0.35);
      rots[BONES.LeftUpLeg] = quatFromAxisAngle([0, 0, 1], t * 0.20);
    } else {
      // LookAtIK: 头部与胸椎实时追踪 LookAt 目标
      const lookDir = [settings.lookAtTargetX, settings.lookAtTargetY, 3.0];
      const yaw = Math.atan2(lookDir[0], lookDir[2]) * 0.65;
      const pitch = -Math.atan2(lookDir[1] - 3.5, 3.0) * 0.65;

      const qYaw = quatFromAxisAngle([0, 1, 0], yaw);
      const qPitch = quatFromAxisAngle([1, 0, 0], pitch);
      const qLook = quatMultiply(qYaw, qPitch);

      rots[BONES.Head] = qLook;
      rots[BONES.Chest] = quatSlerp([0, 0, 0, 1], qLook, 0.40);
    }

    return rots;
  }

  // 严格基于标准数学解算骨骼世界变换与最终蒙皮矩阵
  function solveSkinning(
    skel: BoneNode[],
    invBinds: Float32Array[],
    animRots: { [key: number]: number[] }
  ): Float32Array {
    const worldMats: Float32Array[] = [];
    const skinPalette = new Float32Array(BONE_COUNT * 16);

    for (let i = 0; i < BONE_COUNT; i++) {
      const b = skel[i];
      const rot = animRots[i] || [0, 0, 0, 1];
      const localMat = mat4FromRotationTranslation(rot, b.localPos);

      var worldMat = localMat;
      if (b.parent >= 0) {
        worldMat = mat4Multiply(worldMats[b.parent], localMat);
      }
      worldMats.push(worldMat);

      const skinMat = mat4Multiply(worldMat, invBinds[i]);
      skinPalette.set(skinMat, i * 16);
    }

    return skinPalette;
  }

  // =========================================================================
  // 8. 渲染循环
  // =========================================================================
  let depthTexture: GPUTexture | null = null;
  let animId: number;
  let startTime = performance.now();

  function frame() {
    const now = performance.now();
    const elapsed = (now - startTime) * 0.001;

    const dpr = window.devicePixelRatio || 1;
    const renderWidth = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const renderHeight = Math.max(1, Math.floor(canvas.clientHeight * dpr));

    if (!depthTexture || depthTexture.width !== renderWidth || depthTexture.height !== renderHeight) {
      canvas.width = renderWidth;
      canvas.height = renderHeight;
      if (depthTexture) depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [renderWidth, renderHeight],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    // 1. 评估源动画姿态并完成 IK Retargeter 重定向
    const sourceAnimRots = evaluateClip(settings.animationClip, elapsed);
    const targetAnimRots = { ...sourceAnimRots };

    // 2. 求解两组角色的蒙皮矩阵
    const sourcePalette = solveSkinning(source.bones, source.invBinds, sourceAnimRots);
    const targetPalette = solveSkinning(target.bones, target.invBinds, targetAnimRots);

    bonePaletteData.set(sourcePalette, 0);
    bonePaletteData.set(targetPalette, BONE_COUNT * 16);
    device.queue.writeBuffer(bonePaletteBuffer, 0, bonePaletteData);

    // 3. 相机矩阵计算
    const radTheta = (camera.theta * Math.PI) / 180;
    const radPhi = (camera.phi * Math.PI) / 180;
    const eye = [
      camera.target[0] + camera.radius * Math.cos(radPhi) * Math.sin(radTheta),
      camera.target[1] + camera.radius * Math.sin(radPhi),
      camera.target[2] + camera.radius * Math.cos(radPhi) * Math.cos(radTheta),
    ];
    const proj = mat4Perspective((45 * Math.PI) / 180, renderWidth / renderHeight, 0.1, 100.0);
    const view = mat4LookAt(eye, camera.target, [0, 1, 0]);
    const viewProj = mat4Multiply(proj, view);

    // 4. 【核心修复】：为 Source 与 Target 分别写入独立的 UniformBuffer 数据，彻底分离位置与样式！
    const uniformDataSource = new Float32Array(uniformBufferSize / 4);
    uniformDataSource.set(viewProj, 0);
    uniformDataSource[16] = eye[0]; uniformDataSource[17] = eye[1]; uniformDataSource[18] = eye[2]; uniformDataSource[19] = 1.0;
    uniformDataSource[20] = 0.55; uniformDataSource[21] = 0.85; uniformDataSource[22] = 0.40; uniformDataSource[23] = 0.0;
    uniformDataSource[24] = -2.3; // xOffset: 左侧
    uniformDataSource[25] = 0.0;  // charType: 0 (科技蓝)
    device.queue.writeBuffer(uniformBufferSource, 0, uniformDataSource);

    const uniformDataTarget = new Float32Array(uniformBufferSize / 4);
    uniformDataTarget.set(viewProj, 0);
    uniformDataTarget[16] = eye[0]; uniformDataTarget[17] = eye[1]; uniformDataTarget[18] = eye[2]; uniformDataTarget[19] = 1.0;
    uniformDataTarget[20] = 0.55; uniformDataTarget[21] = 0.85; uniformDataTarget[22] = 0.40; uniformDataTarget[23] = 0.0;
    uniformDataTarget[24] = 2.3;  // xOffset: 右侧
    uniformDataTarget[25] = 1.0;  // charType: 1 (先锋橙)
    device.queue.writeBuffer(uniformBufferTarget, 0, uniformDataTarget);

    const encoder = device.createCommandEncoder();
    const currentView = context.getCurrentTexture().createView();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: currentView,
        clearValue: { r: 0.08, g: 0.11, b: 0.15, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    // 绘制地面
    renderPass.setPipeline(groundPipeline);
    renderPass.setBindGroup(0, bindGroupSource);
    renderPass.draw(6, 1, 0, 0);

    // 1. 绘制左侧源角色 (科技蓝修长机甲)
    renderPass.setPipeline(skinnedPipeline);
    renderPass.setBindGroup(0, bindGroupSource);
    renderPass.setVertexBuffer(0, sourceVertexBuffer);
    renderPass.draw(sourceMesh.length / 10, 1, 0, 0);

    // 2. 绘制右侧重定向目标角色 (先锋橙矮壮机甲)
    renderPass.setBindGroup(0, bindGroupTarget);
    renderPass.setVertexBuffer(0, targetVertexBuffer);
    renderPass.draw(targetMesh.length / 10, 1, 0, 0);

    renderPass.end();
    device.queue.submit([encoder.finish()]);

    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(animId);
    sourceVertexBuffer.destroy();
    targetVertexBuffer.destroy();
    uniformBufferSource.destroy();
    uniformBufferTarget.destroy();
    bonePaletteBuffer.destroy();
    if (depthTexture) depthTexture.destroy();
  };
}