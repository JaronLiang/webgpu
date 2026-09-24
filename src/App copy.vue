<script setup lang="ts">
/// <reference types="@webgpu/types" />
import { ref, onMounted, onUnmounted, nextTick } from "vue";
import { runTriangle } from "./examples/triangle";
import { runTexture } from "./examples/texture";
import { runCube } from "./examples/cube";
import {  runPingPong } from "./examples/pingPong";
import { runComputeParticles } from "./examples/computeParticles";

// 1. 定义用例类型规范，方便后续扩展模块
export interface DemoItem {
  id: string;
  name: string;
  // 统一的渲染入口函数规范
  run: (
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    canvas: HTMLCanvasElement
  ) => (() => void) | void;
}

export interface DemoCategory {
  id: string;
  name: string;
  children: DemoItem[];
}

// 2. 多级分类用例配置列表 (后续新增用例直接在这里添加)
const demoCategories: DemoCategory[] = [
  {
    id: "geometry",
    name: "几何功能",
    children: [
      {
        id: "triangle",
        name: "彩色三角形",
        run: (device, context, format) => runTriangle(device, context, format),
      },
      {
        id: "cube",
        name: "3D 旋转立方体",
        run: (device, context, format, canvas) => runCube(device, context, format, canvas),
      },
    ],
  },
  {
    id: "texture",
    name: "纹理功能",
    children: [
      {
        id: "checkerboard",
        name: "棋盘纹理贴图",
        run: (device, context, format) => runTexture(device, context, format),
      },
      {
        id: "pingpong",
        name: "乒乓渲染",
        run: (device, context, format) => runPingPong(device, context, format),
      },
      
      {
        id: "particle",
        name: "粒子",
        run: (device, context, format) => runComputeParticles(device, context, format),
      },
      
      // 后续如需增加压缩纹理、立方体贴图等，可直接在此扩展：
      // { id: "cube-texture", name: "天空盒纹理", run: ... }
    ],
  },
];

// 状态控制
const currentDemoId = ref<string>("triangle");
const currentDemoTitle = ref<string>("彩色三角形");
const canvasContainerRef = ref<HTMLDivElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);

let device: GPUDevice;
let context: GPUCanvasContext;
let format: GPUTextureFormat;
let currentCleanup: (() => void) | null = null;
let resizeObserver: ResizeObserver | null = null;

// 初始化 WebGPU
async function initWebGPU() {
  if (!navigator.gpu) {
    alert("当前浏览器不支持 WebGPU，请使用最新版 Chrome/Edge 并开启硬件加速");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    alert("未找到适用的 GPU 适配器");
    return;
  }
  device = await adapter.requestDevice();

  const canvas = canvasRef.value!;
  context = canvas.getContext("webgpu") as unknown as GPUCanvasContext;
  format = navigator.gpu.getPreferredCanvasFormat();

  // 尺寸初始化
  resizeCanvas();

  // 监听容器大小改变
  resizeObserver = new ResizeObserver(() => {
    resizeCanvas();
  });
  if (canvasContainerRef.value) {
    resizeObserver.observe(canvasContainerRef.value);
  }

  // 默认启动第一个用例
  const firstCategory = demoCategories[0];
  if (firstCategory && firstCategory.children.length > 0) {
    switchDemo(firstCategory.children[0]);
  }
}

// 调整 Canvas 像素大小，使其保持与容器 1:1 或根据设备 DPI 缩放
function resizeCanvas() {
  if (!canvasRef.value || !canvasContainerRef.value) return;
  const { clientWidth, clientHeight } = canvasContainerRef.value;

  // 避免隐藏或尺寸为 0 时重设
  if (clientWidth === 0 || clientHeight === 0) return;

  canvasRef.value.width = clientWidth * window.devicePixelRatio;
  canvasRef.value.height = clientHeight * window.devicePixelRatio;

  if (device && context) {
    context.configure({
      device,
      format,
      alphaMode: "premultiplied",
    });
  }
}

// 切换用例
async  function switchDemo(item: DemoItem) {
  currentDemoId.value = item.id;
  currentDemoTitle.value = item.name;

  const canvas = canvasRef.value!;

  // 1. 释放上一个用例的定时器/RAF以及显存
  if (currentCleanup) {
    currentCleanup();
    currentCleanup = null;
  }

  // 2. 重新配置当前画布尺寸并执行新的用例
  resizeCanvas();
  const cleanup = item.run(device, context, format, canvas);
  if (typeof cleanup === "function") {
    currentCleanup = cleanup;
  }
}

onMounted(() => {
  nextTick(() => {
    initWebGPU();
  });
});

onUnmounted(() => {
  if (currentCleanup) currentCleanup();
  if (resizeObserver) resizeObserver.disconnect();
});
</script>

<template>
  <div class="layout">
    <!-- 左侧分层导航栏 -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <h1 class="logo">WebGPU Lab</h1>
      </div>

      <nav class="nav-tree">
        <div
          v-for="category in demoCategories"
          :key="category.id"
          class="category-group"
        >
          <!-- 一级分类标题 -->
          <div class="category-title">
            <span class="folder-icon">📂</span>
            <span>{{ category.name }}</span>
          </div>

          <!-- 二级用例列表 -->
          <ul class="sub-list">
            <li
              v-for="item in category.children"
              :key="item.id"
              class="sub-item"
              :class="{ active: currentDemoId === item.id }"
              @click="switchDemo(item)"
            >
              <span class="dot"></span>
              <span class="item-name">{{ item.name }}</span>
            </li>
          </ul>
        </div>
      </nav>
    </aside>

    <!-- 右侧 WebGPU 主视图 -->
    <main class="main-content">
      <header class="top-bar">
        <span class="label">当前用例:</span>
        <span class="active-name">{{ currentDemoTitle }}</span>
      </header>
      <div class="canvas-container" ref="canvasContainerRef">
        <canvas ref="canvasRef"></canvas>
      </div>
    </main>
  </div>
</template>

<style scoped>
/* 全局重置与主布局 */
.layout {
  display: flex;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background-color: #0e0e10;
  color: #e4e4e7;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
    Arial, sans-serif;
}

/* 左侧导航栏 */
.sidebar {
  width: 240px;
  min-width: 240px;
  background-color: #18181b;
  border-right: 1px solid #27272a;
  display: flex;
  flex-direction: column;
  user-select: none;
}

.sidebar-header {
  height: 56px;
  display: flex;
  align-items: center;
  padding: 0 20px;
  border-bottom: 1px solid #27272a;
}

.logo {
  font-size: 16px;
  font-weight: 600;
  letter-spacing: 0.5px;
  color: #38bdf8;
  margin: 0;
}

.nav-tree {
  flex: 1;
  overflow-y: auto;
  padding: 16px 10px;
}

/* 一级目录大类 */
.category-group {
  margin-bottom: 20px;
}

.category-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  color: #71717a;
  padding: 6px 10px;
  letter-spacing: 0.5px;
}

.folder-icon {
  font-size: 13px;
}

/* 二级目录用例 */
.sub-list {
  list-style: none;
  margin: 4px 0 0 0;
  padding: 0;
}

.sub-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
  margin: 2px 0;
  border-radius: 6px;
  font-size: 13.5px;
  color: #a1a1aa;
  cursor: pointer;
  transition: all 0.15s ease;
}

.sub-item:hover {
  background-color: #27272a;
  color: #f4f4f5;
}

.sub-item.active {
  background-color: #0284c7;
  color: #ffffff;
  font-weight: 500;
}

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background-color: #52525b;
}

.sub-item.active .dot {
  background-color: #ffffff;
}

/* 右侧画布区域 */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #09090b;
}

.top-bar {
  height: 56px;
  border-bottom: 1px solid #27272a;
  display: flex;
  align-items: center;
  padding: 0 24px;
  background-color: #121214;
  gap: 8px;
}

.label {
  font-size: 13px;
  color: #71717a;
}

.active-name {
  font-size: 14px;
  font-weight: 500;
  color: #f4f4f5;
}

.canvas-container {
  flex: 1;
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

canvas {
  width: 100%;
  height: 100%;
  display: block;
}
</style>