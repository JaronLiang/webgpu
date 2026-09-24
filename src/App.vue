<script setup lang="ts">
/// <reference types="@webgpu/types" />
import { ref, onMounted, onUnmounted, nextTick } from "vue";
import { SimpleGUI } from "./utils/gui";

import { runTriangle } from "./examples/triangle";
import { runTexture } from "./examples/texture";
import { runCube } from "./examples/cube";
import { runPingPong } from "./examples/pingPong";
import { runComputeParticles } from "./examples/computeParticles";
import { runComputeBarrier } from "./examples/computeBarrier";
import { runSkybox } from "./examples/skybox";
import { runEarlyZ } from "./examples/earlyZ";
import { runDynamicShadowMap } from "./examples/shadowMap";
import { runVolumeRendering } from "./examples/volumeRendering";
import { runTexture3D } from "./examples/texture3D";
import { runOcclusionQuery } from "./examples/occlusionQuery";
import { runMRT } from "./examples/MRT";
import { runReversedZ } from "./examples/reversedZ";
import { runOffscreen } from "./examples/offscreen";
import { runMultiplePipelines } from "./examples/multiplePipelines";
import { runCamera } from "./examples/camera";
import { runInstanceDraw } from "./examples/instanceDraw";
import { runGLTFPBR } from "./examples/gltfPBR";
import { runMemoryAlignment } from "./examples/alignment";
import { runSubgroup } from "./examples/subgroup";
import { runDrawIndexedIndirect } from "./examples/drawIndexedIndirect";
import { runDrawIndirect } from "./examples/drawIndirect";
import { runVolumetricFog } from "./examples/volumetricFog";
import { runAtmosphericScattering } from "./examples/atmosphericScattering";
import { runVolumetricLight } from "./examples/runVolumetricLight";
import { runRaymarching } from "./examples/raymarching";
import {  runGIOptimized } from "./examples/gi";
import { runWeatherRainSnow } from "./examples/weatherRainSnow";
import { runWeatherSkyEnvironments } from "./examples/weatherSkyEnvironments";
import { runIBL } from "./examples/ibl";
import { runOpaqueAndTransparent } from "./examples/opaqueAndTransparent";
import { runWater } from "./examples/water";
import { runSoftShadow } from "./examples/softShadow";
import { runLightTypesShowcase } from "./examples/lightTypesShowcase";
import { runCSMShowcase } from "./examples/csm";
import { runVideoTextureShowcase } from "./examples/VideoTextureShowcase";
import { runImageMarkerShowcase } from "./examples/ImageMarkerShowcase";
import { runTextBillboardShowcase } from "./examples/TextBillboardShowcase";
import { runPureTextBillboardShowcase } from "./examples/PureTextBillboardShowcase";
import { runInteractiveMarkerShowcase } from "./examples/TextureFromCanvas";
import { runBatchMarkersShowcase } from "./examples/BatchMarkersShowcase";
import { runTextureAnimationShowcase } from "./examples/TextureAnimationShowcase";
import { runFireTextureAnimationShowcase } from "./examples/fireAnimationShowcase";
import { runFluidSimulation } from "./examples/FluidSimulation";
import { runGaussianSplatting } from "./examples/GaussianSplatting";

export type CleanupFunction = () => void;

// 1. 【扩展签名】：让 run 接收统一注入的 gui 对象
export interface DemoItem {
  id: string;
  name: string;
  run: (
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    canvas: HTMLCanvasElement,
    gui: SimpleGUI // <-- 注入 GUI
  ) => Promise<CleanupFunction | void> | CleanupFunction | void;
}

export interface DemoCategory {
  id: string;
  name: string;
  children: DemoItem[];
}

const demoCategories: DemoCategory[] = [
  {
    id: "geometry",
    name: "几何功能",
    children: [
      { id: "triangle", name: "彩色三角形", run: (d, c, f) => runTriangle(d, c, f) },
      { id: "cube", name: "3D 旋转立方体", run: (d, c, f, cv) => runCube(d, c, f, cv) },
      { id: "camera", name: "Camera 交互与 GUI 调节", run: (d, c, f, cv, gui) => runCamera(d, c, f, cv, gui) },
      { 
      id: "instanceDraw", 
      name: "10000个正方体 (Instance)", 
      run: (d, c, f, cv, gui) => runInstanceDraw(d, c, f, cv, gui) 
    },
      { id: "runMemoryAlignment", name: "内存对齐", run: (d, c, f) => runMemoryAlignment(d, c, f) },
      { id: "runSubgroup", name: "subGroup", run: (d, c, f) => runSubgroup(d, c, f) },
      { id: "runDrawIndexedIndirect", name: "间接索引绘制", run: (d, c, f) => runDrawIndexedIndirect(d, c, f) },
      { id: "runDrawIndirect", name: "间接绘制", run: (d, c, f) => runDrawIndirect(d, c, f) },     
    ],
  },
  {
  id: "gltfModel",
  name: "glTF 模型与材质",
  children: [
    {
      id: "gltfPBR",
      name: "DamagedHelmet (glTF PBR)",
      run: (d, c, f, cv, gui) => runGLTFPBR(d, c, f, cv, gui),
    },
     {
      id: "opaqueAndTransparent",
      name: "半透明物体",
      run: (d, c, f, cv, gui) => runOpaqueAndTransparent(d, c, f),
    },
    {
      id: "ibl",
      name: "基于图像的光照",
      run: (d, c, f, cv, gui) => runIBL(d, c, f),
    },
  {
      id: "GaussianSplat",
      name: "高斯",
      run: (d, c, f, cv, gui) => runGaussianSplatting(d, c, f),
    },


    
  ],
},
  {
    id: "texture",
    name: "纹理与高级计算",
    children: [
      { id: "checkerboard", name: "棋盘纹理贴图", run: (d, c, f) => runTexture(d, c, f) },
      { id: "pingpong", name: "乒乓渲染", run: (d, c, f) => runPingPong(d, c, f) },
      { id: "particle", name: "Compute 粒子系统", run: (d, c, f) => runComputeParticles(d, c, f) },
      {
        id: "barrier",
        name: "Compute 屏障同步 (控制台)",
        run: async (d, c, f) => {
          const enc = d.createCommandEncoder();
          const pass = enc.beginRenderPass({
            colorAttachments: [{ view: c.getCurrentTexture().createView(), clearValue: { r: 0.05, g: 0.08, b: 0.15, a: 1.0 }, loadOp: "clear", storeOp: "store" }],
          });
          pass.end();
          d.queue.submit([enc.finish()]);
          return await runComputeBarrier(d, c, f);
        },
      },
      { id: "skybox", name: "3D 全景天空盒 (Cubemap)", run: (d, c, f, cv) => runSkybox(d, c, f, cv) },
      { id: "earlyz", name: "Early-Z 提前测试", run: (d, c, f) => runEarlyZ(d, c, f) },
      { id: "shadow", name: "动态光照阴影 (ShadowMap)", run: (d, c, f) => runDynamicShadowMap(d, c, f) },
         { id: "sofashadow", name: "软阴影 ", run: (d, c, f) => runSoftShadow(d, c, f) },
   { id: "LightTypesShowcase", name: "光源 ", run: (d, c, f, cv, gui) => runLightTypesShowcase(d, c, f, cv, gui) },
 { id: "csm", name: "csm ", run: (d, c, f, cv, gui) => runCSMShowcase(d, c, f, cv, gui) },

     { id: "video", name: "视频材质 ", run: (d, c, f, cv, gui) => runVideoTextureShowcase(d, c, f, cv, gui) },

         
      { id: "texture3d", name: "3D 空间体积纹理", run: (d, c, f, cv) => runTexture3D(d, c, f, cv) },
      { id: "occlusionQuery", name: "Occlusion Query 遮挡查询", run: (d, c, f, cv) => runOcclusionQuery(d, c, f, cv) },
      { id: "reversedZ", name: "Reversed-Z 精度对比", run: (d, c, f, cv) => runReversedZ(d, c, f, cv) },
      { id: "MRT", name: "MRT 多渲染目标 G-Buffer", run: (d, c, f, cv) => runMRT(d, c, f, cv) },
      { id: "offscreen", name: "离屏多通道渲染", run: (d, c, f, cv) => runOffscreen(d, c, f, cv) },
      { id: "multiplePipelines", name: "Compute-Render 交互", run: (d, c, f, cv) => runMultiplePipelines(d, c, f, cv) },
  { id: "TextureAnimation", name: "动态纹理动画", run: (d, c, f, cv,gui) => runTextureAnimationShowcase(d, c, f, cv,gui) },
{ id: "fireAnimation", name: "火焰", run: (d, c, f, cv,gui) => runFireTextureAnimationShowcase(d, c, f, cv,gui) },

      
    ],
  },
  {
    id: "volume",
    name: "体渲染",
    children: [
      { id: "volumerun", name: "Raymarching 动态云雾", run: (d, c, f, cv) => runVolumeRendering(d, c, f, cv) },
       { id: "VolumetricFog", name: "体积雾", run: (d, c, f, cv) => runVolumetricFog(d, c, f) },
        { id: "AtmosphericScattering", name: "大气渲染", run: (d, c, f, cv) => runAtmosphericScattering(d, c, f) },
    { id: "VolumetricLight", name: "体积光", run: (d, c, f, cv) => runVolumetricLight(d, c, f) },
 { id: "raymarching", name: "光线步进", run: (d, c, f, cv) => runRaymarching(d, c, f) },
{ id: "runGI", name: "全局gi", run: (d, c, f, cv) => runGIOptimized(d, c, f) },
{ id: "water", name: "水体", run: (d, c, f, cv) => runWater(d, c, f) },
{ id: "FluidSimulation", name: "流体模拟", run: (d, c, f, cv,gui) => runFluidSimulation(d, c, f) },
        
      
    ],
  },
    {
  id: "weather",
  name: "天气环境",
  children: [
    {
      id: "weatherRainSnow",
      name: "雨雪 ",
      run: (d, c, f, cv, gui) => runWeatherRainSnow(d, c, f),
    },
    {
      id: "Environments",
      name: "环境 ",
      run: (d, c, f, cv, gui) => runWeatherSkyEnvironments(d, c, f),
    },

    
    
    
  ],
},

   {
  id: "other",
  name: "覆盖物",
  children: [
    {
      id: "billboard",
      name: "标注 ",
      run: (d, c, f, cv, gui) => runImageMarkerShowcase(d, c, f, cv, gui),
    },
    {
      id: "text",
      name: "文字标签 ",
      run: (d, c, f, cv, gui) => runTextBillboardShowcase(d, c, f, cv, gui),
    },
       {
      id: "textbill",
      name: "纯文字 ",
      run: (d, c, f, cv, gui) => runPureTextBillboardShowcase(d, c, f,cv, gui),
    },
      {
      id: "message",
      name: "信息框 ",
      run: (d, c, f, cv, gui) => runInteractiveMarkerShowcase(d, c, f,cv, gui),
    },
     {
      id: "batch",
      name: "合并批次 ",
      run: (d, c, f, cv, gui) => runBatchMarkersShowcase(d, c, f,cv, gui),
    },

  ],
},

];

// 状态控制
const currentDemoId = ref<string>("triangle");
const currentDemoTitle = ref<string>("彩色三角形");
const canvasContainerRef = ref<HTMLDivElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const guiContainerRef = ref<HTMLDivElement | null>(null);

let device: GPUDevice;
let context: GPUCanvasContext;
let format: GPUTextureFormat;
let currentCleanup: CleanupFunction | null = null;
let resizeObserver: ResizeObserver | null = null;
let guiInstance: SimpleGUI | null = null;

function getSafeCanvas(): HTMLCanvasElement | null {
  return canvasRef.value || document.querySelector("canvas");
}

function resizeCanvas() {
  const canvas = getSafeCanvas();
  if (!canvas || !canvasContainerRef.value) return;

  const { clientWidth, clientHeight } = canvasContainerRef.value;
  if (clientWidth === 0 || clientHeight === 0) return;

  canvas.width = clientWidth * window.devicePixelRatio;
  canvas.height = clientHeight * window.devicePixelRatio;

  if (device && context) {
    context.configure({ device, format, alphaMode: "premultiplied" });
  }
}

async function initWebGPU() {
  if (!navigator.gpu) {
    alert("当前浏览器不支持 WebGPU");
    return;
  }

  await nextTick();
  const canvas = getSafeCanvas();
  if (!canvas) return;

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return;
  device = await adapter.requestDevice();

  context = canvas.getContext("webgpu") as unknown as GPUCanvasContext;
  format = navigator.gpu.getPreferredCanvasFormat();

  // 初始化 GUI 单例
  if (guiContainerRef.value) {
    guiInstance = new SimpleGUI(guiContainerRef.value);
  }

  resizeCanvas();

  if (canvasContainerRef.value) {
    resizeObserver = new ResizeObserver(() => resizeCanvas());
    resizeObserver.observe(canvasContainerRef.value);
  }

  const firstCategory = demoCategories[0];
  if (firstCategory && firstCategory.children.length > 0) {
    await switchDemo(firstCategory.children[0]);
  }
}

async function switchDemo(item: DemoItem) {
  currentDemoId.value = item.id;
  currentDemoTitle.value = item.name;

  const canvas = getSafeCanvas();
  if (!canvas || !device || !context) return;

  // 释放上一用例显存与定时器
  if (currentCleanup) {
    currentCleanup();
    currentCleanup = null;
  }

  // 【核心】：切用例时自动清空 GUI 容器并隐藏，无需各用例手动管理
  if (guiInstance) {
    guiInstance.clear();
  }

  resizeCanvas();

  // 传入 guiInstance
  const cleanup = await item.run(device, context, format, canvas, guiInstance!);
  if (typeof cleanup === "function") {
    currentCleanup = cleanup;
  }
}

onMounted(() => initWebGPU());
onUnmounted(() => {
  if (currentCleanup) currentCleanup();
  if (resizeObserver) resizeObserver.disconnect();
  if (guiInstance) guiInstance.clear();
});
</script>

<template>
  <div class="layout">
    <!-- 侧边导航栏保持不变 -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <h1 class="logo">WebGPU Lab</h1>
      </div>
      <nav class="nav-tree">
        <div v-for="category in demoCategories" :key="category.id" class="category-group">
          <div class="category-title">
            <span class="folder-icon">📂</span>
            <span>{{ category.name }}</span>
          </div>
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

    <!-- 主画布视口 -->
    <main class="main-content">
      <header class="top-bar">
        <span class="label">当前用例:</span>
        <span class="active-name">{{ currentDemoTitle }}</span>
      </header>

      <div class="canvas-container" ref="canvasContainerRef">
        <canvas ref="canvasRef"></canvas>

        <!-- 【核心】：全局统一的 GUI 悬浮挂载面板 -->
        <div class="gui-panel" ref="guiContainerRef" style="display: none;"></div>
      </div>
    </main>
  </div>
</template>

<style scoped>
/* 此处保留你原有的全部样式，只增加 .gui-panel 即可 */
.layout { display: flex; width: 100vw; height: 100vh; overflow: hidden; background-color: #0e0e10; color: #e4e4e7; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.sidebar { width: 250px; min-width: 250px; background-color: #18181b; border-right: 1px solid #27272a; display: flex; flex-direction: column; user-select: none; }
.sidebar-header { height: 56px; display: flex; align-items: center; padding: 0 20px; border-bottom: 1px solid #27272a; }
.logo { font-size: 16px; font-weight: 600; color: #38bdf8; margin: 0; }
.nav-tree { flex: 1; overflow-y: auto; padding: 16px 10px; }
.category-group { margin-bottom: 20px; }
.category-title { display: flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 600; text-transform: uppercase; color: #71717a; padding: 6px 10px; }
.folder-icon { font-size: 13px; }
.sub-list { list-style: none; margin: 4px 0 0 0; padding: 0; }
.sub-item { display: flex; align-items: center; gap: 10px; padding: 8px 14px; margin: 2px 0; border-radius: 6px; font-size: 13.5px; color: #a1a1aa; cursor: pointer; transition: all 0.15s ease; }
.sub-item:hover { background-color: #27272a; color: #f4f4f5; }
.sub-item.active { background-color: #0284c7; color: #ffffff; font-weight: 500; }
.dot { width: 6px; height: 6px; border-radius: 50%; background-color: #52525b; }
.sub-item.active .dot { background-color: #ffffff; }
.main-content { flex: 1; display: flex; flex-direction: column; height: 100%; background: #09090b; }
.top-bar { height: 56px; border-bottom: 1px solid #27272a; display: flex; align-items: center; padding: 0 24px; background-color: #121214; gap: 8px; }
.label { font-size: 13px; color: #71717a; }
.active-name { font-size: 14px; font-weight: 500; color: #f4f4f5; }
.canvas-container { flex: 1; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center; }
canvas { width: 100%; height: 100%; display: block; }

/* 【新增】：GUI 面板容器样式 */
.gui-panel {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 220px;
  background: rgba(24, 24, 27, 0.85);
  backdrop-filter: blur(12px);
  border: 1px solid #3f3f46;
  border-radius: 8px;
  padding: 14px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
  color: #f4f4f5;
  font-size: 12px;
  font-family: monospace;
  z-index: 10;
  user-select: none;
}
</style>