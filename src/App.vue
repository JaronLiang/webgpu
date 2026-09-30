<script setup lang="ts">
/// <reference types="@webgpu/types" />
import { ref, onMounted, onUnmounted, nextTick } from "vue";
import GUI from "lil-gui";
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
import { runGIOptimized } from "./examples/gi";
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
import { runSampler } from "./examples/runSampler";
import { runMipSampler } from "./examples/mipmap";
import { runHdrSkybox } from "./examples/hdrSkybox";
import { runGpuCulling } from "./examples/gpuCulling";
import { runHizOcclusionCulling } from "./examples/hizOcclusionCulling";
import { runDepthTest } from "./examples/depth_test";
import { runRenderToTexture } from "./examples/render_to_texture";
import { runAIInference } from "./examples/aiInferenceGPU";
import { runAIInferenceExt } from "./examples/aiInferenceGPUExt";
import { runFXAA } from "./examples/aa_fxaa";
import { runMSAA } from "./examples/aa_msaa";
import { runTAA } from "./examples/aa_taa";
import { runAntiAliasingShowcase } from "./examples/antiAliasingShowcase";
import { runTSR } from "./examples/sr_tsr";
import { runFSR } from "./examples/sr_fsr";
import { runFeedForward3DGS } from "./examples/gaussian_splatting";
import { runLightProbe } from "./examples/light_probe";
import { runSSR } from "./examples/ssr";
import { runEnvMap } from "./examples/envmap";
import { runSelectiveBloom } from "./examples/selective_bloom";
import { runToneMapping } from "./examples/tonemapping";
import { runSSGI } from "./examples/ssgi";
import { runHBAO } from "./examples/hbao";
import { runSSAO } from "./examples/ssao";
import { runAtomicReduction } from "./examples/atomic_reduction";
import { runNaniteMeshlets } from "./examples/nanite_meshlet_culling";
import { runGLTFPBRqingqi } from "./examples/qingqipbr";
import { runFSR2Demo } from "./examples/fsr2Demo";
import { runSPHFluidDemo } from "./examples/sphFluidDemo";
import { runSmokeDiffusionDemo } from "./examples/smokeDiffusionDemo";
import { runSmokeFireExplosionDemo } from "./examples/smokeFireExplosionDemo";
import { runImageDownsampleDemo } from "./examples/imageDownsampleDemo";
import { runGPUSkinningDemo } from "./examples/gpuSkinningDemo";
import { runCarAnimate } from "./examples/carAnimate";
import { runRayMatching } from "./examples/rayMatchingTest";
import { runComputePro } from "./examples/kcomputeDemo";
import { runComputeBarrierApp } from "./examples/computeBarrierApp";
import { runVXGI } from "./examples/vxgi";
import { runConeTracing } from "./examples/coneTracing";
import { runGBufferVisualizer } from "./examples/gbufferVisualizer";
import { runPlanetaryClouds } from "./examples/planetaryClouds";
import { runGaussianSplattingspz } from "./examples/gaussianSplattingSPZ";

export type CleanupFunction = () => void;

// 1. 兼容类型定义：兼具 lil-gui 和 SimpleGUI 的特性
export type CompatibleGUI = GUI & {
  addSelect?: (target: any, property: string, options: any) => any;
  addTextInfo?: (htmlOrText: string) => any;
  updateDisplay?: () => void;
};

export interface DemoItem {
  id: string;
  name: string;
  run: (
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    canvas: HTMLCanvasElement,
    gui: CompatibleGUI | SimpleGUI | any
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
      { id: "instanceDraw", name: "10000个正方体 (Instance)", run: (d, c, f, cv, gui) => runInstanceDraw(d, c, f, cv, gui) },
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
      { id: "gltfPBR", name: "DamagedHelmet (glTF PBR)", run: (d, c, f, cv, gui) => runGLTFPBR(d, c, f, cv, gui) },
      { id: "qingqi", name: "qingqi (glTF PBR)", run: (d, c, f, cv, gui) => runGLTFPBRqingqi(d, c, f, cv, gui) },
      { id: "opaqueAndTransparent", name: "半透明物体", run: (d, c, f, cv, gui) => runOpaqueAndTransparent(d, c, f, cv, gui) },
      { id: "ibl", name: "基于图像的光照", run: (d, c, f, cv, gui) => runIBL(d, c, f) },
      { id: "GaussianSplat", name: "高斯", run: (d, c, f, cv, gui) => runGaussianSplatting(d, c, f) },
      { id: "GaussianSplatspz", name: "高斯spz", run: (d, c, f, cv, gui) => runGaussianSplattingspz(d, c, f, cv, gui) },
      { id: "forwardGaussianSplat", name: "前馈高斯模拟", run: (d, c, f, cv, gui) => runFeedForward3DGS(d, c, f, cv, gui) },
    ],
  },
  {
    id: "texture",
    name: "材质光照与高级计算",
    children: [
      { id: "checkerboard", name: "棋盘纹理贴图", run: (d, c, f) => runTexture(d, c, f) },
      { id: "pingpong", name: "乒乓渲染", run: (d, c, f) => runPingPong(d, c, f) },
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
      { id: "barrierApp", name: "barrierApp", run: (d, c, f, cv, gui) => runComputeBarrierApp(d, c, f) },
      { id: "automic", name: "automic原子性", run: (d, c, f, cv, gui) => runAtomicReduction(d, c, f, cv, gui) },
      { id: "skybox", name: "3D 全景天空盒 (Cubemap)", run: (d, c, f, cv) => runSkybox(d, c, f, cv) },
      { id: "hdrSkybox", name: "hdr 全景天空盒 (Cubemap)", run: (d, c, f, cv, gui) => runHdrSkybox(d, c, f, cv, gui) },
      { id: "earlyz", name: "Early-Z 提前测试", run: (d, c, f, cv, gui) => runEarlyZ(d, c, f, cv, gui) },
      { id: "shadow", name: "动态光照阴影 (ShadowMap)", run: (d, c, f) => runDynamicShadowMap(d, c, f) },
      { id: "sofashadow", name: "软阴影 ", run: (d, c, f) => runSoftShadow(d, c, f) },
      { id: "LightTypesShowcase", name: "光源 ", run: (d, c, f, cv, gui) => runLightTypesShowcase(d, c, f, cv, gui) },
      { id: "csm", name: "csm ", run: (d, c, f, cv, gui) => runCSMShowcase(d, c, f, cv, gui) },
      { id: "video", name: "视频材质 ", run: (d, c, f, cv, gui) => runVideoTextureShowcase(d, c, f, cv, gui) },
      { id: "sampler", name: "采样 ", run: (d, c, f, cv, gui) => runSampler(d, c, f, cv, gui) },
      { id: "mip", name: "nipmap采样 ", run: (d, c, f, cv, gui) => runMipSampler(d, c, f, cv, gui) },
      { id: "renderTotexture", name: "渲染到纹理 ", run: (d, c, f, cv, gui) => runRenderToTexture(d, c, f, cv, gui) },
      { id: "texture3d", name: "3D 空间体积纹理", run: (d, c, f, cv) => runTexture3D(d, c, f, cv) },
      { id: "occlusionQuery", name: "Occlusion Query 遮挡查询", run: (d, c, f, cv) => runOcclusionQuery(d, c, f, cv) },
      { id: "reversedZ", name: "Reversed-Z 精度对比", run: (d, c, f, cv) => runReversedZ(d, c, f, cv) },
      { id: "MRT", name: "MRT 多渲染目标 G-Buffer", run: (d, c, f, cv) => runMRT(d, c, f, cv) },
      { id: "offscreen", name: "离屏多通道渲染", run: (d, c, f, cv) => runOffscreen(d, c, f, cv) },
      { id: "multiplePipelines", name: "Compute-Render 交互", run: (d, c, f, cv) => runMultiplePipelines(d, c, f, cv) },
      { id: "TextureAnimation", name: "动态纹理动画", run: (d, c, f, cv, gui) => runTextureAnimationShowcase(d, c, f, cv, gui) },
      { id: "fireAnimation", name: "火焰", run: (d, c, f, cv, gui) => runFireTextureAnimationShowcase(d, c, f, cv, gui) },
      { id: "gpuculling", name: "gpu剔除", run: (d, c, f, cv, gui) => runGpuCulling(d, c, f, cv, gui) },
      { id: "hizculling", name: "hiz剔除", run: (d, c, f, cv, gui) => runHizOcclusionCulling(d, c, f, cv, gui) },
      { id: "depthtest", name: "深度测试", run: (d, c, f, cv, gui) => runDepthTest(d, c, f, cv, gui) },
      { id: "mashlet", name: "meshlets简化", run: (d, c, f, cv, gui) => runNaniteMeshlets(d, c, f, cv, gui) },
      { id: "downSample", name: "降采样", run: (d, c, f, cv, gui) => runImageDownsampleDemo(d, c, f, cv, gui) },
      { id: "gbuffer", name: "延迟渲染", run: (d, c, f, cv, gui) => runGBufferVisualizer(d, c, f, cv, gui) },
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
      { id: "ssGI", name: "ssgi", run: (d, c, f, cv, gui) => runSSGI(d, c, f, cv, gui) },
      { id: "vxGI", name: "vxgi", run: (d, c, f, cv, gui) => runVXGI(d, c, f, cv, gui) },
      { id: "coneTracing", name: "锥体追踪 (Cone Tracing)", run: (d, c, f, cv, gui) => runConeTracing(d, c, f, cv, gui) },
      { id: "water", name: "水体", run: (d, c, f, cv) => runWater(d, c, f) },
      { id: "FluidSimulation", name: "流体模拟", run: (d, c, f, cv, gui) => runFluidSimulation(d, c, f) },
      { id: "FluidSimulation2", name: "流体模拟sph", run: (d, c, f, cv, gui) => runSPHFluidDemo(d, c, f, cv, gui) },
      { id: "raymatchingTest", name: "raymatcing调试", run: (d, c, f, cv, gui) => runRayMatching(d, c, f, cv, gui) },
      { id: "planetaryClouds", name: "全球动态体积云 (Planetary Clouds)", run: (d, c, f, cv, gui) => runPlanetaryClouds(d, c, f, cv, gui) },
    ],
  },
  {
    id: "weather",
    name: "天气环境",
    children: [
      { id: "weatherRainSnow", name: "雨雪 ", run: (d, c, f, cv, gui) => runWeatherRainSnow(d, c, f) },
      { id: "Environments", name: "环境 ", run: (d, c, f, cv, gui) => runWeatherSkyEnvironments(d, c, f) },
    ],
  },
  {
    id: "other",
    name: "覆盖物",
    children: [
      { id: "billboard", name: "标注 ", run: (d, c, f, cv, gui) => runImageMarkerShowcase(d, c, f, cv, gui) },
      { id: "text", name: "文字标签 ", run: (d, c, f, cv, gui) => runTextBillboardShowcase(d, c, f, cv, gui) },
      { id: "textbill", name: "纯文字 ", run: (d, c, f, cv, gui) => runPureTextBillboardShowcase(d, c, f, cv, gui) },
      { id: "message", name: "信息框 ", run: (d, c, f, cv, gui) => runInteractiveMarkerShowcase(d, c, f, cv, gui) },
      { id: "batch", name: "合并批次 ", run: (d, c, f, cv, gui) => runBatchMarkersShowcase(d, c, f, cv, gui) },
    ],
  },
  {
    id: "ai",
    name: "ai相关",
    children: [
      { id: "aiBaseextent", name: "推理 ", run: (d, c, f, cv, gui) => runAIInference(d) },
      { id: "aiextent", name: "推理增强 ", run: (d, c, f, cv, gui) => runAIInferenceExt(d) },
    ],
  },
  {
    id: "post",
    name: "全局效果",
    children: [
      { id: "aa", name: "aa抗锯齿 ", run: (d, c, f, cv, gui) => runAntiAliasingShowcase(d, c, f) },
      { id: "tsr", name: "tsr", run: (d, c, f, cv, gui) => runTSR(d, c, f) },
      { id: "fsr", name: "fsr ", run: (d, c, f, cv, gui) => runFSR(d, c, f) },
      { id: "fsr2", name: "fsr2 ", run: (d, c, f, cv, gui) => runFSR2Demo(d, c, f, cv, gui) },
      { id: "probe", name: "光照探针 ", run: (d, c, f, cv, gui) => runLightProbe(d, c, f, cv, gui) },
      { id: "ssr", name: "屏幕空间反射 ", run: (d, c, f, cv, gui) => runSSR(d, c, f, cv, gui) },
      { id: "env", name: "环境贴图 ", run: (d, c, f, cv, gui) => runEnvMap(d, c, f, cv, gui) },
      { id: "bloom", name: "泛光 ", run: (d, c, f, cv, gui) => runSelectiveBloom(d, c, f, cv, gui) },
      { id: "tonemapping", name: "色彩映射 ", run: (d, c, f, cv, gui) => runToneMapping(d, c, f, cv, gui) },
      { id: "hbao", name: "hbao ", run: (d, c, f, cv, gui) => runHBAO(d, c, f, cv, gui) },
      { id: "ssao", name: "ssao ", run: (d, c, f, cv, gui) => runSSAO(d, c, f, cv, gui) },
    ],
  },
  {
    id: "particle",
    name: "粒子系统",
    children: [
      { id: "smoke", name: "烟雾 ", run: (d, c, f, cv, gui) => runSmokeDiffusionDemo(d, c, f, cv, gui) },
      { id: "particle", name: "Compute 粒子系统", run: (d, c, f) => runComputeParticles(d, c, f) },
      { id: "smokeFire", name: "火粒子 ", run: (d, c, f, cv, gui) => runSmokeFireExplosionDemo(d, c, f, cv, gui) },
    ],
  },
  {
    id: "timeAnim",
    name: "时间动画与计算",
    children: [
      { id: "skiinning", name: "gpu皮肤动画 ", run: (d, c, f, cv, gui) => runGPUSkinningDemo(d, c, f, cv, gui) },
      { id: "car", name: "轨迹运动", run: (d, c, f, cv, gui) => runCarAnimate(d, c, f, cv, gui) },
      { id: "kcompute", name: "计算", run: (d, c, f, cv, gui) => runComputePro(d, c, f, cv, gui) },
    ],
  },
];

// 状态控制
const currentDemoId = ref<string>("triangle");
const currentDemoTitle = ref<string>("彩色三角形");
const canvasContainerRef = ref<HTMLDivElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const guiContainerRef = ref<HTMLDivElement | null>(null);

const expandedCategories = ref<Set<string>>(new Set());

function toggleCategory(categoryId: string) {
  if (expandedCategories.value.has(categoryId)) {
    expandedCategories.value.delete(categoryId);
  } else {
    expandedCategories.value.add(categoryId);
  }
}

let device: GPUDevice;
let context: GPUCanvasContext;
let format: GPUTextureFormat;
let currentCleanup: CleanupFunction | null = null;
let resizeObserver: ResizeObserver | null = null;

// 当前活动的 GUI 实例（支持 lil-gui / SimpleGUI）
let activeGui: CompatibleGUI | null = null;

/**
 * 2. GUI 创建工厂：
 * 创建 lil-gui 实例，同时注入 SimpleGUI 的专属方法（addSelect, addTextInfo 等）
 * 使得历史代码完全无感运行，同时完美享受 lil-gui 强大的界面与交互！
 */
function createCompatibleGUI(container: HTMLElement, title: string): CompatibleGUI {
  const gui = new GUI({ container, title, width: 260 }) as CompatibleGUI;

  // 兼容 SimpleGUI 的 addSelect
  gui.addSelect = function (target: any, property: string, options: any) {
    return gui.add(target, property, options);
  };

  // 兼容 SimpleGUI 的 addTextInfo（在 lil-gui 中创建一个只读的展示项或 HTML DOM）
  gui.addTextInfo = function (htmlOrText: string) {
    const infoDiv = document.createElement("div");
    infoDiv.innerHTML = htmlOrText;
    infoDiv.style.padding = "6px 8px";
    infoDiv.style.fontSize = "11px";
    infoDiv.style.lineHeight = "1.5";
    infoDiv.style.color = "#a1a1aa";
    infoDiv.style.borderBottom = "1px solid rgba(255,255,255,0.08)";
    infoDiv.style.marginBottom = "4px";
    gui.$children.prepend(infoDiv);
    return infoDiv;
  };

  // 兼容 updateDisplay
  gui.updateDisplay = function () {
    gui.controllersRecursive().forEach((c) => c.updateDisplay());
  };

  return gui;
}

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
    try {
      currentCleanup();
    } catch (e) {
      console.error("Cleanup error:", e);
    }
    currentCleanup = null;
  }

  // 销毁旧 GUI 并重置容器
  if (activeGui) {
    activeGui.destroy();
    activeGui = null;
  }

  if (guiContainerRef.value) {
    guiContainerRef.value.innerHTML = "";
    // 创建新的具备 SimpleGUI 兼容性的 lil-gui 实例
    activeGui = createCompatibleGUI(guiContainerRef.value, item.name);
  }

  resizeCanvas();

  const cleanup = await item.run(device, context, format, canvas, activeGui!);
  if (typeof cleanup === "function") {
    currentCleanup = cleanup;
  }
}

onMounted(() => initWebGPU());
onUnmounted(() => {
  if (currentCleanup) currentCleanup();
  if (resizeObserver) resizeObserver.disconnect();
  if (activeGui) {
    activeGui.destroy();
    activeGui = null;
  }
});
</script>

<template>
  <div class="layout">
    <!-- 侧边导航栏 -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <h1 class="logo">WebGPU Lab</h1>
      </div>
      <nav class="nav-tree">
        <div v-for="category in demoCategories" :key="category.id" class="category-group">
          <!-- 点击分类标题即可展开 / 折叠 -->
          <div class="category-title" @click="toggleCategory(category.id)">
            <span class="arrow-icon" :class="{ open: expandedCategories.has(category.id) }">▶</span>
            <span class="folder-icon">📂</span>
            <span class="category-name">{{ category.name }}</span>
            <span class="count-badge">{{ category.children.length }}</span>
          </div>

          <!-- 使用 v-show 控制二级子项显隐 -->
          <ul v-show="expandedCategories.has(category.id)" class="sub-list">
            <li
              v-for="item in category.children"
              :key="item.id"
              class="sub-item"
              :class="{ active: currentDemoId === item.id }"
              @click.stop="switchDemo(item)"
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
        <!-- lil-gui 悬浮容器 -->
        <div class="gui-panel" ref="guiContainerRef"></div>
      </div>
    </main>
  </div>
</template>

<style scoped>
.layout { display: flex; width: 100vw; height: 100vh; overflow: hidden; background-color: #0e0e10; color: #e4e4e7; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.sidebar { width: 250px; min-width: 250px; background-color: #18181b; border-right: 1px solid #27272a; display: flex; flex-direction: column; user-select: none; }
.sidebar-header { height: 56px; display: flex; align-items: center; padding: 0 20px; border-bottom: 1px solid #27272a; }
.logo { font-size: 16px; font-weight: 600; color: #38bdf8; margin: 0; }
.nav-tree { flex: 1; overflow-y: auto; padding: 12px 10px; }
.category-group { margin-bottom: 6px; }

.category-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12.5px;
  font-weight: 600;
  color: #a1a1aa;
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}
.category-title:hover {
  background-color: #27272a;
  color: #f4f4f5;
}
.category-name {
  flex: 1;
}

.arrow-icon {
  font-size: 9px;
  color: #71717a;
  display: inline-block;
  transition: transform 0.2s ease;
  transform: rotate(0deg);
}
.arrow-icon.open {
  transform: rotate(90deg);
}
.folder-icon { font-size: 13px; }

.count-badge {
  font-size: 11px;
  color: #71717a;
  background: #27272a;
  padding: 1px 6px;
  border-radius: 10px;
}

.sub-list { list-style: none; margin: 2px 0 6px 12px; padding: 0; border-left: 1px solid #27272a; }
.sub-item { display: flex; align-items: center; gap: 10px; padding: 7px 12px; margin: 2px 0 2px 6px; border-radius: 6px; font-size: 13px; color: #a1a1aa; cursor: pointer; transition: all 0.15s ease; }
.sub-item:hover { background-color: #27272a; color: #f4f4f5; }
.sub-item.active { background-color: #0284c7; color: #ffffff; font-weight: 500; }
.dot { width: 5px; height: 5px; border-radius: 50%; background-color: #52525b; }
.sub-item.active .dot { background-color: #ffffff; }

.main-content { flex: 1; display: flex; flex-direction: column; height: 100%; background: #09090b; }
.top-bar { height: 56px; border-bottom: 1px solid #27272a; display: flex; align-items: center; padding: 0 24px; background-color: #121214; gap: 8px; }
.label { font-size: 13px; color: #71717a; }
.active-name { font-size: 14px; font-weight: 500; color: #f4f4f5; }
.canvas-container { flex: 1; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center; }
canvas { width: 100%; height: 100%; display: block; }

/* lil-gui 面板悬浮定位 */
.gui-panel {
  position: absolute;
  top: 16px;
  right: 16px;
  z-index: 100;
  user-select: none;
}

/* 微调 lil-gui 视觉融入暗黑极简风格 */
:deep(.lil-gui) {
  --background-color: rgba(24, 24, 27, 0.88);
  --text-color: #e4e4e7;
  --title-background-color: rgba(39, 39, 42, 0.95);
  --widget-color: #27272a;
  --hover-color: #3f3f46;
  --focus-color: #0284c7;
  --number-color: #38bdf8;
  --string-color: #4ade80;
  font-family: inherit;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #3f3f46;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(10px);
}
</style>