# Stem App — 茎秆微观结构语义分割 Android App

## 项目概述

将茎秆微观结构语义分割模型部署到 Android 手机端，支持相机拍摄识别和相册导入识别，并设计为通用框架，可动态加载不同的 ONNX 分割模型。

---

## 模型信息

| 项目 | 值 |
|------|------|
| **原始模型文件** | `data/best_epoch_weights_vb3.onnx` |
| **模型大小** | 94.98 MB (FP32) |
| **输入规格** | `[1, 3, 1024, 1024]` — RGB 图像 |
| **输出规格** | `[1, 2, 1024, 1024]` — 2通道语义分割图 |
| **模型架构** | UNet + ECA Attention |
| **训练框架** | PyTorch（有原始 .pth 权重） |
| **当前精度** | mIoU 89% |
| **现有工具** | `data/Genealized stem pre.exe`（Windows 端预测程序） |

---

## 架构决策

| 决策项 | 选择 |
|--------|------|
| **平台** | Android 原生 App (Kotlin) |
| **推理引擎** | ONNX Runtime Mobile (`onnxruntime-android`) |
| **模型加载方式** | 动态加载（从手机存储选择 `.onnx` 文件）→ **通用框架** |
| **压缩策略** | FP16 量化 + 代码内缩放到 512x512 推理 + 输出上采样回原分辨率 |
| **相机功能** | 拍照模式（取景 → 拍照 → 推理 → 显示结果） |
| **开发投入** | 完整 App 体验 |

---

## 通用框架设计

### 设计原则
- App 本身是一个通用的 **ONNX 语义分割推理引擎**
- 不绑定特定模型，用户可从手机文件系统加载任意 `.onnx` 模型文件
- 模型需满足：**1个图像输入 + 1个分割图输出** 的基本格式

### 支持的模型格式
- **输入:** `[1, C, H, W]` — 任意尺寸的 RGB 图像（代码自动缩放）
- **输出:** `[1, N, H, W]` — N 通道分割图（代码自动 argmax 取最大概率类别）
- **数据类型:** `tensor(float)` 即 FP32

### 模型热切换
- 用户通过文件选择器选择 `.onnx` 文件
- App 自动加载并初始化推理会话
- 支持随时切换不同模型

---

## App 功能清单

### 1. 主界面
- 相机预览（CameraX API）
- 拍照按钮
- 导入图片按钮（从相册）
- 模型选择按钮

### 2. 模型管理
- 从手机文件系统选择 `.onnx` 模型文件
- 显示当前加载的模型信息（名称、输入输出尺寸）
- 支持热切换模型

### 3. 拍照推理
- 打开相机取景
- 点击拍照按钮
- 自动将照片缩放到模型输入尺寸
- 运行 ONNX Runtime 推理
- 对输出进行 argmax + 上采样回原图尺寸
- 叠加分割蒙版显示在结果页

### 4. 相册导入
- 从系统相册选择图片
- 执行推理流程（同拍照推理）
- 显示分割结果

### 5. 结果展示
- 原图 / 分割结果 / 叠加图 三种模式切换
- 保存结果到相册
- 分享结果

---

## 模型压缩方案

### 阶段一：FP16 量化（Python 脚本）
```python
import onnx
from onnxconverter_common import float16

model = onnx.load("data/best_epoch_weights_vb3.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "data/model_fp16.onnx")
```
- 预期大小：~47 MB（缩小约 50%）
- 预期精度损失：<0.5%

### 阶段二：输入缩放（App 代码内实现）
- 拍照/导入图片后，代码将图像缩放到 512x512
- 推理完成后，将 512x512 的分割结果上采样回原图尺寸
- 计算量减少约 4 倍

### 阶段三（可选）：INT8 量化
- 如果 FP16 + 512x512 仍不够快，进一步做 INT8 量化
- 需要校准数据集（少量标注图片即可）
- 预期大小：~24 MB
- 预期精度损失：<2%

---

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| **语言** | Kotlin |
| **UI 框架** | Jetpack Compose |
| **相机 API** | CameraX |
| **推理引擎** | onnxruntime-android |
| **图像处理** | Bitmap / Android Graphics |
| **文件选择** | Android SAF (Storage Access Framework) |
| **最低 SDK** | API 26 (Android 8.0) |
| **目标 SDK** | API 34 (Android 14) |

---

## 项目结构（规划）

```
stem_app_android/
├── app/
│   ├── src/main/
│   │   ├── java/com/stemapp/
│   │   │   ├── MainActivity.kt
│   │   │   ├── ui/
│   │   │   │   ├── CameraScreen.kt
│   │   │   │   ├── ResultScreen.kt
│   │   │   │   └── ModelSelectScreen.kt
│   │   │   ├── ml/
│   │   │   │   ├── OnnxInference.kt        # 推理引擎封装
│   │   │   │   └── ImagePreprocessor.kt     # 图像预处理/后处理
│   │   │   ├── model/
│   │   │   │   └── ModelManager.kt          # 模型加载管理
│   │   │   └── utils/
│   │   │       └── BitmapUtils.kt
│   │   ├── res/
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts
├── build.gradle.kts
├── settings.gradle.kts
└── gradle.properties
```

---

## 实施步骤

### 阶段一：模型压缩（Python）
- [ ] 编写 FP16 量化脚本
- [ ] 验证 FP16 模型精度（对比 FP32 输出差异）
- [ ] 确定最终使用的模型文件

### 阶段二：Android 项目搭建
- [ ] 创建 Android 项目（Kotlin + Jetpack Compose）
- [ ] 配置 onnxruntime-android 依赖
- [ ] 配置 CameraX 依赖
- [ ] 配置 AndroidManifest（相机权限、文件读取权限）

### 阶段三：核心功能开发
- [ ] 实现 `OnnxInference.kt` — ONNX Runtime 推理封装
- [ ] 实现 `ImagePreprocessor.kt` — 图像缩放、归一化、argmax、上采样
- [ ] 实现 `ModelManager.kt` — 模型文件选择与加载
- [ ] 实现相机拍照功能（CameraX）
- [ ] 实现相册导入功能
- [ ] 实现推理管线（拍照 → 预处理 → 推理 → 后处理 → 显示）

### 阶段四：UI 开发
- [ ] 主界面（相机预览 + 操作按钮）
- [ ] 模型选择界面（文件浏览器）
- [ ] 结果展示界面（原图/分割/叠加切换）
- [ ] 保存/分享功能

### 阶段五：测试与打包
- [ ] 真机测试（中端 Android 手机）
- [ ] 性能优化（推理速度、内存占用）
- [ ] 打包 APK

---

## 注意事项

1. **模型文件大小：** FP16 量化后约 47MB，APK 安装包不包含模型文件，用户首次使用需从文件系统加载
2. **推理速度：** 中端手机 512x512 输入，预期推理时间 0.5-1.5 秒
3. **内存管理：** 大图（相机照片通常 4000x3000）需先缩放到 512x512 再推理，避免 OOM
4. **权限：** 需要 `CAMERA` 和 `READ_EXTERNAL_STORAGE` / `READ_MEDIA_IMAGES` 权限
5. **通用性：** 任何符合 `[1,C,H,W] → [1,N,H,W]` 格式的 ONNX 分割模型均可加载使用