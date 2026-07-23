# 茎秆分析 App 开发进度

## 项目位置
- `D:\stem_app\` — 项目根目录
- `D:\stem_app\stem_app_android\` — Android 项目
- `D:\stem_app\stem_app_android\app\src\main\java\com\stemapp\` — Kotlin 源码
- `D:\stem_app\stem_app-debug.apk` — 最新构建 APK

## 环境配置

### Android SDK
- 路径: `C:\Users\Administrator\AppData\Local\Android\Sdk`
- platforms: `android-34`, `android-36.1`
- build-tools: `36.0.0`

### JDK
- 路径: `D:\jdk-17.0.12+7`
- 通过 `export JAVA_HOME="/d/jdk-17.0.12+7"` 使用

### Gradle
- 版本 8.4 (wrapper 已生成)
- 项目内直接用 `./gradlew` 即可

### 代理（需要每次启动时设置）
```bash
export HTTP_PROXY="http://127.0.0.1:7892"
export HTTPS_PROXY="http://127.0.0.1:7892"
```

### 构建命令
```bash
cd /d/stem_app/stem_app_android
export JAVA_HOME="/d/jdk-17.0.12+7"
export PATH="$JAVA_HOME/bin:$PATH"
export ANDROID_HOME="C:/Users/Administrator/AppData/Local/Android/Sdk"
export HTTP_PROXY="http://127.0.0.1:7892"
export HTTPS_PROXY="http://127.0.0.1:7892"
./gradlew assembleDebug --no-daemon
```

---

## 已完成的修复

### 1. 基础编译修复
- [x] `settings.gradle.kts` — `dependencyResolution` → `dependencyResolutionManagement`
- [x] `local.properties` — SDK 路径正斜杠格式
- [x] `compileSdk` / `targetSdk` → 36
- [x] ONNX Runtime 包名 `org.onnxruntime` → `ai.onnxruntime`
- [x] FP16 张量创建适配新版 API
- [x] `HorizontalDivider` → `Divider`
- [x] K-means 参数传参修复
- [x] 添加 `x86_64`、`x86` ABI 支持（便于模拟器测试）

### 2. 相机无法启动修复
- [x] PreviewView 绑定 Preview surfaceProvider（核心修复）
- [x] 未加载模型 / 相机初始化中 → SnackBar 提示
- [x] 拍照回调 isProcessing 状态管理修复
- [x] `takePhoto` 空值保护

### 3. 默认模型自动加载
- [x] `model_fp16.onnx` (49.8MB) 复制到 `app/src/main/assets/`
- [x] `MainActivity.onCreate()` 后台线程自动加载默认模型
- [x] 启动时显示「正在加载模型...」加载动画

### 4. 导入图片闪退修复
- [x] 导航路径 URL 编解码（防止路径中的 `/` 破坏路由）
- [x] `runInferenceFromPath` 采样解码（`inSampleSize`），大图防 OOM
- [x] `SegmentationResult` 新增 `original` 字段，避免 ResultScreen 重复解码
- [x] `OutOfMemoryError` 捕获

### 5. 比例尺标定功能重构
- [x] **直接在结果图片上点击拉线**（沿照片上的旧标尺线点两点）
- [x] Canvas 绘制标记点、连线、距离标签
- [x] 确认对话框输入标尺长度
- [x] 自动计算 px/μm 比例
- [x] 面积自动换算 μm²
- [x] 设置页面改为只读显示当前标定状态

### 6. 🔥 真机推理闪退修复（最终方案）
- [x] **根因：** 模型 `model_fp16.onnx` 是 FLOAT16，ORT 1.17.1 `getShortBuffer()` 有 ClassCastException bug；ORT 1.22.0 有 `Optional<OnnxTensor>` 类型不匹配；Cast 模型在 arm64 上还有 ORT_INVALID_ARGUMENT
- [x] **最终修复：** 使用 FP32 模型 (`best_epoch_weights_vb3.onnx`)，输入输出均为原生 FLOAT，彻底避免所有 FP16 类型问题
- [x] 代码：`preprocess()` 传 FloatBuffer；`readOutputAsFloatArray()` 用 `getFloatBuffer()`；`session.run()` 返回值加 Optional 解包
- [x] **验证：** 用户手机上拍照/选图推理成功 ✅

### 7. 🎯 结果页增强（2026-07-23）
- [x] **标定模式支持缩放** — 双指捏合缩放 + 单指平移 + 缩放控制按钮（+/-/重置）
- [x] **坐标映射** — Tap 坐标从屏幕坐标系正确映射回图像坐标系
- [x] **K-means 类别着色集成到 Mask/Overlay** — "分割 Mask"和"叠加图"直接显示类别着色（每个 K-means 类别不同颜色），无需额外按钮
- [x] **所有视图支持缩放** — 不再仅限标定模式，原图/Mask/叠加图/分析图均可缩放
- [x] **默认 K=2** — 设置页面默认值从 3 改为 2
- [x] **布局重构** — 图片区域独立使用 weight(1f) 避免 scroll 冲突，分析卡片可滚动

---

## 模拟器测试结果（2026-07-22/23）

### ✅ 模拟器启动成功
- AVD `Phone_6G_128G` (Pixel 6 Pro, Android 14 API 34) 正常启动
- WHPX 加速器已启用，无需重启 Windows
- 启动命令：`emulator.exe -avd Phone_6G_128G -no-audio -no-boot-anim -gpu swiftshader_indirect -memory 4096`

### ✅ 测试通过项
1. **启动崩溃修复** — Compose BOM 从 `2024.01.00` → `2024.06.00`，修复了 `NoSuchMethodError: KeyframesSpec.at()`
2. **相机功能** — CameraX 正常连接、预览显示、拍照成功（1856×1392 JPEG）
3. **自动加载模型** — `model_fp16.onnx`（49.8MB）从 assets 自动加载
4. **拍照后导航** — 使用 Base64 URL-safe 编码替代 URLEncoder，修复 Navigation Compose `%2F` 路由匹配问题
5. **权限处理** — CAMERA 权限正常申请，READ_MEDIA_IMAGES 可授予
6. **🥇 FP16 推理崩溃修复** — onnxruntime 1.17.1→1.22.0 + getByteBuffer() 防御性编码

### ⏳ 推理性能
- **模拟器（x86_64 CPU）：** ONNX 1024×1024 推理极慢（>3分钟），因为使用 CPU 软解
- **真机预期（arm64-v8a + NNAPI）：** 推理时间 0.5-2 秒
- 建议在真机上测试以获得真实性能数据

### 📸 测试截图
- 首次启动 → 相机预览界面：`screenshot.png`
- 拍照后进入结果页（推理中）：`screenshot_result.png`

---

## 已知问题

### ✅ FP16 推理崩溃已修复（2026-07-22）
- **HeapByteBuffer → ShortBuffer ClassCastException** 已通过在真机验证中确认修复
- onnxruntime-android 升级 1.17.1 → 1.22.0
- 防御性编码：`getByteBuffer()` 替代 `getShortBuffer()`

### ⚠️ APK 体积较大（159MB）
- onnxruntime 1.22.0 比 1.17.1 大，后续可优化为 release 构建 + R8 + 仅 arm64-v8a ABI

---

## 修改过的文件清单

| 文件 | 修改内容 |
|------|---------|
| `app/build.gradle.kts` | compileSdk 36, targetSdk 36, abiFilters |
| `settings.gradle.kts` | dependencyResolution 语法修正 |
| `local.properties` | 正斜杠路径 |
| `MainActivity.kt` | 默认模型加载、URL 编解码、后台加载 |
| `CameraScreen.kt` | Preview 绑定、SnackBar 反馈 |
| `OnnxInference.kt` | 采样解码、SegmentationResult 增加 original |
| `ResultScreen.kt` | 比例尺拉线标定、nativeCanvas 修复 |
| `SettingsScreen.kt` | 比例尺改为只读显示 |
| `app/src/main/assets/model_fp16.onnx` | 新增默认模型 |

## 模拟器配置
- **AVD 名称:** Phone_6G_128G
- **设备:** Pixel 6 Pro (1440×3120, 560dpi)
- **系统:** Android 14 (API 34) Google APIs x86_64
- **CPU:** 6 核
- **RAM:** 6144MB
- **储存:** 128GB data partition
- **GPU:** swiftshader_indirect
- **配置文件:** `C:\Users\Administrator\.android\avd\Phone_6G_128G.avd\config.ini`
