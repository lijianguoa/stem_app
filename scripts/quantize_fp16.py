"""
FP16 Quantization Script for ONNX Model
将 FP32 ONNX 模型转换为 FP16 半精度，减小模型体积约 50%
"""
import onnx
from onnxconverter_common import float16
import numpy as np
import os

# 路径配置
MODEL_DIR = "data"
INPUT_MODEL = os.path.join(MODEL_DIR, "best_epoch_weights_vb3.onnx")
OUTPUT_MODEL = os.path.join(MODEL_DIR, "model_fp16.onnx")

def quantize_fp16():
    print("=" * 50)
    print("FP16 Quantization")
    print("=" * 50)
    
    # 1. 加载原始模型
    print(f"\n[1/4] Loading FP32 model: {INPUT_MODEL}")
    model = onnx.load(INPUT_MODEL)
    
    # 获取原始模型信息
    orig_size = os.path.getsize(INPUT_MODEL) / (1024 * 1024)
    print(f"    Original model size: {orig_size:.2f} MB")
    
    # 2. 检查模型
    print("\n[2/4] Checking model validity...")
    onnx.checker.check_model(model)
    print("    Model check passed!")
    
    # 3. 转换为 FP16
    print("\n[3/4] Converting to FP16...")
    model_fp16 = float16.convert_float_to_float16(model)
    
    # 检查 FP16 模型
    onnx.checker.check_model(model_fp16)
    print("    FP16 conversion successful!")
    
    # 4. 保存 FP16 模型
    print(f"\n[4/4] Saving FP16 model: {OUTPUT_MODEL}")
    onnx.save(model_fp16, OUTPUT_MODEL)
    
    # 输出信息
    fp16_size = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
    reduction = (1 - fp16_size / orig_size) * 100
    
    print(f"\n{'=' * 50}")
    print(f"COMPLETE!")
    print(f"  FP32 size: {orig_size:.2f} MB")
    print(f"  FP16 size: {fp16_size:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  Saved to:  {OUTPUT_MODEL}")
    print(f"{'=' * 50}")

def compare_outputs():
    """验证 FP16 模型输出与 FP32 的差异"""
    import onnxruntime as ort
    
    print("\n" + "=" * 50)
    print("Accuracy Verification (FP32 vs FP16)")
    print("=" * 50)
    
    # 创建随机输入（使用模型原始输入尺寸 1024x1024）
    np.random.seed(42)
    dummy_input = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
    
    # FP32 推理
    print("\n[1/2] Running FP32 inference...")
    session_fp32 = ort.InferenceSession(INPUT_MODEL)
    input_name = session_fp32.get_inputs()[0].name
    output_name = session_fp32.get_outputs()[0].name
    output_fp32 = session_fp32.run([output_name], {input_name: dummy_input})[0]
    print(f"    FP32 output shape: {output_fp32.shape}")
    
    # FP16 推理（输入需要转换为 float16）
    print("\n[2/2] Running FP16 inference...")
    session_fp16 = ort.InferenceSession(OUTPUT_MODEL)
    dummy_input_fp16 = dummy_input.astype(np.float16)
    output_fp16 = session_fp16.run([output_name], {input_name: dummy_input_fp16})[0]
    print(f"    FP16 output shape: {output_fp16.shape}")
    
    # 将 FP16 输出转回 FP32 以便比较
    output_fp16 = output_fp16.astype(np.float32)
    
    # 计算差异
    diff = np.abs(output_fp32 - output_fp16)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    # 对 argmax 结果进行比较（分割任务关注的是类别预测）
    pred_fp32 = np.argmax(output_fp32, axis=1)
    pred_fp16 = np.argmax(output_fp16, axis=1)
    pixel_diff = np.mean(pred_fp32 != pred_fp16) * 100
    
    print(f"\n{'=' * 50}")
    print(f"DIFFERENCE ANALYSIS")
    print(f"  Max absolute diff: {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")
    print(f"  Std diff: {std_diff:.6f}")
    print(f"  Pixel prediction disagreement: {pixel_diff:.4f}%")
    print(f"  (Lower is better, <0.5% is excellent)")
    print(f"{'=' * 50}")

if __name__ == "__main__":
    quantize_fp16()
    if os.path.exists(OUTPUT_MODEL):
        compare_outputs()
    else:
        print("\nFP16 model not found, skipping comparison.")