"""
茎秆微观结构语义分割 - 预测脚本
支持 FP32 和 FP16 ONNX 模型
用法: python scripts/predict.py --image path/to/image.jpg [--model data/model_fp16.onnx]
"""
import argparse
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import time

def parse_args():
    parser = argparse.ArgumentParser(description="茎秆微观结构语义分割预测")
    parser.add_argument("--image", "-i", type=str, required=True,
                        help="输入图片路径")
    parser.add_argument("--model", "-m", type=str,
                        default="data/model_fp16.onnx",
                        help="ONNX 模型路径（支持 FP32 和 FP16）")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出结果保存路径（默认不保存）")
    parser.add_argument("--input_size", type=int, default=None,
                        help="模型输入尺寸（默认自动从模型读取）")
    return parser.parse_args()

def load_model(model_path):
    """加载 ONNX 模型，自动检测 FP16/FP32"""
    print(f"加载模型: {model_path}")
    session = ort.InferenceSession(model_path)
    
    # 获取输入输出信息
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    input_name = input_info.name
    output_name = output_info.name
    input_shape = input_info.shape
    output_shape = output_info.shape
    input_type = input_info.type
    
    # 检测是否为 FP16
    is_fp16 = "float16" in input_type
    
    print(f"  输入: {input_name} {input_type} {input_shape}")
    print(f"  输出: {output_name} {output_info.type} {output_shape}")
    print(f"  数据类型: {'FP16 (半精度)' if is_fp16 else 'FP32 (单精度)'}")
    
    return session, input_name, output_name, input_shape, output_shape, is_fp16

def preprocess(image_path, target_height, target_width, is_fp16=False):
    """预处理图片"""
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size  # (width, height)
    
    # 缩放到模型输入尺寸
    img_resized = img.resize((target_width, target_height), Image.LANCZOS)
    
    # 转换为 numpy 数组并归一化到 [0, 1]
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # CHW 格式: (C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # 添加 batch 维度: (1, C, H, W)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 如果是 FP16 模型，转换为 float16
    if is_fp16:
        img_array = img_array.astype(np.float16)
    
    return img_array, orig_size

def postprocess(output_data, orig_size, num_classes):
    """后处理：argmax 获取分割结果"""
    # output_data shape: (1, num_classes, H, W)
    # argmax 取概率最大的类别
    pred = np.argmax(output_data[0], axis=0)  # (H, W)
    
    # 生成彩色分割图
    colors = np.array([
        [255, 0, 0],      # 类别0: 红色
        [0, 255, 0],      # 类别1: 绿色
        [0, 0, 255],      # 类别2: 蓝色
        [255, 255, 0],    # 类别3: 黄色
        [255, 0, 255],    # 类别4: 品红
        [0, 255, 255],    # 类别5: 青色
        [255, 136, 0],    # 类别6: 橙色
        [136, 0, 255],    # 类别7: 紫色
    ], dtype=np.uint8)
    
    # 创建彩色掩码
    mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for c in range(min(num_classes, len(colors))):
        mask[pred == c] = colors[c]
    
    # 缩放到原始图片尺寸
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.resize(orig_size, Image.NEAREST)
    
    # 统计各类别像素数
    total_pixels = pred.size
    class_counts = {}
    for c in range(num_classes):
        count = int(np.sum(pred == c))
        if count > 0:
            class_counts[c] = {
                "pixels": count,
                "percentage": count / total_pixels * 100
            }
    
    return np.array(mask_img), class_counts

def create_overlay(original_path, mask_array):
    """创建分割叠加图"""
    original = Image.open(original_path).convert("RGB")
    original_array = np.array(original)
    
    # 半透明叠加
    overlay = (original_array * 0.6 + mask_array * 0.4).astype(np.uint8)
    return overlay

def main():
    args = parse_args()
    
    # 检查文件
    if not os.path.exists(args.image):
        print(f"错误: 图片文件不存在: {args.image}")
        return
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return
    
    # 1. 加载模型
    session, input_name, output_name, input_shape, output_shape, is_fp16 = \
        load_model(args.model)
    
    # 获取输入输出尺寸
    input_height = input_shape[2] if args.input_size is None else args.input_size
    input_width = input_shape[3] if args.input_size is None else args.input_size
    num_classes = output_shape[1]
    
    print(f"\n推理配置:")
    print(f"  输入尺寸: {input_width}x{input_height}")
    print(f"  类别数: {num_classes}")
    print(f"  图片: {args.image}")
    
    # 2. 预处理
    print("\n预处理图片...")
    input_tensor, orig_size = preprocess(args.image, input_height, input_width, is_fp16)
    print(f"  原始尺寸: {orig_size[0]}x{orig_size[1]}")
    print(f"  输入张量: {input_tensor.shape} dtype={input_tensor.dtype}")
    
    # 3. 推理
    print("\n推理中...")
    start_time = time.time()
    outputs = session.run([output_name], {input_name: input_tensor})
    inference_time = time.time() - start_time
    print(f"  耗时: {inference_time:.3f} 秒")
    
    # 4. 后处理
    print("\n后处理...")
    output_data = outputs[0]
    
    # 如果是 FP16 输出，转回 FP32 以便处理
    if output_data.dtype == np.float16:
        output_data = output_data.astype(np.float32)
    
    mask_array, class_counts = postprocess(output_data, orig_size, num_classes)
    
    # 5. 创建叠加图
    overlay_array = create_overlay(args.image, mask_array)
    
    # 6. 输出统计信息
    print(f"\n{'='*40}")
    print(f"分割结果统计")
    print(f"{'='*40}")
    print(f"推理耗时: {inference_time:.3f}s")
    print(f"图片尺寸: {orig_size[0]}x{orig_size[1]}")
    print(f"各类别像素占比:")
    for cls, info in sorted(class_counts.items()):
        print(f"  类别 {cls}: {info['pixels']:>8} px ({info['percentage']:.1f}%)")
    print(f"{'='*40}")
    
    # 7. 保存结果
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        basename = os.path.splitext(os.path.basename(args.image))[0]
        
        # 保存分割图
        mask_path = os.path.join(args.output, f"{basename}_mask.png")
        Image.fromarray(mask_array).save(mask_path)
        print(f"\n分割图已保存: {mask_path}")
        
        # 保存叠加图
        overlay_path = os.path.join(args.output, f"{basename}_overlay.png")
        Image.fromarray(overlay_array).save(overlay_path)
        print(f"叠加图已保存: {overlay_path}")
    else:
        # 不保存时显示图片（如果有 GUI 环境）
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            original = Image.open(args.image)
            axes[0].imshow(original)
            axes[0].set_title("原图")
            axes[0].axis("off")
            axes[1].imshow(mask_array)
            axes[1].set_title("分割结果")
            axes[1].axis("off")
            axes[2].imshow(overlay_array)
            axes[2].set_title("叠加图")
            axes[2].axis("off")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("\n提示: 安装 matplotlib 可显示结果: pip install matplotlib")

if __name__ == "__main__":
    main()