#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from PIL import Image
import argparse
from pathlib import Path

def is_progressive_jpeg(file_path):
    """检查JPEG文件是否为Progressive格式"""
    try:
        with Image.open(file_path) as img:
            if img.format == 'JPEG':
                return hasattr(img, 'is_progressive') and img.is_progressive
    except Exception:
        pass
    return False

def convert_to_baseline_jpeg(input_path, output_path, quality=95):
    """
    将图片转换为Baseline JPEG格式
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        quality: JPEG质量(1-100)
    """
    try:
        with Image.open(input_path) as img:
            # 如果图片有透明通道(RGBA)，转换为RGB
            if img.mode in ('RGBA', 'LA', 'P'):
                # 创建白色背景
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
                    img = background
                else:
                    img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 保存为Baseline JPEG (progressive=False是默认值)
            img.save(output_path, 'JPEG', quality=quality, progressive=False, optimize=True)
            return True
    except Exception as e:
        print(f"转换失败 {input_path}: {str(e)}")
        return False

def batch_convert_images(source_dir, target_dir, quality=95):
    """
    批量转换图片为Baseline JPEG
    
    Args:
        source_dir: 源目录路径
        target_dir: 目标目录路径
        quality: JPEG质量
    """
    # 支持的图片格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 检查源目录是否存在
    if not source_path.exists():
        print(f"错误: 源目录不存在 - {source_dir}")
        return False
    
    # 创建目标目录
    target_path.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    total_count = 0
    progressive_count = 0
    
    print(f"开始扫描目录: {source_dir}")
    print("="*60)
    
    # 遍历源目录中的所有文件
    for file_path in source_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            total_count += 1
            
            # 构建相对路径，保持目录结构
            relative_path = file_path.relative_to(source_path)
            output_file = target_path / relative_path.with_suffix('.jpg')
            
            # 确保输出目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 检查是否为Progressive JPEG
            is_progressive = is_progressive_jpeg(file_path)
            if is_progressive:
                progressive_count += 1
                file_type = "Progressive JPEG"
            else:
                file_type = file_path.suffix.upper()[1:] if file_path.suffix else "未知格式"
            
            print(f"处理: {relative_path} ({file_type})")
            
            # 转换图片
            if convert_to_baseline_jpeg(file_path, output_file, quality):
                converted_count += 1
                print(f"  ✓ 成功转换为: {output_file.relative_to(target_path)}")
            else:
                print(f"  ✗ 转换失败")
            
            print()
    
    print("="*60)
    print(f"转换完成!")
    print(f"总文件数: {total_count}")
    print(f"Progressive JPEG文件数: {progressive_count}")
    print(f"成功转换: {converted_count}")
    print(f"转换失败: {total_count - converted_count}")
    print(f"输出目录: {target_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='将图片转换为Baseline JPEG格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python convert_to_baseline.py /home/HwHiAiUser/gp/demo/ascend-yolov8-sample/data /output/path
  python /home/HwHiAiUser/gp/YOLO/convert.py /home/HwHiAiUser/gp/YOLO/DATASETS/IRSTD_1K/imgs_test /home/HwHiAiUser/gp/YOLO/DATASETS/IRSTD_1K/imgs_test_jpg --quality 100
        """
    )
    
    parser.add_argument(
        'source_dir',
        nargs='?',
        default='/home/HwHiAiUser/gp/demo/ascend-yolov8-sample/data',
        help='源目录路径 (默认: /home/HwHiAiUser/gp/demo/ascend-yolov8-sample/data)'
    )
    
    parser.add_argument(
        'target_dir',
        help='目标目录路径'
    )
    
    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        choices=range(1, 101),
        metavar='1-100',
        help='JPEG质量 (1-100, 默认: 95)'
    )
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.target_dir:
        print("错误: 请指定目标目录路径")
        parser.print_help()
        sys.exit(1)
    
    print("图片格式转换工具")
    print("="*60)
    print(f"源目录: {args.source_dir}")
    print(f"目标目录: {args.target_dir}")
    print(f"JPEG质量: {args.quality}")
    print("="*60)
    
    # 执行转换
    success = batch_convert_images(args.source_dir, args.target_dir, args.quality)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
