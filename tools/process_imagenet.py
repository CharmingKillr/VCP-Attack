import os
import shutil
from pathlib import Path

def organize_imagenet_val(val_dir, val_txt, target_dir):
    """将ImageNet验证集图片分类到1000个文件夹
    
    Args:
        val_dir: 存放原始验证集图片的目录（包含50,000张JPEG）
        val_txt: val.txt文件的路径（格式：文件名 类别ID）
        target_dir: 目标根目录（将在此创建1000个子文件夹）
    """
    # 创建目标目录结构
    target_dir = Path(target_dir)
    target_dir.mkdir(exist_ok=True)
    
    # 创建1000个类别子文件夹 (000-999)
    for i in range(1000):
        (target_dir / f"{i:03d}").mkdir(exist_ok=True)
    
    # 读取标签文件
    with open(val_txt) as f:
        lines = f.readlines()
    
    # 移动文件到对应类别文件夹
    for line in lines:
        filename, class_id = line.strip().split()
        src_path = Path(val_dir) / filename
        dst_path = target_dir / f"{int(class_id):03d}" / filename
        
        if src_path.exists():
            shutil.copy(str(src_path), str(dst_path))
            print(f"Copied: {filename} -> {class_id}/")
        else:
            print(f"Warning: {filename} not found!")

if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    organize_imagenet_val(
        val_dir="ImageNet_val_path",      # 原始验证集目录
        val_txt="./val.txt",             # 标签文件路径
        target_dir="../data/ImageNet_val_1000"  # 输出目录
    )