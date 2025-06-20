import os
import shutil

def merge_and_rename_folders(source_dirs, target_dir):
    # 创建目标文件夹
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    file_index = 1  # 文件编号起始值
    for folder in source_dirs:
        # 遍历源文件夹中的文件
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if os.path.isfile(file_path):  # 确保仅处理文件
                # 生成新的文件名，带前缀0（例如 001、002）
                new_name = f"{str(file_index).zfill(3)}{os.path.splitext(file_name)[1]}"
                new_path = os.path.join(target_dir, new_name)
                
                # 移动文件到目标文件夹并重命名
                shutil.move(file_path, new_path)
                file_index += 1

# 示例用法
source_dirs = [
    r"D:\pixiv图包\129683100_喧闹的代价_魅音の代わりに…肉体で弁償して",
    r"D:\pixiv图包\129724229_双狼小乐的3P日常（上）",
    r"D:\pixiv图包\129725206_双狼小乐的3P日常（下）",
    r"D:\pixiv图包\129773404__Ani-knights_",
    r"D:\pixiv图包\130134623_Vulpisfoglia",
    r"D:\pixiv图包\130156705_？",
    r"D:\pixiv图包\130592237_草你们",
    r"D:\pixiv图包\130798888_草你们",
    r"D:\pixiv图包\喂奶"
    ]  # 源文件夹列表
target_dir = "merged_folder"  # 目标文件夹
merge_and_rename_folders(source_dirs, target_dir)
