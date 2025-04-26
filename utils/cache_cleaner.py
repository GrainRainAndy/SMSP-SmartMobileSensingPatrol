import os
import shutil
import argparse

def clean_cache(
    cache_dir='Cache',
    delete_all=False,
    delete_images_only=True,
    verbose=True
):
    """
    清理缓存目录，并统计删除的文件大小

    参数:
        cache_dir (str): 缓存目录路径
        delete_all (bool): 是否删除整个目录
        delete_images_only (bool): 是否只删除图像文件
        verbose (bool): 是否打印日志
    """
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', cache_dir))
    if not os.path.exists(abs_path):
        if verbose:
            print(f"缓存目录 {abs_path} 不存在")
        return

    total_bytes = 0
    deleted_files = 0

    if delete_all:
        # 统计整个目录大小
        for root, _, files in os.walk(abs_path):
            for file in files:
                try:
                    total_bytes += os.path.getsize(os.path.join(root, file))
                except:
                    continue
        shutil.rmtree(abs_path)
        if verbose:
            print(f"已删除整个缓存目录：{abs_path}")
            print(f"共释放空间：{_format_size(total_bytes)}")
        return

    # 仅删除图像文件
    for root, _, files in os.walk(abs_path):
        for file in files:
            if delete_images_only and not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            file_path = os.path.join(root, file)
            try:
                total_bytes += os.path.getsize(file_path)
                os.remove(file_path)
                deleted_files += 1
            except:
                continue

    if verbose:
        print(f"共清理 {deleted_files} 个图像文件")
        print(f"️共释放空间：{_format_size(total_bytes)}")
        print(f"清理目录：{abs_path}")

def _format_size(bytes_num):
    if bytes_num > 1024 ** 2:
        return f"{bytes_num / (1024 ** 2):.2f} MB"
    elif bytes_num > 1024:
        return f"{bytes_num / 1024:.2f} KB"
    else:
        return f"{bytes_num} B"

# ------------------------
# CLI 入口函数
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理缓存目录")
    parser.add_argument('--delete_all', action='store_true', help='删除整个 Cache 文件夹')
    parser.add_argument('--dir', type=str, default='Cache', help='要清理的缓存目录')
    args = parser.parse_args()

    confirm = input(f"确定清理缓存目录 '{args.dir}'？(y/[n]): ")
    if confirm.lower() == 'y':
        clean_cache(cache_dir=args.dir, delete_all=args.delete_all)
    else:
        print("清理操作已取消")

