import os
from collections import defaultdict

def count_images_per_class(root_dir):
    class_counts = defaultdict(int)

    class_dirs = sorted(os.listdir(root_dir))

    for class_name in class_dirs:
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            img_files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            class_counts[class_name] = len(img_files)

    return class_counts

# 🟡 使用範例：檢查你的 train 資料夾
train_dir = "../data/train"  # 路徑根據你實際專案調整
class_counts = count_images_per_class(train_dir)

# 🔵 印出每個類別的圖片數量
for class_name, count in class_counts.items():
    print(f"Class {class_name}: {count} images")

# 🔴 如果你想畫圖看看分布（可選）
import matplotlib.pyplot as plt

classes = list(class_counts.keys())
counts = list(class_counts.values())

plt.figure(figsize=(15, 5))
plt.bar(classes, counts)
plt.xticks(rotation=90)
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Number of Images per Class in Train Set")
plt.tight_layout()
plt.show()
