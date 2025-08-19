import os
import shutil
import random
from tqdm import tqdm

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)

    classes = os.listdir(input_dir)
    for cls in classes:
        cls_path = os.path.join(input_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val   = int(val_ratio * n_total)

        train_imgs = images[:n_train]
        val_imgs   = images[n_train:n_train + n_val]
        test_imgs  = images[n_train + n_val:]

        for split_name, split_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_class_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_class_dir, exist_ok=True)

            for img_name in tqdm(split_imgs, desc=f'{split_name}/{cls}'):
                src_path = os.path.join(cls_path, img_name)
                dst_path = os.path.join(split_class_dir, img_name)
                shutil.copy2(src_path, dst_path)

if __name__ == "__main__":
    input_dir = "dataset"            # Thư mục gốc chứa good/average/bad
    output_dir = "dataset_split"     # Thư mục mới để lưu train/val/test
    split_dataset(input_dir, output_dir)
