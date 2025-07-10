import os
import cv2
import numpy as np

def augment_image(img):
    augmented = []

    # 1. Horizontal Flip
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)

    # 2. Rotation (+15 degrees)
    h, w = img.shape[:2]
    M_rot = cv2.getRotationMatrix2D((w/2, h/2), 15, 1)
    rotated = cv2.warpAffine(img, M_rot, (w, h))
    augmented.append(rotated)

    # 2. Rotation (+345 degrees)
    h, w = img.shape[:2]
    M_rot = cv2.getRotationMatrix2D((w/2, h/2), 345, 1)
    rotated = cv2.warpAffine(img, M_rot, (w, h))
    augmented.append(rotated)

    

    # 4. Increase Contrast
    contrast = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    augmented.append(contrast)

    return augmented

def augment_dataset(base_dir):
    filters = ['Bilateral', 'CLAHE', 'Gamma', 'Otsu', 'Otsu_Canny']
    stages = ['STAGE1', 'Stage-2', 'stage 3', 'stage 4' ]

    for filter_name in filters:
        for stage in stages:
            folder_path = os.path.join(base_dir, filter_name, stage)
            if not os.path.exists(folder_path):
                continue

            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.tif', '.tiff')):
                    img_path = os.path.join(folder_path, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Error loading {img_path}")
                        continue

                    # Apply augmentations
                    aug_images = augment_image(img)
                    name, ext = os.path.splitext(filename)

                    for i, aug_img in enumerate(aug_images):
                        new_filename = f"{name}_aug{i+1}{ext}"
                        cv2.imwrite(os.path.join(folder_path, new_filename), aug_img)

                    print(f"Augmented: {filename} -> {len(aug_images)} new images")

if __name__ == "__main__":
    base_dir = r"C:\Users\Asus\Desktop\DATASETS_FOR_FILTERS\Filtered_outputs_new"
    augment_dataset(base_dir)
