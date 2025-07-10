import cv2
import numpy as np
import os

def normalize_image_8bit(img):
    img = img.astype('float32')
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype('uint8')

def bilateral_filter(image):
    return cv2.bilateralFilter(image, d=5, sigmaColor=25, sigmaSpace=25)

def clahe_filter(image):
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    return clahe.apply(image)

def gamma_correction(image, gamma=1.3):
    inv_gamma = 1.3 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype('uint8')
    return cv2.LUT(image, table)



def normal_threshold(image, thresh_value=220):

    _, thresh_img = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    return thresh_img

def canny_edge_detection(image, lower_threshold=240, upper_threshold=255):
    
    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    return edges



def process_all_images(input_folders, output_base, gamma_value=1.3):
    for stage_name, folder_path in input_folders.items():
        print(f"Processing stage: {stage_name}")
        # Create subfolders for each filter
        filters = ['Bilateral', 'CLAHE', 'Gamma', 'Otsu', 'Otsu_Canny']
        for f in filters:
            os.makedirs(os.path.join(output_base, f, stage_name), exist_ok=True)

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.tif', '.tiff')):
                input_path = os.path.join(folder_path, filename)
                # Load image as is (could be 16-bit)
                img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"Could not read {input_path}, skipping.")
                    continue

                # Normalize to 8-bit
                img_8bit = normalize_image_8bit(img)

                # Apply each filter
                filtered_bilateral = bilateral_filter(img_8bit)
                filtered_clahe = clahe_filter(img_8bit)
                filtered_gamma = gamma_correction(img_8bit, gamma=gamma_value)
                filtered_thresh = normal_threshold(img_8bit)
                filtered_canny = canny_edge_detection(img_8bit)

                # Save filtered images
                cv2.imwrite(os.path.join(output_base, 'Bilateral', stage_name, filename), filtered_bilateral)
                cv2.imwrite(os.path.join(output_base, 'CLAHE', stage_name, filename), filtered_clahe)
                cv2.imwrite(os.path.join(output_base, 'Gamma', stage_name, filename), filtered_gamma)
                cv2.imwrite(os.path.join(output_base, 'Otsu', stage_name, filename), filtered_thresh)
                cv2.imwrite(os.path.join(output_base, 'Otsu_Canny', stage_name, filename), filtered_canny)

        print(f"Completed processing stage: {stage_name}")

if __name__ == "__main__":
    input_folders = {
        'STAGE1' : r"C:\Users\Asus\Desktop\DATASETS_FOR_FILTERS\Input_folder\STAGE1",
        'Stage-2': r"C:\Users\Asus\Desktop\DATASETS_FOR_FILTERS\Input_folder\Stage-2",
        'stage 3' : r"C:\Users\Asus\Desktop\DATASETS_FOR_FILTERS\Input_folder\stage 3",
        'stage 4' : r"C:\Users\Asus\Desktop\DATASETS_FOR_FILTERS\Input_folder\stage 4"
    }
    output_base = r"C:\Users\Asus\Desktop\DATASETS_FOR_FILTERS\Filtered_outputs_new"
    gamma_value = 1.3  # you can change gamma based on your optimization

    process_all_images(input_folders, output_base, gamma_value)
