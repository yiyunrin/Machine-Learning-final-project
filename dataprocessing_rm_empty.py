import os
import cv2
import numpy as np

sz = 28
def resize_image(img, target_size=(sz, sz)):
    # 先將圖片轉為二值圖像
    _, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 找出圖像的非零像素的最小矩形邊界框
    coords = cv2.findNonZero(img_binary)
    x, y, w, h = cv2.boundingRect(coords)
    
    # 裁剪出有效區域並進行縮放
    img_crop = img_binary[y:y+h, x:x+w]
    resized_img = cv2.resize(img_crop, target_size, interpolation=cv2.INTER_AREA)

    return resized_img


def process_images(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # 縮放圖片尺寸
                img = resize_image(img, target_size=(sz, sz))

                output_path = os.path.join(output_subfolder, file)
                cv2.imwrite(output_path, img)


input_folder = "./ori_dataset"
output_folder = "./dataset"
process_images(input_folder, output_folder)
print("圖片轉換完成")