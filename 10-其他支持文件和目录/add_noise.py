import os
import cv2
import numpy as np
import random

def add_gaussian_noise(image):
    h, w, c = image.shape
    mean = 0
    sigma = 20
    noise = np.random.normal(mean, sigma, (h, w, c))
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image):
    s_vs_p = 0.05
    amount = 0.01
    noisy_img = np.copy(image)

    # 添加salt噪声
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_img[coords[0], coords[1], :] = [255, 255, 255]

    # 添加pepper噪声
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_img[coords[0], coords[1], :] = [0, 0, 0]

    return noisy_img

def process_images(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        print("Processing:", input_path)

        image = cv2.imread(input_path)

        # 添加不同类型的对抗噪声
        noisy_image_gaussian = add_gaussian_noise(image)
        noisy_image_salt_and_pepper = add_salt_and_pepper_noise(image)

        # 保存处理后的图像
        #cv2.imwrite(os.path.join(output_dir, f"{filename}"), noisy_image_gaussian)  #高斯噪声
        #cv2.imwrite(os.path.join(output_dir, f"{filename}"), noisy_image_salt_and_pepper)  #椒盐噪声

if __name__ == '__main__':
    input_dir = ''  # 输入数据文件夹路径
    output_dir = ''  # 输出数据文件夹路径
    process_images(input_dir, output_dir)
