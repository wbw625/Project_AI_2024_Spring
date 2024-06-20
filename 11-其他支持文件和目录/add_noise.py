import os
import cv2
import numpy as np
import random

def add_gaussian_noise(image):
    h, w, c = image.shape
    mean = 0
    sigma = 25
    noise = np.random.normal(mean, sigma, (h, w, c))
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image):
    s_vs_p = 0.5
    amount = 0.04
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

def add_poisson_noise(img, scale=1.0, gray_noise=False):
    if gray_noise:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.clip((img * 255.0).round(), 0, 255) / 255
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    out = np.float32(np.random.poisson(img * vals) / float(vals))
    noise = out - img
    if gray_noise:
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    return noise * scale

def add_speckle_noise(image):
    h, w, c = image.shape
    gauss = np.random.randn(h, w, c)
    noisy_img = image + image * gauss
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    return noisy_img

def process_images(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        print("Processing:", input_path)

        image = cv2.imread(input_path)

        # 添加不同类型的对抗噪声
        noisy_image_gaussian = add_gaussian_noise(image)
        noisy_image_salt_and_pepper = add_salt_and_pepper_noise(image)
        noisy_image_poisson = add_poisson_noise(image)
        noisy_image_speckle = add_speckle_noise(image)

        # 保存处理后的图像
        cv2.imwrite(os.path.join(output_dir, f"{filename}"), noisy_image_gaussian)  #高斯噪声
        #cv2.imwrite(os.path.join(output_dir, f"{filename}"), noisy_image_salt_and_pepper)  #椒盐噪声
        #cv2.imwrite(os.path.join(output_dir, f"{filename}"), noisy_image_poisson)   #泊松噪声
        #cv2.imwrite(os.path.join(output_dir, f"{filename}"), noisy_image_speckle)   #散斑噪声

if __name__ == '__main__':
    input_dir = ''  # 输入数据文件夹路径
    output_dir = ''  # 输出数据文件夹路径
    process_images(input_dir, output_dir)
