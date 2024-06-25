import cv2
import numpy as np
from matplotlib import pyplot as plt

# 确认图像路径正确
image_path = 'image.png'

# 尝试加载图像
original_image = cv2.imread(image_path)
if original_image is None:
    raise ValueError(f"Cannot load image, please check the path: {image_path}")

original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# 添加高斯噪声
def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy

# 添加椒盐噪声
def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    row, col, ch = image.shape
    noisy = np.copy(image)
    # Salt noise
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 255

    # Pepper noise
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0
    return noisy

# 均值滤波
def apply_mean_filter(image, kernel_size):
    filtered = cv2.blur(image, (kernel_size, kernel_size))
    return filtered

# 中值滤波
def apply_median_filter(image, kernel_size):
    filtered = cv2.medianBlur(image, kernel_size)
    return filtered

# 保存放大图像对比细节并保存
def save_comparison_images(images, titles, zoom_factor=2, region=(100, 100), output_file = ""):
    assert len(images) == len(titles)
    x, y = region
    w, h = 100 // zoom_factor, 100 // zoom_factor
    
    fig, axes = plt.subplots(1, len(images), figsize=(20, 10))
    for ax, img, title in zip(axes, images, titles):
        zoomed_img = img[y:y+h, x:x+w]
        ax.imshow(zoomed_img)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

# 添加噪声
noisy_image_gaussian = add_gaussian_noise(original_image)
noisy_image_salt_pepper = add_salt_pepper_noise(original_image)

# 定义不同大小的模板尺寸
kernel_sizes = [3, 5, 7]

# 处理高斯噪声图像并保存结果
gaussian_images = [noisy_image_gaussian]
gaussian_titles = ["Noisy Image (Gaussian)"]

for kernel_size in kernel_sizes:
    filtered_image_mean = apply_mean_filter(noisy_image_gaussian, kernel_size)
    filtered_image_median = apply_median_filter(noisy_image_gaussian, kernel_size)
    gaussian_images.extend([filtered_image_mean, filtered_image_median])
    gaussian_titles.extend([f"Mean Filter {kernel_size}x{kernel_size}", f"Median Filter {kernel_size}x{kernel_size}"])

save_comparison_images(gaussian_images, gaussian_titles, zoom_factor=2, region=(300, 300), output_file='./test-1/gaussian_comparison.png')

# 处理椒盐噪声图像并保存结果
salt_pepper_images = [noisy_image_salt_pepper]
salt_pepper_titles = ["Noisy Image (Salt & Pepper)"]

for kernel_size in kernel_sizes:
    filtered_image_mean = apply_mean_filter(noisy_image_salt_pepper, kernel_size)
    filtered_image_median = apply_median_filter(noisy_image_salt_pepper, kernel_size)
    salt_pepper_images.extend([filtered_image_mean, filtered_image_median])
    salt_pepper_titles.extend([f"Mean Filter {kernel_size}x{kernel_size}", f"Median Filter {kernel_size}x{kernel_size}"])

save_comparison_images(salt_pepper_images, salt_pepper_titles, zoom_factor=2, region=(300, 300), output_file='./test-1/salt_pepper_comparison.png')
