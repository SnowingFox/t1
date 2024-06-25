import cv2
import numpy as np
from matplotlib import pyplot as plt

# 加载图像
image_path = 'image.png'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if original_image is None:
    raise ValueError(f"Cannot load image, please check the path: {image_path}")

# 保存图像到本地文件
def save_image(file_path, image):
    cv2.imwrite(file_path, image)

# Sobel 算子
def sobel_operator(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude_euclidean = np.sqrt(sobelx**2 + sobely**2)
    magnitude_manhattan = np.abs(sobelx) + np.abs(sobely)
    
    _, binary_image_euclidean = cv2.threshold(magnitude_euclidean, 50, 255, cv2.THRESH_BINARY)
    _, binary_image_manhattan = cv2.threshold(magnitude_manhattan, 50, 255, cv2.THRESH_BINARY)
    
    return sobelx, sobely, magnitude_euclidean, magnitude_manhattan, binary_image_euclidean, binary_image_manhattan

# Prewitt 算子
def prewitt_operator(image):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    
    prewittx = cv2.filter2D(image, -1, kernelx)
    prewitty = cv2.filter2D(image, -1, kernely)
    
    magnitude_euclidean = np.sqrt(prewittx**2 + prewitty**2)
    magnitude_manhattan = np.abs(prewittx) + np.abs(prewitty)
    
    _, binary_image_euclidean = cv2.threshold(magnitude_euclidean, 50, 255, cv2.THRESH_BINARY)
    _, binary_image_manhattan = cv2.threshold(magnitude_manhattan, 50, 255, cv2.THRESH_BINARY)
    
    return prewittx, prewitty, magnitude_euclidean, magnitude_manhattan, binary_image_euclidean, binary_image_manhattan

# 拉普拉斯算子
def laplacian_operator(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    
    _, binary_image = cv2.threshold(laplacian, 50, 255, cv2.THRESH_BINARY)
    
    return laplacian, binary_image

# 结果保存路径
output_dir = './test-2'

# Sobel 算子处理结果
sobelx, sobely, magnitude_euclidean, magnitude_manhattan, binary_image_euclidean, binary_image_manhattan = sobel_operator(original_image)
save_image(f'{output_dir}sobel_x.png', sobelx)
save_image(f'{output_dir}sobel_y.png', sobely)
save_image(f'{output_dir}sobel_magnitude_euclidean.png', magnitude_euclidean)
save_image(f'{output_dir}sobel_magnitude_manhattan.png', magnitude_manhattan)
save_image(f'{output_dir}sobel_binary_euclidean.png', binary_image_euclidean)
save_image(f'{output_dir}sobel_binary_manhattan.png', binary_image_manhattan)

# Prewitt 算子处理结果
prewittx, prewitty, magnitude_euclidean, magnitude_manhattan, binary_image_euclidean, binary_image_manhattan = prewitt_operator(original_image)
save_image(f'{output_dir}prewitt_x.png', prewittx)
save_image(f'{output_dir}prewitt_y.png', prewitty)
save_image(f'{output_dir}prewitt_magnitude_euclidean.png', magnitude_euclidean)
save_image(f'{output_dir}prewitt_magnitude_manhattan.png', magnitude_manhattan)
save_image(f'{output_dir}prewitt_binary_euclidean.png', binary_image_euclidean)
save_image(f'{output_dir}prewitt_binary_manhattan.png', binary_image_manhattan)

# 拉普拉斯算子处理结果
laplacian, binary_image = laplacian_operator(original_image)
save_image(f'{output_dir}laplacian.png', laplacian)
save_image(f'{output_dir}laplacian_binary.png', binary_image)
