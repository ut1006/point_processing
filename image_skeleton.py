import cv2
import numpy as np
from skimage.morphology import skeletonize
import os

def detect_branch_points(image_path):
    # 画像の読み込み
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # 二値化
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    
    # スケルトン化
    skeleton = skeletonize(binary_image // 255).astype(np.uint8) * 255
    
    # 隣接ピクセル数の計算
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    neighbors = cv2.filter2D(skeleton, -1, kernel)
    
    # ブランチポイントの検出
    branch_points = np.where(neighbors >= 30, 255, 0).astype(np.uint8)
    
    # 画像をファイルに保存
    cv2.imwrite('original_image.png', image)
    cv2.imwrite('skeleton.png', skeleton)
    cv2.imwrite('branch_points.png', branch_points)
    
    print("Images have been saved to current directory.")

if __name__ == "__main__":
    detect_branch_points('/home/kamada/work/3d-reconstruction/point_processing/20240702162444.jpeg')
