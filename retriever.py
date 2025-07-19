import numpy as np
import cv2
from PIL import Image
from scipy.spatial import distance
import matplotlib.pyplot as plt

def euclidean(a, b):
    return np.linalg.norm(a - b)

def cosine_distance(a, b):
    return distance.cosine(a, b)

def perform_search(queryFeatures, index, maxResults=64):
    results = []
    for i in range(len(index["features"])):
        d = euclidean(queryFeatures, index["features"][i])
        results.append((d, i))
    results = sorted(results)[:maxResults]
    return results

def build_montages(image_list, image_shape, montage_shape):
    image_montages = []
    montage_image = np.zeros((image_shape[1]*montage_shape[1], image_shape[0]*montage_shape[0], 3), dtype=np.uint8)
    cursor_pos = [0, 0]
    for img in image_list:
        img = cv2.resize(img, image_shape)
        montage_image[cursor_pos[1]:cursor_pos[1]+image_shape[1], cursor_pos[0]:cursor_pos[0]+image_shape[0]] = img
        cursor_pos[0] += image_shape[0]
        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                image_montages.append(montage_image)
                montage_image = np.zeros_like(montage_image)
    if not np.array_equal(montage_image, np.zeros_like(montage_image)):
        image_montages.append(montage_image)
    return image_montages

def visualize_retrieval_result(query_image_path, result_indexes, image_paths, output_path='retrieval_result.png'):
    """
    Hiển thị ảnh truy vấn + ảnh kết quả tìm kiếm và lưu ra file ảnh.
    
    Parameters:
        query_image_path (str): Đường dẫn ảnh truy vấn
        result_indexes (List[int]): Danh sách index ảnh kết quả
        image_paths (List[str]): Danh sách đường dẫn toàn bộ ảnh
        output_path (str): Đường dẫn lưu ảnh đầu ra
    """
    query_img = np.array(Image.open(query_image_path))
    result_imgs = [np.array(Image.open(image_paths[i])) for i in result_indexes]

    fig, ax = plt.subplots(nrows=2, figsize=(15, 10))

    ax[0].imshow(query_img)
    ax[0].set_title("Query Image")
    ax[0].axis("off")

    if result_imgs:
        montage = build_montages(result_imgs, (256, 256), (5, 2))[0]
        ax[1].imshow(montage)
        ax[1].set_title("Top Matching Images")
        ax[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📸 Kết quả truy xuất đã lưu vào {output_path}")
    plt.close()

