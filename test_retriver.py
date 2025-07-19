from retriever import perform_search, visualize_retrieval_result
import pickle

# Load features + paths
with open("features.pkl", "rb") as f:
    features = pickle.load(f)
with open("image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)

# Chọn ảnh truy vấn
query_idx = 7
query_feature = features[query_idx]
query_path = image_paths[query_idx]

# Tìm kiếm ảnh tương tự
index = {"features": features}
results = perform_search(query_feature, index, maxResults=10)
result_indexes = [i for _, i in results]

# Hiển thị và lưu kết quả
visualize_retrieval_result(query_path, result_indexes, image_paths)

