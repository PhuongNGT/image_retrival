from feature_extractor import get_latent_features1D
from cluster_analysis import elbow_method, silhouette_analysis
import pandas as pd
from pathlib import Path
import numpy as np
from torchvision import transforms
import torch
from model import ConvAutoencoder_v2  # hoặc ConvAutoencoder nếu bạn dùng loại nhỏ hơn

# Load ảnh
datasetPath = Path("dataset/")
df = pd.DataFrame()
df["image"] = [str(p) for p in datasetPath.glob("*.jpg")]

# Transform giống khi train
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load model đã train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder_v2().to(device)
model.load_state_dict(torch.load("conv_autoencoderv2_200ep.pt", map_location=device)['model_state_dict'])
model.eval()

# Trích xuất đặc trưng
images = df.image.values
latent_features1d = get_latent_features1D(model, images, transformations, device=device)
latent_features1d = np.array(latent_features1d)

# (Tùy chọn) lưu lại để dùng sau
np.save("latent_features.npy", latent_features1d)

# Elbow method
elbow_method(latent_features1d, k_range=range(4, 10), save_path="test_elbow.png")

# Silhouette analysis
silhouette_analysis(latent_features1d, k_range=range(3, 10), save_prefix="test_silhouette_")

