from bow_clustering import plot_elbow_BoW
from pathlib import Path
import pandas as pd
import cv2

datasetPath = Path("dataset/")
df = pd.DataFrame()
df["image"] = [str(p) for p in datasetPath.glob("*.jpg")]  # hoặc "*.png" tùy bạn

images = df.image.values
orb = cv2.ORB_create()

plot_elbow_BoW(images, orb, k_range=range(4, 10), save_path="orb_bow_elbow.png")

