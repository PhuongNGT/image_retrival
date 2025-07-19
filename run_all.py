# run_all.py
import os
import subprocess
import webbrowser
import time
import shutil
# Danh sách các file Python chính
main_files = [
    "train.py",
    "build_index.py",
    "cluster_analysis.py",
    "generate_dashboard.py",
    "bow_clustering.py",
    "feature_extractor.py",
    "retriever.py",
    "model.py",
    "app.py"
]

# Các file test
test_files = [
    "test_feature_extractor.py",
    "test_retriver.py",
    "test_cluster_analysis.py",
    "test_bow.py"
]

# Danh sách các file HTML
html_files = [
    "templates/index.html",
    "templates/results.html",
    "templates/cluster_dashboard.html"
]



print("✅ [1] Running core modules...\n")
for f in main_files[:-1]:  # Exclude app.py (launch last)
    if os.path.exists(f):
        print(f"🚀 Running {f}")
        subprocess.run(["python", f])
    else:
        print(f"❌ File not found: {f}")
print("\n🔧 [2] Running all test files before launch...\n")
for test in test_files:
    if os.path.exists(test):
        print(f"🧪 Running test: {test}")
        subprocess.run(["python", test])
    else:
        print(f"⚠️ Test file not found: {test}")

print("\n📁 [3] Verifying HTML templates...\n")
for f in html_files:
    print(f"✅ Found: {f}" if os.path.exists(f) else f"❌ Missing: {f}")


# Chuyển file hình ảnh vào static/
print("\n🗂️ Moving output plots to static/...\n")
img_outputs = [
    "test_elbow.png",
    "test_silhouette_5.png",
    "orb_bow_elbow.png",
    "loss_plot.png",
    "retrieval_result.png"
]

for img in img_outputs:
    if os.path.exists(img):
        dest = os.path.join("static", img)
        shutil.move(img, dest)
        print(f"✅ Moved {img} → {dest}")
    else:
        print(f"⚠️ Not found: {img}")
print("\n🌐 [4] Launching Flask app (http://127.0.0.1:5000)")
time.sleep(2)
webbrowser.open("http://127.0.0.1:5000")
os.system("python app.py")  # Run Flask app (waits for Ctrl+C)

