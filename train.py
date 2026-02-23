import kagglehub

# Download latest version
path = kagglehub.dataset_download("mohamedgamal07/reduced-mnist")

print("Path to dataset files:", path)