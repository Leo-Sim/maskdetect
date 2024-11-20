import kagglehub

# Download latest version

path = kagglehub.dataset_download("omkargurav/face-mask-dataset")

print("Path to dataset files:", path)