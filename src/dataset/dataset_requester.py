import kagglehub

# Download opensource dataset from Kaggle

path = kagglehub.dataset_download("ashishjangra27/face-mask-12k-images-dataset")

print("Path to dataset files:", path)