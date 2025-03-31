import kagglehub

# Download latest version
path = kagglehub.dataset_download("azizbali/football-players-faces-dataset")

print("Path to dataset files:", path)