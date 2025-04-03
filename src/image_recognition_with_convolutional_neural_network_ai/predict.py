import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Caminhos
CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE, "..", "..", ".."))
IMAGE_PATH = os.path.join(PROJECT_ROOT, "dataset", "Kross", "Kross200.png")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "cnn_model.h5")
IMAGE_SIZE = (64, 64)

# Verifica se o arquivo existe
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Imagem não encontrada em: {IMAGE_PATH}")

# Carrega o modelo
model = load_model(MODEL_PATH)

# Pré-processa a imagem
img = image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predição
pred = model.predict(img_array)
classe = np.argmax(pred)

# Mapeia classes
class_labels = ['Cristiano', 'Kross', 'Messi', 'Pogba', 'Salah']
print("Classe prevista:", class_labels[classe])
