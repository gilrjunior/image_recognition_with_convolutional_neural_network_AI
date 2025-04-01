import os
import numpy as np
from PIL import Image

# Obtém o diretório do arquivo atual
current_dir = os.path.dirname(os.path.abspath(__file__))

targets_filename = "targets.csv"

# Constrói o caminho absoluto para o arquivo de dados
training_file_path = os.path.join(current_dir, f"../../dataset/")
targets_file_path = os.path.join(current_dir, f"../../data/{targets_filename}")

# Tamanho para redimensionar as imagens (se necessário)
img_height = 64
img_width = 64

def load_image(player_name):
    """Carrega uma imagem PNG, redimensiona e converte em matriz NumPy normalizada"""
    img = Image.open(f"{training_file_path}/{player_name}/{player_name}001.png").convert('L')  # ou "L" para grayscale
    img = img.resize((img_width, img_height))        # redimensionar se necessário
    img_array = np.array(img) / 255.0                # normaliza os pixels para [0, 1]
    return img_array