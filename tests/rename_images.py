import os

# Caminho para a pasta com as imagens
pasta = 'dataset/Pogba'

# Pega todos os arquivos e ordena
arquivos = sorted(os.listdir(pasta))

# Filtra apenas imagens (jpg, png, etc.)
imagens = [f for f in arquivos if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Renomeia um por um no formato n001.jpg
for i, nome_antigo in enumerate(imagens, start=1):
    nome_novo = f"Pogba{i:03d}.png"
    caminho_antigo = os.path.join(pasta, nome_antigo)
    caminho_novo = os.path.join(pasta, nome_novo)
    os.rename(caminho_antigo, caminho_novo)
    print(f"{nome_antigo} -> {nome_novo}")

print("✅ Renomeação concluída com sucesso!")