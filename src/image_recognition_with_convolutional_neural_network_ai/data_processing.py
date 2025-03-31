import os

# Obtém o diretório do arquivo atual
current_dir = os.path.dirname(os.path.abspath(__file__))

inputs_name = []

# Constrói o caminho absoluto para o arquivo de dados
training_file_path = os.path.join(current_dir, f"../../data/{inputs_filename}")
targets_file_path = os.path.join(current_dir, f"../../data/{targets_filename}")