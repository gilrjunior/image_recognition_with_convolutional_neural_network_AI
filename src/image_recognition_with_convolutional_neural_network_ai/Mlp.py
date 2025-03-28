import numpy as np
class Mlp:
    def __init__(self, number_neurons, learning_rate):

        # Parâmetros
        self.number_neurons = number_neurons
        self.learning_rate = learning_rate
        self.threshold = 0.00

        # Carrega os dados
        self.inputs = dp.load_inputs()
        self.targets = dp.load_targets()
        
        # Camada oculta: vi (bias) e wi (pesos) - cada um com shape
        self.vi = np.random.uniform(-0.5, 0.5, (self.number_neurons, 1))
        self.wi = np.random.uniform(-0.5, 0.5, (self.number_neurons, self.inputs.shape[1]))
        
        # Camada de saída: vy (bias) escalar e wy (pesos)
        self.vy = np.random.uniform(-0.5, 0.5, (self.targets.shape[1], 1))
        self.wy = np.random.uniform(-0.5, 0.5, (self.targets.shape[1], self.number_neurons))

        # Supondo que self.targets tenha shape (10,10)
        # num_samples = self.inputs.shape[0]  # 900
        # # Cria um array de índices que se repete de 0 a 9
        # indices = np.arange(num_samples) % 10
        # # Expande os targets para ter 900 linhas, cada uma sendo o one-hot correto
        # self.full_targets = self.targets[indices, :]  # shape (900, 10)
        self.expanded_targets = self.targets[np.arange(self.inputs.shape[0]) % 10, :]

    def sigmoid(self, x):
        return 1.7159 * np.tanh((2.0/3.0) * x)

    def sigmoid_derivative(self, x):
        return (2.0/3.0) * 1.7159 * (1 - np.tanh((2.0/3.0) * x)**2)

    def forward(self, x):
        # 1) Camada oculta
        net_in = self.vi + np.dot(self.wi, x)
        z_out = 1.7159 * np.tanh((2.0 / 3.0) * net_in)

        # 2) Camada de saída
        yin = self.vy + np.dot(self.wy, z_out)
        y = 1.7159 * np.tanh((2.0 / 3.0) * yin)

        # Limiarização
        y = np.where(y > self.threshold, 1, -1)
    
    # Rede montada sem uso de toolbox, porém não está sendo utilizada, pois é necessário adaptação para os inputs.
    def batch_train(self, min_error, batch_size=32, patience=10):
        # Realiza o split dos dados: 70% treino, 20% validação, 10% teste (teste não é usado aqui)
        num_samples = self.inputs.shape[0]
        n_train = int(num_samples * 0.7)
        n_val = int(num_samples * 0.2)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        X_train = self.inputs[train_indices]
        Y_train = self.expanded_targets[train_indices]
        X_val = self.inputs[val_indices]
        Y_val = self.expanded_targets[val_indices]
        epochs = 0

        # Treinamento
        while epochs <= 100000:
            epochs += 1
            epoch_error = 0.0
            num_train = X_train.shape[0]

            # Embaralha os dados de treinamento
            train_perm = np.random.permutation(num_train)
            X_train_shuffled = X_train[train_perm]
            Y_train_shuffled = Y_train[train_perm]

            # Processa os dados em mini-batches
            for start in range(0, num_train, batch_size):
                end = start + batch_size
                # Transpõe para que cada mini-batch tenha shape (n_features, batch_size)
                batch_inputs = X_train_shuffled[start:end].T      
                batch_targets = Y_train_shuffled[start:end].T      

                # Feedforward
                net_in = self.vi + np.dot(self.wi, batch_inputs)
                z = self.sigmoid(net_in)
                yin = self.vy + np.dot(self.wy, z)
                y = self.sigmoid(yin)

                # Calcula o erro para o mini-batch
                error = batch_targets - y
                batch_error = 0.5 * np.sum(error**2)
                epoch_error += batch_error

                # Backpropagation
                delta_k = error * self.sigmoid_derivative(yin)
                delta_wy = self.learning_rate * np.dot(delta_k, z.T)
                delta_vy = self.learning_rate * np.sum(delta_k, axis=1, keepdims=True)
                delta_in = np.dot(self.wy.T, delta_k)
                delta_j = delta_in * self.sigmoid_derivative(net_in)
                delta_wi = self.learning_rate * np.dot(delta_j, batch_inputs.T)
                delta_vi = self.learning_rate * np.sum(delta_j, axis=1, keepdims=True)

                # Atualiza os pesos e vieses
                self.wy += delta_wy
                self.vy += delta_vy
                self.wi += delta_wi
                self.vi += delta_vi

            print(f"Época {epochs} - Erro Treino: {epoch_error:.4f} | Learning Rate: {self.learning_rate:.6f}")

        print("Treinamento finalizado em", epochs, "épocas.")
