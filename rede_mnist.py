import numpy as np
import gzip

def relu(x):
    return np.maximum(0, x)

def relu_derivada(x):
    return (x > 0).astype(float)

def softmax(x):
    exp = np.exp(x - x.max(axis=1, keepdims = True))
    return exp / exp.sum(axis=1, keepdims = True)

print("Funções de ativação definidas")


np.random.seed(42)


W1 = np.random.randn(784, 128) * np.sqrt(2 / 784)
b1 = np.zeros((1, 128))


W2 = np.random.randn(128, 64) * np.sqrt(2 / 128)
b2 = np.zeros((1, 64))


W3 = np.random.randn(64, 10) * np.sqrt(2 / 64)
b3 = np.zeros((1, 10))


print(f"W1: {W1.shape}")
print(f"W2: {W2.shape}")
print(f"W3: {W3.shape}")
print("Pesos inicializados")

def forward(X):
    # Camada 1
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    # Camada 2
    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)

    # Camada 3 (saída)
    z3 = np.dot(a2, W3) + b3
    a3 = softmax(z3)

    return z1, a1, z2, a2, z3, a3
print("Forward pass definido")


def carregar_imagens(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 784) / 255.0

def carregar_labels(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

X_treino = carregar_imagens("train-images-idx3-ubyte.gz")
y_treino = carregar_labels("train-labels-idx1-ubyte.gz")

# Testa com as primeiras 5 imagens
z1, a1, z2, a2, z3, a3 = forward(X_treino[:5])

print(f"\nTestando com 5 imagens:")
print(f"Labels reais:   {y_treino[:5]}")
print(f"\nSaída da rede (probabilidades):")
for i in range(5):
    previsao = np.argmax(a3[i])
    confianca = a3[i][previsao] * 100
    print(f"  Imagem {i}: previu {previsao} ({confianca:.1f}% confiança) | real: {y_treino[i]}")


# ── One-Hot Encoding ─────────────────────────────────
def one_hot(y, num_classes=10):
    resultado = np.zeros((len(y), num_classes))
    resultado[np.arange(len(y)), y] = 1
    return resultado


taxa_aprendizado = 0.01
epocas = 20
batch_size = 64  
m = len(X_treino)

y_oh = one_hot(y_treino)

for epoca in range(epocas):
    
    indices = np.random.permutation(m)
    X_shuffle = X_treino[indices]
    y_shuffle = y_oh[indices]

    loss_total = 0

    for i in range(0, m, batch_size):
        X_batch = X_shuffle[i:i+batch_size]
        y_batch = y_shuffle[i:i+batch_size]

        # Forward
        z1, a1, z2, a2, z3, a3 = forward(X_batch)

        # Loss (Cross-Entropy)
        loss = -np.mean(np.sum(y_batch * np.log(a3 + 1e-8), axis=1))
        loss_total += loss

        # Backward
        mb = len(X_batch)
        dz3 = a3 - y_batch
        dW3 = np.dot(a2.T, dz3) / mb
        db3 = np.sum(dz3, axis=0, keepdims=True) / mb

        da2 = np.dot(dz3, W3.T)
        dz2 = da2 * relu_derivada(z2)
        dW2 = np.dot(a1.T, dz2) / mb
        db2 = np.sum(dz2, axis=0, keepdims=True) / mb

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * relu_derivada(z1)
        dW1 = np.dot(X_batch.T, dz1) / mb
        db1 = np.sum(dz1, axis=0, keepdims=True) / mb

        # Update
        W3 -= taxa_aprendizado * dW3
        b3 -= taxa_aprendizado * db3
        W2 -= taxa_aprendizado * dW2
        b2 -= taxa_aprendizado * db2
        W1 -= taxa_aprendizado * dW1
        b1 -= taxa_aprendizado * db1

    loss_media = loss_total / (m // batch_size)
    print(f"Época {epoca+1:2d}/{epocas} | Loss: {loss_media:.4f}")

print("\nTreino concluído")


X_teste = carregar_imagens("t10k-images-idx3-ubyte.gz")
y_teste = carregar_labels("t10k-labels-idx1-ubyte.gz")

_, _, _, _, _, a3_teste = forward(X_teste)

previsoes = np.argmax(a3_teste, axis=1)
acertos = np.sum(previsoes == y_teste)
acuracia = acertos / len(y_teste) * 100

print(f"\nResultados no conjunto de teste:")
print(f"  Total de imagens: {len(y_teste)}")
print(f"  Acertos:          {acertos}")
print(f"  Acurácia:         {acuracia:.2f}%")

print(f"\nTestando as primeiras 10 imagens:")
for i in range(10):
    status = "✓" if previsoes[i] == y_teste[i] else "✗"
    print(f"  {status} previu {previsoes[i]} | real: {y_teste[i]}")
