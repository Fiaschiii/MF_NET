import numpy as np

print("Início do código...")
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(42)
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

taxa_aprendizado = 0.5
epocas = 10000

for epoca in range(epocas):
    # Forward
    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(a1, W2) + b2
    a2 = 1 / (1 + np.exp(-z2))

    # Loss
    loss = -np.mean(y * np.log(a2) + (1 - y) * np.log(1 - a2))

    # Backward
    m = X.shape[0]
    erro_saida = a2 - y
    dW2 = np.dot(a1.T, erro_saida) / m
    db2 = np.sum(erro_saida, axis=0, keepdims=True) / m
    erro_escondida = np.dot(erro_saida, W2.T) * (z1 > 0).astype(float)
    dW1 = np.dot(X.T, erro_escondida) / m
    db1 = np.sum(erro_escondida, axis=0, keepdims=True) / m

    # Update
    W1 -= taxa_aprendizado * dW1
    b1 -= taxa_aprendizado * db1
    W2 -= taxa_aprendizado * dW2
    b2 -= taxa_aprendizado * db2

    # Mostra progresso a cada 1000 épocas
    if epoca % 1000 == 0:
        print(f"Época {epoca:5d} | Loss: {loss:.6f}")

# Resultado final
print("\nPrevisões finais:")
for i in range(4):
    print(f"  {X[i]} → {a2[i][0]:.4f} (esperado: {y[i][0]})")

print("Fim do código.")    