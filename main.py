import numpy as np

print("Início do código...")

x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])


np.random.seed(42)

W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))



print("W1 shape:", W1.shape)
print("W2 shape:", W2.shape)
print("\nW1:\n", W1)

z1 = np.dot(x, W1) + b1
a1 = np.maximum(0, z1)


z2 = np.dot(a1, W2) + b2
a2 = 1 / (1 + np.exp(-z2))

print("Previsões da rede (antes de treinar):")
print(a2)
print("\nRespostas esperadas:")
print(y)

loss = -np.mean(y * np.log(a2) + (1 - y) * np.log(1 - a2))

print(f"\nLoss (erro): {loss:.4f}")


erro_saida = a2 - y

print("\nErro na sáida (previsão - esperado):")
print(erro_saida)

m = x.shape[0]

dW2 = np.dot(a1.T, erro_saida) / m 
db2 = np.sum(erro_saida, axis=0, keepdims=True) / m


print("\nGradiente de W2 (o quanto cada peso deve mudar):")
print(dW2)
print("\nGradiente de b2:")
print(db2)

erro_escondida = np.dot(erro_saida, W2.T)
derivada_relu = (z1 > 0).astype(float)
erro_escondida = erro_escondida * derivada_relu

print("\nErro propagado para a camada escondida:")
print(erro_escondida)

dW1 = np.dot(x.T, erro_escondida) / m
db1 = np.sum(erro_escondida, axis=0, keepdims=True) / m

print("\nGradiente de W1:")
print(dW1)
print("\nGradiente de b1:")
print(db1)


taxa_aprendizado = 0.5

W2 = W2 - taxa_aprendizado * dW2
b2 = b2 - taxa_aprendizado * db2
W1 = W1 - taxa_aprendizado * dW1
b1 = b1 - taxa_aprendizado * db1        

print("\nW1 ATUALIZADO:")
print(W1)
print("\nW2 ATUALIZADO:")
print(W2)

print("Código finalizado!")