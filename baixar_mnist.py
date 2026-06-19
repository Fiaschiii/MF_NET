import urllib.request
import gzip
import numpy as np
import os

print("Início do código...")


def baixar_arquivo(url, destino):
    if not os.path.exists(destino):
        print(f"Baixando {destino}...")
        urllib.request.urlretrieve(url, destino)
        print(f"  ✓ Concluído")
    else:
        print(f"  ✓ {destino} já existe")

def carregar_imagens(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 784)

def carregar_labels(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# URLs oficiais do MNIST
base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
arquivos = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

for arq in arquivos:
    baixar_arquivo(base + arq, arq)

X_treino = carregar_imagens("train-images-idx3-ubyte.gz")
y_treino = carregar_labels("train-labels-idx1-ubyte.gz")
X_teste  = carregar_imagens("t10k-images-idx3-ubyte.gz")
y_teste  = carregar_labels("t10k-labels-idx1-ubyte.gz")

print(f"\nShape X_treino: {X_treino.shape}")
print(f"Shape y_treino: {y_treino.shape}")
print(f"Shape X_teste:  {X_teste.shape}")
print(f"Shape y_teste:  {y_teste.shape}")
print(f"\nPrimeiro label: {y_treino[0]}")
print(f"Pixel mínimo: {X_treino[0].min()}")
print(f"Pixel máximo: {X_treino[0].max()}")


X_treino = X_treino / 255.0
X_teste = X_teste / 255.0

print(f"\nApós normalização:")
print(f"Pixel mínimo: {X_treino[0].min()}")
print(f"Pixel máximo: {X_treino[0].max():.4f}")
print(f"Pixel médio:  {X_treino[0].mean():.4f}")


print(f"\nVisualizando o dígito '{y_treino[0]}':")
img = (X_treino[0] * 255).reshape(28, 28)

for linha in img:
    for pixel in linha:
        if pixel > 128: 
            print("██", end="")
        elif pixel > 50:
            print("▓▓", end="")
        else:
            print(" ", end="")
    print()       


def one_hot(y, num_classes=10):
    resultado = np.zeros((len(y), num_classes))
    resultado[np.arange(len(y)), y] = 1
    return resultado

y_treino_oh = one_hot(y_treino)
y_teste_oh = one_hot(y_teste)

print(f"\nLabel original: {y_treino[0]}")
print(f"One-hot: {y_treino_oh[0]}")
