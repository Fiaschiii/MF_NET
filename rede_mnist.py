import numpy as np


def relu(x):
    return np.maximum(0, x)

def relu_derivada(x):
    return (x > 0).astype(float)

def sofmax(x):
    exp = np.exp(x - x.max(axis=1, keepdims = True))
    return exp / exp.sum(axis=1, keepdims = True)

print("Funções de ativação definidas")






