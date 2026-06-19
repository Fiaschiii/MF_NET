import tkinter as tk
import numpy as np

W1 = np.load("W1.npy")
b1 = np.load("b1.npy")
W2 = np.load("W2.npy")
b2 = np.load("b2.npy")
W3 = np.load("W3.npy") 
b3 = np.load("b3.npy")

print("Pesos e vieses carregados")
print(f"W1: {W1.shape}, W2: {W2.shape}, W3: {W3.shape}")


janela = tk.Tk()
janela.title("MFNet - Reconhecedor de Dígitos")

canvas = tk.Canvas(janela, width=280, height=280, bg="black")
canvas.pack(pady=10)

def desenhar(event):
    x, y = event.x, event.y
    raio = 8
    canvas.create_oval(
        x - raio, y - raio, x + raio, y + raio,
        fill="white", outline="white"
    )

canvas.bind("<B1-Motion>", desenhar)

def limpar():
    canvas.delete("all")

def prever():
    print("Botão Prever clicado! (ainda sem IA conectada)")

def relu(x):
    return np.maximum(0, x) 

def softmax(x):
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

def forward(X):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, W3) + b3
    a3 = softmax(z3)
    return a3

# Frame para os botões ficarem lado a lado
frame_botoes = tk.Frame(janela)
frame_botoes.pack(pady=10)

botao_prever = tk.Button(frame_botoes, text="Prever", command=prever, width=10)
botao_prever.pack(side="left", padx=5)

botao_limpar = tk.Button(frame_botoes, text="Limpar", command=limpar, width=10)
botao_limpar.pack(side="left", padx=5)


janela.mainloop()