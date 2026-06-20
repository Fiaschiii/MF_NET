import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw


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
    raio = 14
    canvas.create_oval(
        x - raio, y - raio, x + raio, y + raio,
        fill="white", outline="white"
    )

    desenho_pil.ellipse(
        [x - raio, y - raio, x + raio, y - raio],
        fill = 255
    )


canvas.bind("<B1-Motion>", desenhar)

def limpar():
    canvas.delete("all")
    desenho_pil.rectangle([0, 0, 280, 280], fill=0)

def prever():
    imagem_pequena = imagem_pil.resize((28, 28))
    array = np.array(imagem_pequena)

    array = array / 255.0
    array = array.reshape(1, 784)


    probabilidade = forward(array)
    digito_previsto = np.argmax(probabilidade)
    confianca = probabilidade[0][digito_previsto] * 100

    print(f"\nPrevisão: {digito_previsto} ({confianca:.1f}% confiança)")
    print(f"Todas as propabilidades:")
    for i in range(10):
        print(f"  {i}: {probabilidade[0][i]*100:.1f}%")

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

imagem_pil = Image.new("L", (280, 280), color=0)
desenho_pil = ImageDraw.Draw(imagem_pil)


janela.mainloop()