import tkinter as tk

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

# Frame para os botões ficarem lado a lado
frame_botoes = tk.Frame(janela)
frame_botoes.pack(pady=10)

botao_prever = tk.Button(frame_botoes, text="Prever", command=prever, width=10)
botao_prever.pack(side="left", padx=5)

botao_limpar = tk.Button(frame_botoes, text="Limpar", command=limpar, width=10)
botao_limpar.pack(side="left", padx=5)


janela.mainloop()