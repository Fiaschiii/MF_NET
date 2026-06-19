import tkinter as tk

janela = tk.Tk()
janela.title("MFNet - Reconhecedor de Dígitos")

canvas = tk.Canvas(janela, width=280, height=280, bg="black")
canvas.pack(pady=10)

def desenhar(event):
    print(f"Mouse em: {event.x}, {event.y}")  # debug
    x, y = event.x, event.y
    raio = 8
    canvas.create_oval(
        x - raio, y - raio, x + raio, y + raio,
        fill="white", outline="white"
    )

canvas.bind("<B1-Motion>", desenhar)

janela.mainloop()