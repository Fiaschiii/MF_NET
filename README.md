# MFNet 🧠

Uma rede neural construída **inteiramente do zero**, em Python puro com NumPy — sem PyTorch, TensorFlow ou qualquer outra biblioteca de Machine Learning.

O projeto reconhece dígitos escritos à mão (0-9) através de uma interface gráfica onde você desenha com o mouse e a rede prevê o número em tempo real.

## ✨ Demonstração

```
Desenha um dígito → MFNet processa → Previsão com % de confiança
```

## 📦 O que foi construído

O projeto foi dividido em 3 etapas, cada uma construindo sobre a anterior:

### Etapa 1 — Fundamentos (`main.py`)
Implementação de uma rede neural do zero para resolver o problema **XOR**, clássico por não ser linearmente separável.

- Forward pass (propagação direta)
- Funções de ativação: ReLU e Sigmoid
- Função de perda: Binary Cross-Entropy
- Backpropagation (regra da cadeia, calculada manualmente)
- Gradient Descent
- **Resultado:** 100% de acurácia

### Etapa 2 — Dados reais (`baixar_mnist.py`, `rede_mnist.py`)
Treinamento em dados reais com o dataset **MNIST** (70.000 imagens de dígitos manuscritos).

- Download e parsing do dataset original (formato binário `.gz`)
- Normalização dos pixels (0-255 → 0-1)
- One-Hot Encoding dos labels
- Arquitetura com 2 camadas escondidas: `784 → 128 → 64 → 10`
- Ativação Softmax na saída (classificação multi-classe)
- Treino com mini-batches
- **Resultado:** 96.73% de acurácia em 10.000 imagens nunca vistas

### Etapa 3 — Interface interativa (`interface.py`)
Aplicação gráfica com `tkinter` onde a MFNet treinada reconhece dígitos desenhados ao vivo.

- Canvas de desenho capturado via mouse
- Conversão da imagem (280×280 → 28×28) com Pillow
- Inferência em tempo real usando os pesos salvos da Etapa 2
- Exibição da previsão com nível de confiança

## 🏗️ Arquitetura da rede (MNIST)

```
Entrada (784 pixels)
       ↓
Camada oculta 1 (128 neurônios) — ReLU
       ↓
Camada oculta 2 (64 neurônios) — ReLU
       ↓
Saída (10 neurônios) — Softmax
```

## 🚀 Como rodar

### Pré-requisitos
- Python 3.10+

### Instalação

```bash
git clone https://github.com/Fiaschiii/MF_NET.git
cd MF_NET
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Executando cada etapa

**Etapa 1 — Rede neural do zero (XOR):**
```bash
python main.py
```

**Etapa 2 — Treinar no MNIST:**
```bash
python baixar_mnist.py    # baixa e prepara os dados
python rede_mnist.py      # treina a rede e salva os pesos (.npy)
```

**Etapa 3 — Interface gráfica:**
```bash
python interface.py
```
> ⚠️ É necessário rodar a Etapa 2 antes — a interface carrega os pesos salvos (`W1.npy`, `W2.npy`, `W3.npy`, etc).

## 🛠️ Tecnologias

| Ferramenta | Uso |
|---|---|
| `numpy` | Toda a matemática da rede neural (forward, backward, gradientes) |
| `Pillow` | Conversão da imagem desenhada para o formato de entrada da rede |
| `tkinter` | Interface gráfica (nativo do Python) |

Nenhuma biblioteca de Machine Learning (PyTorch, TensorFlow, scikit-learn) foi utilizada na construção da rede — todo o forward pass, backpropagation e gradient descent foram implementados manualmente.

## 📚 Conceitos aplicados

- Álgebra linear (produto matricial, transposição)
- Funções de ativação (ReLU, Sigmoid, Softmax)
- Funções de perda (Binary Cross-Entropy, Categorical Cross-Entropy)
- Backpropagation e regra da cadeia
- Gradient Descent com mini-batches
- Inicialização de pesos (Xavier/Glorot)
- One-Hot Encoding
- Normalização de dados

## 💡 Dica de uso

Para melhores resultados na Etapa 3, desenhe o dígito:
- **Grande**, ocupando boa parte do canvas
- Com **traço grosso**
- **Centralizado**

Isso aproxima o desenho do estilo das imagens usadas no treino (MNIST), aumentando a confiança da previsão.

## 👤 Autor

Desenvolvido por **Miguel Fiaschi** como projeto de estudo prático sobre redes neurais, construído peça por peça para entender cada conceito por trás do Machine Learning moderno.

---

*Projeto educacional — construído do zero para aprender, não para produção.*
