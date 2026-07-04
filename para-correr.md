## Antes de correr

### 1. Versao Python e C++

* **Versao Mínima:** `Python 3.10`
* **Versao Máxima:** `Python 3.12`

Recomenda-se evitar o Python 3.13+ por enquanto.

Como o projeto utiliza o Qiskit Aer -> Garantir o Visual C++ Redistributable atualizado.
Caso haja erros de alocação no simulador, a instalação ou atualização do pacote redistribuível da Microsoft resolverá o problema, possivelmente.

---

### 2. Instalação de pacakges

Para instalar dependências necessárias de uma só vez, executar comando:

```bash
pip install torch numpy opencv-python scikit-learn matplotlib qiskit qiskit-aer qiskit-machine-learning
```

### 3. Detalhe das Dependências 

| Package | Files Used | Purpose |
|---------|------------|---------|
| `torch` | `quantum_model.py`, `train.py`, `evaluate.py` | PyTorch framework used to build the training pipeline and manage optimization gradients. |
| `qiskit` | `frqi.py`, `quantum_model.py` | IBM's main quantum computing framework used to create quantum circuits and manipulate parameterized quantum gates. |
| `qiskit-aer` | `quantum_model.py` | High-performance local C++ simulator (`AerSimulator`) used to accelerate quantum circuit execution during training and inference. |
| `qiskit-machine-learning` | `quantum_model.py` | Provides native integration (`EstimatorQNN` and `TorchConnector`) between Qiskit and PyTorch. |
| `opencv-python (cv2)` | `generator.py`, `utils.py` | Used to draw synthetic polyp ellipses and read/write grayscale PNG images. |
| `numpy` | Global | Used for array manipulation, mathematical operations, flattening image data, and converting pixel vectors into rotation angles (radians). |
| `scikit-learn` | `evaluate.py` | Used to compute standard evaluation metrics, including the confusion matrix, ROC curve, and AUC score. |
| `matplotlib` | `evaluate.py` | Used to generate and save evaluation plots such as the confusion matrix and ROC curve. |

## Fluxo

### 1. Generate

```bash
python generator.py
```
Cria duas pastas `dataset/train` e `dataset/test` com imagens 4x4
Para alterar as imagens para 8x8, ir ao ficheiro `generator.py` e alterar o `image_size` para 8

```python
...
class SyntheticDatasetGenerator:

    def __init__(
        self,
        image_size=4, # alterar este valor
        train_samples=40,
        ...
```

### 2. Train

este step demorou me umas 4 horas a correr para imagens 4x4 em 40 samples

```bash
python train.py
```
Chama automaticamente o ficheiro `utils.py` para configurar a estrutura de diretórios (criando as pastas `results/models`), normaliza as imagens convertendo os valores dos píxeis em ângulos para a representação FRQI, executa o ciclo de otimização do treino utilizando o PyTorch e o Qiskit e guarda os pesos do modelo treinado em `results/models/quantum_polyp_model.pt`.

### 3. Evaluate

```bash
python evaluate.py
```

Carrega os pesos do modelo previamente treinado, executa a inferência sobre o conjunto de teste, apresenta as métricas de avaliação e guarda os gráficos gerados diretamente na pasta `results/plots/`.
