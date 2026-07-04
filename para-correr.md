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

| Package | Ficheiros Usado | Objetivo |
|---------|------------|---------|
| `torch` | `quantum_model.py`, `train.py`, `evaluate.py` | Framework PyTorch utilizado para construir o pipeline de treino e gerir os gradientes de otimização. |
| `qiskit` | `frqi.py`, `quantum_model.py` | Framework principal de computação quântica da IBM, utilizado para criar circuitos quânticos e manipular portas quânticas parametrizadas. |
| `qiskit-aer` | `quantum_model.py` | Simulador local de elevado desempenho em C++ (`AerSimulator`), utilizado para acelerar a execução de circuitos quânticos durante o treino e a inferência. |
| `qiskit-machine-learning` | `quantum_model.py` | Fornece a integração nativa (`EstimatorQNN` e `TorchConnector`) entre o Qiskit e o PyTorch. |
| `opencv-python (cv2)` | `generator.py`, `utils.py` | Utilizado para desenhar as elipses dos pólipos sintéticos e para ler e escrever imagens PNG em tons de cinzento. |
| `numpy` | Global | Utilizado para manipulação de matrizes, operações matemáticas, achatamento dos dados das imagens e conversão dos vetores de píxeis em ângulos de rotação (radianos). |
| `scikit-learn` | `evaluate.py` | Utilizado para calcular as métricas padrão de avaliação, incluindo a matriz de confusão, a curva ROC e a métrica AUC. |
| `matplotlib` | `evaluate.py` | Utilizado para gerar e guardar os gráficos de avaliação, como a matriz de confusão e a curva ROC. |

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
