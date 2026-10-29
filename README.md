# Before Running

## 1. Python and C++ Versions

- **Minimum version:** `Python 3.10`
- **Maximum version:** `Python 3.12`

It is recommended to avoid Python 3.13+ for now.

Since the project uses Qiskit Aer, make sure the Visual C++ Redistributable is up to date.

If allocation errors occur in the simulator, installing or updating Microsoft's redistributable package may resolve the issue.

---

## 2. Installing Packages

To install all the required dependencies at once, run the following command:

```bash
pip install torch numpy opencv-python scikit-learn matplotlib qiskit qiskit-aer qiskit-machine-learning
```

## 3. Dependency Details

| Package | Files Used | Purpose |
|---|---|---|
| `torch` | `quantum_model.py`, `train.py`, `evaluate.py` | PyTorch framework used to build the training pipeline and manage optimization gradients. |
| `qiskit` | `frqi.py`, `quantum_model.py` | IBM's main quantum computing framework, used to create quantum circuits and manipulate parameterized quantum gates. |
| `qiskit-aer` | `quantum_model.py` | High-performance local C++ simulator (`AerSimulator`), used to accelerate quantum circuit execution during training and inference. |
| `qiskit-machine-learning` | `quantum_model.py` | Provides native integration (`EstimatorQNN` and `TorchConnector`) between Qiskit and PyTorch. |
| `opencv-python (cv2)` | `generator.py`, `utils.py` | Used to draw synthetic polyp ellipses and read and write grayscale PNG images. |
| `numpy` | Global | Used for matrix manipulation, mathematical operations, flattening image data, and converting pixel vectors into rotation angles (radians). |
| `scikit-learn` | `evaluate.py` | Used to calculate standard evaluation metrics, including the confusion matrix, ROC curve, and AUC metric. |
| `matplotlib` | `evaluate.py` | Used to generate and save evaluation plots, such as the confusion matrix and ROC curve. |

---

# Workflow

## 1. Generate

Run:

```bash
python generator.py
```

This creates two folders, `dataset/train` and `dataset/test`, containing 4×4 images.

To change the image size to 8×8, open `generator.py` and change the value of `image_size` to `8`.

For example:

```python
...
class SyntheticDatasetGenerator:

    def __init__(
        self,
        image_size=4,  # Change this value
        train_samples=40,
        ...
```

## 2. Train

This step took me approximately 4 hours to run with 4×4 images and 40 samples.

Run:

```bash
python train.py
```

This automatically calls `utils.py` to configure the directory structure, creating the `results/models` folders.

It then normalizes the images by converting pixel values into angles for the FRQI representation, runs the training optimization loop using PyTorch and Qiskit, and saves the trained model weights to:

`results/models/quantum_polyp_model.pt`

## 3. Evaluate

Run:

```bash
python evaluate.py
```

This loads the previously trained model weights, performs inference on the test dataset, displays the evaluation metrics, and saves the generated plots directly to the `results/plots/` folder.
