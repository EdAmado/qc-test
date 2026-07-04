"""
evaluate.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from quantum_model import DynamicHybridQNN
from train import preprocess_data
from utils import load_dataset, save_metrics, save_plot
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, dataloader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images) # Outputs in [-1, 1]
            
            # Normalize outputs to [0, 1] pseudo-probabilities for ROC metrics
            probs = (outputs + 1.0) / 2.0 
            preds = (outputs > 0.0).int() # Threshold at 0.0 instead of 0.5
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    return np.array(y_true).flatten(), np.array(y_pred).flatten(), np.array(y_prob).flatten()

def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc
    }
    return metrics, cm, fpr, tpr

def plot_all(cm, fpr, tpr, roc_auc):
    # Confusion Matrix Plot
    fig1, ax1 = plt.subplots()
    ax1.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Actuals', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    save_plot(fig1, "confusion_matrix.png")
    
    # ROC Curve Plot
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f'Quantum Model (AUC = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    save_plot(fig2, "roc_curve.png")

if __name__ == "__main__":
    # Load evaluation dataset
    X_test_raw, y_test_raw = load_dataset("dataset/test")
    X_test, y_test = preprocess_data(X_test_raw, y_test_raw)
    
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)
    
    # Load and build model structure
    model = DynamicHybridQNN(image_size=4, reps=2)
    model.load_state_dict(torch.load("results/models/quantum_polyp_model.pt"))
    
    y_true, y_pred, y_prob = evaluate_model(model, test_loader)
    metrics, cm, fpr, tpr = calculate_metrics(y_true, y_pred, y_prob)
    
    print("\nEvaluation Performance Metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
        
    save_metrics(metrics, "evaluation_metrics.json")
    plot_all(cm, fpr, tpr, metrics["auc"])