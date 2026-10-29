"""
evaluate.py
Updated to profile quantum circuit resources and execution timelines.
"""

import argparse
import time
import json
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
    
    # Start tracking evaluation timeline
    start_eval_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)  # Outputs in [-1, 1]
            probs = (outputs + 1.0) / 2.0 
            preds = (outputs > 0.0).int() 
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    end_eval_time = time.perf_counter()
    total_eval_duration = end_eval_time - start_eval_time

    return (np.array(y_true).flatten(), 
            np.array(y_pred).flatten(), 
            np.array(y_prob).flatten(), 
            total_eval_duration)


def extract_quantum_resources(model):
    """Dynamically extracts underlying Qiskit circuit performance properties."""
    # TorchConnector wraps EstimatorQNN -> accessible via neural_network property
    qnn_circuit = model.quantum.neural_network.circuit
    
    resources = {
        "num_qubits": qnn_circuit.num_qubits,
        "circuit_depth": qnn_circuit.depth(),
        "total_gates": qnn_circuit.size(),
        "operations": dict(qnn_circuit.count_ops())
    }
    return resources


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


def plot_all(cm, fpr, tpr, roc_auc, encoding_name, total_time, avg_img_time, resources):
    # 1. Confusion Matrix Plot
    fig1, ax1 = plt.subplots()
    ax1.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Actuals', fontsize=12)
    plt.title(f'Confusion Matrix ({encoding_name.upper()})', fontsize=14)
    save_plot(fig1, f"confusion_matrix_{encoding_name}.png")
    plt.close(fig1)
    
    # 2. ROC Curve Plot
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f'Quantum Model (AUC = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({encoding_name.upper()})')
    plt.legend(loc='lower right')
    save_plot(fig2, f"roc_curve_{encoding_name}.png")
    plt.close(fig2)

    # 3. NEW: Execution & Quantum Resource Performance Plot
    fig3, (ax_time, ax_gates) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time Metric sub-bars
    time_metrics = ['Total Eval (s)', 'Per Image (ms)']
    time_values = [total_time, avg_img_time]
    colors = ['#4AA469', '#2A7BB6']
    ax_time.bar(time_metrics, time_values, color=colors, width=0.4)
    ax_time.set_title("Runtime Engine Profiles")
    ax_time.set_ylabel("Execution Scale Magnitude")
    for bar in ax_time.patches:
        ax_time.annotate(f"{bar.get_height():.2f}", 
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha='center', va='bottom', fontweight='bold')

    # Structural Circuits sub-bars
    struct_metrics = ['Qubits', 'Depth', 'Total Gates']
    struct_values = [resources['num_qubits'], resources['circuit_depth'], resources['total_gates']]
    ax_gates.bar(struct_metrics, struct_values, color='#E67E22', width=0.5)
    ax_gates.set_title("Quantum Resource Requirements")
    ax_gates.set_ylabel("Count Dimensions")
    for bar in ax_gates.patches:
        ax_gates.annotate(f"{int(bar.get_height())}", 
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f"Performance Analysis Benchmarks: {encoding_name.upper()}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig3, f"computational_profile_{encoding_name}.png")
    plt.close(fig3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid QNN Model Evaluation Engine")
    parser.add_argument(
        "--encoding", 
        type=str, 
        default="frqi", 
        choices=["frqi", "amplitude", "neqr", "qwt"],
        help="Target encoding approach variant to evaluate"
    )
    args = parser.parse_args()

    # Load evaluation dataset using the correct preprocessing protocol mapping
    X_test_raw, y_test_raw = load_dataset("dataset/test")
    X_test, y_test = preprocess_data(X_test_raw, y_test_raw, args.encoding)
    
    total_images = len(X_test)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)
    
    # Rebuild dynamic model architecture skeleton
    model = DynamicHybridQNN(image_size=8, reps=2, encoding=args.encoding)
    model.load_state_dict(torch.load(f"results/models/quantum_polyp_model_{args.encoding}.pt"))
    
    # Run structural assessment and inferences
    y_true, y_pred, y_prob, total_eval_time = evaluate_model(model, test_loader)
    quantum_resources = extract_quantum_resources(model)
    metrics, cm, fpr, tpr = calculate_metrics(y_true, y_pred, y_prob)
    
    avg_image_time_ms = (total_eval_time / total_images) * 1000

    # Consolidate complete logging stats dictionary
    metrics["eval_total_time_sec"] = total_eval_time
    metrics["eval_avg_image_time_ms"] = avg_image_time_ms
    metrics["quantum_hardware_profile"] = quantum_resources

    print(f"\n==============================================")
    print(f" PERFORMANCE PROFILE REPORT ({args.encoding.upper()})")
    print(f"==============================================")
    for k, v in metrics.items():
        if k != "quantum_hardware_profile":
            print(f" - {k:<25}: {v:.4f}")
    print(f" - Quantum Width (Qubits)  : {quantum_resources['num_qubits']}")
    print(f" - Quantum Depth           : {quantum_resources['circuit_depth']}")
    print(f" - Quantum Operation Size  : {quantum_resources['total_gates']}")
    print(f"==============================================\n")
        
    save_metrics(metrics, f"evaluation_metrics_{args.encoding}.json")
    plot_all(cm, fpr, tpr, metrics["auc"], args.encoding, total_eval_time, avg_image_time_ms, quantum_resources)